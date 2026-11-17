"""Batch-vectorize native DIB history windows during learner replay, not execution.

All layers and helper equations are the external agent's. Recurrent backbone
states remain causal; other-agent/future context still enters the native
training-only auxiliary loss, never the deployed actor.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def hypotheses_without_decode(agent, context, mu, logstd):
    """Native hypothesis equations; omit the unused reconstruction in old-MAC."""
    if getattr(agent.args,'dib_tactical',False):
        return agent._mhf_forward(context,mu,logstd,decode=False)
    belief = torch.cat((context, mu, 2. * logstd), dim=-1)
    belief = belief.unsqueeze(1).expand(-1, agent.n_hypotheses, -1)
    queries = agent.hypothesis_queries.unsqueeze(0).expand(mu.shape[0], -1, -1)
    trunk = agent._finite(agent.hypothesis_trunk(torch.cat((belief, queries), dim=-1)))
    delta = agent._finite(agent.hypothesis_delta(trunk), clip=agent.latent_clip)
    gates = agent._finite(agent.hypothesis_gate(trunk).squeeze(-1), clip=20.)
    z = agent._finite(mu.unsqueeze(1) + torch.exp(logstd).unsqueeze(1) * torch.tanh(delta),
                      clip=agent.latent_clip)
    pi = torch.nan_to_num(torch.softmax(gates, dim=-1), nan=0., posinf=0., neginf=0.)
    return z, agent._normalise_weights(pi)


def replay_sequence(mac, batch, auxiliary):
    agent, args = mac.agent, mac.args
    obs = batch["obs"][:, :-1].float()
    b, t, n, o = obs.shape
    bn = b * n
    def flatten(value):
        return value.transpose(1, 2).reshape(bn * t, *value.shape[3:])
    def unflatten(value):
        return value.reshape(b, n, t, *value.shape[1:]).transpose(1, 2)
    obs_f = agent._finite(flatten(obs))
    previous = torch.cat([torch.zeros_like(batch["actions_onehot"][:, :1]),
                          batch["actions_onehot"][:, :t - 1]], dim=1)
    last_action = agent._finite(flatten(previous))
    ids = torch.eye(n, device=obs.device, dtype=obs.dtype)[None, :, None].expand(b, n, t, n).reshape(bn * t, n)
    if getattr(args,'dib_tactical',False):
        base_x=agent.encode_local(obs_f,last_action,ids).reshape(bn,t,agent.hidden_dim)
    else:
        base_inputs = torch.cat([obs_f, ids], dim=-1) if agent.use_agent_id else obs_f
        base_x = F.relu(agent.base_policy_encoder(base_inputs)).reshape(bn, t, agent.hidden_dim)
    def recurrent(inputs, cell):
        # GRUCell and a single-layer GRU use the same recurrence/parameters.
        # The sequence kernel batches input projections and avoids T separate
        # Python/autograd dispatches. Keep the native per-step finite recovery
        # path for exceptional non-finite trajectories.
        initial = inputs.new_zeros(1, bn, agent.hidden_dim)
        weights = [cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh]
        output, _ = torch._VF.gru(inputs, initial, weights, True, 1, 0., cell.training, False, True)
        if bool(torch.isfinite(output).all()):
            return output.reshape(bn * t, agent.hidden_dim)
        hidden = inputs.new_zeros(bn, agent.hidden_dim)
        values = []
        for value in inputs.unbind(1):
            hidden = agent._finite(cell(value, hidden))
            values.append(hidden)
        return torch.stack(values, dim=1).reshape(bn * t, agent.hidden_dim)
    base_h = recurrent(base_x, agent.base_policy_gru)
    base_logits = (agent.action_logits(base_h,obs_f) if getattr(args,'dib_tactical',False)
                   else agent.base_actor_head(base_h)).unsqueeze(1)
    warmup = (getattr(mac, "_learning_clock", 0) < args.dib_warmup_steps and args.dib_stage == "auto")
    if warmup and agent.residual_actor_gate == 0:
        # Auxiliary objectives are disabled and the zero-gated branch has no
        # influence on task gradients. Its zero-initialized output stays zero.
        # Preserve Adam's per-parameter clocks: native zero-gated parameters
        # receive zero gradients, not None. Discover graph connectivity with
        # one native step once, rather than changing their optimizer history.
        if auxiliary and torch.is_grad_enabled():
            if not hasattr(mac, "_warmup_connected"):
                parameters = tuple(p for p in agent.parameters() if p.requires_grad)
                probe, _ = agent(obs[:1, 0], previous[:1, 0], agent.init_hidden(1))
                gradients = torch.autograd.grad(probe["logits_k"].sum() + probe["weights"].sum(),
                                                parameters, allow_unused=True)
                mac._warmup_connected = tuple(p for p, g in zip(parameters, gradients) if g is not None)
            anchor = sum(p.reshape(-1)[0] * 0 for p in mac._warmup_connected)
            base_logits = base_logits + anchor
        residual_logits = base_logits.new_zeros(bn * t, agent.n_hypotheses, agent.n_actions)
        logits = agent._finite(base_logits.expand_as(residual_logits), clip=args.dib_logit_clip)
        weights = base_logits.new_full((bn * t, agent.n_hypotheses), 1 / agent.n_hypotheses)
        aux = {"base_logits": unflatten(base_logits), "residual_logits": unflatten(residual_logits)}
    else:
        pieces = [obs_f, last_action] + ([ids] if agent.use_agent_id else [])
        x_f = agent._finite(agent.obs_action_encoder(torch.cat(pieces, dim=-1)))
        x = x_f.reshape(bn, t, agent.embed_dim)
        policy_h = recurrent(x, agent.policy_gru)
        padded = torch.cat([x.new_zeros(bn, agent.l_max - 1, agent.embed_dim), x], dim=1)
        windows = padded.unfold(1, agent.l_max, 1).permute(0, 1, 3, 2).reshape(
            bn * t, agent.l_max, agent.embed_dim)
        mask = (agent.history_distances[None, :] <= torch.arange(t, device=obs.device)[:, None]).to(x.dtype)
        mask = mask[None].expand(bn, -1, -1).reshape(bn * t, agent.l_max)
        obs_seq = obs_f.reshape(bn, t, o)
        prev_obs = torch.cat([obs_seq.new_zeros(bn, 1, o), obs_seq[:, :-1]], dim=1).reshape(bn * t, o)
        has_prev = (torch.arange(t, device=obs.device) > 0).to(x.dtype)[None, :, None].expand(bn, -1, -1).reshape(bn * t, 1)
        if getattr(args,'dib_tactical',False):
            states=policy_h.reshape(bn,t,agent.hidden_dim)
            padded_states=torch.cat((states.new_zeros(bn,agent.l_max-1,agent.hidden_dim),states),1)
            history=padded_states.unfold(1,agent.l_max,1).permute(0,1,3,2).reshape(bn*t,agent.l_max,agent.hidden_dim)
            context,ahs=agent.streaming_context(history,mask,obs_f,last_action,prev_obs,has_prev)
        elif agent.use_ahs:
            from algorithms.dib_mappo.auxiliary import history_context
            query = agent.query_proj(x_f)
            def projected_windows(layer):
                projected = layer(padded)
                return projected.unfold(1, agent.l_max, 1).permute(0, 1, 3, 2).reshape(
                    bn * t, agent.l_max, agent.ahs_hidden_dim)
            key_windows = projected_windows(agent.key_proj)
            value_windows = projected_windows(agent.value_proj)
            # Recompute native AHS activations in backward instead of retaining
            # every 16-token window's GRU graph. Bounded tiles also keep CPU
            # matrix products/cache use independent of the episode length.
            keys = ("probe_mu", "probe_logstd", "transition_pred", "transition_target", "rho", "soft_mask")
            def history_forward(*inputs):
                context, values = history_context(agent, *inputs)
                return (context,) + tuple(values[key] for key in keys)
            outputs = []
            tiles = zip(*(value.split(256, dim=0) for value in
                          (windows, mask, obs_f, last_action, prev_obs, has_prev,
                           query, key_windows, value_windows)))
            for inputs in tiles:
                outputs.append(checkpoint(history_forward, *inputs) if torch.is_grad_enabled()
                               else history_forward(*inputs))
            combined = [torch.cat(values, dim=0) for values in zip(*outputs)]
            context, ahs = combined[0], dict(zip(keys, combined[1:]))
        else:
            context, ahs = agent._fixed_history_forward(windows, mask, obs_f, prev_obs, has_prev)
        mu, logstd = agent._student_forward(context)
        if agent.use_mhf:
            if auxiliary and not getattr(args,'dib_tactical',False):
                z, prior, y_k = agent._mhf_forward(context, mu, logstd)
            else:
                z, prior = hypotheses_without_decode(agent, context, mu, logstd)
            weights, filter_error = agent._completed_transition_filter(prior, z, prev_obs, obs_f, last_action, has_prev)
        else:
            z = mu.unsqueeze(1)
            prior = mu.new_ones(mu.shape[0], 1)
            if auxiliary:
                y_k = agent._finite(agent.belief_decoder(z))
            weights, filter_error = prior, torch.zeros_like(prior)
        actor_h = policy_h.unsqueeze(1).expand(-1, agent.n_hypotheses, -1)
        actor_c = context.unsqueeze(1).expand(-1, agent.n_hypotheses, -1)
        residual_logits = agent.actor_head(torch.cat([actor_h, actor_c, z], dim=-1))
        logits = agent._finite(base_logits + agent.residual_actor_scale * agent.residual_actor_gate * residual_logits,
                               clip=args.dib_logit_clip)
        trust = None
        if getattr(args, "dib_robust_entities", False):
            from algorithms.dib_mappo.robust import compose_logits
            logits,residual_logits,trust = compose_logits(agent,base_logits,residual_logits,base_h,context)
        aux = {"x": unflatten(x_f), "c": unflatten(context),
            "probe_mu": unflatten(ahs["probe_mu"]), "probe_logstd": unflatten(ahs["probe_logstd"]),
            "transition_pred": unflatten(ahs["transition_pred"]), "transition_target": unflatten(ahs["transition_target"]),
            "transition_has_prev": unflatten(has_prev), "rho": unflatten(ahs["rho"]),
            "soft_mask": unflatten(ahs["soft_mask"]), "student_mu": unflatten(mu), "student_logstd": unflatten(logstd),
            "z": unflatten(z), "pi": unflatten(prior), "weights": unflatten(weights),
            "filter_error": unflatten(filter_error), "base_logits": unflatten(base_logits),
            "residual_logits": unflatten(residual_logits), "logits_k": unflatten(logits)} if auxiliary else None
        if auxiliary and not getattr(args,'dib_tactical',False):
            aux['y_hat_s']=unflatten(agent.belief_decoder(mu));aux['y_hat_k']=unflatten(y_k)
        if auxiliary and trust is not None:
            aux['belief_trust'] = unflatten(trust)
    logits, weights = unflatten(logits), unflatten(weights)
    available = batch["avail_actions"][:, :t].float()
    logits = torch.nan_to_num(logits, nan=0., posinf=1e10, neginf=-1e10)
    probability_k = torch.softmax(logits.masked_fill(available.unsqueeze(3) == 0, -1e10), dim=-1)
    probability_k = torch.nan_to_num(probability_k, nan=0., posinf=1., neginf=0.)
    probabilities = (weights.unsqueeze(-1) * probability_k).sum(dim=3) * available
    probabilities = torch.nan_to_num(probabilities, nan=0., posinf=1., neginf=0.)
    mass = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(mass > 1e-8, probabilities / mass.clamp_min(1e-8),
                                available / available.sum(dim=-1, keepdim=True).clamp_min(1.))
    if auxiliary:
        aux["action_prob_k"], aux["action_probs"] = probability_k, probabilities
    else:
        aux = None
    return probabilities, aux


def install(mac):
    original_init = mac.init_hidden
    original_forward = mac.forward
    original_auxiliary = mac.compute_auxiliary_losses
    def init_hidden(batch_size):
        original_init(batch_size)
        mac._sequence_cache = None
        mac._sequence_auxiliary = None
    def forward(batch, t, test_mode=False, return_aux=False):
        if not getattr(mac, "_replay_enabled", False):
            return original_forward(batch, t, test_mode=test_mode, return_aux=return_aux)
        if mac._sequence_cache is None:
            probabilities, auxiliary = replay_sequence(mac, batch, return_aux)
            mac._sequence_auxiliary = auxiliary
            # One UnbindBackward per tensor, not T independent SliceBackward
            # nodes each materializing an entire zero-filled sequence gradient.
            mac._sequence_cache = (probabilities.unbind(1),
                {key: value.unbind(1) for key, value in auxiliary.items()} if auxiliary is not None else None)
        probabilities, auxiliary = mac._sequence_cache
        if return_aux:
            return probabilities[t], {key: value[t] for key, value in auxiliary.items()}
        return probabilities[t]
    def auxiliary_losses(aux_steps, batch, mask):
        if not getattr(mac, "_replay_enabled", False) or mac._sequence_auxiliary is None:
            return original_auxiliary(aux_steps, batch, mask)
        # The native MAC restacks every timestep of every auxiliary tensor.
        # These are already the same complete tensors in the replay cache.
        return mac.agent.compute_auxiliary_losses(mac._sequence_auxiliary,
            batch["obs"], batch["actions_onehot"], mask)
    mac.init_hidden, mac.forward = init_hidden, forward
    mac.compute_auxiliary_losses = auxiliary_losses
    mac._sequence_cache = None
    mac._sequence_auxiliary = None


def learn(policy, batch, clock, episodes):
    for mac in (policy.mac, policy.learner.old_mac):
        mac._learning_clock = clock
        mac._replay_enabled = True
    try:
        policy.learner.train(batch, clock, episodes)
    finally:
        for mac in (policy.mac, policy.learner.old_mac):
            mac._sequence_cache = None
            mac._sequence_auxiliary = None
            mac._replay_enabled = False
