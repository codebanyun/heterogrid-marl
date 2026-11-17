"""Exact DIB teacher objective with only its consumed BiGRU centre states.

The native teacher consumes the centre output of a one-layer BiGRU. Forward
states after the centre and reverse states before it cannot influence that
output, so they need not be evaluated. No time horizon, target or gradient to
the centre is removed. Original parameters and state-dict names are retained.
"""
import math
import torch
import torch.nn.functional as F


class SelfDistances(torch.autograd.Function):
    """Native self-cdist values, accumulate its two tied-input adjoints once.

    Native cdist(y, y) allocates two full reconstruction-sized gradients. Each
    unordered pair can instead add the same (g_ij + g_ji) adjoint directly to
    the two endpoints, preserving the zero-distance subgradient of cdist.
    """
    @staticmethod
    def forward(ctx, values):
        distances = torch.cdist(values, values, p=2)
        ctx.save_for_backward(values, distances)
        return distances

    @staticmethod
    def backward(ctx, gradient):
        values, distances = ctx.saved_tensors
        result = torch.zeros_like(values)
        for i in range(values.shape[1]):
            for j in range(i + 1, values.shape[1]):
                distance = distances[:, i, j]
                positive = distance > 0
                denominator = torch.where(positive, distance, torch.ones_like(distance))
                weight = ((gradient[:, i, j] + gradient[:, j, i]) / denominator) * positive
                delta = (values[:, i] - values[:, j]) * weight.unsqueeze(-1)
                result[:, i] += delta
                result[:, j] -= delta
        return result


def mhf_losses(agent, aux, obs_t, next_obs, actions_t, y, valid):
    pi, weights = aux["pi"].clamp_min(1e-8), aux["weights"].clamp_min(1e-8)
    error = agent._safe_error(aux["y_hat_k"], y.unsqueeze(-2)).mean(-1)
    sigma = max(getattr(agent.args, "dib_target_sigma", getattr(agent.args, "dib_mix_temperature", 1.)), 1e-6)
    mixture = agent._masked_mean(-torch.logsumexp(torch.log(pi) - error / (2. * sigma * sigma), -1), valid)
    if agent.n_hypotheses > 1:
        flat = aux["y_hat_k"].reshape(-1, agent.n_hypotheses, agent.y_dim)
        distance = agent._finite(SelfDistances.apply(flat))
        eye = torch.eye(agent.n_hypotheses, device=flat.device, dtype=torch.bool).unsqueeze(0)
        separation = F.relu(getattr(agent.args, "dib_sep_margin", 1.) - distance).pow(2)
        separation = separation.masked_fill(eye, 0.).sum((1, 2)) / (agent.n_hypotheses * (agent.n_hypotheses - 1))
        separation_loss = agent._masked_mean(separation.view_as(valid), valid)
    else:
        separation_loss = aux["y_hat_k"].sum() * 0.
    entropy_loss = -agent._masked_mean(-(pi * torch.log(pi)).sum(-1), valid)
    prediction = agent.predict_next_obs(
        obs_t.unsqueeze(-2).expand(-1, -1, -1, agent.n_hypotheses, -1),
        actions_t.unsqueeze(-2).expand(-1, -1, -1, agent.n_hypotheses, -1), aux["z"])
    error_next = agent._safe_error(prediction, next_obs.unsqueeze(-2)).mean(-1)
    sigma_o = max(getattr(agent.args, "dib_obs_sigma", 1.), 1e-6)
    forward = agent._masked_mean(-torch.logsumexp(torch.log(weights) - error_next / (2. * sigma_o * sigma_o), -1), valid)
    hyp = mixture + getattr(agent.args, "dib_lambda_sep", .02) * separation_loss
    return {"hyp_loss": hyp, "hyp_mixture_loss": mixture, "hyp_separation_loss": separation_loss,
        "mhf_loss": hyp + getattr(agent.args, "dib_lambda_fwd_internal", 1.) * forward,
        "mhf_entropy_loss": entropy_loss, "fwd_loss": forward, "mhf_forward_loss": forward}


def history_context(agent, history, mask, obs, last_action, prev_obs, has_prev, query, keys, values):
    """Native AHS equations with shared per-token Q/K/V linear projections."""
    sequence, _ = agent.ahs_gru(history)
    sequence = agent._finite(sequence)
    minimum = mask[:, -agent.l_min:].unsqueeze(-1)
    pool = (sequence[:, -agent.l_min:] * minimum).sum(1) / minimum.sum(1).clamp_min(1.)
    probe_mu, probe_logstd = agent.probe_head(pool).chunk(2, -1)
    probe_mu = agent._finite(probe_mu)
    probe_logstd = agent._finite(probe_logstd, clip=5.).clamp(-5., 2.)
    entropy = (.5 * (math.log(2. * math.pi * math.e) + 2. * probe_logstd)).mean(-1, keepdim=True)
    if getattr(agent.args, "dib_robust_entities", False):
        from algorithms.dib_mappo.robust import probe_entropy
        probe_logstd = probe_logstd.clamp(-2., 1.)
        entropy = probe_entropy(agent, probe_mu, probe_logstd)
    prediction = agent._finite(agent.transition_predictor(torch.cat((agent.obs_norm(prev_obs), last_action), -1)))
    target = agent._finite(agent.obs_norm(obs))
    error = agent._safe_error(prediction, target).mean(-1, keepdim=True) * has_prev
    if getattr(agent.args, "dib_robust_entities", False):
        from algorithms.dib_mappo.robust import local_normalized_error
        error = local_normalized_error(agent,prediction,target,obs).unsqueeze(-1)*has_prev
    ambiguity = agent._finite(agent.ambiguity_norm(torch.cat((entropy.detach(), error.detach()), -1)))
    rho_unit = torch.sigmoid(agent.boundary_mlp(torch.cat((sequence[:, -1], ambiguity), -1)))
    rho = agent.l_min + (agent.l_max - agent.l_min) * rho_unit
    distance = agent.history_distances.to(history.device).view(1, agent.l_max)
    soft_mask = torch.sigmoid((rho - distance - .5) / agent.kappa) * mask
    scores = (query.unsqueeze(1) * keys).sum(-1) / math.sqrt(agent.ahs_hidden_dim)
    bias = agent.relative_bias[agent.history_distances.to(history.device).long()].view(1, agent.l_max)
    scores = scores + bias
    scores = scores + torch.log(soft_mask.clamp_min(1e-6))
    alpha = torch.nan_to_num(torch.softmax(scores, -1), nan=0., posinf=0., neginf=0.)
    alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-8)
    context = agent._finite((alpha.unsqueeze(-1) * values).sum(1))
    return context, {"probe_mu": probe_mu, "probe_logstd": probe_logstd,
        "transition_pred": prediction, "transition_target": target, "rho": rho, "soft_mask": soft_mask}


def teacher_centre(agent, tokens):
    b, t, n, e = tokens.shape
    kb = getattr(agent.args, "dib_teacher_kb", 8)
    kf = getattr(agent.args, "dib_teacher_kf", 4)
    sequence = tokens.permute(0, 2, 1, 3).reshape(b * n, t, e)
    padded = F.pad(sequence.transpose(1, 2), (kb, kf), mode="replicate")
    windows = padded.unfold(2, kb + kf + 1, 1).permute(0, 2, 3, 1).reshape(
        b * n * t, kb + kf + 1, e)
    gru = agent.teacher_gru
    initial = windows.new_zeros(1, windows.shape[0], gru.hidden_size)
    def direction(inputs, suffix):
        weights = [getattr(gru, key + suffix) for key in
                   ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
        output, hidden = torch._VF.gru(inputs, initial, weights, True, 1, 0., gru.training, False, True)
        return output[:, -1]
    forward = direction(windows[:, :kb + 1], "")
    reverse = direction(windows[:, kb:].flip(1), "_reverse")
    return torch.cat((forward, reverse), dim=-1)


def teacher_losses(agent, aux, obs_t, next_obs, actions_t, prev_actions, y, valid):
    # Equations and returned logging aliases match the native implementation.
    b, t, n, _ = aux["x"].shape
    stats = agent._finite(agent.teacher_head(teacher_centre(agent, aux["x"])))
    mu, logstd = stats.chunk(2, dim=-1)
    mu = agent._finite(mu, clip=agent.latent_clip)
    logstd = agent._finite(logstd, clip=5.).clamp(-5., 2.)
    mu = mu.view(b, n, t, agent.z_dim).permute(0, 2, 1, 3)
    logstd = logstd.view(b, n, t, agent.z_dim).permute(0, 2, 1, 3)
    teacher_recon_loss = agent._masked_mean(agent._safe_error(agent.belief_decoder(mu), y).mean(-1), valid)
    prior_kl = agent._normal_kl(mu, logstd, torch.zeros_like(mu), torch.zeros_like(logstd)).mean(-1)
    prior_loss = agent._masked_mean(prior_kl, valid)
    # The native actor already computed this same decoder output. Reuse its
    # graph, including decoder-weight and student gradients, not a detach.
    student_recon = agent._finite(aux["y_hat_s"])
    consistency = agent._safe_error(student_recon, agent.belief_decoder(mu.detach())).mean(-1)
    consistency_loss = agent._masked_mean(consistency, valid)
    kl = agent._normal_kl(mu.detach(), logstd.detach(), aux["student_mu"], aux["student_logstd"]).mean(-1)
    kl_loss = agent._masked_mean(kl, valid)
    norm_obs = agent.obs_norm(obs_t)
    pred_next = agent._finite(agent.teacher_forward_dynamics(torch.cat((mu, norm_obs, actions_t), -1)))
    pred_prev = agent._finite(agent.teacher_backward_dynamics(torch.cat((mu, norm_obs, prev_actions), -1)))
    previous = agent.obs_norm(torch.cat((obs_t[:, :1], obs_t[:, :-1]), 1))
    following = agent.obs_norm(next_obs)
    previous_mask = torch.ones_like(valid)
    previous_mask[:, 0] = 0.
    dynamics_loss = (agent._masked_mean(agent._safe_error(pred_prev, previous).mean(-1), valid * previous_mask)
                     + agent._masked_mean(agent._safe_error(pred_next, following).mean(-1), valid))
    args = agent.args
    teacher = (teacher_recon_loss
        + getattr(args, "dib_teacher_dyn_coef", getattr(args, "dib_cbd_dyn_coef", .1)) * dynamics_loss
        + getattr(args, "dib_teacher_prior_beta", .001) * prior_loss)
    distill = (getattr(args, "dib_distill_kl_coef", getattr(args, "dib_cbd_kl_coef", 1.)) * kl_loss
        + getattr(args, "dib_distill_output_coef", getattr(args, "dib_cbd_output_coef", .5)) * consistency_loss)
    cbd = teacher + getattr(args, "dib_cbd_kl_coef", 1.) * kl_loss + getattr(args, "dib_cbd_output_coef", .5) * consistency_loss
    return {"teacher_loss": teacher, "teacher_recon_loss": teacher_recon_loss,
        "teacher_dyn_loss": dynamics_loss, "teacher_prior_kl_loss": prior_loss,
        "distill_loss": distill, "distill_kl_loss": kl_loss,
        "distill_output_consistency_loss": consistency_loss, "cbd_loss": cbd,
        "cbd_teacher_recon_loss": teacher_recon_loss, "cbd_teacher_dyn_loss": dynamics_loss,
        "cbd_teacher_prior_kl_loss": prior_loss, "cbd_distill_kl_loss": kl_loss,
        "cbd_output_consistency_loss": consistency_loss}


def install(agent):
    from types import MethodType
    gru = agent.teacher_gru
    if gru.num_layers != 1 or not gru.bidirectional or not gru.bias or gru.dropout != 0:
        raise ValueError("Centre-state optimization requires the native one-layer DIB v2 teacher")
    agent._compute_teacher_losses = MethodType(teacher_losses, agent)
    agent._compute_mhf_losses = MethodType(mhf_losses, agent)
