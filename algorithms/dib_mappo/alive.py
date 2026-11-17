"""HeteroGrid active-agent masks around the native DIB PPO update.

The train body is adapted from DIB-MAPPO/src/learners/dib_ppo_learner.py
(SHA256 c67b4ead241cdd3ce66d2b97691e06b3ecf9ace398eb49673ca1f268c31ce237).
Only actor/auxiliary validity and their diagnostics change. In particular,
team critic targets, terminal handling, PPO clipping/KL, auxiliary equations,
optimizers and target-update clocks remain upstream-defined.
"""
from types import MethodType

import torch as th


def alive_at_action(batch, length):
    """Own HP is local information. HeteroGrid puts normalised HP at index 3.

    The mask refers to the pre-action observation: an action that kills its
    actor is still learned. Padding is handled separately by filled/terminal.
    This adapter must not be installed on arbitrary LBF/SMAC feature layouts.
    """
    return (batch["obs"][:, :length, :, 3] > 0).to(dtype=th.float32)


def train(self, batch, t_env, episode_num):
    rewards = batch["reward"][:, :-1]
    actions = batch["actions"][:, :-1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

    if self.args.standardise_rewards:
        self.rew_ms.update(rewards)
        rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

    if self.args.common_reward:
        assert rewards.size(2) == 1, "Expected singular agent dimension for common rewards"
        rewards = rewards.expand(-1, -1, self.n_agents)

    if mask.sum() == 0:
        self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
        self.logger.console_logger.error(
            "DIB MAPPO Learner: mask.sum() == 0 at t_env {}".format(t_env)
        )
        return

    # Death is not a terminal for the shared team value. Keep all team
    # reward/n-step targets; exclude inactive agents only from actor/auxiliary
    # losses, advantage normalisation and policy diagnostics.
    critic_mask = mask.repeat(1, 1, self.n_agents)
    agent_mask = critic_mask * alive_at_action(batch, critic_mask.size(1))
    active_count = agent_mask.sum()
    self.logger.log_stat("active_actor_fraction", (active_count / critic_mask.sum().clamp_min(1)).item(), t_env)
    self.logger.log_stat("active_actor_samples", active_count.item(), t_env)
    self.logger.log_stat("dead_actor_samples", (critic_mask.sum() - active_count).item(), t_env)
    if active_count.item() == 0:
        self.logger.log_stat("empty_active_batch", 1.0, t_env)
        return

    actor_gate = self.mac.set_residual_actor_gate_for_t(t_env)
    self.old_mac.set_residual_actor_gate(actor_gate)
    entropy_coef = self._entropy_coef(t_env)

    with th.no_grad():
        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            old_mac_out.append(self.old_mac.forward(batch, t=t))
        old_pi = th.stack(old_mac_out, dim=1)
        old_pi = old_pi.clone()
        old_pi[agent_mask == 0] = 1.0
        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

    stage = self._get_stage(t_env)
    epoch_stats = {
        "pg_loss": [],
        "belief_loss_raw": [],
        "belief_loss_scheduled": [],
        "belief_loss": [],
        "belief_cap_scale": [],
        "agent_loss": [],
        "agent_grad_norm": [],
        "ppo_ratio_mean": [],
        "ppo_clip_fraction": [],
        "ppo_approx_kl": [],
        "advantage_raw_mean": [],
        "advantage_raw_std": [],
    }
    critic_stats_all = None
    skipped_agent_update = False
    aux_losses = None
    advantages = None
    pi = None
    epochs_completed = 0
    ppo_early_stop = False
    base_logit_abs = 0.0
    residual_logit_abs = 0.0

    for epoch_index in range(self.args.epochs):
        mac_out = []
        aux_steps = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs, aux = self.mac.forward(batch, t=t, return_aux=True)
            mac_out.append(agent_outs)
            aux_steps.append(aux)
        pi = th.stack(mac_out, dim=1)
        pi = pi.clone()
        pi[agent_mask == 0] = 1.0
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)
        log_ratio = (log_pi_taken - old_log_pi_taken).clamp(-20.0, 20.0)
        ratios = th.exp(log_ratio)
        valid_count = agent_mask.sum()
        clip_fraction = (
            ((ratios - 1.0).abs() > self.args.eps_clip).float() * agent_mask
        ).sum() / valid_count
        approx_kl = (((ratios - 1.0) - log_ratio) * agent_mask).sum() / valid_count
        ratio_mean = (ratios * agent_mask).sum() / valid_count

        target_kl = getattr(self.args, "ppo_target_kl", 0.0)
        if epoch_index > 0 and target_kl > 0 and approx_kl.item() > target_kl:
            ppo_early_stop = True
            break

        advantages, critic_train_stats = self.train_critic_sequential(
            self.critic, self.target_critic, batch, rewards, critic_mask
        )
        advantages = advantages.detach()
        if critic_stats_all is None:
            critic_stats_all = {key: [] for key in critic_train_stats}
        for key, values in critic_train_stats.items():
            critic_stats_all[key].extend(values)

        valid_advantages = advantages[agent_mask > 0]
        advantage_raw_mean = valid_advantages.mean()
        advantage_raw_std = valid_advantages.std(unbiased=False)
        if getattr(self.args, "normalise_advantages", False):
            advantages = (
                (advantages - advantage_raw_mean)
                / advantage_raw_std.clamp_min(1e-6)
            ) * agent_mask

        surr1 = ratios * advantages
        surr2 = th.clamp(
            ratios, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip
        ) * advantages
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        pg_loss = -(
            (th.min(surr1, surr2) + entropy_coef * entropy) * agent_mask
        ).sum() / agent_mask.sum()

        raw_belief_loss = pg_loss.new_tensor(0.0)
        scheduled_belief_loss = pg_loss.new_tensor(0.0)
        belief_loss = pg_loss.new_tensor(0.0)
        belief_cap_scale = pg_loss.new_tensor(1.0)
        aux_losses = None
        if stage != "warmup":
            aux_losses = self.mac.compute_auxiliary_losses(aux_steps, batch, agent_mask)
            if "bbd_core_loss" in aux_losses:
                bbd_weight = getattr(
                    self.args, "lambda_bbd", getattr(self.args, "lambda_cbd", 1.0)
                )
                raw_belief_loss = (
                    getattr(self.args, "lambda_ahs", bbd_weight)
                    * aux_losses["ahs_loss"]
                    + bbd_weight * aux_losses["bbd_core_loss"]
                    + getattr(
                        self.args, "lambda_hyp", getattr(self.args, "lambda_dmbi", 1.0)
                    )
                    * aux_losses["hyp_loss"]
                    + getattr(self.args, "lambda_fwd", 0.0)
                    * aux_losses["fwd_loss"]
                )
            else:
                raw_belief_loss = (
                    getattr(
                        self.args, "lambda_bbd", getattr(self.args, "lambda_cbd", 1.0)
                    )
                    * aux_losses["bbd_loss"]
                    + getattr(
                        self.args, "lambda_hyp", getattr(self.args, "lambda_dmbi", 1.0)
                    )
                    * aux_losses["hyp_loss"]
                    + getattr(self.args, "lambda_fwd", 0.0)
                    * aux_losses["fwd_loss"]
                )
            scheduled_belief_loss = self._belief_scale(t_env) * raw_belief_loss
            belief_loss, belief_cap_scale = self._limit_belief_loss(
                scheduled_belief_loss, pg_loss
            )

        agent_loss = pg_loss + belief_loss
        self.agent_optimiser.zero_grad()
        if th.isfinite(agent_loss):
            agent_loss.backward()
            if getattr(self.args, "dib_robust_entities", False):
                from algorithms.dib_mappo.robust import clip_actor_gradients
                grad_norm = clip_actor_gradients(self)
            else:
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.agent_params, self.args.grad_norm_clip, error_if_nonfinite=False)
            if th.isfinite(grad_norm):
                self.agent_optimiser.step()
            else:
                skipped_agent_update = True
                self.agent_optimiser.zero_grad(set_to_none=True)
        else:
            grad_norm = agent_loss.new_tensor(float("nan"))
            skipped_agent_update = True
            self.agent_optimiser.zero_grad(set_to_none=True)

        epoch_stats["pg_loss"].append(pg_loss.item())
        epoch_stats["belief_loss_raw"].append(raw_belief_loss.item())
        epoch_stats["belief_loss_scheduled"].append(scheduled_belief_loss.item())
        epoch_stats["belief_loss"].append(belief_loss.item())
        epoch_stats["belief_cap_scale"].append(belief_cap_scale.item())
        epoch_stats["agent_loss"].append(agent_loss.item())
        epoch_stats["agent_grad_norm"].append(grad_norm.item())
        epoch_stats["ppo_ratio_mean"].append(ratio_mean.item())
        epoch_stats["ppo_clip_fraction"].append(clip_fraction.item())
        epoch_stats["ppo_approx_kl"].append(approx_kl.item())
        epoch_stats["advantage_raw_mean"].append(advantage_raw_mean.item())
        epoch_stats["advantage_raw_std"].append(advantage_raw_std.item())
        epochs_completed += 1

        base_logits = th.stack([step["base_logits"] for step in aux_steps], dim=1)
        residual_logits = th.stack(
            [step["residual_logits"] for step in aux_steps], dim=1
        )
        base_values = base_logits.squeeze(3).abs().mean(dim=-1)
        residual_values = residual_logits.abs().mean(dim=(-1, -2))
        base_logit_abs = (base_values * agent_mask).sum().item() / agent_mask.sum().item()
        residual_logit_abs = (
            (residual_values * agent_mask).sum().item() / agent_mask.sum().item()
        )

    self.old_mac.load_state(self.mac)

    self.critic_training_steps += 1
    if (
        self.args.target_update_interval_or_tau > 1
        and (self.critic_training_steps - self.last_target_update_step)
        / self.args.target_update_interval_or_tau
        >= 1.0
    ):
        self._update_targets_hard()
        self.last_target_update_step = self.critic_training_steps
    elif self.args.target_update_interval_or_tau <= 1.0:
        self._update_targets_soft(self.args.target_update_interval_or_tau)

    if t_env - self.log_stats_t >= self.args.learner_log_interval:
        for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
            values = critic_stats_all[key]
            self.logger.log_stat(key, sum(values) / len(values), t_env)

        self.logger.log_stat(
            "advantage_mean",
            (advantages * agent_mask).sum().item() / agent_mask.sum().item(),
            t_env,
        )
        for key, values in epoch_stats.items():
            self.logger.log_stat(key, sum(values) / len(values), t_env)
        self.logger.log_stat("agent_update_skipped", float(skipped_agent_update), t_env)
        self.logger.log_stat("ppo_epochs", epochs_completed, t_env)
        self.logger.log_stat("ppo_early_stop", float(ppo_early_stop), t_env)
        self.logger.log_stat("entropy_coef", entropy_coef, t_env)
        self.logger.log_stat("dib_actor_residual_gate", actor_gate, t_env)
        self.logger.log_stat("base_logit_abs", base_logit_abs, t_env)
        self.logger.log_stat("residual_logit_abs", residual_logit_abs, t_env)
        self.logger.log_stat("dib_stage", {"warmup": 0, "belief": 1, "joint": 2}[stage], t_env)
        self.logger.log_stat(
            "belief_loss_scale",
            self._belief_scale(t_env) if stage != "warmup" else 0.0,
            t_env,
        )
        self.logger.log_stat(
            "pi_max",
            (pi.max(dim=-1)[0] * agent_mask).sum().item() / agent_mask.sum().item(),
            t_env,
        )
        if aux_losses is not None:
            for key, value in aux_losses.items():
                self.logger.log_stat(key, value.item(), t_env)
        self.log_stats_t = t_env



def install(learner):
    """Opt-in; do not modify shared external sources or existing workers."""
    if not hasattr(learner, "_unmasked_train"):
        learner._unmasked_train = learner.train
        learner.train = MethodType(train, learner)
