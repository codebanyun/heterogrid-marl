"""Narrow correction for the native learner's missing filtered target sync."""
import torch

from learners.actor_critic_learner import ActorCriticVLearner


class TargetSyncActorCriticVLearner(ActorCriticVLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filtered_target_updates = 0

    def _update_targets(self):
        self._update_targets_hard()

    def _update_targets_hard(self):
        super()._update_targets_hard()
        if self.args.use_w and self.args.use_w_critic:
            self.target_critic_w.load_state_dict(self.critic_w.state_dict())
            self.filtered_target_updates += 1

    def _update_targets_soft(self, tau):
        super()._update_targets_soft(tau)
        if self.args.use_w and self.args.use_w_critic:
            with torch.no_grad():
                for target, source in zip(self.target_critic_w.parameters(), self.critic_w.parameters()):
                    target.mul_(1 - tau).add_(source, alpha=tau)
            self.filtered_target_updates += 1
