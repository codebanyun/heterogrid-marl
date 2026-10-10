"""Example checkpoint adapter, not a trained policy supplied with the environment.

Expected .npz: weight [n_actions, flattened_obs_size], bias [n_actions].
Replace this module with your own MAPPO/PyTorch actor adapter when available.
"""
import numpy as np

from heterogrid.core import GridCombat
from heterogrid.smac import flatten_observation


class SavedLinearPolicy:
    def __init__(self, config, team, checkpoint):
        if checkpoint is None:
            raise ValueError("A trained checkpoint path is required")
        core = GridCombat(config)
        allies = len(core.names[team])
        size = core.obs_size(team) + allies * (config.message_dim + 2)
        with np.load(checkpoint, allow_pickle=False) as data:
            self.weight = np.asarray(data["weight"], dtype=np.float32)
            self.bias = np.asarray(data["bias"], dtype=np.float32)
        if self.weight.shape != (core.n_actions(team), size) or self.bias.shape != (core.n_actions(team),):
            raise ValueError("Checkpoint shapes do not match this team/configuration")
        if not np.all(np.isfinite(self.weight)) or not np.all(np.isfinite(self.bias)):
            raise ValueError("Checkpoint contains non-finite weights")

    def reset(self, seed=None):
        pass  # Clear recurrent state here for a recurrent MAPPO actor.

    def act(self, observations, infos):
        actions = {}
        for name, obs in observations.items():
            logits = self.weight @ flatten_observation(obs) + self.bias
            logits = np.where(obs["action_mask"], logits, -np.inf)
            actions[name] = int(np.argmax(logits))
        return actions


def make_policy(*, config, team, checkpoint):
    return SavedLinearPolicy(config, team, checkpoint)
