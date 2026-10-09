"""Local-observation-only baselines and an optional pretrained policy hook."""
from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np

from .config import EnvConfig, ROLES
from .core import ALLY_FEATURES, ATTACK_OFFSET, SELF_FEATURES, TARGET_FEATURES


class TeamPolicy(Protocol):
    """Wrap any saved MARL actor in this small protocol (PyTorch is optional).

    act receives ONLY the team's local observation dicts, not the environment/state.
    Stateful policies must implement reset(seed=...) to clear recurrent state.
    """
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, observations: dict, infos: dict) -> dict[str, int]: ...


class ScriptedPolicy:
    def __init__(self, config: EnvConfig, team="red", mode="rule"):
        if mode not in ("dummy", "static", "random", "rule"):
            raise ValueError("Unknown scripted policy")
        self.config, self.team, self.mode = config, team, mode
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observations, infos=None):
        return {name: self._act_one(name, obs) for name, obs in observations.items()}

    def _act_one(self, name, obs):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        if self.mode == "dummy":
            return 0
        if self.mode == "random":
            return int(self.rng.choice(legal))
        vector = obs["observation"]
        if not np.any(vector):
            return 0
        count = len(getattr(self.config, self.team))
        offset = SELF_FEATURES + ALLY_FEATURES * (count - 1)
        targets = vector[offset:].reshape(-1, TARGET_FEATURES)
        available_attacks = np.flatnonzero(mask[ATTACK_OFFSET:])
        if len(available_attacks):
            target = min(available_attacks, key=lambda j: targets[j, 5])
            return ATTACK_OFFSET + int(target)
        if self.mode == "static":
            return 0
        size = np.maximum(np.asarray(self.config.map_size) - 1, 1)
        known = [r for r in targets if r[0] and r[5] > 0]
        role = ROLES[int(np.argmax(vector[4:7]))]
        if known:
            row = min(known, key=lambda r: np.linalg.norm(r[2:5] * size))
            relative = row[2:5] * size
            if role == "scout" and np.linalg.norm(relative) <= self.config.role_stats[role].vision * 0.6:
                return 0
        else:
            # A fixed patrol point uses public map geometry, not hidden enemy positions.
            own = vector[:3] * size
            index = int(name.rsplit("_", 1)[1])
            tick = int(round(vector[10] * self.config.max_steps))
            goal = np.asarray([size[0] * (0.72 if self.team == "red" else 0.28),
                               size[1] * ((index % 3 + 1) / 4), size[2] * 0.5])
            if tick > self.config.max_steps // 3:
                goal[1] = size[1] * (0.2 if (tick // 12 + index) % 2 else 0.8)
            relative = goal - own
        for axis in np.argsort(-np.abs(relative)):
            if abs(relative[axis]) >= 0.5:
                action = 1 + 2 * int(axis) + int(relative[axis] < 0)
                if mask[action]:
                    return action
        return 0


def load_policy(factory: str, config: EnvConfig, team="blue", checkpoint=None) -> TeamPolicy:
    """Load a user-supplied module:factory without assuming a network architecture."""
    module_name, separator, attr = factory.partition(":")
    if not separator:
        raise ValueError("Policy factory must have the form module:function")
    builder = getattr(importlib.import_module(module_name), attr)
    policy = builder(config=config, team=team, checkpoint=checkpoint)
    if not callable(getattr(policy, "act", None)) or not callable(getattr(policy, "reset", None)):
        raise TypeError("Pretrained policy must implement act(observations, infos) and reset(seed=...)")
    return policy
