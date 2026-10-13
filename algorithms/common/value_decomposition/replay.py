"""Lossless float32 episode ring: do not duplicate consecutive observations."""
from __future__ import annotations

import numpy as np


class EpisodeReplay:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.episodes = []
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.episodes)

    def add(self, episode):
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            self.episodes[self.cursor] = episode
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, size, actions):
        selected = [self.episodes[i] for i in self.rng.integers(0, len(self), size=size)]
        length = max(len(e["u"]) for e in selected)
        n, obs = selected[0]["o"].shape[1:]
        state = selected[0]["s"].shape[-1]
        shapes = {"o": (n, obs), "o_next": (n, obs), "s": (state,), "s_next": (state,),
                  "u": (n, 1), "r": (1,), "avail_u": (n, actions), "avail_u_next": (n, actions),
                  "u_onehot": (n, actions), "padded": (1,), "terminated": (1,)}
        batch = {k: np.zeros((size, length) + shape, dtype=np.float32) for k, shape in shapes.items()}
        batch["padded"].fill(1)
        batch["terminated"].fill(1)
        # Keep padded targets numerically benign even though their loss is masked.
        batch["avail_u"][..., 0] = 1
        batch["avail_u_next"][..., 0] = 1
        for row, e in enumerate(selected):
            t = len(e["u"])
            batch["o"][row, :t], batch["o_next"][row, :t] = e["o"][:-1], e["o"][1:]
            batch["s"][row, :t], batch["s_next"][row, :t] = e["s"][:-1], e["s"][1:]
            batch["avail_u"][row, :t], batch["avail_u_next"][row, :t] = e["avail"][:-1], e["avail"][1:]
            batch["u"][row, :t, :, 0] = e["u"]
            batch["u_onehot"][row, :t] = np.eye(actions, dtype=np.float32)[e["u"]]
            batch["r"][row, :t, 0] = e["r"]
            batch["padded"][row, :t] = 0
            batch["terminated"][row, :t] = 0
            # A scored timeout is a finite-horizon terminal, as in the MAPPO integration.
            batch["terminated"][row, t - 1] = 1
        return batch, length
