"""PettingZoo Parallel API. Only the red team is externally controlled by default."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import EnvConfig
from .core import GridCombat
from .policies import ScriptedPolicy, TeamPolicy


class HeteroGridParallelEnv(ParallelEnv):
    metadata = {"name": "heterogrid_v0", "render_modes": ["ansi", "rgb_array"],
                "render_fps": 10, "is_parallelizable": True}

    def __init__(self, config: EnvConfig | dict | None = None, *, render_mode=None,
                 opponent_policy: TeamPolicy | None = None, record_replay=False):
        self.config = copy.deepcopy(config if isinstance(config, EnvConfig) else EnvConfig.from_dict(config or {}))
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Use render_mode=None, 'ansi' or 'rgb_array'; browser viewing is a separate process")
        self.render_mode = render_mode
        self.core = GridCombat(self.config)
        self.possible_agents = self.core.names["red"].copy()
        self.agents = []
        self.record_replay = record_replay
        self.frames = []
        self._last_observations = {}
        self._last_infos = {}
        self._reset_done = False
        if self.config.opponent == "pretrained":
            if opponent_policy is None:
                raise ValueError("opponent='pretrained' requires an opponent_policy implementing TeamPolicy")
            if not callable(getattr(opponent_policy, "act", None)) or not callable(getattr(opponent_policy, "reset", None)):
                raise TypeError("opponent_policy must implement act and reset")
            self.opponent_policy = opponent_policy
        else:
            if opponent_policy is not None:
                raise ValueError("Set opponent='pretrained' when supplying opponent_policy")
            self.opponent_policy = ScriptedPolicy(self.config, "blue", self.config.opponent)
        n_allies, dim = len(self.possible_agents), self.config.message_dim
        self.observation_spaces = {n: spaces.Dict({
            "observation": spaces.Box(-1, 1, (self.core.obs_size("red"),), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.core.n_actions("red")),
            "messages": spaces.Box(-1, 1, (n_allies, dim), dtype=np.float32),
            "message_mask": spaces.MultiBinary(n_allies),
            "message_age": spaces.Box(0, 1, (n_allies,), dtype=np.float32),
        }) for n in self.possible_agents}
        self.action_spaces = {n: spaces.Discrete(self.core.n_actions("red")) for n in self.possible_agents}
        self.state_space = spaces.Box(0, 1, (11 * len(self.core.all_names) + 1,), dtype=np.float32)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _require_reset(self):
        if not self._reset_done:
            raise RuntimeError("Call reset() before accessing environment state")

    def _info(self, name):
        unit = self.core.units[name]
        info = {"role": unit.spec.role, "platform": unit.spec.platform,
                "comm_online": self.core.online[name], "step": self.core.step_count}
        if self.core.finished:
            info.update({"battle_won": self.core.winner == "red", "winner": self.core.winner,
                         "termination_reason": self.core.reason, "episode_limit": self.core.reason == "timeout"})
        return info

    def _observe(self, names):
        return {n: self.core.observation(n) for n in names}, {n: self._info(n) for n in names}

    def reset(self, seed=None, options=None):
        options = options or {}
        # PettingZoo's standard API tests pass arbitrary options to check forwarding.
        # Recognize positions; ignore unused options without changing spaces/config.
        self.core.reset(seed=seed, positions=options.get("positions"))
        self._reset_done = True
        self.agents = self.possible_agents.copy()
        if seed is not None:
            for i, name in enumerate(self.possible_agents):
                self.action_space(name).seed(seed + i)
                self.observation_space(name).seed(seed + i)
        self.opponent_policy.reset(seed=seed)
        if self.config.target_sharing == "auto":
            self.core.share_visible_targets("red")
            self.core.share_visible_targets("blue")
        self.frames = [self.core.snapshot()] if self.record_replay else []
        self._last_observations, self._last_infos = self._observe(self.agents)
        return self._last_observations, self._last_infos

    def exchange_messages(self, messages):
        """Optional communication phase before standard step(). Returns refreshed obs/info.

        No physics, random link resampling or reward occurs here. Messages can only
        originate from controlled, living red units. Already received messages remain
        readable through their TTL even if the link subsequently fails.
        """
        self._require_reset()
        self.core.exchange_messages(messages, allowed_senders=set(self.agents))
        self._last_observations, self._last_infos = self._observe(self.agents)
        return self._last_observations, self._last_infos

    def share_visible_targets(self):
        """Explicit broadcast baseline; learned communication should use exchange_messages."""
        self._require_reset()
        self.core.share_visible_targets("red")
        self._last_observations, self._last_infos = self._observe(self.agents)
        return self._last_observations, self._last_infos

    def step(self, actions):
        self._require_reset()
        if not self.agents:
            if actions:
                raise ValueError("Episode has ended; step requires an empty action dict")
            return {}, {}, {}, {}, {}
        acting = self.agents.copy()
        if set(actions) != set(acting):
            raise ValueError("step() requires exactly one action for each current env.agents entry")
        # A scripted opponent may use the same legal one-hop reports as the red baseline.
        # Pretrained opponents get these reports only in explicit auto-sharing mode.
        if self.config.opponent in ("static", "random", "rule"):
            self.core.share_visible_targets("blue")
        blue = [n for n in self.core.names["blue"] if self.core.units[n].alive]
        blue_obs, blue_info = self._observe(blue)
        blue_actions = self.opponent_policy.act(blue_obs, blue_info)
        if not isinstance(blue_actions, dict) or set(blue_actions) != set(blue):
            raise ValueError("Opponent policy must return one action for every living blue unit")
        rewards, metrics = self.core.step({**actions, **blue_actions})
        timed_out = self.core.reason == "timeout"
        terminations = {n: (not self.core.units[n].alive) or (self.core.finished and not timed_out) for n in acting}
        truncations = {n: bool(timed_out and not terminations[n]) for n in acting}
        team_rewards = {n: float(rewards["red"]) for n in acting}
        self.agents = [n for n in acting if not terminations[n] and not truncations[n]]
        if self.config.target_sharing == "auto" and not self.core.finished:
            self.core.share_visible_targets("red")
            self.core.share_visible_targets("blue")
        # Final observations are returned for agents that died/terminated on this step.
        observations, infos = self._observe(acting)
        for name in metrics["invalid_actions"]:
            if name in infos:
                infos[name]["invalid_action"] = True
        if self.record_replay:
            self.frames.append(self.core.snapshot())
        self._last_observations = {n: observations[n] for n in self.agents}
        self._last_infos = {n: infos[n] for n in self.agents}
        return observations, team_rewards, terminations, truncations, infos

    def state(self):
        self._require_reset()
        return self.core.state()

    def get_comm_mask(self):
        """Transport/diagnostic matrix [sender,receiver]; not an actor observation."""
        self._require_reset()
        return self.core.comm_mask("red")

    def observation_layout(self):
        allies = len(self.possible_agents) - 1
        return {"self": {"offset": 0, "size": 12},
                "allies": {"offset": 12, "count": allies, "stride": 8},
                "targets": {"offset": 12 + 8 * allies, "count": len(self.core.names["blue"]), "stride": 11},
                "ally_ids": self.possible_agents.copy(), "target_ids": self.core.names["blue"].copy()}

    def snapshot(self):
        self._require_reset()
        return self.core.snapshot()

    def save_replay(self, path):
        self._require_reset()
        if not self.record_replay:
            raise RuntimeError("Construct the environment with record_replay=True to save full replays")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 1, "config": self.config.to_dict(), "frames": self.frames},
                                   ensure_ascii=False), encoding="utf-8")
        return str(path.resolve())

    def render(self):
        self._require_reset()
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            lines = [f"HeteroGrid step={self.core.step_count} winner={self.core.winner} reason={self.core.reason}"]
            lines += [f"{n:8} {u.spec.role:6} {tuple(u.position)} HP={u.hp:g}/{u.stats.hp:g}"
                      for n, u in self.core.units.items()]
            return "\n".join(lines)
        from .rendering import render_rgb
        return render_rgb(self.core.snapshot())

    def close(self):
        pass


def parallel_env(**kwargs):
    return HeteroGridParallelEnv(**kwargs)


def env(**kwargs):
    """AEC conversion for algorithms that consume the sequential PettingZoo API."""
    from .aec import as_aec
    return as_aec(parallel_env(**kwargs))
