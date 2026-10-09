"""Fixed red-team slots and scalar team reward for PyMARL/SMAC-style runners."""
from __future__ import annotations

import numpy as np

from .env import HeteroGridParallelEnv


def flatten_observation(obs):
    return np.concatenate([obs["observation"], obs["messages"].ravel(),
                           obs["message_mask"].astype(np.float32), obs["message_age"]]).astype(np.float32)


class SMACAdapter:
    def __init__(self, config=None, **kwargs):
        self.env = HeteroGridParallelEnv(config, **kwargs)
        self.n_agents = len(self.env.possible_agents)
        self.episode_limit = self.env.config.max_steps
        self._seed = None
        self._reset = False
        self._stats = {"battles_game": 0, "battles_won": 0, "timeouts": 0}

    def seed(self, seed=None):
        self._seed = seed
        return [seed]

    def reset(self, seed=None, options=None):
        chosen = seed if seed is not None else self._seed
        self._seed = None
        self.env.reset(seed=chosen, options=options)
        self._reset = True
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if len(actions) != self.n_agents:
            raise ValueError("SMAC step expects a fixed-length action list")
        if not self.env.agents:
            raise RuntimeError("Reset before stepping or after an episode ends")
        selected = {n: actions[i] for i, n in enumerate(self.env.possible_agents) if n in self.env.agents}
        self.env.step(selected)
        core = self.env.core
        done = core.finished
        info = {"battle_won": core.winner == "red", "winner": core.winner,
                "termination_reason": core.reason, "episode_limit": core.reason == "timeout",
                "dead_allies": sum(not core.units[n].alive for n in core.names["red"]),
                "dead_enemies": sum(not core.units[n].alive for n in core.names["blue"]),
                "invalid_actions": core.last_metrics["invalid_actions"]}
        if done:
            self._stats["battles_game"] += 1
            self._stats["battles_won"] += int(core.winner == "red")
            self._stats["timeouts"] += int(core.reason == "timeout")
        # Never sum the identical per-agent PettingZoo rewards.
        return float(core.last_metrics["team_rewards"]["red"]), done, info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        self.env._require_reset()
        name = self.env.possible_agents[agent_id]
        return flatten_observation(self.env.core.observation(name))

    def get_obs_size(self):
        return self.env.core.obs_size("red") + self.n_agents * (self.env.config.message_dim + 2)

    def get_state(self):
        return self.env.state()

    def get_state_size(self):
        return self.env.state_space.shape[0]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        self.env._require_reset()
        return self.env.core.action_mask(self.env.possible_agents[agent_id]).tolist()

    def get_total_actions(self):
        return self.env.core.n_actions("red")

    def get_env_info(self):
        return {"state_shape": self.get_state_size(), "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(), "n_agents": self.n_agents,
                "episode_limit": self.episode_limit}

    def exchange_messages(self, messages):
        self.env.exchange_messages(messages)
        return self.get_obs()

    def get_comm_mask(self):
        return self.env.get_comm_mask()

    def get_stats(self):
        result = dict(self._stats)
        result["win_rate"] = result["battles_won"] / max(1, result["battles_game"])
        return result

    def render(self):
        return self.env.render()

    def save_replay(self, path):
        return self.env.save_replay(path)

    def close(self):
        return self.env.close()
