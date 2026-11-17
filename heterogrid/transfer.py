"""Opt-in, fixed-capacity task interface; underlying game rules are unchanged.

Initial roster sizes are public, as in the legacy variable-length interface.
Slot presence never encodes hidden health, position, role or current survival.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import EnvConfig
from .smac import SMACAdapter


@dataclass(frozen=True)
class TransferCapacity:
    max_red: int
    max_blue: int
    message_dim: int = 8

    def __post_init__(self):
        for name in ("max_red", "max_blue", "message_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return {name: getattr(self, name) for name in ("max_red", "max_blue", "message_dim")}


class TaskSuite:
    """Expanded, serializable task definitions with disjoint named splits."""
    def __init__(self, tasks: Mapping[str, EnvConfig | dict], splits: Mapping[str, list[str]],
                 capacity: TransferCapacity | dict | None = None):
        if not tasks or any(not isinstance(k, str) or not k for k in tasks):
            raise ValueError("tasks must have nonempty string identifiers")
        self._tasks = {k: copy.deepcopy(v if isinstance(v, EnvConfig) else EnvConfig.from_dict(v))
                       for k, v in tasks.items()}
        dims = {c.message_dim for c in self._tasks.values()}
        if len(dims) != 1:
            raise ValueError("All tasks must use the same message_dim")
        if capacity is None:
            capacity = TransferCapacity(max(len(c.red) for c in self._tasks.values()),
                                        max(len(c.blue) for c in self._tasks.values()), next(iter(dims)))
        self.capacity = capacity if isinstance(capacity, TransferCapacity) else TransferCapacity(**capacity)
        for name, c in self._tasks.items():
            if (len(c.red) > self.capacity.max_red or len(c.blue) > self.capacity.max_blue or
                    c.message_dim != self.capacity.message_dim):
                raise ValueError(f"Task {name!r} exceeds capacity or has a different message_dim")
        if not splits or any(not isinstance(k, str) or not k for k in splits):
            raise ValueError("splits must have nonempty string identifiers")
        self._splits = {}
        used = set()
        for name, ids in splits.items():
            if not isinstance(ids, (list, tuple)) or not ids or len(ids) != len(set(ids)):
                raise ValueError(f"Split {name!r} must be a nonempty, unique task list")
            if set(ids) - self._tasks.keys():
                raise ValueError(f"Split {name!r} contains an unknown task")
            if used.intersection(ids):
                raise ValueError("Task identifiers must not overlap across splits")
            used.update(ids)
            self._splits[name] = tuple(ids)
        if used != self._tasks.keys():
            raise ValueError("Every task must belong to exactly one split")

    @property
    def task_ids(self):
        return tuple(self._tasks)

    @property
    def episode_limit(self):
        return max(c.max_steps for c in self._tasks.values())

    def ids(self, split="train"):
        if split not in self._splits:
            raise ValueError(f"Unknown split {split!r}")
        return self._splits[split]

    def config(self, task_id):
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task {task_id!r}")
        return copy.deepcopy(self._tasks[task_id])

    def to_dict(self):
        return {"schema_version": 1, "capacity": self.capacity.to_dict(),
                "tasks": {k: c.to_dict() for k, c in self._tasks.items()},
                "splits": {k: list(v) for k, v in self._splits.items()}}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 1 or set(data) - {"schema_version", "capacity", "tasks", "splits"}:
            raise ValueError("Expected task suite schema_version=1")
        return cls(data["tasks"], data["splits"], data.get("capacity"))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        """Never overwrite an existing experiment/suite definition."""
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


class TransferSMACAdapter:
    """Stable red/blue slots across tasks, selected only at episode boundaries.

    This is a separate SMAC-style API, not a replacement PettingZoo environment.
    ``split`` controls sampling. An explicit ``task_ids`` whitelist is used for
    deliberate target adaptation; reset cannot escape the selected whitelist.
    """
    schema = "heterogrid-transfer-v1"

    def __init__(self, suite: TaskSuite, split="train", *, task_ids=None, record_replay=False,
                 opponent_factory=None):
        self.suite = TaskSuite.from_dict(suite.to_dict())
        self.capacity = self.suite.capacity
        self.allowed_tasks = tuple(task_ids) if task_ids is not None else self.suite.ids(split)
        if not self.allowed_tasks or len(self.allowed_tasks) != len(set(self.allowed_tasks)) or set(self.allowed_tasks) - set(self.suite.task_ids):
            raise ValueError("task_ids must be a nonempty unique subset of suite tasks")
        self.n_agents = self.capacity.max_red
        self.episode_limit = self.suite.episode_limit
        self.record_replay = record_replay
        self.opponent_factory = opponent_factory
        self.task_id = None
        self._inner = None
        self._started = False
        self._closed = False
        self._pending_seed = None
        self._task_rng = np.random.default_rng()
        self._world_rng = np.random.default_rng()
        self._stats = {"battles_game": 0, "battles_won": 0, "timeouts": 0}

    @property
    def env(self):
        self._require_reset()
        return self._inner.env

    def _require_reset(self):
        if not self._started or self._closed:
            raise RuntimeError("Call reset() before accessing the task environment")

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def reset(self, seed=None, options=None):
        if self._closed:
            raise RuntimeError("Adapter is closed")
        options = dict(options or {})
        if set(options) - {"task_id", "positions"}:
            raise ValueError("Transfer reset options are task_id and positions")
        chosen = seed if seed is not None else self._pending_seed
        self._pending_seed = None
        if chosen is not None:
            streams = np.random.SeedSequence(chosen).spawn(2)
            self._task_rng, self._world_rng = (np.random.default_rng(s) for s in streams)
        requested = options.get("task_id")
        running = self._started and not self._inner.env.core.finished
        if requested is None:
            requested = self.task_id if running else str(self._task_rng.choice(self.allowed_tasks))
        if requested not in self.allowed_tasks:
            raise ValueError(f"Task {requested!r} is outside this adapter's allowed task set")
        if running and requested != self.task_id:
            raise RuntimeError("Task changes are allowed only after the current episode has ended")
        if self._inner is None or requested != self.task_id:
            config = self.suite.config(requested)
            kwargs = {"record_replay": self.record_replay}
            if config.opponent == "pretrained":
                if self.opponent_factory is None:
                    raise ValueError("Pretrained tasks need opponent_factory(task_id, config)")
                kwargs["opponent_policy"] = self.opponent_factory(requested, copy.deepcopy(config))
            replacement = SMACAdapter(config=config, **kwargs)
            if self._inner is not None:
                self._inner.close()
            self._inner, self.task_id = replacement, requested
        world_seed = chosen if chosen is not None else int(self._world_rng.integers(0, 2**31 - 1))
        self._inner.reset(seed=world_seed, options={k: v for k, v in options.items() if k != "task_id"})
        self._started = True
        return self.get_obs(), self.get_state()

    def observation_layout(self):
        r, b, d = self.capacity.max_red, self.capacity.max_blue, self.capacity.message_dim
        layout, offset = {}, 0
        for name, shape in (("self", (12,)), ("allies", (r-1, 8)), ("targets", (b, 11)),
                            ("messages", (r, d)), ("message_mask", (r,)), ("message_age", (r,)),
                            ("ally_slots", (r-1,)), ("target_slots", (b,)), ("self_slot", (1,)),
                            ("map_inverse_span", (3,))):
            size = int(np.prod(shape))
            layout[name] = {"offset": offset, "size": size, "shape": list(shape)}
            offset += size
        return layout

    def get_entity_obs_agent(self, agent_id):
        """Only transforms the original local observation, never global state.

        Slots mark the public initial roster. Unknown enemies retain zero local
        features; known/direct/reported flags remain those produced by the core.
        """
        self._require_reset()
        if not 0 <= agent_id < self.n_agents:
            raise IndexError("agent_id outside red capacity")
        c = self.capacity
        data = {"self": np.zeros(12, np.float32), "allies": np.zeros((c.max_red-1,8), np.float32),
                "targets": np.zeros((c.max_blue,11), np.float32),
                "messages": np.zeros((c.max_red,c.message_dim), np.float32),
                "message_mask": np.zeros(c.max_red,np.float32), "message_age": np.zeros(c.max_red,np.float32),
                "ally_slots": np.zeros(c.max_red-1,np.float32), "target_slots": np.zeros(c.max_blue,np.float32),
                "self_slot": np.zeros(1,np.float32), "map_inverse_span": np.zeros(3,np.float32)}
        nr, nb = self._inner.n_agents, len(self._inner.env.config.blue)
        if agent_id >= nr:
            return data
        name = self._inner.env.possible_agents[agent_id]
        obs = self._inner.env.core.observation(name)
        raw = obs["observation"]
        data["self"][:] = raw[:12]
        data["allies"][:nr-1] = raw[12:12+8*(nr-1)].reshape(nr-1,8)
        data["targets"][:nb] = raw[12+8*(nr-1):].reshape(nb,11)
        data["messages"][:nr] = obs["messages"]
        data["message_mask"][:nr] = obs["message_mask"]
        data["message_age"][:nr] = obs["message_age"]
        data["ally_slots"][:nr-1] = 1
        data["target_slots"][:nb] = 1
        data["self_slot"][0] = 1
        # Declared map extents are public in this opt-in protocol. This permits
        # local-distance calibration without reading hidden entity coordinates.
        data["map_inverse_span"][:] = 1/np.maximum(np.asarray(self._inner.env.config.map_size)-1,1)
        return data

    def get_obs_agent(self, agent_id):
        values = self.get_entity_obs_agent(agent_id)
        return np.concatenate([values[name].ravel() for name in self.observation_layout()]).astype(np.float32)

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_size(self):
        return sum(field["size"] for field in self.observation_layout().values())

    def get_state_size(self):
        return 12*(self.capacity.max_red+self.capacity.max_blue)+1

    def get_state(self):
        self._require_reset()
        nr, nb = self._inner.n_agents, len(self._inner.env.config.blue)
        raw = self._inner.get_state()
        rows = raw[:-1].reshape(nr+nb,11)
        red = np.zeros((self.capacity.max_red,11),np.float32)
        blue = np.zeros((self.capacity.max_blue,11),np.float32)
        red[:nr],blue[:nb] = rows[:nr],rows[nr:]
        present = np.zeros(self.capacity.max_red+self.capacity.max_blue,np.float32)
        present[:nr]=1;present[self.capacity.max_red:self.capacity.max_red+nb]=1
        return np.concatenate((red.ravel(),blue.ravel(),raw[-1:],present)).astype(np.float32)

    def get_total_actions(self):
        return 7+self.capacity.max_blue

    def get_avail_agent_actions(self, agent_id):
        self._require_reset()
        if not 0 <= agent_id < self.n_agents:
            raise IndexError("agent_id outside red capacity")
        mask = np.zeros(self.get_total_actions(),np.int8)
        if agent_id < self._inner.n_agents:
            raw = self._inner.get_avail_agent_actions(agent_id)
            mask[:len(raw)] = raw
        else:
            mask[0]=1
        return mask.tolist()

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_active_mask(self):
        """Training mask; it is not appended to local actor observations."""
        self._require_reset()
        result=np.zeros((self.n_agents,1),np.float32)
        for i,name in enumerate(self._inner.env.possible_agents):
            result[i,0]=float(self._inner.env.core.units[name].alive)
        return result

    def step(self, actions):
        self._require_reset()
        if len(actions)!=self.n_agents:
            raise ValueError("Expected max_red actions, including padded no-op slots")
        # A reference to a padded target/agent is an interface error, not a game action.
        for i,a in enumerate(actions):
            if (isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or
                    not 0<=a<self.get_total_actions()):
                raise ValueError("Action outside transfer action space")
            if i>=self._inner.n_agents and a!=0:
                raise ValueError("Padded red slots can only wait")
            if i<self._inner.n_agents and a>=self._inner.get_total_actions():
                raise ValueError("Cannot address a padded target slot")
        reward,done,info=self._inner.step(list(actions[:self._inner.n_agents]))
        info.update(task_id=self.task_id,actual_n_agents=self._inner.n_agents,
                    actual_n_enemies=len(self._inner.env.config.blue))
        if done:
            self._stats["battles_game"]+=1
            self._stats["battles_won"]+=int(info["battle_won"])
            self._stats["timeouts"]+=int(info["termination_reason"]=="timeout")
        return reward,done,info

    def get_env_info(self):
        return {"state_shape":self.get_state_size(),"obs_shape":self.get_obs_size(),
                "n_actions":self.get_total_actions(),"n_agents":self.n_agents,
                "episode_limit":self.episode_limit}

    def interface_spec(self):
        return {"schema":self.schema,"capacity":self.capacity.to_dict(),
                "observation_layout":self.observation_layout(),"env_info":self.get_env_info()}

    def get_stats(self):
        return {**self._stats,"win_rate":self._stats["battles_won"]/max(1,self._stats["battles_game"])}

    def exchange_messages(self,messages):
        self._require_reset();self._inner.exchange_messages(messages)
        return self.get_obs()

    def get_comm_mask(self):
        self._require_reset()
        result=np.zeros((self.n_agents,self.n_agents),np.int8)
        raw=self._inner.get_comm_mask();result[:len(raw),:len(raw)]=raw
        return result

    def save_replay(self,path):
        self._require_reset()
        return self._inner.save_replay(path)

    def render(self):
        self._require_reset()
        return self._inner.render()

    def close(self):
        if self._inner is not None:
            self._inner.close()
        self._closed=True
