"""Serializable environment settings; no algorithm dependencies."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROLES = ("scout", "strike", "hybrid")
PLATFORMS = ("uav", "usv", "robot", "robot_dog")


@dataclass(frozen=True)
class RoleStats:
    hp: float
    vision: float
    attack_range: float
    damage: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.hp <= 0:
            raise ValueError("hp must be positive")


def default_stats():
    return {
        "scout": RoleStats(60, 7, 0, 0),
        "strike": RoleStats(100, 3, 6, 20),
        "hybrid": RoleStats(80, 5, 4, 12),
    }


@dataclass(frozen=True)
class UnitSpec:
    role: str
    platform: str = "uav"

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"Unknown role {self.role!r}; choose from {ROLES}")
        if self.platform not in PLATFORMS:
            raise ValueError(f"Unknown platform {self.platform!r}; choose from {PLATFORMS}")


def default_roster():
    return [UnitSpec("scout", "uav"), UnitSpec("scout", "usv"),
            UnitSpec("strike", "robot"), UnitSpec("strike", "robot_dog"),
            UnitSpec("hybrid", "uav"), UnitSpec("hybrid", "robot_dog")]


@dataclass(frozen=True)
class EnvConfig:
    map_size: tuple[int, int, int] = (24, 24, 6)
    red: list[UnitSpec] = field(default_factory=default_roster)
    blue: list[UnitSpec] = field(default_factory=default_roster)
    role_stats: dict[str, RoleStats] = field(default_factory=default_stats)
    max_steps: int = 200
    opponent: str = "rule"
    communication: str = "distance"  # none, full, distance, random
    comm_radius: float = 10.0
    dropout_probability: float = 0.25
    target_sharing: str = "manual"  # manual or auto
    target_ttl: int = 2
    message_dim: int = 8
    reward_mode: str = "dense"
    damage_reward: float = 0.01
    kill_reward: float = 1.0
    win_reward: float = 10.0
    loss_reward: float = -10.0
    timeout_as_loss: bool = False

    def __post_init__(self):
        if len(self.map_size) != 3 or any(type(n) is not int or not 1 <= n < 2**31 for n in self.map_size):
            raise ValueError("map_size must contain three positive 32-bit integers")
        if not self.red or not self.blue:
            raise ValueError("Both teams must contain at least one unit")
        if any(not isinstance(s, UnitSpec) for s in self.red + self.blue):
            raise ValueError("Team rosters must contain UnitSpec objects")
        if set(self.role_stats) != set(ROLES) or any(not isinstance(s, RoleStats) for s in self.role_stats.values()):
            raise ValueError("role_stats must define scout, strike and hybrid")
        for name in ("max_steps", "target_ttl", "message_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.opponent not in ("static", "random", "rule", "pretrained", "dummy"):
            raise ValueError("opponent must be static, random, rule, pretrained or dummy")
        if self.communication not in ("none", "full", "distance", "random"):
            raise ValueError("Unknown communication mode")
        if self.target_sharing not in ("manual", "auto") or self.reward_mode not in ("dense", "sparse"):
            raise ValueError("Invalid target_sharing or reward_mode")
        if not math.isfinite(self.comm_radius) or self.comm_radius < 0:
            raise ValueError("comm_radius must be finite and nonnegative")
        if not math.isfinite(self.dropout_probability) or not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability must be in [0, 1]")
        for name in ("damage_reward", "kill_reward", "win_reward", "loss_reward"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.damage_reward < 0 or self.kill_reward < 0 or self.win_reward < 0 or self.loss_reward > 0:
            raise ValueError("Damage/kill/win rewards must be >= 0 and loss_reward <= 0")
        if type(self.timeout_as_loss) is not bool:
            raise ValueError("timeout_as_loss must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvConfig":
        data = dict(data)
        if "map_size" in data:
            data["map_size"] = tuple(data["map_size"])
        for team in ("red", "blue"):
            if team in data:
                data[team] = [UnitSpec(s) if isinstance(s, str) else UnitSpec(**s) for s in data[team]]
        if "role_stats" in data:
            data["role_stats"] = {k: RoleStats(**v) for k, v in data["role_stats"].items()}
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "EnvConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
