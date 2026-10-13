"""The shared, algorithm-independent 13-task sensitivity matrix."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from heterogrid import EnvConfig, UnitSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"map_size": ["B0", "M1", "M2", "M3"],
          "agent_count": ["B0", "N1", "N2", "N3"],
          "communication": ["B0", "C1", "C2", "C3", "C4", "C5", "C6"]}
LABELS = {"B0": "Baseline: 24x24x6, 6v6, radius 10", "M1": "12x12x3",
          "M2": "36x36x9", "M3": "48x48x12", "N1": "3v3", "N2": "9v9", "N3": "12v12",
          "C1": "No communication", "C2": "Full communication", "C3": "Radius 5",
          "C4": "Radius 15", "C5": "Random outage 0.25", "C6": "Random outage 0.50"}


def balanced_roster(count):
    if count < 3 or count % 3:
        raise ValueError("Roster size must be a positive multiple of three")
    # Preserve the baseline's unit IDs, roles and platform labels exactly.
    roles = ("scout", "strike", "hybrid")
    platforms = (("uav", "usv"), ("robot", "robot_dog"), ("uav", "robot_dog"))
    return [UnitSpec(role, platforms[r][i % 2]) for r, role in enumerate(roles)
            for i in range(count // 3)]


def scenarios():
    baseline = EnvConfig.load(PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json")
    tasks = {"B0": baseline}
    for name, size in (("M1", (12, 12, 3)), ("M2", (36, 36, 9)), ("M3", (48, 48, 12))):
        tasks[name] = replace(baseline, map_size=size)
    for name, count in (("N1", 3), ("N2", 9), ("N3", 12)):
        tasks[name] = replace(baseline, red=balanced_roster(count), blue=balanced_roster(count))
    tasks.update(C1=replace(baseline, communication="none"),
                 C2=replace(baseline, communication="full"),
                 C3=replace(baseline, comm_radius=5.0), C4=replace(baseline, comm_radius=15.0),
                 C5=replace(baseline, communication="random", dropout_probability=0.25),
                 C6=replace(baseline, communication="random", dropout_probability=0.50))
    return tasks
