"""The shared, algorithm-independent 13-task sensitivity matrix."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from heterogrid import CommunicationConfig, EnvConfig
from heterogrid.scenarios import balanced_roster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"map_size": ["B0", "M1", "M2", "M3"],
          "agent_count": ["B0", "N1", "N2", "N3"],
          "communication": ["B0", "C1", "C2", "C3", "C4", "C5", "C6"]}
LABELS = {"B0": "Baseline: 24x24x6, 6v6, radius 10", "M1": "12x12x3",
          "M2": "36x36x9", "M3": "48x48x12", "N1": "3v3", "N2": "9v9", "N3": "12v12",
          "C1": "No communication", "C2": "Full communication", "C3": "Radius 5",
          "C4": "Radius 15", "C5": "Random outage 0.25", "C6": "Random outage 0.50"}


def scenarios(fixed_blue=False):
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
    if fixed_blue:
        blue = CommunicationConfig(mode="distance", radius=10.0, dropout_probability=0.0)
        # Resolve the old red treatment first, then neutralize shared fields so
        # each communication contrast differs in red_communication only.
        tasks = {name: replace(config, communication=baseline.communication,
                    comm_radius=baseline.comm_radius, dropout_probability=baseline.dropout_probability,
                    red_communication=config.communication_for("red"), blue_communication=blue)
                 for name, config in tasks.items()}
    return tasks


def communication_protocol(fixed_blue=False):
    return {"scope": "red_only" if fixed_blue else "both_teams",
            "blue_fixed": {"mode": "distance", "radius": 10.0, "dropout_probability": 0.0}
                          if fixed_blue else None,
            "outage_topology": "full", "outage_unit": "node_per_tick",
            "independent_team_rng": bool(fixed_blue)}
