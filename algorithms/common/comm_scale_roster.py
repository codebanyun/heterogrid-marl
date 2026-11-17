"""The fixed communication/scale/roster replication protocol."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.common.episode_log import episode_rows
from algorithms.common.experiment_log import write_atomic
from heterogrid import EnvConfig

SEEDS = (42, 43, 44)
SCALES = {"B0": ((24, 24, 6), 6), "M2": ((36, 36, 9), 6),
          "N2": ((24, 24, 6), 9), "D2": ((27, 27, 7), 9)}
ROSTERS = {"R_scout": (3, 1, 2), "R_strike": (2, 3, 1), "R_hybrid": (1, 2, 3)}
CONDITIONS = tuple(f"{comm}_{scale}" for comm in ("R5", "P50", "C1S") for scale in SCALES) + tuple(
    "C1S_" + name for name in ROSTERS)
PROTOCOL = {"target_steps_per_seed": 5_000_000, "validation_seed": 1_910_042,
            "heldout_seed": 2_800_042, "eval_episodes": 64, "heldout_episodes": 128,
            "eval_interval": 250_000, "log_interval": 10_000, "replay_episodes": 1,
            "environment_seed_mode": "episode-index", "environment_episode_index_origin": 0,
            "training_seed_rule": "1000000 + training_seed * 100000000 + episode_index",
            "terminate_on_no_firepower": False, "primary_model": "latest.pt",
            "budget_policy": "finish the first complete on-policy batch reaching 5M; actual overshoot < 2000 team steps",
            "reward_comparison_caveat": "N2 uses none; D2 uses reference. Their difference is not a pure density effect."}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def environment_config(value):
    data = dict(value)
    if data.pop("terminate_on_no_firepower", False) is not False or data.pop("draw_reward", 0.0) != 0:
        raise ValueError("Only legacy termination and zero draw reward are supported by this study")
    return EnvConfig.from_dict(data)


def validate_config(name, value):
    if name not in CONDITIONS:
        raise ValueError("Unknown protocol condition: " + name)
    # This checkout implements elimination/timeout termination only. Never silently enable a new rule.
    env = environment_config(value)
    comm, scale = name.split("_", 1)
    size, number = SCALES.get(scale, ((24, 24, 6), 6))
    counts = ROSTERS.get(scale, (number // 3,) * 3)
    if env.map_size != size or env.max_steps != 200 or env.opponent != "rule":
        raise ValueError("Map, horizon or opponent differs from protocol: " + name)
    if env.red != env.blue or len(env.red) != number:
        raise ValueError("Both rosters must match, without changing their order: " + name)
    if tuple(Counter(unit.role for unit in env.red)[role] for role in ("scout", "strike", "hybrid")) != counts:
        raise ValueError("Wrong role counts: " + name)
    if env.role_stats != EnvConfig().role_stats:
        raise ValueError("Role capabilities must remain unchanged")
    if env.target_sharing != "auto" or env.target_ttl != 2:
        raise ValueError("Automatic target sharing and TTL=2 are required")
    expected = {"R5": (("distance", 5, 0), ("distance", 10, 0)),
                "P50": (("full", 10, .5), ("distance", 10, 0)),
                "C1S": (("none", 10, 0), ("none", 10, 0))}[comm]
    for team, (mode, radius, dropout) in zip(("red", "blue"), expected):
        cfg = env.communication_for(team)
        if getattr(env, team + "_communication") is None:
            raise ValueError("Both communication overrides must be explicit")
        if cfg.mode != mode or cfg.dropout_probability != dropout or (mode == "distance" and cfg.radius != radius):
            raise ValueError("Wrong " + team + " communication: " + name)
    normalization = "reference" if scale == "D2" else "none"
    if env.reward_normalization != normalization:
        raise ValueError("Wrong reward normalization: " + name)
    if (env.reward_reference_team_size, env.reward_reference_team_hp) != (6, 480):
        raise ValueError("Reward reference must remain 6 agents / 480 HP")
    base = EnvConfig.load(Path(__file__).resolve().parents[2] / "configs/scenarios/6v6_rule_distance.json")
    for field in ("reward_mode", "damage_reward", "kill_reward", "win_reward", "loss_reward", "timeout_as_loss",
                  "time_penalty", "discovery_reward", "sharing_reward", "information_reward_cap", "scripted_target_sharing",
                  "message_dim"):
        if getattr(env, field) != getattr(base, field):
            raise ValueError("Changed fixed environment setting: " + field)
    return env


def supplied_roster_configs(base):
    rosters = {
        "R_scout": [("scout", "uav"), ("scout", "usv"), ("scout", "uav"),
                    ("strike", "robot"), ("hybrid", "uav"), ("hybrid", "robot_dog")],
        "R_strike": [("scout", "uav"), ("scout", "usv"), ("strike", "robot"),
                     ("strike", "robot_dog"), ("strike", "robot"), ("hybrid", "uav")],
        "R_hybrid": [("scout", "uav"), ("strike", "robot"), ("strike", "robot_dog"),
                     ("hybrid", "uav"), ("hybrid", "robot_dog"), ("hybrid", "uav")],
    }
    result = {}
    for name, units in rosters.items():
        value = json.loads(json.dumps(base))
        value.update(red=[dict(role=role, platform=platform) for role, platform in units],
                     blue=[dict(role=role, platform=platform) for role, platform in units],
                     terminate_on_no_firepower=False, draw_reward=0.0)
        validate_config("C1S_" + name, value)
        result["C1S_" + name] = value
    return result


def scale_configs(project):
    project = Path(project)
    base = EnvConfig.load(project / "configs/scenarios/6v6_rule_distance.json").to_dict()
    nine = EnvConfig.load(project / "configs/scaling/density_9v9.json").to_dict()
    result = {}
    for name in CONDITIONS[:12]:
        comm, scale = name.split("_", 1)
        value = json.loads(json.dumps(base if SCALES[scale][1] == 6 else nine))
        value.update(map_size=list(SCALES[scale][0]), diagnostics_enabled=True,
                     reward_normalization="reference" if scale == "D2" else "none")
        red = {"mode": {"R5": "distance", "P50": "full", "C1S": "none"}[comm],
               "radius": 5 if comm == "R5" else 10, "dropout_probability": .5 if comm == "P50" else 0}
        blue = {"mode": "none" if comm == "C1S" else "distance", "radius": 10, "dropout_probability": 0}
        value.update(red_communication=red, blue_communication=blue)
        validate_config(name, value)
        result[name] = value
    return result


def training_command(manifest, job):
    algorithm = job["algorithm"]
    run = manifest["algorithms"][algorithm]
    command = [manifest["python"], "-u", "-m", "algorithms." + algorithm + ".train",
               "--" + ("smpe-root" if algorithm == "smpe" else "epymarl-root"), run["backend_root"],
               "--config", str(Path(run["output"]) / "configs" / (job["condition"] + ".json")),
               "--output", job["path"], "--seed", str(job["seed"]), "--device", job["device"],
               "--steps", str(manifest["protocol"]["target_steps_per_seed"]),
               "--eval-seed", str(manifest["protocol"]["validation_seed"]),
               "--heldout-seed", str(manifest["protocol"]["heldout_seed"]),
               "--environment-seed-mode", "episode-index", "--greedy-validation-only", "--skip-initial-heldout"]
    for key in ("eval_episodes", "heldout_episodes", "eval_interval", "log_interval", "replay_episodes"):
        command += ["--" + key.replace("_", "-"), str(manifest["protocol"][key])]
    if algorithm == "smpe":
        command += ["--compact-episode-log", "--save-interval", str(manifest["protocol"]["target_steps_per_seed"])]
    return command


def learning_statistics(metrics, budget):
    series = metrics["test_battle_won_mean"]
    actual = metrics["test_actual_environment_steps"]["values"]
    pairs = list(zip(series["steps"], series["values"], actual))
    if len(pairs) != len(actual) or pairs[0][0] != 0 or pairs[-1][0] != budget:
        raise ValueError("Incomplete validation curve")
    integrate = getattr(np, "trapezoid", None) or np.trapz
    auc = float(integrate(series["values"], series["steps"]))
    first = next(((scheduled, used) for scheduled, win, used in pairs if win >= .8), None)
    return {"validation_win_auc_team_steps": auc, "validation_win_auc_normalized": auc / budget,
            "reached_80pct": first is not None, "first_80pct_scheduled_steps": first[0] if first else None,
            "first_80pct_actual_steps": int(first[1]) if first else None}


def finalize_run(path):
    from algorithms.common.compact_results import compact, plan
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    algorithm = config["name"]
    if algorithm == "smpe":
        from algorithms.smpe.audit import audit
    else:
        from algorithms.epymarl_mappo.audit import audit
    audit(path, recompute=True)
    cli = config["cli"]
    if cli["environment_seed_mode"] != "episode-index":
        raise ValueError("Wrong training seed mode")
    count = 0
    # Collection completes episodes out of order; verify allocation indices, not completion order.
    indices = set()
    base = 1_000_000 + cli["seed"] * 100_000_000
    for row in episode_rows(path):
        index = row["environment_episode_index"]
        if index in indices or row["seed"] != base + index:
            raise ValueError("Duplicate or incorrect training environment seed")
        indices.add(index)
        count += 1
    if indices != set(range(count)):
        raise ValueError("Missing training episode indices")
    summary = json.loads((path / "summary.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    report = learning_statistics(metrics, cli["steps"])
    record = json.loads((path / "run.json").read_text())
    total_seconds = (datetime.fromisoformat(record["stop_time"]) - datetime.fromisoformat(record["start_time"])).total_seconds()
    report.update(algorithm=algorithm, training_seed=cli["seed"], environment_steps=summary["environment_steps"],
                  requested_environment_steps=cli["steps"], training_seconds=summary["training_seconds"],
                  environment_steps_per_second=summary["environment_steps_per_second"], training_seed_audit="PASSED",
                  total_wall_seconds=total_seconds, wall_environment_steps_per_second=summary["environment_steps"] / total_seconds,
                  validation_kill_ratio={"steps": metrics["test_kills_mean"]["steps"],
                      "values": [value / len(config["environment"]["blue"]) for value in metrics["test_kills_mean"]["values"]]}, models={})
    for model in ("latest", "best"):
        report["models"][model] = {}
        for mode in ("greedy", "sampled"):
            result = summary["heldout"][model][mode]
            if [row["seed"] for row in result["results"]] != list(range(cli["heldout_seed"], cli["heldout_seed"] + cli["heldout_episodes"])):
                raise ValueError("Wrong heldout seeds")
            report["models"][model][mode] = {key: result[key] for key in (
                "win_rate", "timeout_rate", "contact_rate", "mean_first_contact_step", "mean_useful_reports")}
            report["models"][model][mode]["mean_kill_ratio"] = result["mean_kills"] / len(config["environment"]["blue"])
    write_atomic(path / "protocol_metrics.json", report)
    item = plan(path)
    if item is None:
        raise RuntimeError("Completed run cannot be safely compacted")
    compact(item)
    audit(path, recompute=True)
    return report
