"""MAPPO/SMPE D0-D3 study with reference rewards and bilateral diagnostics."""
import csv
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

from algorithms.common import communication_study as shared
from algorithms.common.episode_log import episode_rows
from algorithms.common.experiment_log import write_atomic
from heterogrid.scenarios import scaling_scenarios

CONDITION_NAMES = ("D0_baseline", "D1_3v3", "D2_9v9", "D3_12v12")


def conditions(base):
    baseline = replace(shared.conditions(base)["B0_baseline"], dropout_probability=base.dropout_probability,
                       reward_normalization="reference",
                       reward_reference_team_size=6, reward_reference_team_hp=480.0, diagnostics_enabled=True)
    generated = scaling_scenarios(baseline, counts=(6, 3, 9, 12), mode="constant_density")
    return {label: generated[str(count) + "v" + str(count)]
            for label, count in zip(CONDITION_NAMES, (6, 3, 9, 12))}


def audit_density(path, recompute=False):
    path = Path(path)
    result = shared.audit_run(path, recompute=recompute)
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    env = config["environment"]
    assert env["diagnostics_enabled"] and env["reward_normalization"] == "reference"
    assert env["reward_reference_team_size"] == 6 and env["reward_reference_team_hp"] == 480
    assert env["opponent"] == "rule" and env["max_steps"] == 200
    for team in ("red", "blue"):
        assert env[team + "_communication"] == dict(mode="distance", radius=10.0, dropout_probability=0.0)
    assert config["cli"]["compact_episode_log"]
    assert (path / "training_episodes.jsonl.gz").exists()
    assert not (path / "training_episodes.json").exists() and not (path / "training_episodes.jsonl").exists()
    count = completed_steps = 0
    for row in episode_rows(path):
        count += 1
        completed_steps += row["length"]
        assert abs(sum(row["reward_components"].values()) - row["return"]) < 1e-7
        for team in ("red", "blue"):
            diagnostics = row["team_episode_metrics"][team]["diagnostics"]
            assert diagnostics["schema_version"] == 1
            assert diagnostics["initial_agents"] == len(env[team])
    assert count == summary["training_episodes"] and count > 0
    remaining = summary["environment_steps"] - completed_steps
    if config["name"] == "smpe":
        assert remaining == 0
    else:
        assert 0 <= remaining < config["cli"]["num_envs"] * env["max_steps"]
    evaluated = 0
    for stage in (summary["heldout_before"], summary["heldout"]["latest"], summary["heldout"]["best"]):
        for result_set in stage.values():
            assert result_set["episodes"] == config["cli"]["heldout_episodes"]
            for row in result_set["results"]:
                evaluated += 1
                assert set(row["team_episode_metrics"]) == {"red", "blue"}
                for team in ("red", "blue"):
                    assert row["team_episode_metrics"][team]["diagnostics"]["initial_agents"] == len(env[team])
    result.update(training_diagnostic_episodes=count, heldout_diagnostic_episodes=evaluated,
                  uncompleted_training_steps=remaining, compressed_episode_log=True)
    write_atomic(path / "audit.json", result)
    return result


def flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        if key == "schema_version":
            continue
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + "."))
        else:
            result[name] = value
    return result


def diagnostic_summary(path, runs):
    rows, by_seed = [], {}
    for run in runs:
        heldout = run["summary"]["heldout"]["latest"]["greedy"]
        by_seed[run["seed"]] = flatten(heldout["mean_team_episode_metrics"])
    for metric in sorted({key for values in by_seed.values() for key in values}):
        values = [by_seed[run["seed"]].get(metric) for run in runs]
        observed = [value for value in values if value is not None]
        n = len(observed)
        std = statistics.stdev(observed) if n > 1 else None
        row = dict(metric=metric, mean=statistics.mean(observed) if n else None, std=std,
                   sem=std / math.sqrt(n) if n > 1 else None, n_seeds=n)
        row.update({"seed_" + str(run["seed"]): value for run, value in zip(runs, values)})
        rows.append(row)
    write_atomic(path / "diagnostics.json", {"phase": "heldout.latest.greedy", "weighting": "mean across training seeds",
                 "runs": [{"seed": run["seed"], "episodes": run["summary"]["heldout"]["latest"]["greedy"]["episodes"],
                           "observed_episodes": run["summary"]["heldout"]["latest"]["greedy"]["team_episode_metrics_observed_episodes"]}
                          for run in runs], "metrics": rows})
    with (path / "diagnostics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "mean", "std", "sem", "n_seeds"] +
                                ["seed_" + str(run["seed"]) for run in runs])
        writer.writeheader()
        writer.writerows(rows)


def summarize_density(path, runs):
    if path.parent.name == "mappo":
        from algorithms.mappo.study import summarize
    else:
        from algorithms.smpe.matrix import summarize
    result = summarize(path, runs)
    diagnostic_summary(path, runs)
    with (path / "RESULTS.md").open("a", encoding="utf-8") as file:
        file.write("\n本组为密度受控 D 组：双方距离通信半径 10、无失联，规则蓝方，200 步，角色比例 1:1:1。\n"
                   "参考奖励归一化仅缩放伤害和击杀；诊断不输入 actor，不改变奖励或随机流。\n"
                   "最终贪心独立测试的双方诊断见 diagnostics.csv 和 diagnostics.json；空值保留，另列有效种子及局数。\n"
                   "全部逐局训练诊断无损保存在 training_episodes.jsonl.gz，不重复保存整数组。\n"
                   "地图取整、接敌距离、通信连通性及边界效应仍会影响难度，不与旧 N 组直接作纯密度归因。\n")
    return result


def main():
    shared.main(profile="density")


if __name__ == "__main__":
    main()
