"""Audit SMPE updates, full policy artifacts, evaluation budgets and replays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithms.common.audit import audit_records, audit_replays
from algorithms.common.experiment_log import write_atomic


def audit(path, recompute=False):
    path = Path(path)
    run, config, summary, metrics, _ = audit_records(path)
    cli, settings = config["cli"], config["smpe_args"]
    maximum_overshoot = settings["batch_size_run"] * config["environment"]["max_steps"]
    assert config["t_max"] <= summary["environment_steps"] < config["t_max"] + maximum_overshoot
    for name in ("actor", "critic", "critic_w") + (("agent_models", "filters") if summary["model_updates"] else ()):
        assert summary["parameter_deltas_l2"][name] > 0, name
    if config["t_max"] >= maximum_overshoot * 8:
        assert summary["model_updates"] > 0
    expected = [0] + (list(range(cli["eval_interval"], cli["steps"], cli["eval_interval"])) if cli["eval_interval"] else []) + [cli["steps"]]
    for name in ("test_return_mean", "sampled_test_return_mean", "test_actual_environment_steps"):
        assert metrics[name]["steps"] == expected
    actual = metrics["test_actual_environment_steps"]["values"]
    assert all(scheduled <= used < scheduled + maximum_overshoot for scheduled, used in zip(expected, actual))
    assert actual[-1] == summary["environment_steps"]
    assert not (path / "trajectories").exists() and not (path / "evaluations").exists()
    episode_steps = episode_count = 0
    with (path / "training_episodes.jsonl").open() as file:
        for line in file:
            row = json.loads(line)
            assert row["invalid_red_actions"] == 0
            np.testing.assert_allclose(sum(row["reward_components"].values()), row["return"], atol=1e-7)
            episode_steps += row["length"]
            episode_count += 1
    assert episode_steps == summary["environment_steps"] and episode_count == summary["training_episodes"]
    saved = torch.load(path / "models/latest.pt", map_location="cpu")
    assert saved["environment_steps"] == summary["environment_steps"]
    assert set(saved["modules"]) == {"actor", "critic", "critic_w", "target_critic", "target_critic_w",
                                     "agent_models", "filters", "target_filters"}
    assert {"obs", "state", "reward", "learner_reward"} <= set(saved["normalizers"])
    assert len(saved["hash_counts"]["z"]) == len(config["environment"]["red"])
    replays = audit_replays(path, config, recompute)
    result = {"path": str(path.resolve()), "status": "PASSED", "environment_steps": summary["environment_steps"],
              "overshoot": summary["environment_steps"] - cli["steps"], "model_updates": summary["model_updates"],
              "metric_series": len(metrics), "source_snapshots": len(run["experiment"]["sources"]), **replays}
    write_atomic(path / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(audit(path, args.recompute)), flush=True)


if __name__ == "__main__":
    main()
