"""Audit native EPyMARL MAPPO updates, checkpoints, rewards and replays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithms.common.audit import audit_records, audit_replays
from algorithms.common.episode_log import episode_rows
from algorithms.common.experiment_log import write_atomic


def audit(path, recompute=False):
    path = Path(path)
    run, config, summary, metrics, _ = audit_records(path)
    cli, args = config["cli"], config["algorithm_args"]
    assert config["name"] == "epymarl_mappo"
    maximum = args["batch_size_run"] * config["environment"]["max_steps"]
    assert cli["steps"] <= summary["environment_steps"] < cli["steps"] + maximum
    assert summary["ppo_optimizer_steps"] == summary["updates"] * args["epochs"]
    assert all(value > 0 for value in summary["parameter_deltas_l2"].values())
    count = steps = 0
    for row in episode_rows(path):
        count += 1
        steps += row["length"]
        assert row["invalid_red_actions"] == 0
        np.testing.assert_allclose(sum(row["reward_components"].values()), row["return"], atol=1e-7)
        if config["environment"].get("diagnostics_enabled"):
            assert set(row["team_episode_metrics"]) == {"red", "blue"}
    assert count == summary["training_episodes"] and steps == summary["environment_steps"]
    assert count == summary["updates"] * args["batch_size_run"]
    expected = [0] + list(range(cli["eval_interval"], cli["steps"], cli["eval_interval"])) + [cli["steps"]]
    series = ["test_battle_won_mean", "test_actual_environment_steps"]
    if not cli.get("greedy_validation_only", False):
        series.append("sampled_test_battle_won_mean")
    for key in series:
        assert metrics[key]["steps"] == expected
    actual = metrics["test_actual_environment_steps"]["values"]
    assert all(scheduled <= used < scheduled + maximum for scheduled, used in zip(expected, actual))
    saved = torch.load(path / "models/latest.pt", map_location="cpu")
    assert saved["algorithm"] == "epymarl_mappo" and saved["environment_steps"] == steps
    assert set(saved["modules"]) == {"actor", "old_actor", "critic", "target_critic"}
    for key, value in saved["modules"]["actor"].items():
        assert torch.equal(value, saved["modules"]["old_actor"][key])
    for optimizer in saved["optimizers"].values():
        for state in optimizer["state"].values():
            assert int(state["step"]) == summary["ppo_optimizer_steps"]
    assert ("reward" in saved["normalizers"]) == args["standardise_rewards"]
    assert saved["learner_counters"]["critic_training_steps"] == summary["updates"]
    assert not (path / "evaluations").exists() and not (path / "trajectories").exists()
    for phase in (summary["heldout_before"], summary["heldout"]["latest"], summary["heldout"]["best"]):
        for result in phase.values():
            assert result["episodes"] == cli["heldout_episodes"]
            assert [row["seed"] for row in result["results"]] == list(range(cli["heldout_seed"], cli["heldout_seed"] + cli["heldout_episodes"]))
    result = {"status": "PASSED", "path": str(path.resolve()), "environment_steps": steps,
              "overshoot": steps - cli["steps"], "training_episodes": count, "updates": summary["updates"],
              "ppo_optimizer_steps": summary["ppo_optimizer_steps"], "source_snapshots": len(run["experiment"]["sources"]),
              **audit_replays(path, config, recompute)}
    write_atomic(path / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--recompute", action="store_true")
    cli = parser.parse_args()
    print(json.dumps(audit(cli.path, cli.recompute)))
