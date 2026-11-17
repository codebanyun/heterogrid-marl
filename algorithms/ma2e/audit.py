"""Audit interaction accounting, MAE updates, target networks and deterministic replays."""
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
    args, cli = config["algorithm_args"], config["cli"]
    assert config["name"] == "ma2e" and args["use_MT_mode"]
    maximum = args["batch_size_run"] * config["environment"]["max_steps"]
    assert cli["steps"] <= summary["environment_steps"] < cli["steps"] + maximum
    assert summary["updates"] > 0 and summary["mae_updates"] > 0 and summary["mae_parameter_delta_l2"] > 0
    assert all(value > 0 for value in summary["parameter_deltas_l2"].values())
    count = steps = pretraining = pretraining_episodes = 0
    for row in episode_rows(path):
        count += 1
        steps += row["length"]
        if row["phase"] == "mae_pretraining":
            pretraining += row["length"]
            pretraining_episodes += 1
        assert row["invalid_red_actions"] == 0
        np.testing.assert_allclose(sum(row["reward_components"].values()), row["return"], atol=1e-7)
        if config["environment"].get("diagnostics_enabled"):
            assert set(row["team_episode_metrics"]) == {"red", "blue"}
    assert count == summary["training_episodes"] and steps == summary["environment_steps"]
    assert pretraining == summary["pretraining_environment_steps"]
    assert pretraining_episodes == summary["pretraining_episodes"]
    expected = [0] + list(range(cli["eval_interval"], cli["steps"], cli["eval_interval"])) + [cli["steps"]]
    for key in ("test_battle_won_mean", "sampled_test_battle_won_mean", "test_actual_environment_steps"):
        assert metrics[key]["steps"] == expected
    assert all(t <= used < t + maximum for t, used in zip(expected, metrics["test_actual_environment_steps"]["values"]))
    saved = torch.load(path / "models/latest.pt", map_location="cpu")
    assert saved["algorithm"] == "ma2e" and saved["environment_steps"] == steps
    assert set(saved["modules"]) == {"actor", "target_actor", "mixer", "target_mixer"}
    assert saved["counters"]["updates"] == summary["updates"]
    for state in saved["optimizer"]["state"].values():
        assert int(state["step"]) == summary["updates"]
    network = json.loads((path / "network.json").read_text())
    assert not network["actor_uses_global_state"] and not network["actor_uses_other_agents_observations"]
    assert not (path / "evaluations").exists() and not (path / "trajectories").exists()
    for phase in (summary["heldout_before"], summary["heldout"]["latest"], summary["heldout"]["best"]):
        for result in phase.values():
            assert result["episodes"] == cli["heldout_episodes"]
            assert [r["seed"] for r in result["results"]] == list(range(cli["heldout_seed"], cli["heldout_seed"] + cli["heldout_episodes"]))
    result = {"status": "PASSED", "path": str(path.resolve()), "environment_steps": steps,
        "pretraining_environment_steps": pretraining, "updates": summary["updates"],
        "mae_updates": summary["mae_updates"], "mae_parameter_delta_l2": summary["mae_parameter_delta_l2"],
        "source_snapshots": len(run["experiment"]["sources"]), **audit_replays(path, config, recompute)}
    write_atomic(path / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--recompute", action="store_true")
    cli = parser.parse_args()
    print(json.dumps(audit(cli.path, cli.recompute)))
