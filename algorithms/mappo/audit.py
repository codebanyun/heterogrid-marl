"""Audit a completed MAPPO experiment's records, rollouts and viewer replays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from algorithms.common.audit import audit_records, audit_replays
from heterogrid.core import REWARD_COMPONENTS


def audit(path, recompute=False):
    path = Path(path)
    run, config, summary, metrics, info = audit_records(path)
    assert summary["environment_steps"] == config["t_max"]
    assert summary["actor_parameter_delta_l2"] > 0 and summary["critic_parameter_delta_l2"] > 0
    cli = config["cli"]
    if cli["eval_interval"]:
        expected = [0] + list(range(cli["eval_interval"], cli["steps"], cli["eval_interval"])) + [cli["steps"]]
        assert metrics["test_return_mean"]["steps"] == expected
        assert metrics["sampled_test_return_mean"]["steps"] == expected
    rollouts = sorted((path / "trajectories").glob("rollout_*.npz"))
    interval = cli.get("trajectory_interval", 0)
    batch = cli["num_envs"] * cli["rollout_length"]
    expected_updates = [u for u in range(1, summary["updates"] + 1)
                        if not interval or (u * batch) % interval == 0 or u in (1, summary["updates"])]
    if cli.get("no_trajectories", False):
        expected_updates = []
    assert [int(f.stem.split("_")[-1]) for f in rollouts] == expected_updates
    steps = 0
    for file in rollouts:
        with np.load(file) as data:
            for name in data.files:
                assert np.isfinite(data[name]).all(), (file, name)
            actions = data["actions"][..., 0].astype(np.int64)
            assert (actions == data["actions"][..., 0]).all()
            masks = data["available_actions"][:-1]
            assert ((actions >= 0) & (actions < masks.shape[-1])).all()
            assert np.take_along_axis(masks, actions[..., None], axis=-1).all(), file
            rewards = sum(data["reward_" + key] for key in REWARD_COMPONENTS)
            np.testing.assert_allclose(data["rewards"], np.broadcast_to(rewards[..., None, None], data["rewards"].shape), atol=2e-6)
            assert data["obs"].shape[0] == actions.shape[0] + 1
            np.testing.assert_array_equal(data["masks"][1:, :, 0, 0] == 0, data["environment_done"])
            steps += actions.shape[0] * actions.shape[1]
    assert steps == len(expected_updates) * batch
    if not interval and not cli.get("no_trajectories", False):
        assert steps == summary["environment_steps"]
    replay_report = audit_replays(path, config, recompute)
    result = {"path": str(path.resolve()), "environment_steps": summary["environment_steps"],
              "stored_rollout_steps": steps, "rollouts": len(rollouts),
              "replays": replay_report["replays"], "metric_series": len(metrics), "source_snapshots": len(run["experiment"]["sources"]),
              "recomputed_replays": replay_report["recomputed_replays"], "status": "PASSED"}
    (path / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--recompute", action="store_true", help="Re-execute one complete replay per evaluation group")
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(audit(path, recompute=args.recompute)), flush=True)


if __name__ == "__main__":
    main()
