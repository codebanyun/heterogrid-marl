"""Audit a completed MAPPO experiment's records, rollouts and viewer replays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, SMACAdapter
from heterogrid.core import REWARD_COMPONENTS


def audit(path, recompute=False):
    path = Path(path)
    run = json.loads((path / "run.json").read_text())
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    info = json.loads((path / "info.json").read_text())
    assert run["status"] == "COMPLETED", run["status"]
    assert summary["environment_steps"] == config["t_max"]
    assert summary["checkpoint_reload_equal"] and summary["invalid_red_actions"] == 0
    assert summary["actor_parameter_delta_l2"] > 0 and summary["critic_parameter_delta_l2"] > 0
    assert Path(config["registered_path"]).resolve() == path.resolve()
    for source in run["experiment"]["sources"]:
        assert hashlib.sha256((path / source["snapshot"]).read_bytes()).hexdigest() == source["sha256"]
    for name, metric in metrics.items():
        assert len(metric["steps"]) == len(metric["values"]) == len(metric["timestamps"])
        assert np.isfinite(metric["values"]).all()
        assert metric["steps"] == sorted(metric["steps"])
        assert info[name] == metric["values"] and info[name + "_T"] == metric["steps"]
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
    if not interval:
        assert steps == summary["environment_steps"]
    replays = sorted((path / "replays").rglob("episode_*.json"))
    recomputed_groups = set()
    for file in replays:
        replay = json.loads(file.read_text())
        row = replay["evaluation"]
        transitions = replay["transitions"]
        assert replay["schema_version"] == 1
        assert len(replay["frames"]) == row["length"] + 1
        assert len(transitions) == row["length"]
        assert transitions[-1]["done"]
        assert row["invalid_red_actions"] == 0
        np.testing.assert_allclose(sum(t["team_reward"] for t in transitions), row["return"], atol=1e-7)
        for transition in transitions:
            np.testing.assert_allclose(sum(transition["reward_components"].values()), transition["team_reward"], atol=1e-7)
        if recompute and file.parent not in recomputed_groups:
            env = SMACAdapter(config=EnvConfig.from_dict(config["environment"]))
            try:
                env.reset(seed=row["seed"])
                assert env.env.core.snapshot() == replay["frames"][0], (file, "reset")
                for index, transition in enumerate(transitions, start=1):
                    reward, done, _ = env.step(transition["actions"])
                    assert env.env.core.snapshot() == replay["frames"][index], (file, index)
                    np.testing.assert_allclose(reward, transition["team_reward"], atol=1e-7)
                    assert done == transition["done"]
            finally:
                env.close()
            recomputed_groups.add(file.parent)
    result = {"path": str(path.resolve()), "environment_steps": summary["environment_steps"],
              "stored_rollout_steps": steps, "rollouts": len(rollouts),
              "replays": len(replays), "metric_series": len(metrics), "source_snapshots": len(run["experiment"]["sources"]),
              "recomputed_replays": len(recomputed_groups), "status": "PASSED"}
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
