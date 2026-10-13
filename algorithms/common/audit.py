"""Common record integrity and deterministic environment replay checks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, SMACAdapter
from heterogrid.core import REWARD_COMPONENTS


def audit_records(path):
    path = Path(path)
    run = json.loads((path / "run.json").read_text())
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    info = json.loads((path / "info.json").read_text())
    assert run["status"] == "COMPLETED", run["status"]
    assert summary["checkpoint_reload_equal"] and summary["invalid_red_actions"] == 0
    assert Path(config["registered_path"]).resolve() == path.resolve()
    for source in run["experiment"]["sources"]:
        assert hashlib.sha256((path / source["snapshot"]).read_bytes()).hexdigest() == source["sha256"]
    for name, metric in metrics.items():
        assert len(metric["steps"]) == len(metric["values"]) == len(metric["timestamps"])
        assert np.isfinite(metric["values"]).all()
        assert metric["steps"] == sorted(metric["steps"])
        assert info[name] == metric["values"] and info[name + "_T"] == metric["steps"]
    return run, config, summary, metrics, info


def audit_replays(path, config, recompute=False):
    path = Path(path)
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
    return {"replays": len(replays), "recomputed_replays": len(recomputed_groups)}
