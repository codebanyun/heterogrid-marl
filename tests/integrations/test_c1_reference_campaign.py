import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from algorithms.common.c1_reference_campaign import commands, compact_completed, completed_reference, report
from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, c1_reference_scenarios, scenarios
try:
    from algorithms.common.value_decomposition.train import aggregate, episode_row
except ImportError:
    aggregate = episode_row = None
from heterogrid import EnvConfig
from heterogrid.core import GridCombat, REWARD_COMPONENTS


class C1ReferenceTests(unittest.TestCase):
    def test_complete_historical_both_none_is_reused_but_changed_objective_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "qmix/20261011_sensitivity_5m/C1/seed_42"
            (run / "models").mkdir(parents=True)
            for model in ("final", "best"):
                (run / "models" / (model + ".pt")).write_bytes(model.encode())
            cfg = scenarios(False)["C1"]
            write_atomic(run / "env.json", cfg.to_dict())
            stats = {"episodes": 128, "base_seed": 200042}
            write_atomic(run / "summary.json", {"environment_steps": 5000000,
                "heldout": {model: {mode: stats for mode in ("greedy", "sampled")} for model in ("final", "best")}})
            target = replace(c1_reference_scenarios()["C1S"], diagnostics_enabled=True)
            ref = completed_reference(root, "qmix", "C1S", 42, target, 5000000)
            self.assertEqual(ref["source_run"], str(run.resolve()))
            self.assertEqual(ref["new_training_steps"], 0)
            self.assertEqual(ref["heldout_seed"], 200042)
            self.assertIsNone(completed_reference(root, "qmix", "C1S", 42,
                              replace(target, terminate_on_no_firepower=True), 5000000))
            self.assertIsNone(completed_reference(root, "qmix", "C1S", 42, target, 4000000))
            self.assertIsNone(completed_reference(root, "qmix", "C1", 42, target, 5000000))
            (run / "models/best.pt").unlink()
            self.assertIsNone(completed_reference(root, "qmix", "C1S", 42, target, 5000000))

    def test_only_blue_communication_changes_and_historical_matrix_unchanged(self):
        original = scenarios(True)
        tasks = c1_reference_scenarios()
        self.assertEqual(len(original), 13)
        self.assertEqual(tasks["C1"], original["C1"])
        a, b = tasks["C1"].to_dict(), tasks["C1S"].to_dict()
        self.assertEqual({key for key in a if a[key] != b[key]}, {"blue_communication"})
        self.assertEqual(tasks["C1S"].communication_for("red").mode, "none")
        self.assertEqual(tasks["C1S"].communication_for("blue").mode, "none")
        self.assertFalse(tasks["C1S"].terminate_on_no_firepower)
        for cfg in c1_reference_scenarios(early_termination=True).values():
            self.assertTrue(cfg.terminate_on_no_firepower)
        self.assertEqual(scenarios(True), original)

    def test_shipped_reference_configs_match_generator(self):
        tasks = c1_reference_scenarios()
        early = c1_reference_scenarios(early_termination=True)
        for filename, expected in (("6v6_rule_both_none.json", tasks["C1S"]),
                                   ("6v6_rule_both_none_early.json", early["C1S"]),
                                   ("6v6_rule_red_none_early.json", early["C1"])):
            self.assertEqual(EnvConfig.load(PROJECT_ROOT / "configs/references" / filename), expected)

    def test_reference_keeps_same_initial_positions_and_red_local_perception(self):
        cores = [GridCombat(cfg) for cfg in c1_reference_scenarios().values()]
        for core in cores:
            core.reset(42)
            self.assertFalse(core.comm_mask("red").any())
        self.assertFalse(cores[1].comm_mask("blue").any())
        for name in cores[0].all_names:
            np.testing.assert_array_equal(cores[0].units[name].position, cores[1].units[name].position)
        for name in cores[0].names["red"]:
            for key, value in cores[0].observation(name).items():
                np.testing.assert_array_equal(value, cores[1].observation(name)[key])

    def test_baseline_command_preserves_hyperparameters_and_uses_requested_seed(self):
        metadata = {"cli": {"algorithm": "qmix", "seed": 42, "steps": 5000000,
                            "batch_size": 32, "update_interval": 200, "buffer_size": 1000,
                            "marl_root": "/old", "config": "/old.json", "output": "/old_run"}}
        args = SimpleNamespace(steps=5000000, eval_interval=250000, eval_episodes=64,
                               heldout_episodes=128, eval_seed=910042, heldout_seed=1800042)
        cmd = commands("qmix", metadata, "/frozen", "/C1S.json", "/new", args, 44)
        for flag, value in (("--seed", "44"), ("--batch-size", "32"), ("--update-interval", "200"),
                            ("--heldout-seed", "1800042"), ("--marl-root", "/frozen")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertEqual(metadata["cli"]["seed"], 42)
        for algorithm, profile in (("dib_v2", "tactical_v2"), ("dib_v3", "resilient_v3")):
            cmd = commands(algorithm, None, "/backend", "/C1S.json", "/new", args, 43)
            self.assertEqual(cmd[cmd.index("--training-profile") + 1], profile)
            self.assertEqual(cmd[cmd.index("--tactical-preset") + 1], "belief_first")
            self.assertNotIn("--resume", cmd)

    def test_compaction_preserves_models_and_exact_decoded_episode_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "models").mkdir()
            for name in ("final", "best", "initial", "step_002000000"):
                (root / "models" / (name + ".pt")).write_bytes(name.encode())
            write_atomic(root / "summary.json", {"finished": True})
            original = b'{"episode":1}\n{"episode":2}\n'
            (root / "training_episodes.jsonl").write_bytes(original)
            compact_completed(root)
            self.assertEqual((root / "models/latest.pt").read_bytes(), b"final")
            self.assertEqual((root / "models/final.pt").read_bytes(), b"final")
            self.assertTrue((root / "models/final.pt").is_symlink())
            self.assertEqual((root / "models/best.pt").read_bytes(), b"best")
            self.assertFalse((root / "models/initial.pt").exists())
            self.assertEqual(gzip.open(root / "training_episodes.jsonl.gz", "rb").read(), original)
            compact_completed(root)  # Idempotent.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                compact_completed(tmp)

    def test_report_never_aggregates_incomplete_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for seed in (42, 43, 44):
                out = root / ("seed_" + str(seed)); out.mkdir()
                jobs.append({"algorithm": "qmix", "scenario": "C1S", "seed": seed, "output": str(out)})
                if seed == 44:
                    continue
                stats = {"win_rate": .5, "timeout_rate": .5, "mean_length": 200}
                write_atomic(out / "summary.json", {"heldout": {model: {mode: stats for mode in ("greedy", "sampled")}
                             for model in ("final", "best")}, "training_seconds": 1,
                             "environment_steps_per_second": 5000000})
            state = {"jobs": jobs, "roots": {"qmix": str(root)}, "seeds": [42, 43, 44],
                     "scenarios": ["C1S"], "termination_protocol": "legacy_elimination"}
            report(root, state)
            self.assertEqual(json.loads((root / "aggregate.json").read_text())["results"], [])
            self.assertIn("pending", (root / "RESULTS.md").read_text())

    @unittest.skipIf(aggregate is None, "Optional training dependencies unavailable")
    def test_reason_aggregates_separate_exhaustion_draws_from_timeouts(self):
        core = GridCombat(replace(EnvConfig(), diagnostics_enabled=True))
        core.reset(42)
        rows = []
        for reason, winner in (("no_firepower", "blue"), ("mutual_no_firepower", None), ("timeout", None)):
            core.reason, core.winner = reason, winner
            rows.append(episode_row(SimpleNamespace(env=SimpleNamespace(core=core)), 42, 0,
                                    dict.fromkeys(REWARD_COMPONENTS, 0)))
        result = aggregate(rows)
        for key in ("timeout_rate", "no_firepower_rate", "mutual_no_firepower_rate", "mutual_draw_rate"):
            self.assertAlmostEqual(result[key], 1/3)


if __name__ == "__main__":
    unittest.main()
