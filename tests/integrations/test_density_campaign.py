"""D-group reuse is explicit and supplementary reports never replace old results."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from algorithms.common.density_campaign import assert_baseline_equivalent, training_command
from algorithms.common.value_decomposition.report import summarize
from heterogrid import EnvConfig


class DensityCampaignTests(unittest.TestCase):
    def test_d0_reuse_requires_equivalent_task_and_exact_unit_reward_factors(self):
        base = EnvConfig()
        new = replace(base, reward_normalization="reference", diagnostics_enabled=True)
        assert_baseline_equivalent(base, new)
        with self.assertRaises(ValueError):
            assert_baseline_equivalent(base, replace(new, comm_radius=5))
        with self.assertRaises(ValueError):
            assert_baseline_equivalent(base, replace(new, reward_reference_team_hp=960))
        with self.assertRaises(ValueError):
            assert_baseline_equivalent(base, replace(new, diagnostics_enabled=False))

    def test_training_cli_preserves_original_budget_and_hyperparameters(self):
        source = {"cli": {"algorithm": "qmix", "marl_root": "/old", "config": "/B0.json",
                           "output": "/B0/seed_42", "steps": 5000000, "seed": 42,
                           "batch_size": 32, "warmup_episodes": 32, "update_interval": 200}}
        command = training_command("qmix", source, "/frozen", "/D3.json", "/D3/seed_42")
        for flag, value in (("--marl-root", "/frozen"), ("--config", "/D3.json"),
                            ("--steps", "5000000"), ("--batch-size", "32"), ("--update-interval", "200")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(source["cli"]["config"], "/B0.json")

    def test_supplementary_report_reads_alias_curves_and_leaves_old_reports_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = {"algorithm": "qmix", "seeds": [42, 43, 44], "steps": 5000000,
                "eval_episodes": 64, "heldout_episodes": 128, "labels": {"D0": "reused B0", "D1": "3v3"},
                "groups": {"density_scale": ["D0", "D1"]}, "validation_sources": {"D0": "B0"},
                "report_notes": ["D0 reuses B0, not a new replicate"]}
            (root / "D_experiment.json").write_text(json.dumps(experiment))
            (root / "RESULTS.md").write_text("original results")
            (root / "aggregate.json").write_text("original aggregate")
            (root / "status.json").write_text("original status")
            (root / "figures").mkdir()
            (root / "figures/map_size.png").write_bytes(b"original figure")
            for seed in (42, 43, 44):
                path = root / "B0" / f"seed_{seed}" / "evaluations/validation_000000000"
                path.mkdir(parents=True)
                (path / "summary.json").write_text(json.dumps({"win_rate": .5, "episodes": 64}))
            with patch("algorithms.common.value_decomposition.report.plot_curves") as plot:
                result = summarize(root, experiment_file="D_experiment.json", report_directory="D_report")
                self.assertEqual(result, {"completed": 0, "planned": 6})
                self.assertEqual(set(plot.call_args[0][1]), {"D0", "D1"})
                self.assertEqual(plot.call_args[0][1]["D0"][0]["n"], 3)
                self.assertEqual(plot.call_args[0][1]["D1"], [])
            self.assertEqual((root / "RESULTS.md").read_text(), "original results")
            self.assertEqual((root / "aggregate.json").read_text(), "original aggregate")
            self.assertEqual((root / "status.json").read_text(), "original status")
            self.assertEqual((root / "figures/map_size.png").read_bytes(), b"original figure")
            self.assertIn("not a new replicate", (root / "D_report/RESULTS.md").read_text())
