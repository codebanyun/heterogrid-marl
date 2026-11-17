"""Campaign contracts using temporary references and no training subprocesses."""
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from algorithms.ma2e import campaign
from heterogrid import EnvConfig


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReferenceFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.references = tuple(self.root / name for name in ("communication", "sensitivity", "density"))
        self.groups = (
            ["B0_baseline"] + [f"C{i}_case" for i in range(1, 7)],
            [f"M{i}_case" for i in range(1, 4)] + [f"N{i}_case" for i in range(1, 4)],
            [f"D{i}_case" for i in range(4)],
        )
        self.protocol = dict(validation_seed=100042, heldout_seed=200042,
                             eval_episodes=64, heldout_episodes=128,
                             eval_interval=250368, log_interval=10240)
        self.configs, self.sources = {}, {}
        base = json.loads(json.dumps(EnvConfig().to_dict()))
        for group, (reference, labels) in enumerate(zip(self.references, self.groups)):
            write_json(reference / "experiment.json", dict(self.protocol, conditions=labels))
            write_json(reference / "status.json", {"state": "COMPLETED"})
            for index, label in enumerate(labels):
                config = copy.deepcopy(base)
                config["map_size"] = [24 + index, 24 + group, 6]
                if group == 0:
                    config["blue_communication"] = dict(mode="distance", radius=10, dropout_probability=0)
                if group == 2:
                    config.update(reward_normalization="reference", diagnostics_enabled=True)
                write_json(reference / "configs" / (label + ".json"), config)
                self.configs[label], self.sources[label] = config, str(reference.resolve())


class ReferenceConfigTests(ReferenceFixture):
    def test_exactly_seventeen_scenarios_preserve_every_config_and_source(self):
        variants, sources, protocol = campaign.reference_configs(self.references)
        self.assertEqual(len(variants), 17)
        self.assertEqual(set(variants), {label for group in self.groups for label in group})
        self.assertEqual(variants, self.configs)
        self.assertEqual(sources, self.sources)
        for key, value in self.protocol.items():
            self.assertEqual(protocol[key], value)
        for label in self.groups[0]:
            with self.subTest(label=label):
                blue = EnvConfig.from_dict(variants[label]).communication_for("blue")
                self.assertEqual((blue.mode, blue.radius, blue.dropout_probability), ("distance", 10, 0))
        for label in self.groups[2]:
            with self.subTest(label=label):
                self.assertEqual(variants[label]["reward_normalization"], "reference")
                self.assertTrue(variants[label]["diagnostics_enabled"])

    def test_every_bc_scenario_requires_explicit_fixed_blue_communication(self):
        mutations = [None, dict(mode="full", radius=10, dropout_probability=0),
                     dict(mode="distance", radius=9, dropout_probability=0),
                     dict(mode="distance", radius=10, dropout_probability=0.1)]
        for label in self.groups[0]:
            path = self.references[0] / "configs" / (label + ".json")
            for blue in mutations:
                with self.subTest(label=label, blue=blue):
                    config = dict(self.configs[label], blue_communication=blue)
                    write_json(path, config)
                    try:
                        with self.assertRaisesRegex(ValueError, "blue communication must be explicitly fixed"):
                            campaign.reference_configs(self.references)
                    finally:
                        write_json(path, self.configs[label])

    def test_every_density_scenario_requires_normalization_and_diagnostics(self):
        for label in self.groups[2]:
            path = self.references[2] / "configs" / (label + ".json")
            for change in ({"reward_normalization": "none"}, {"diagnostics_enabled": False}):
                with self.subTest(label=label, change=change):
                    write_json(path, dict(self.configs[label], **change))
                    try:
                        with self.assertRaisesRegex(ValueError, "reference rewards and bilateral diagnostics"):
                            campaign.reference_configs(self.references)
                    finally:
                        write_json(path, self.configs[label])

    def test_incomplete_references_are_rejected(self):
        for reference in self.references:
            with self.subTest(reference=reference.name):
                write_json(reference / "status.json", {"state": "RUNNING"})
                try:
                    with self.assertRaisesRegex(ValueError, "must be completed"):
                        campaign.reference_configs(self.references)
                finally:
                    write_json(reference / "status.json", {"state": "COMPLETED"})

    def test_missing_scenario_is_rejected(self):
        write_json(self.references[0] / "experiment.json", dict(self.protocol, conditions=self.groups[0][:-1]))
        with self.assertRaisesRegex(ValueError, "Expected B0"):
            campaign.reference_configs(self.references)

    def test_mismatched_evaluation_protocols_are_rejected(self):
        path = self.references[1] / "experiment.json"
        for key in ("validation_seed", "heldout_seed", "eval_episodes", "heldout_episodes"):
            with self.subTest(key=key):
                protocol = dict(self.protocol, conditions=self.groups[1])
                protocol[key] += 1
                write_json(path, protocol)
                with self.assertRaisesRegex(ValueError, "protocols differ: " + key):
                    campaign.reference_configs(self.references)

    def test_manifest_hashes_all_scenarios_and_forwards_planned_resource_settings(self):
        project, external = self.root / "project", self.root / "upstream/src"
        for name in ("algorithms", "heterogrid", "configs"):
            (project / name).mkdir(parents=True)
        for name in ("backend.py", "train.py", "audit.py", "launch.py"):
            path = project / "algorithms/ma2e" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        for name in ("components", "config", "controllers", "envs", "learners", "modules", "utils"):
            (external / name).mkdir(parents=True)
        backend = types.ModuleType("algorithms.ma2e.backend")
        backend.CORRECTIONS = {"fixture": "record integration corrections"}
        backend.PROJECT_ROOT = project
        backend.resolve_root = mock.Mock(return_value=external)
        backend.settings = mock.Mock(return_value={"device": "cpu", "use_cuda": False})
        output = self.root / "campaign"
        args = ["campaign", "--ma2e-root", str(external), "--output", str(output),
                "--communication-reference", str(self.references[0]),
                "--sensitivity-reference", str(self.references[1]),
                "--density-reference", str(self.references[2]),
                "--workers", "4", "--gpu-max-workers", "3", "--gpu-min-free-mb", "8000",
                "--projected-storage-mb", "1900", "--minimum-free-disk-gb", "0.4"]
        with mock.patch.dict("sys.modules", {"algorithms.ma2e.backend": backend}), \
                mock.patch.object(campaign.sys, "argv", args), \
                mock.patch.object(campaign, "git_provenance", return_value={"commit": "fixture"}), \
                mock.patch.object(campaign.shutil, "disk_usage", return_value=types.SimpleNamespace(free=4 * 1024 ** 3)), \
                mock.patch.object(campaign.subprocess, "Popen", return_value=mock.Mock(pid=123)) as process, \
                mock.patch.object(campaign.subprocess, "run") as subprocess_run, \
                mock.patch.object(campaign, "run_queue") as queue, \
                contextlib.redirect_stdout(io.StringIO()):
            campaign.main()
        subprocess_run.assert_not_called()
        queue.assert_not_called()
        process.assert_called_once()
        self.assertEqual(process.call_args.args[0][1:5],
                         [campaign.sys.executable, "-u", "-m", "algorithms.ma2e.campaign"])
        self.assertIn("--manage", process.call_args.args[0])
        manifest = json.loads((output / "experiment.json").read_text())
        self.assertEqual(manifest["conditions"], list(self.configs))
        self.assertEqual(manifest["condition_references"], self.sources)
        self.assertEqual(manifest["seeds"], [42, 43, 44])
        self.assertEqual(manifest["jobs"], 51)
        self.assertEqual(manifest["target_steps_per_seed"], 5000192)
        self.assertEqual((manifest["workers"], manifest["gpu_max_workers"], manifest["gpu_min_free_mb"]), (4, 3, 8000))
        self.assertEqual(manifest["device"], "auto")
        self.assertEqual(manifest["projected_storage_bytes"], 1900 * 1024 ** 2)
        self.assertEqual(manifest["minimum_free_disk_bytes"], int(0.4 * 1024 ** 3))
        self.assertEqual(manifest["protocol_overrides"], {})
        expected_hashes = {label: hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                           for label, config in self.configs.items()}
        self.assertEqual(manifest["config_hashes"], expected_hashes)
        for label, config in self.configs.items():
            with self.subTest(label=label):
                frozen = json.loads((output / "configs" / (label + ".json")).read_text())
                self.assertEqual(frozen, config)
                digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
                self.assertEqual(manifest["config_hashes"][label], digest)


class TrainingCommandTests(unittest.TestCase):
    def setUp(self):
        self.output = Path("/campaign")
        self.manifest = dict(python="/python", ma2e_root="/frozen/ma2e/src", target_steps_per_seed=5000192,
                             validation_seed=100042, heldout_seed=200042, eval_episodes=64,
                             heldout_episodes=128, eval_interval=250368, log_interval=10240, replay_episodes=1)

    def assert_command(self, manifest, seed, device, gpu_memory, gpu_workers):
        job = dict(condition="D3_12v12", seed=seed, path=f"/campaign/D3_12v12/seed_{seed}")
        command = campaign.training_command(manifest, self.output, job)
        self.assertEqual(command[:4], ["/python", "-u", "-m", "algorithms.ma2e.launch"])
        expected = {"--ma2e-root": "/frozen/ma2e/src", "--config": "/campaign/configs/D3_12v12.json",
                    "--output": job["path"], "--steps": "5000192", "--seed": str(seed), "--device": device,
                    "--gpu-min-free-mb": str(gpu_memory), "--gpu-max-workers": str(gpu_workers),
                    "--eval-seed": "100042", "--heldout-seed": "200042", "--eval-episodes": "64",
                    "--heldout-episodes": "128", "--eval-interval": "250368", "--log-interval": "10240",
                    "--replay-episodes": "1"}
        self.assertEqual(len(command), 4 + 2 * len(expected))
        self.assertEqual(dict(zip(command[4::2], command[5::2])), expected)

    def test_default_auto_device_gpu_limits_and_all_seed_budgets(self):
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                self.assert_command(self.manifest, seed, "auto", 10000, 2)

    def test_explicit_devices_and_gpu_limits_are_forwarded_unchanged(self):
        for device in ("auto", "cpu", "cuda"):
            for seed in (42, 43, 44):
                with self.subTest(device=device, seed=seed):
                    manifest = dict(self.manifest, device=device, gpu_min_free_mb=8000, gpu_max_workers=3)
                    self.assert_command(manifest, seed, device, 8000, 3)


class ResultsTests(unittest.TestCase):
    def test_reports_latest_greedy_seed_mean_and_sem_not_individual_or_best_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "experiment.json", dict(conditions=["B0_baseline", "D3_12v12"], seeds=[42, 43, 44]))
            for condition, rates in (("B0_baseline", (0.1, 0.2, 0.6)), ("D3_12v12", (1.0, 1.0, 1.0))):
                for seed, rate in zip((42, 43, 44), rates):
                    summary = {"heldout": {"latest": {"greedy": {"win_rate": rate}, "sampled": {"win_rate": 0.99}},
                                           "best": {"greedy": {"win_rate": 0.99}}},
                               "heldout_before": {"greedy": {"win_rate": 0.99}}}
                    write_json(output / condition / f"seed_{seed}" / "summary.json", summary)
            campaign.write_results(output)
            text = (output / "RESULTS.md").read_text(encoding="utf-8")
            rows = [line for line in text.splitlines() if line.startswith(("| B0_baseline |", "| D3_12v12 |"))]
            self.assertEqual(rows, ["| B0_baseline | 30.00 \u00b1 15.28 | 3 |", "| D3_12v12 | 100.00 \u00b1 0.00 | 3 |"])
            for seed in (42, 43, 44):
                self.assertNotIn(f"seed_{seed}", text)
            self.assertNotIn("99.00", text)

    def test_single_seed_does_not_claim_zero_sem(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "experiment.json", dict(conditions=["B0_baseline"], seeds=[42]))
            write_json(output / "B0_baseline/seed_42/summary.json", {"heldout": {"latest": {"greedy": {"win_rate": 0.25}}}})
            campaign.write_results(output)
            text = (output / "RESULTS.md").read_text(encoding="utf-8")
            row = next(line for line in text.splitlines() if line.startswith("| B0_baseline |"))
            self.assertIn("25.00", row)
            self.assertIn("SEM", row)
            self.assertTrue(row.endswith("| 1 |"))
            self.assertNotIn("\u00b1", row)


if __name__ == "__main__":
    unittest.main()
