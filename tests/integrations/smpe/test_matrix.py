import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.smpe import matrix
from heterogrid import EnvConfig


class SMPEMatrixTests(unittest.TestCase):
    def test_fixed_blue_reference_replaces_only_b_and_c(self):
        from algorithms.common.communication_study import conditions
        from algorithms.smpe.settings import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            communication = conditions(EnvConfig.load(DEFAULT_CONFIG))
            (root / "experiment.json").write_text(json.dumps(dict(algorithm="smpe", conditions=list(communication))))
            (root / "status.json").write_text(json.dumps(dict(state="COMPLETED")))
            for name, config in communication.items():
                (root / "configs" / (name + ".json")).write_text(json.dumps(config.to_dict()))
            original = {name: {"original": name} for name in ("M1_small", "M2_large", "M3_xlarge", "N1_3v3", "N2_9v9", "N3_12v12")}
            _, combined = matrix.combine_communication_reference(original, root)
            self.assertEqual(len(combined), 13)
            self.assertTrue(all(combined[name] == value for name, value in original.items()))
            config = EnvConfig.from_dict(combined["C1_none"])
            self.assertEqual(config.communication_for("red").mode, "none")
            self.assertEqual(config.communication_for("blue").mode, "distance")
            bad = combined["C1_none"]
            bad["blue_communication"]["radius"] = 5
            (root / "configs/C1_none.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                matrix.combine_communication_reference(original, root)

    def test_legacy_replay_compatibility_allows_only_added_snapshot_fields(self):
        from heterogrid import SMACAdapter
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "experiment.json").write_text(json.dumps(dict(seeds=[42])))
            config = EnvConfig(max_steps=2)
            (root / "configs/M1_small.json").write_text(json.dumps(config.to_dict()))
            path = root / "M1_small/seed_42/replays/heldout_latest_greedy"
            path.mkdir(parents=True)
            env = SMACAdapter(config=config)
            try:
                env.reset(seed=42)
                frames, transitions = [], []
                for step in range(3):
                    frame = env.env.core.snapshot()
                    frame.pop("independent_communication")
                    frame.pop("team_communication")
                    frames.append(frame)
                    if step < 2:
                        actions = [0] * env.n_agents
                        reward, done, _ = env.step(actions)
                        transitions.append(dict(actions=actions, team_reward=reward, done=done))
                replay = dict(evaluation=dict(seed=42), frames=frames, transitions=transitions)
                file = path / "episode_000.json"
                file.write_text(json.dumps(replay))
                result = matrix.verify_replay_compatibility(root, ["M1_small"])
                self.assertEqual(result[0]["frames"], 3)
                replay["transitions"][0]["team_reward"] += 1
                file.write_text(json.dumps(replay))
                with self.assertRaises(AssertionError):
                    matrix.verify_replay_compatibility(root, ["M1_small"])
            finally:
                env.close()

    def test_completed_density_reference_can_use_smpe_configs(self):
        from algorithms.common.density_study import conditions
        from algorithms.smpe.settings import DEFAULT_CONFIG
        for algorithm in ("mappo", "smpe"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "configs").mkdir()
                configs = conditions(EnvConfig.load(DEFAULT_CONFIG))
                manifest = dict(algorithm=algorithm, study_group="density", conditions=list(configs))
                (root / "experiment.json").write_text(json.dumps(manifest))
                for name, config in configs.items():
                    (root / "configs" / (name + ".json")).write_text(json.dumps(config.to_dict()))
                original, copied = matrix.reference_configs(root)
                self.assertEqual(original, manifest)
                self.assertEqual(copied, json.loads(json.dumps({name: config.to_dict() for name, config in configs.items()})))

    def test_reference_configuration_copy_is_lossless(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            names = ["condition_" + str(index) for index in range(13)]
            manifest = dict(algorithm="mappo", conditions=names)
            (root / "experiment.json").write_text(json.dumps(manifest))
            value = json.loads(json.dumps(EnvConfig().to_dict()))
            for name in names:
                (root / "configs" / (name + ".json")).write_text(json.dumps(value))
            original, configs = matrix.reference_configs(root)
            self.assertEqual(original, manifest)
            self.assertEqual(len(configs), 13)
            self.assertTrue(all(config == value for config in configs.values()))
            manifest["conditions"][-1] = names[0]
            (root / "experiment.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                matrix.reference_configs(root)

    def test_code_snapshot_is_independent_and_excludes_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, backend, output = root / "project", root / "backend", root / "output"
            for base, names in ((project, ("algorithms", "heterogrid", "configs")),
                                (backend, ("controllers", "learners", "model_learners", "modules", "components", "utils", "config", "saves"))):
                for name in names:
                    (base / name).mkdir(parents=True)
                    (base / name / "example.py").write_text("original = True\n")
            (project / "heterogrid/__pycache__").mkdir()
            (project / "heterogrid/__pycache__/example.pyc").touch()
            with patch.object(matrix, "PROJECT_ROOT", project):
                frozen_project, frozen_backend = matrix.freeze_code(output, backend)
            (backend / "modules/example.py").write_text("changed = True\n")
            self.assertEqual((frozen_backend / "modules/example.py").read_text(), "original = True\n")
            self.assertFalse((frozen_backend / "saves").exists())
            self.assertFalse((frozen_project / "heterogrid/__pycache__").exists())

    def test_native_training_command_preserves_protocol(self):
        manifest = dict(python="python", smpe_root="/frozen/backend", device="cpu", target_steps_per_seed=5000192,
                        validation_seed=100042, heldout_seed=200042, eval_interval=250368, save_interval=2000384,
                        log_interval=10240, eval_episodes=64, heldout_episodes=128, replay_episodes=2)
        command = matrix.training_command(manifest, Path("/output"),
                                          dict(condition="N3_12v12", seed=44, path="/output/N3_12v12/seed_44"))
        self.assertEqual(command[command.index("--steps") + 1], "5000192")
        self.assertEqual(command[command.index("--smpe-root") + 1], "/frozen/backend")
        self.assertEqual(command[command.index("--config") + 1], "/output/configs/N3_12v12.json")
        self.assertNotIn("--mappo-root", command)
        self.assertNotIn("--compact-episode-log", command)
        manifest["compact_episode_log"] = True
        compressed = matrix.training_command(manifest, Path("/output"),
                                             dict(condition="D3_12v12", seed=42, path="/output/D3/seed_42"))
        self.assertIn("--compact-episode-log", compressed)

    def test_configuration_hash_is_order_independent(self):
        self.assertEqual(matrix.canonical_hash({"a": 1, "b": 2}), matrix.canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(matrix.canonical_hash({"a": 1}), matrix.canonical_hash({"a": 2}))

    def test_repair_comparison_uses_final_heldout_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "old", root / "new"
            for directory, wins in ((old, (.25, .50, .75)), (new, (.50, .75, 1.0))):
                directory.mkdir()
                manifest = dict(seeds=[42, 43, 44], conditions=["D0_baseline"], reference=str(old))
                (directory / "experiment.json").write_text(json.dumps(manifest))
                for seed, win in zip(manifest["seeds"], wins):
                    path = directory / "D0_baseline" / ("seed_" + str(seed))
                    path.mkdir(parents=True)
                    data = {"heldout": {"latest": {"greedy": {"win_rate": win}},
                                        "best": {"greedy": {"win_rate": 0}}}}
                    (path / "summary.json").write_text(json.dumps(data))
            matrix.write_reference_comparison(new)
            report = (new / "RESULTS.md").read_text()
            self.assertIn("50.00", report)
            self.assertIn("75.00", report)
            self.assertIn("14.43", report)
            self.assertFalse((old / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
