import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.smpe import matrix
from heterogrid import EnvConfig


class SMPEMatrixTests(unittest.TestCase):
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

    def test_configuration_hash_is_order_independent(self):
        self.assertEqual(matrix.canonical_hash({"a": 1, "b": 2}), matrix.canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(matrix.canonical_hash({"a": 1}), matrix.canonical_hash({"a": 2}))


if __name__ == "__main__":
    unittest.main()
