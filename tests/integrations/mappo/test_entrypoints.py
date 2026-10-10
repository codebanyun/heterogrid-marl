import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.mappo import ablate, study
from algorithms.mappo.settings import DEFAULT_CONFIG


class MAPPOEntryPointTests(unittest.TestCase):
    def backend(self, root):
        backend = root / "backend"
        (backend / "onpolicy").mkdir(parents=True)
        (backend / "onpolicy/config.py").touch()
        return backend

    def test_study_passes_portable_paths_to_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = self.backend(root)
            output = root / "study"
            calls = []

            def completed_worker(command, **kwargs):
                calls.append(command)
                directory = Path(command[command.index("--output") + 1])
                directory.mkdir()
                (directory / "config.json").write_text(json.dumps({"registered_path": str(directory)}))
                (directory / "summary.json").write_text('{}')

            arguments = ["study", "--mappo-root", str(backend), "--config", str(DEFAULT_CONFIG),
                         "--output", str(output), "--steps", "16384", "--seeds", "42", "--workers", "1"]
            with patch("sys.argv", arguments), patch.object(study.subprocess, "run", side_effect=completed_worker), \
                    patch.object(study, "summarize") as summarize, contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][2], "algorithms.mappo.train")
            self.assertEqual(calls[0][calls[0].index("--mappo-root") + 1], str(backend))
            self.assertEqual(calls[0][calls[0].index("--config") + 1], str(DEFAULT_CONFIG))
            self.assertNotIn("--results-root", calls[0])
            summarize.assert_called_once()

    def test_ablation_passes_config_backend_and_optional_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = self.backend(root)
            output = root / "ablation"
            index = root / "results"
            calls = []

            def completed_worker(command, **kwargs):
                calls.append(command)
                directory = Path(command[command.index("--output") + 1])
                directory.mkdir()
                row = dict(win_rate=0, timeout_rate=1, mean_return=-10, mean_damage=0,
                           mean_kills=0, contact_rate=0, mean_first_contact_step=None,
                           mean_discovered_targets=0, mean_useful_reports=0)
                (directory / "summary.json").write_text(json.dumps({mode: row for mode in ("before", "after", "greedy")}))

            arguments = ["ablate", "--mappo-root", str(backend), "--results-root", str(index),
                         "--output", str(output), "--steps", "512", "--seeds", "42", "--workers", "1"]
            with patch("sys.argv", arguments), patch.object(ablate.subprocess, "run", side_effect=completed_worker), \
                    contextlib.redirect_stdout(io.StringIO()):
                ablate.main()
            self.assertEqual(len(calls), 3)
            for command in calls:
                self.assertEqual(command[2], "algorithms.mappo.train")
                self.assertEqual(command[command.index("--mappo-root") + 1], str(backend))
                self.assertEqual(command[command.index("--results-root") + 1], str(index))
                self.assertTrue(Path(command[command.index("--config") + 1]).is_file())
            self.assertEqual(len(json.loads((output / "summary.json").read_text())["runs"]), 3)


if __name__ == "__main__":
    unittest.main()
