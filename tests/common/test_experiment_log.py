import json
import tempfile
import unittest
from pathlib import Path

from algorithms.common.experiment_log import ExperimentRecorder


class ExperimentRecorderTest(unittest.TestCase):
    def test_registration_metrics_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / "results"
            group.mkdir()
            (group / "1").symlink_to(root / "missing")
            output = root / "output"
            output.mkdir()
            recorder = ExperimentRecorder(output, group, {"name": "mappo"}, [Path(__file__)], capture=False)
            try:
                self.assertEqual(recorder.registered_path.name, "2")
                self.assertEqual(recorder.registered_path.resolve(), output.resolve())
                recorder.log(512, {"return_mean": 1.5, "absent": None})
                metrics = json.loads((output / "metrics.json").read_text())
                info = json.loads((output / "info.json").read_text())
                self.assertEqual(metrics["return_mean"]["steps"], [512])
                self.assertEqual(info["return_mean"], [1.5])
                self.assertEqual(info["return_mean_T"], [512])
                self.assertNotIn("absent", metrics)
                self.assertEqual(len(list((output / "_sources").iterdir())), 1)
                with self.assertRaises(ValueError):
                    recorder.log(1024, {"bad": float("nan")})
                recorder.complete({"done": True})
                run = json.loads((output / "run.json").read_text())
                self.assertEqual(run["status"], "COMPLETED")
                self.assertIsNotNone(run["stop_time"])
            finally:
                recorder.close()

    def test_local_recording_does_not_require_a_results_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            recorder = ExperimentRecorder(output, None, {"name": "mappo"}, capture=False)
            try:
                self.assertEqual(recorder.registered_path, output.resolve())
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(Path(config["registered_path"]), output.resolve())
                recorder.log(512, {"battle_won_mean": 0.5})
                recorder.complete({"done": True})
                self.assertFalse(any(p.is_symlink() for p in output.iterdir()))
            finally:
                recorder.close()

    def test_unfinished_run_is_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            recorder = ExperimentRecorder(output, root / "results", {"name": "mappo"}, capture=False)
            recorder.close()
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "FAILED")

    def test_study_summary_cannot_overwrite_run(self):
        from algorithms.mappo.study import summarize

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = output / "summary.json"
            summary.write_text('{"original": true}')
            with self.assertRaises(ValueError):
                summarize(output, [{"path": str(output)}])
            self.assertEqual(json.loads(summary.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
