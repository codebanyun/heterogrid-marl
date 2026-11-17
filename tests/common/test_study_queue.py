import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from algorithms.common import study_queue


class SharedWorkerTests(unittest.TestCase):
    def test_only_live_external_workers_reserve_slots_and_duplicates_are_counted_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            path.write_text(json.dumps({"runs": {
                "a": {"state": "RUNNING", "pid": 101}, "duplicate": {"state": "RUNNING", "pid": 101},
                "dead": {"state": "RUNNING", "pid": 102}, "finished": {"state": "COMPLETED", "pid": 103}}}))
            def alive(pid, signal):
                if pid == 102:
                    raise ProcessLookupError()
            with patch.object(study_queue.os, "kill", side_effect=alive):
                self.assertEqual(study_queue.external_running_workers([path]), 1)
            with self.assertRaises(FileNotFoundError):
                study_queue.external_running_workers([path.with_name("missing.json")])

    def test_queue_waits_for_external_slot_then_starts_one_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external.json"
            external.write_text(json.dumps({"runs": {"d": {"state": "RUNNING", "pid": 101}}}))
            manifest = dict(conditions=["B0"], seeds=[42], workers=1, repository=str(root),
                            minimum_free_disk_bytes=0, external_status_paths=[str(external)])
            (root / "experiment.json").write_text(json.dumps(manifest))
            def spawn(*args, **kwargs):
                path = root / "B0/seed_42"
                path.mkdir(parents=True)
                (path / "config.json").write_text(json.dumps({"registered_path": str(path)}))
                (path / "summary.json").write_text("{}")
                return Mock(pid=202, poll=Mock(return_value=0))
            def release_slot(*args):
                external.write_text(json.dumps({"runs": {"d": {"state": "COMPLETED", "pid": 101}}}))
            def summarize(path, runs):
                (path / "summary.json").write_text("{}")
            with patch.object(study_queue.os, "kill"), \
                    patch.object(study_queue.time, "sleep", side_effect=release_slot) as sleep, \
                    patch.object(study_queue.subprocess, "Popen", side_effect=spawn) as process, \
                    patch.object(study_queue, "latest_progress", return_value={}):
                study_queue.run_queue(root, lambda *args: ["python"], Mock(), summarize)
                self.assertGreaterEqual(sleep.call_count, 2)
                self.assertEqual(process.call_count, 1)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["state"], "COMPLETED")
            self.assertEqual(status["counts"]["COMPLETED"], 1)


if __name__ == "__main__":
    unittest.main()
