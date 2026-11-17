import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.common.compact_results import compact, plan


class CompactResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        (self.run / "run.json").write_text(json.dumps({"status": "COMPLETED"}))
        (self.run / "summary.json").write_text(json.dumps({"training_episodes": 2}))
        (self.run / "metrics.json").write_text('{"wins": [0, 1]}')
        (self.run / "models").mkdir()
        for name in ("initial.pt", "best.pt", "latest.pt", "step_0002.pt"):
            (self.run / "models" / name).write_bytes(name.encode())
        self.records = [{"length": 2}, {"length": 3}]
        self.raw = "".join(json.dumps(row) + "\n" for row in self.records).encode()
        (self.run / "training_episodes.jsonl").write_bytes(self.raw)
        (self.run / "training_episodes.json").write_text(json.dumps(self.records))
        for name in ("before", "heldout_latest_greedy", "heldout_best_greedy", "heldout_best_sampled"):
            group = self.run / "replays" / name
            group.mkdir(parents=True)
            for index in range(2):
                (group / f"episode_{index:03}.json").write_text(json.dumps({
                    "schema_version": 1, "frames": [{}, {}], "transitions": [{"done": True}]}))
                (group / f"actions_{index:03}.json").write_text('[]')
            (group / "summary.json").write_text('{"win_rate": 1}')

    def test_only_two_models_and_two_replays_remain(self):
        original = (self.run / "metrics.json").read_bytes()
        report = compact(plan(self.run))
        self.assertEqual({p.name for p in (self.run / "models").iterdir()}, {"latest.pt", "best.pt"})
        self.assertEqual(len(list((self.run / "replays").rglob('episode_*.json'))), 2)
        self.assertEqual(len(list((self.run / "replays").rglob('actions_*.json'))), 2)
        self.assertEqual(len(list((self.run / "replays").rglob('summary.json'))), 4)
        self.assertFalse((self.run / "training_episodes.json").exists())
        self.assertFalse((self.run / "training_episodes.jsonl").exists())
        with gzip.open(self.run / "training_episodes.jsonl.gz", "rb") as stream:
            self.assertEqual(stream.read(), self.raw)
        self.assertEqual((self.run / "metrics.json").read_bytes(), original)
        self.assertEqual(report["episode_log"]["episode_steps"], 5)
        self.assertFalse(plan(self.run)["remove"])

    def test_running_and_failed_are_not_eligible(self):
        for state in ("RUNNING", "FAILED", "INTERRUPTED"):
            (self.run / "run.json").write_text(json.dumps({"status": state}))
            self.assertIsNone(plan(self.run))

    def test_live_output_is_never_eligible(self):
        with patch('algorithms.common.compact_results.active_outputs', return_value={str(self.run.resolve())}):
            self.assertIsNone(plan(self.run, include_interrupted=True))

    def test_conflicting_compressed_log_keeps_originals(self):
        with gzip.open(self.run / "training_episodes.jsonl.gz", "wb") as stream:
            stream.write(b"different\n")
        with self.assertRaises(ValueError):
            compact(plan(self.run))
        self.assertTrue((self.run / "models/initial.pt").exists())
        self.assertTrue((self.run / "training_episodes.jsonl").exists())

    def test_incomplete_episode_count_keeps_originals(self):
        (self.run / "summary.json").write_text('{"training_episodes": 3}')
        with self.assertRaises(AssertionError):
            compact(plan(self.run))
        self.assertTrue((self.run / "training_episodes.json").exists())
        self.assertTrue((self.run / "models/initial.pt").exists())

    def test_missing_best_prevents_deletion(self):
        (self.run / "models/best.pt").unlink()
        self.assertIsNone(plan(self.run))

    def test_interrupted_requires_explicit_option(self):
        (self.run / "run.json").write_text('{"status": "INTERRUPTED"}')
        self.assertIsNone(plan(self.run))
        self.assertIsNotNone(plan(self.run, include_interrupted=True))

    def test_state_change_after_planning_aborts(self):
        item = plan(self.run)
        (self.run / "run.json").write_text('{"status": "RUNNING"}')
        with self.assertRaises(ValueError):
            compact(item)
        self.assertTrue((self.run / "models/initial.pt").exists())

    def test_symbolic_link_is_rejected(self):
        (self.run / "linked_model.pt").symlink_to(self.run / "models/latest.pt")
        with self.assertRaises(ValueError):
            plan(self.run)

    def test_existing_compressed_log_is_preserved_byte_for_byte(self):
        compressed = self.run / "training_episodes.jsonl.gz"
        with gzip.open(compressed, "wb") as stream:
            stream.write(self.raw)
        (self.run / "training_episodes.jsonl").unlink()
        original = compressed.read_bytes()
        report = compact(plan(self.run))
        self.assertEqual(compressed.read_bytes(), original)
        self.assertTrue(report["episode_log"]["preserved_without_modification"])


if __name__ == "__main__":
    unittest.main()
