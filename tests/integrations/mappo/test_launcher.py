import json
import tempfile
import unittest
from pathlib import Path

from algorithms.mappo.launch import align_steps, latest_progress


class MAPPOStudyLauncherTest(unittest.TestCase):
    def test_batch_alignment_never_undertrains(self):
        self.assertEqual(align_steps(10000000), 10000384)
        self.assertEqual(align_steps(512), 512)
        for budget in (1, 513, 250000, 2000000, 10000000):
            actual = align_steps(budget)
            self.assertGreaterEqual(actual, budget)
            self.assertLess(actual - budget, 512)
            self.assertEqual(actual % 512, 0)

    def test_progress_tolerates_partial_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(latest_progress(output)["environment_steps"], 0)
            row = {"environment_steps": 512, "episodes": 8, "recent_return": -8.1, "recent_win_rate": 0}
            (output / "metrics.jsonl").write_text(json.dumps(row) + '\n{"environment_steps":')
            self.assertEqual(latest_progress(output), row)


if __name__ == "__main__":
    unittest.main()
