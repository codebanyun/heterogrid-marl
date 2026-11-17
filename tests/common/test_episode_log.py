import tempfile
import unittest
from pathlib import Path

from algorithms.common.episode_log import episode_rows, open_episode_log, recent_buffer


class EpisodeLogTests(unittest.TestCase):
    def test_plain_and_compressed_streams_preserve_all_rows(self):
        import json
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as directory:
                rows = [{"episode": i, "team_episode_metrics": {"red": {"diagnostics": {"missing": None, "value": i / 7}}}}
                        for i in range(100)]
                with open_episode_log(directory, compressed) as file:
                    for row in rows:
                        file.write(json.dumps(row, allow_nan=False) + "\n")
                self.assertEqual(list(episode_rows(directory)), rows)
                self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_compact_buffer_keeps_exact_recent_window(self):
        for compact in (False, True):
            recent = recent_buffer(compact)
            for index in range(100):
                recent.append(index)
            self.assertEqual(list(recent)[-32:], list(range(68, 100)))
            self.assertEqual(len(recent), 32 if compact else 100)


if __name__ == "__main__":
    unittest.main()
