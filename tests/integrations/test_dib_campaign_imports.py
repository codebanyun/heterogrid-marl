"""Campaign inspection must work with only the environment's core dependencies."""
import subprocess
import sys
import unittest
from pathlib import Path


class CampaignImportTests(unittest.TestCase):
    def test_matrix_and_selection_import_without_training_dependencies(self):
        code = """
import importlib.abc
import sys

class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'yaml', 'matplotlib'}:
            raise ModuleNotFoundError('Optional dependency unavailable: ' + fullname)

sys.meta_path.insert(0, CoreOnly())
from algorithms.dib_mappo.campaign import matrix
from algorithms.dib_mappo.full_campaign import task_matrix
from algorithms.dib_mappo.tactical_campaign import choose

assert len(matrix()[0]) == 13
assert len(task_matrix()) == 17
assert callable(choose)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
