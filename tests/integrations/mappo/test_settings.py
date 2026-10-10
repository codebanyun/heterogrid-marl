import argparse
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.mappo.settings import DEFAULT_CONFIG, add_backend_argument, validate_backend


class MAPPOSettingsTest(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        add_backend_argument(parser)
        return parser

    def test_default_scenario_exists(self):
        self.assertTrue(DEFAULT_CONFIG.is_file())

    def test_backend_can_be_selected_by_environment_or_argument(self):
        with patch.dict(os.environ, {"MAPPO_ROOT": "from_environment"}):
            parser = self.parser()
            self.assertEqual(parser.parse_args([]).mappo_root, Path("from_environment"))
            self.assertEqual(parser.parse_args(["--mappo-root", "from_argument"]).mappo_root, Path("from_argument"))

    def test_missing_backend_fails_without_machine_specific_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            parser = self.parser()
            self.assertIsNone(parser.parse_args([]).mappo_root)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                validate_backend(parser, None)
            self.assertEqual(raised.exception.code, 2)

    def test_backend_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parser = self.parser()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_backend(parser, root)
            (root / "onpolicy").mkdir()
            (root / "onpolicy/config.py").touch()
            self.assertEqual(validate_backend(parser, root), root.resolve())


if __name__ == "__main__":
    unittest.main()
