import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from algorithms.mappo import matrix
from algorithms.mappo.launch import align_steps
from algorithms.mappo.settings import DEFAULT_CONFIG
from heterogrid import EnvConfig


class MAPPOMatrixTests(unittest.TestCase):
    def setUp(self):
        self.base = EnvConfig.load(DEFAULT_CONFIG)
        self.variants = matrix.conditions(self.base)

    def test_thirteen_distinct_one_factor_conditions(self):
        self.assertEqual(len(self.variants), 13)
        self.assertEqual(self.variants["B0_baseline"], self.base)
        self.assertEqual(len({json.dumps(c.to_dict(), sort_keys=True) for c in self.variants.values()}), 13)
        for name, config in self.variants.items():
            changed = {key for key in config.to_dict() if config.to_dict()[key] != self.base.to_dict()[key]}
            allowed = ({"map_size"} if name.startswith("M") else {"red", "blue"} if name.startswith("N")
                       else {"communication", "comm_radius", "dropout_probability"} if name.startswith("C") else set())
            self.assertLessEqual(changed, allowed)
            self.assertEqual(config.opponent, "rule")

    def test_balanced_deterministic_rosters(self):
        for count in (3, 6, 9, 12):
            roster = matrix.team_roster(self.base.red, count)
            self.assertEqual(Counter(unit.role for unit in roster), dict(scout=count // 3, strike=count // 3, hybrid=count // 3))
            self.assertEqual(roster, matrix.team_roster(self.base.red, count))
        self.assertEqual(matrix.team_roster(self.base.red, 6), self.base.red)
        with self.assertRaises(ValueError):
            matrix.team_roster(self.base.red, 4)
        with self.assertRaises(ValueError):
            matrix.conditions(replace(self.base, communication="full"))

    def test_jobs_and_training_commands(self):
        manifest = dict(conditions=list(self.variants), seeds=[42, 43, 44], python="python",
                        mappo_root="/portable/backend", device="cpu", actual_steps_per_seed=align_steps(5000000),
                        eval_interval=250368, save_interval=2000384, log_interval=10240,
                        eval_episodes=64, heldout_episodes=128, replay_episodes=2)
        jobs = matrix.jobs_for(manifest, Path("/results/study"))
        self.assertEqual(len(jobs), 39)
        self.assertEqual(len({job["path"] for job in jobs}), 39)
        self.assertEqual(len({job["condition"] for job in jobs[:13]}), 13)
        self.assertEqual(manifest["actual_steps_per_seed"], 5000192)
        command = matrix.training_command(manifest, Path("/results/study"), jobs[-1])
        self.assertEqual(command[command.index("--steps") + 1], "5000192")
        self.assertIn("--no-trajectories", command)
        self.assertIn("--no-evaluation-files", command)
        manifest["team_sizes"] = {name: len(config.red) for name, config in self.variants.items()}
        ordered = matrix.jobs_for(manifest, Path("/results/study"))
        self.assertTrue(all(job["condition"] == "N3_12v12" for job in ordered[:3]))
        self.assertTrue(all(job["condition"] == "N1_3v3" for job in ordered[-3:]))

    def test_manager_bounds_concurrency_and_continues_after_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                manifest = dict(conditions=["B0_baseline", "C1_none"], seeds=[42, 43, 44],
                                python="python", mappo_root="/backend", repository=str(output),
                                device="cpu", actual_steps_per_seed=512, workers=2, minimum_free_disk_bytes=1,
                                eval_interval=512, save_interval=512, log_interval=512,
                                eval_episodes=1, heldout_episodes=1, replay_episodes=0)
                (output / "experiment.json").write_text(json.dumps(manifest))
                live, peak, launches = [], [], []

                class Child:
                    def __init__(self, command, **kwargs):
                        self.pid = 100 + len(launches)
                        self.code = 1 if fail and not launches else 0
                        self.polls = 0
                        live.append(self)
                        peak.append(len(live))
                        launches.append(command)
                        path = Path(command[command.index("--output") + 1])
                        path.mkdir()
                        (path / "config.json").write_text(json.dumps({"registered_path": str(path)}))
                        (path / "summary.json").write_text('{}')

                    def poll(self):
                        self.polls += 1
                        if self.polls < 2:
                            return None
                        if self in live:
                            live.remove(self)
                        return self.code

                def summarize(path, runs):
                    (path / "summary.json").write_text('{}')

                with patch.object(matrix.subprocess, "Popen", Child), patch.object(matrix.time, "sleep"), \
                        patch("algorithms.mappo.audit.audit"), patch("algorithms.mappo.study.summarize", side_effect=summarize), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            matrix.manage(output)
                    else:
                        matrix.manage(output)
                status = json.loads((output / "status.json").read_text())
                self.assertEqual(len(launches), 6)
                self.assertEqual(max(peak), 2)
                self.assertEqual(status["state"], "FAILED" if fail else "COMPLETED")
                self.assertEqual(status["counts"]["FAILED"], int(fail))

    def test_launcher_freezes_matrix_and_detaches_nohup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "backend"
            (backend / "onpolicy").mkdir(parents=True)
            (backend / "onpolicy/config.py").touch()
            output = root / "study"
            arguments = ["matrix", "--mappo-root", str(backend), "--output", str(output), "--workers", "12"]
            with patch("sys.argv", arguments), patch.object(matrix.subprocess, "Popen") as popen, \
                    contextlib.redirect_stdout(io.StringIO()):
                popen.return_value.pid = 123
                matrix.main()
            manifest = json.loads((output / "experiment.json").read_text())
            self.assertEqual(manifest["jobs"], 39)
            self.assertEqual(manifest["workers"], 12)
            self.assertEqual(len(list((output / "configs").glob("*.json"))), 13)
            self.assertEqual(popen.call_args[0][0][0], "nohup")
            self.assertTrue(popen.call_args[1]["start_new_session"])


if __name__ == "__main__":
    unittest.main()
