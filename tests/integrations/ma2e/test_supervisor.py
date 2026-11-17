"""Resource/adoption contracts; no training or signals to real launchers."""
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from algorithms.ma2e import supervisor


class ProcessTokenTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux /proc required")
    def test_real_subprocess_token_is_stable_and_disappears_after_exit(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            token = supervisor.process_token(child.pid)
            self.assertIsNotNone(token)
            self.assertTrue(token.isdecimal())
            self.assertIsNone(child.poll())
            self.assertEqual(supervisor.process_token(child.pid), token)
        finally:
            # EOF lets only this test-owned child exit, without sending a signal.
            child.communicate(timeout=5)
        self.assertEqual(child.returncode, 0)
        self.assertIsNone(supervisor.process_token(child.pid))


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.row = self.job("small", 42)
        self.row.update(pid=987654, requested_device="auto")
        self.token = "12345678"
        # Every signal in these tests must stop here, even if an assertion fails.
        signal_patch = mock.patch.object(supervisor.os, "kill")
        self.send_signal = signal_patch.start()
        self.addCleanup(signal_patch.stop)

    def job(self, condition, seed=42):
        return dict(id=f"{condition}/seed_{seed}", condition=condition, seed=seed,
                    path=str(self.root / condition / f"seed_{seed}"))

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def decision(self, row, devices, code=None):
        record = dict(requested_device=row.get("requested_device", "auto"),
                      attempts=[dict(device=device, pid=800000 + index)
                                for index, device in enumerate(devices)])
        if code is not None:
            record.update(return_code=code, status="completed" if code == 0 else "failed")
        path = Path(row["path"]).parent / "logs" / f"seed_{row['seed']}.device.json"
        self.write_json(path, record)
        return path

    def adopt(self, row=None):
        row = self.row if row is None else row
        command = [sys.executable, "-u", "-m", "algorithms.ma2e.launch", "--output", row["path"]]
        with mock.patch.object(supervisor, "process_token", return_value=self.token), \
                mock.patch.object(Path, "read_bytes", return_value=b"\0".join(
                    part.encode() for part in command) + b"\0"):
            return supervisor.AdoptedProcess(row, self.token)

    def test_adoption_rejects_stale_or_dead_identity_before_reading_command(self):
        for current in (None, "reused-pid-token"):
            with self.subTest(current=current), \
                    mock.patch.object(supervisor, "process_token", return_value=current), \
                    mock.patch.object(Path, "read_bytes") as read_command:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    supervisor.AdoptedProcess(self.row, self.token)
                read_command.assert_not_called()
        self.send_signal.assert_not_called()

    def test_adoption_rejects_unrelated_commands_even_with_matching_arguments(self):
        commands = (
            [sys.executable, "-m", "algorithms.ma2e.train", "--output", self.row["path"]],
            [sys.executable, "-m", "algorithms.ma2e.launch", "--output", self.row["path"] + "_other"],
            [sys.executable, "-m", "algorithms.ma2e.launch_extra", self.row["path"]],
            [sys.executable, "-c", "pass", "algorithms.ma2e.launch", self.row["path"]],
        )
        for command in commands:
            with self.subTest(command=command), \
                    mock.patch.object(supervisor, "process_token", return_value=self.token), \
                    mock.patch.object(Path, "read_bytes", return_value=b"\0".join(
                        part.encode() for part in command) + b"\0"):
                with self.assertRaisesRegex(RuntimeError, "unrelated process"):
                    supervisor.AdoptedProcess(self.row, self.token)
        self.send_signal.assert_not_called()

    def test_handover_can_observe_a_launcher_that_completed_during_transfer(self):
        self.decision(self.row, ["cuda"], code=0)
        output = Path(self.row["path"])
        self.write_json(output / "summary.json", {})
        self.write_json(output / "run.json", {"status": "COMPLETED"})
        with mock.patch.object(supervisor, "process_token", return_value=None):
            child = supervisor.AdoptedProcess(self.row, self.token, allow_exited=True)
            self.assertEqual(child.poll(), 0)
        self.send_signal.assert_not_called()

    def test_handover_still_refuses_reused_pid_even_when_exited_is_allowed(self):
        with mock.patch.object(supervisor, "process_token", return_value="different-token"):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                supervisor.AdoptedProcess(self.row, self.token, allow_exited=True)

    def test_peak_ram_grows_with_team_observation_cache_and_buffer(self):
        info = dict(n_agents=6, obs_shape=128, state_shape=256, n_actions=9, episode_limit=100)
        args = dict(buffer_size=256, MT_n_embd=32, MT_batchsize=32,
                    MT_traj_length=8, batch_size=16)
        baseline = supervisor.peak_ram(info, args, "cpu")
        self.assertIsInstance(baseline, int)
        full_cache = 4 * info["n_agents"] ** 2 * args["MT_n_embd"] * 101 * args["buffer_size"]
        self.assertGreater(baseline, full_cache)
        for key in ("n_agents", "obs_shape", "state_shape", "n_actions", "episode_limit"):
            with self.subTest(dimension=key):
                self.assertGreater(supervisor.peak_ram(dict(info, **{key: info[key] * 2}), args, "cpu"),
                                   baseline)
        for key in ("MT_n_embd", "buffer_size"):
            with self.subTest(dimension=key):
                self.assertGreater(supervisor.peak_ram(info, dict(args, **{key: args[key] * 2}), "cpu"),
                                   baseline)
        self.assertGreater(supervisor.peak_ram(info, args, "cuda"), baseline)
        self.assertEqual(supervisor.peak_ram(info, args, "auto"),
                         supervisor.peak_ram(info, args, "cuda"))

    def test_capacity_reserves_future_growth_and_floor_not_just_current_rss(self):
        gib = supervisor.GIB
        active = [dict(self.job(f"team_{i}"), pid=900000 + i, requested_device="auto")
                  for i in range(4)]
        self.decision(active[0], ["cuda", "cpu"])
        self.decision(active[1], ["cuda"])
        self.decision(active[2], ["cpu"])
        peaks = {(row["condition"], device): size * gib
                 for row, device, size in zip(active, ("cpu", "cuda", "cpu", "auto"), (8, 6, 4, 2))}
        rss = {800001: gib, 800000: 2 * gib, active[3]["pid"]: gib}
        # Give the third run a distinct trainer PID, already above its forecast.
        decision_path = self.decision(active[2], ["cpu"])
        record = json.loads(decision_path.read_text())
        record["attempts"][-1]["pid"] = 800002
        self.write_json(decision_path, record)
        rss[800002] = 5 * gib
        with mock.patch.object(supervisor, "process_rss", side_effect=rss.__getitem__) as measured:
            self.assertEqual(supervisor.capacity(24 * gib, active, peaks, 4 * gib), 8 * gib)
            self.assertEqual(measured.call_args_list,
                             [mock.call(pid) for pid in (800001, 800000, 800002, active[3]["pid"])])
            self.assertEqual(supervisor.capacity(10 * gib, active, peaks, 4 * gib), -6 * gib)
            rss[800001] += 2 * gib
            self.assertEqual(supervisor.capacity(24 * gib, active, peaks, 4 * gib), 10 * gib)
        self.assertEqual(supervisor.capacity(24 * gib, [], {}, 4 * gib), 20 * gib)

    def test_cpu_selection_is_smallest_fitting_with_agent_limit_and_no_mutation(self):
        large, medium, small_b, small_a = [self.job(name) for name in ("large", "medium", "small_b", "small_a")]
        jobs = [large, medium, small_b, small_a]
        original = list(jobs)
        control = dict(gpu_max_workers=1, cpu_max_agents=12,
                       team_sizes=dict(large=16, medium=12, small_b=6, small_a=6))
        peaks = {(job["condition"], "cpu"): size
                 for job, size in zip(jobs, (1, 5, 3, 3))}
        self.decision(self.row, ["cuda"])
        self.assertEqual(supervisor.choose_job(jobs, [self.row], control, 3, peaks), (small_a, "cpu"))
        peaks[("small_a", "cpu")] = 7
        peaks[("small_b", "cpu")] = 6
        self.assertEqual(supervisor.choose_job(jobs, [self.row], control, 5, peaks), (medium, "cpu"))
        self.assertEqual(supervisor.choose_job(jobs, [self.row], control, 4, peaks), (None, None))
        self.assertEqual(supervisor.choose_job([], [self.row], control, 100, peaks), (None, None))
        self.assertEqual(jobs, original)

    def test_gpu_limit_uses_latest_device_records_and_reserves_unknown_auto(self):
        running = [dict(self.job(f"running_{i}"), requested_device="auto") for i in range(4)]
        # Two wrappers fell back to CPU; one requested CPU but actually reports CUDA.
        for row in running[:2]:
            self.decision(row, ["cuda", "cpu"])
        running[2]["requested_device"] = "cpu"
        self.decision(running[2], ["cuda"])
        self.decision(running[3], ["cuda"])
        job = self.job("candidate")
        control = dict(gpu_max_workers=2, cpu_max_agents=12, team_sizes=dict(candidate=6))
        peaks = {("candidate", "auto"): 10, ("candidate", "cpu"): 5}
        self.assertEqual(supervisor.choose_job([job], running, control, 10, peaks), (job, "cpu"))
        decision_path = self.decision(running[3], ["cuda", "cpu"])
        self.assertEqual(supervisor.choose_job([job], running, control, 10, peaks), (job, "auto"))
        self.assertEqual(supervisor.choose_job([job], running, control, 9, peaks), (None, None))
        decision_path.unlink()
        self.assertEqual(supervisor.choose_job([job], running, control, 10, peaks), (job, "cpu"))

    def test_priority_retries_start_before_other_queued_jobs(self):
        normal, retry = self.job("normal"), self.job("retry", 43)
        control = dict(gpu_max_workers=5, cpu_max_agents=3,
                       team_sizes=dict(normal=6, retry=6), priority_jobs=[retry["id"]],
                       gpu_only_jobs=[retry["id"]])
        peaks = {("normal", "auto"): 4, ("retry", "cuda"): 5}
        jobs = [normal, retry]
        self.assertEqual(supervisor.choose_job(jobs, [], control, 10, peaks), (retry, "cuda"))
        self.assertEqual(jobs, [normal, retry])

    def test_gpu_only_retries_do_not_fall_back_when_gpu_slots_are_full(self):
        retry = self.job("retry", 44)
        self.decision(self.row, ["cuda"])
        control = dict(gpu_max_workers=1, cpu_max_agents=12,
                       team_sizes=dict(retry=6), gpu_only_jobs=[retry["id"]])
        self.assertEqual(supervisor.choose_job([retry], [self.row], control, 100,
                                              {("retry", "cpu"): 1}), (None, None))

    def test_retry_priority_does_not_bypass_memory_admission(self):
        normal, retry = self.job("normal"), self.job("retry", 43)
        control = dict(gpu_max_workers=5, cpu_max_agents=3,
                       team_sizes=dict(normal=6, retry=6), priority_jobs=[retry["id"]],
                       gpu_only_jobs=[retry["id"]])
        peaks = {("normal", "auto"): 4, ("retry", "cuda"): 5}
        self.assertEqual(supervisor.choose_job([normal, retry], [], control, 4, peaks),
                         (normal, "auto"))

    def test_adopted_poll_requires_exit_decision_summary_and_completed_run(self):
        cases = (
            ("no decision", None, True, "COMPLETED", 1),
            ("no return code", "missing", True, "COMPLETED", 1),
            ("failed decision", 7, True, "COMPLETED", 7),
            ("no summary", 0, False, "COMPLETED", 1),
            ("no run record", 0, True, None, 1),
            ("incomplete run", 0, True, "RUNNING", 1),
            ("failed run", 0, True, "FAILED", 1),
            ("complete", 0, True, "COMPLETED", 0),
        )
        for index, (name, code, summary, status, expected) in enumerate(cases):
            with self.subTest(case=name):
                row = dict(self.job(f"case_{index}"), pid=self.row["pid"])
                child = self.adopt(row)
                output = Path(row["path"])
                if code is not None:
                    self.decision(row, ["cpu"], None if code == "missing" else code)
                if summary:
                    self.write_json(output / "summary.json", {})
                if status is not None:
                    self.write_json(output / "run.json", dict(status=status))
                with mock.patch.object(supervisor, "process_token", return_value=self.token):
                    self.assertIsNone(child.poll())
                for gone in (None, "replacement-token"):
                    child = self.adopt(row)
                    with mock.patch.object(supervisor, "process_token", return_value=gone):
                        self.assertEqual(child.poll(), expected)
                        self.assertEqual(child.returncode, expected)
                    with mock.patch.object(supervisor, "process_token") as lookup:
                        self.assertEqual(child.poll(), expected)
                        lookup.assert_not_called()

    def test_adopted_signals_are_guarded_by_process_identity(self):
        child = self.adopt()
        for current in (self.token, "replacement-token", None):
            for method, signum in ((child.terminate, signal.SIGTERM), (child.kill, signal.SIGKILL)):
                with self.subTest(current=current, signal=signum), \
                        mock.patch.object(supervisor, "process_token", return_value=current):
                    self.send_signal.reset_mock()
                    method()
                    if current == self.token:
                        self.send_signal.assert_called_once_with(self.row["pid"], signum)
                    else:
                        self.send_signal.assert_not_called()

    def test_adopted_wait_times_out_while_alive_and_returns_recorded_exit(self):
        child = self.adopt()
        with mock.patch.object(supervisor, "process_token", return_value=self.token), \
                mock.patch.object(supervisor.time, "monotonic", side_effect=[100, 101]), \
                mock.patch.object(supervisor.time, "sleep") as sleep:
            with self.assertRaises(subprocess.TimeoutExpired):
                child.wait(timeout=1)
            sleep.assert_not_called()
        self.decision(self.row, ["cpu"], code=9)
        with mock.patch.object(supervisor, "process_token", return_value=None):
            self.assertEqual(child.wait(timeout=0), 9)
            self.assertEqual(child.wait(), 9)
        self.send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
