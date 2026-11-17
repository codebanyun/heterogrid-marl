import contextlib
import io
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from algorithms.ma2e import launch


class FakeChild:
    def __init__(self, finish, code=0):
        self.pid = 987654
        self.returncode = None
        self.finish = finish
        self.code = code

    def wait(self, timeout=None):
        if self.returncode is None:
            self.finish()
            self.returncode = self.code
        return self.returncode

    def poll(self):
        return self.returncode


class GPUSelectionTest(unittest.TestCase):
    def test_gpu_query_is_bounded_and_targets_gpu_zero(self):
        with mock.patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "12000\n")) as run:
            self.assertEqual(launch.gpu_free_memory_mb(), 12000)
        self.assertIn("--id=0", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_unavailable_or_invalid_gpu_query(self):
        for code, stdout in ((1, "12000"), (0, "N/A"), (0, "-1"), (0, "10000\n20000"), (0, "")):
            with self.subTest(code=code, stdout=stdout):
                with mock.patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], code, stdout)):
                    self.assertIsNone(launch.gpu_free_memory_mb())
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5)):
            with mock.patch.object(launch.subprocess, "run", side_effect=error):
                self.assertIsNone(launch.gpu_free_memory_mb())

    def test_cpu_skips_gpu_query(self):
        with mock.patch.object(launch, "gpu_free_memory_mb") as query:
            self.assertEqual(launch.choose_attempt("cpu", 10000, 2), ("cpu", "CPU requested", None, None))
        query.assert_not_called()

    def test_auto_and_explicit_cuda_fall_back_when_unavailable_or_low_memory(self):
        for requested in ("auto", "cuda"):
            for free in (None, 9999):
                with self.subTest(requested=requested, free=free):
                    with mock.patch.object(launch, "gpu_free_memory_mb", return_value=free):
                        with mock.patch.object(launch, "acquire_lease") as acquire:
                            device, reason, measured, lease = launch.choose_attempt(requested, 10000, 2)
                    self.assertEqual(device, "cpu")
                    self.assertTrue(reason)
                    self.assertEqual(measured, free)
                    self.assertIsNone(lease)
                    acquire.assert_not_called()

    def test_leases_are_nonblocking_limited_and_reusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = launch.acquire_lease(2, root)
            second = launch.acquire_lease(2, root)
            try:
                self.assertEqual(first.path.name, "heterogrid-ma2e-gpu0.slot_0")
                self.assertEqual(second.path.name, "heterogrid-ma2e-gpu0.slot_1")
                self.assertIsNone(launch.acquire_lease(2, root))
                for requested in ("auto", "cuda"):
                    with mock.patch.object(launch, "gpu_free_memory_mb", return_value=20000):
                        device, reason, _, lease = launch.choose_attempt(requested, 10000, 2, root)
                    self.assertEqual(device, "cpu")
                    self.assertIn("busy", reason)
                    self.assertIsNone(lease)
                first.release()
                with mock.patch.object(launch, "gpu_free_memory_mb", return_value=10000):
                    device, _, _, reused = launch.choose_attempt("auto", 10000, 2, root)
                try:
                    self.assertEqual(device, "cuda")
                    self.assertEqual(reused.path, first.path)
                finally:
                    reused.release()
            finally:
                first.release()
                second.release()
            self.assertTrue(first.path.exists())


class CudaOOMTest(unittest.TestCase):
    def test_recognizes_cuda_memory_failures_only(self):
        positives = (
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB",
            "RuntimeError: CUDA error: out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
            "cublas allocation failure", "CudaMemoryError", "CUDA: out of memory",
        )
        negatives = ("CPU out of memory", "out of memory", "CUDA illegal memory access", "ValueError", "")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.log"
            for message in positives + negatives:
                with self.subTest(message=message):
                    path.write_text(message)
                    self.assertEqual(launch.is_cuda_oom(path), message in positives)

    def test_only_inspects_final_four_kibibytes_and_handles_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.log"
            path.write_bytes(b"CUDA out of memory\n" + b"x" * 4096)
            self.assertFalse(launch.is_cuda_oom(path))
            path.write_bytes(b"x" * 10000 + b"\xffCUDA out of memory\xfe")
            self.assertTrue(launch.is_cuda_oom(path))


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "seed_42"
        self.decision_path = self.root / "logs" / "seed_42.device.json"
        self.args = ["--output", str(self.output), "--seed", "42", "--ma2e-root", "/backend",
                     "--config", "env.json", "--steps", "5000000"]
        self.stdout = io.StringIO()

    def decision(self):
        return json.loads(self.decision_path.read_text())

    def run_main(self, args=None):
        with contextlib.redirect_stdout(self.stdout):
            return launch.main(self.args if args is None else args)

    def popen_factory(self, results, observer=None):
        calls = []

        def popen(command, **kwargs):
            index = len(calls)
            calls.append((command, kwargs))
            code, text, create_output = results[index]
            row = self.decision()["attempts"][-1]
            self.assertEqual(row["status"], "starting")
            self.assertFalse(self.output.exists())
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertNotEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertTrue(kwargs["start_new_session"])

            def finish():
                row = self.decision()["attempts"][-1]
                self.assertEqual(row["pid"], 987654)
                self.assertEqual(row["status"], "running")
                if observer:
                    observer(index)
                if create_output:
                    self.output.mkdir()
                    (self.output / "checkpoint.bin").write_text(f"attempt {index}")
                kwargs["stdout"].write(text.encode())

            return FakeChild(finish, code)

        return popen, calls

    def test_cpu_command_forwarding_logs_and_provenance(self):
        popen, calls = self.popen_factory([(0, "all stdout and stderr\n", True)])
        with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
            with mock.patch.object(launch, "gpu_free_memory_mb") as query:
                self.assertEqual(self.run_main(["--device", "cpu", *self.args]), 0)
        query.assert_not_called()
        command, kwargs = calls[0]
        self.assertEqual(command, [launch.sys.executable, "-u", "-m", "algorithms.ma2e.train", *self.args,
                                   "--device", "cpu"])
        self.assertEqual((self.root / "logs" / "seed_42.cpu.log").read_text(), "all stdout and stderr\n")
        self.assertEqual(self.decision()["status"], "completed")
        self.assertEqual(self.decision()["lease_wait_seconds"], 0)
        self.assertEqual(self.decision()["gpu_min_free_mb"], 10000)
        self.assertEqual(self.decision()["gpu_max_workers"], 2)
        self.assertIn("attempt_started", self.stdout.getvalue())

    def test_cuda_oom_archives_partial_output_and_retries_fresh_cpu(self):
        lease = launch.acquire_lease(1, self.root)

        def observe(index):
            if index == 0:
                self.assertIsNone(launch.acquire_lease(1, self.root))
            else:
                self.assertTrue(lease.handle.closed)
                archived = Path(self.decision()["attempts"][0]["archived_output"])
                self.assertEqual((archived / "checkpoint.bin").read_text(), "attempt 0")

        popen, calls = self.popen_factory([(1, "CUDA out of memory\n", True), (0, "done\n", True)], observe)
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test GPU", 12000, lease)):
            with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
                self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][:-1], calls[1][0][:-1])
        self.assertEqual(calls[0][0][-1], "cuda")
        self.assertEqual(calls[1][0][-1], "cpu")
        self.assertEqual(calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual((self.output / "checkpoint.bin").read_text(), "attempt 1")
        decision = self.decision()
        self.assertEqual(decision["status"], "completed")
        self.assertEqual([row["return_code"] for row in decision["attempts"]], [1, 0])
        self.assertTrue(decision["attempts"][0]["cuda_oom"])
        archived = Path(decision["attempts"][0]["archived_output"])
        self.assertEqual(archived.parent, self.root / "failed_attempts")
        self.assertTrue(archived.name.startswith("seed_42_cuda_"))
        self.assertEqual(Path(decision["attempts"][0]["log"]).read_text(), "CUDA out of memory\n")
        self.assertTrue(lease.handle.closed)

    def test_cuda_oom_before_output_creation_still_retries(self):
        popen, calls = self.popen_factory([(1, "CUBLAS_STATUS_ALLOC_FAILED", False), (0, "done", True)])
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test GPU", 12000, None)):
            with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
                self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(self.decision()["attempts"][0]["archived_output"])

    def test_gpu_only_refuses_cpu_admission_without_starting_training(self):
        with mock.patch.object(launch, "choose_attempt", return_value=("cpu", "GPU busy", 1000, None)), \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            self.assertEqual(self.run_main(["--no-cpu-fallback", *self.args]), 75)
        popen.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.decision()["attempts"], [])
        self.assertEqual(self.decision()["return_code"], 75)

    def test_gpu_only_cuda_oom_preserves_output_and_never_launches_cpu(self):
        lease = launch.acquire_lease(1, self.root)
        popen, calls = self.popen_factory([(1, "CUDA out of memory", True)])
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test GPU", 12000, lease)), \
                mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
            self.assertEqual(self.run_main(["--no-cpu-fallback", *self.args]), 1)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--no-cpu-fallback", calls[0][0])
        self.assertTrue(self.output.exists())
        self.assertTrue(lease.handle.closed)
        self.assertEqual(self.decision()["status"], "failed")

    def test_gpu_only_success_strips_wrapper_flag_from_training_command(self):
        popen, calls = self.popen_factory([(0, "done", True)])
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test GPU", 12000, None)), \
                mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
            self.assertEqual(self.run_main(["--no-cpu-fallback", *self.args]), 0)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--no-cpu-fallback", calls[0][0])
        self.assertEqual(calls[0][0][-1], "cuda")

    def test_other_failures_and_success_with_oom_text_do_not_retry(self):
        for device, code, text in (("cuda", 7, "invalid config"), ("cpu", 3, "CUDA out of memory"),
                                   ("cuda", 0, "CUDA out of memory was handled"),
                                   ("cuda", -9, "killed")):
            with self.subTest(device=device, code=code):
                with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                    self.output = Path(temporary) / "seed_42"
                    self.decision_path = Path(temporary) / "logs" / "seed_42.device.json"
                    args = ["--output", str(self.output)]
                    popen, calls = self.popen_factory([(code, text, True)])
                    with mock.patch.object(launch, "choose_attempt", return_value=(device, "test", None, None)):
                        with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
                            self.assertEqual(self.run_main(args), code if code >= 0 else 128 - code)
                    self.assertEqual(len(calls), 1)
                    self.assertFalse((self.output.parent / "failed_attempts").exists())
                    self.assertTrue(self.output.exists())

    def test_cpu_retry_failure_is_returned_without_third_attempt(self):
        popen, calls = self.popen_factory([(1, "CUDA out of memory", True), (5, "CUDA out of memory", True)])
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test GPU", 12000, None)):
            with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
                self.assertEqual(self.run_main(), 5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.decision()["status"], "failed")

    def test_existing_output_and_dangling_symlink_refused_without_mutation(self):
        for kind in ("directory", "file", "symlink"):
            with self.subTest(kind=kind):
                path = self.root / kind
                if kind == "directory":
                    path.mkdir()
                elif kind == "file":
                    path.write_text("user data")
                else:
                    path.symlink_to(self.root / "missing")
                with mock.patch.object(launch.subprocess, "Popen") as popen:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        self.run_main(["--output", str(path)])
                self.assertEqual(error.exception.code, 2)
                popen.assert_not_called()
                self.assertTrue(path.exists() or path.is_symlink())
                self.assertFalse((self.root / "logs").exists())

    def test_existing_logs_and_provenance_are_preserved(self):
        logs = self.root / "logs"
        logs.mkdir()
        self.decision_path.write_text("old provenance")
        old_log = logs / "seed_42.cpu.log"
        old_log.write_text("old log")

        def popen(command, **kwargs):
            self.assertFalse(self.output.exists())
            return FakeChild(lambda: kwargs["stdout"].write(b"new log"))

        with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
            self.assertEqual(self.run_main(["--device=cpu", *self.args]), 0)
        self.assertEqual(self.decision_path.read_text(), "old provenance")
        self.assertEqual(old_log.read_text(), "old log")
        self.assertEqual(len(list(logs.glob("seed_42*.json"))), 2)
        self.assertEqual(len(list(logs.glob("seed_42*.log"))), 2)

    def test_invalid_limits_and_missing_output(self):
        for args in (["--gpu-max-workers=0"], ["--gpu-min-free-mb=-1"], ["--device=bogus"], []):
            with self.subTest(args=args):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.run_main(args)

    def test_equals_options_and_wrapper_options_are_not_forwarded(self):
        popen, calls = self.popen_factory([(0, "done", False)])
        args = ["--device=auto", "--gpu-max-workers=3", "--gpu-min-free-mb=11000",
                f"--output={self.output}", "--seed=42", "--ma2e-root=/backend", "--unknown-train-option=yes"]
        with mock.patch.object(launch, "choose_attempt", return_value=("cpu", "busy", 12000, None)) as choose:
            with mock.patch.object(launch.subprocess, "Popen", side_effect=popen):
                self.assertEqual(self.run_main(args), 0)
        choose.assert_called_once_with("auto", 11000, 3)
        self.assertEqual(calls[0][0][4:], [*args[3:], "--device", "cpu"])

    def test_spawn_failure_records_failure_and_releases_lease(self):
        lease = launch.acquire_lease(1, self.root)
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test", 12000, lease)):
            with mock.patch.object(launch.subprocess, "Popen", side_effect=OSError("cannot spawn")):
                self.assertEqual(self.run_main(), 1)
        self.assertTrue(lease.handle.closed)
        self.assertEqual(self.decision()["status"], "failed")
        self.assertEqual(self.decision()["attempts"][0]["status"], "failed")
        self.assertIn("cannot spawn", self.decision()["error"])

    def test_signal_is_forwarded_child_reaped_no_retry_and_handler_restored(self):
        original = signal.getsignal(signal.SIGTERM)
        lease = launch.acquire_lease(1, self.root)
        child = FakeChild(lambda: None, -signal.SIGTERM)
        waits = []

        def wait(timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                raise subprocess.TimeoutExpired("fake child", timeout)
            self.assertIsNone(launch.acquire_lease(1, self.root))
            child.returncode = -signal.SIGTERM
            return child.returncode

        child.wait = wait
        with mock.patch.object(launch, "choose_attempt", return_value=("cuda", "test", 12000, lease)):
            with mock.patch.object(launch.subprocess, "Popen", return_value=child) as popen:
                with mock.patch.object(launch.os, "killpg") as kill:
                    self.assertEqual(self.run_main(), 128 + signal.SIGTERM)
        kill.assert_called_once_with(child.pid, signal.SIGTERM)
        popen.assert_called_once()
        self.assertIsNotNone(child.poll())
        self.assertTrue(lease.handle.closed)
        self.assertEqual(self.decision()["status"], "interrupted")
        self.assertEqual(self.decision()["attempts"][0]["status"], "interrupted")
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_stubborn_child_is_killed_and_reaped(self):
        path = self.root / "attempt.log"
        path.touch()
        child = FakeChild(lambda: None, -signal.SIGKILL)
        calls = []

        def wait(timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("fake child", timeout)
            child.returncode = -signal.SIGKILL
            return child.returncode

        child.wait = wait
        with mock.patch.object(launch.subprocess, "Popen", return_value=child):
            with mock.patch.object(launch.os, "killpg") as kill:
                with mock.patch.object(launch.time, "monotonic", side_effect=[0, 0, launch.STOP_TIMEOUT + 1]):
                    with contextlib.redirect_stdout(self.stdout):
                        code = launch.run_attempt(["fake"], path, {"device": "cpu"}, lambda: None, [signal.SIGTERM])
        self.assertEqual(code, -signal.SIGKILL)
        self.assertEqual(kill.call_args_list, [mock.call(child.pid, signal.SIGTERM), mock.call(child.pid, signal.SIGKILL)])
        self.assertIsNotNone(child.poll())

    def test_unexpected_provenance_error_still_terminates_and_reaps_child(self):
        path = self.root / "attempt.log"
        path.touch()
        child = FakeChild(lambda: None, -signal.SIGTERM)
        with mock.patch.object(launch.subprocess, "Popen", return_value=child):
            with mock.patch.object(launch.os, "killpg") as kill:
                with self.assertRaisesRegex(OSError, "disk full"):
                    launch.run_attempt(["fake"], path, {"device": "cpu"},
                                       mock.Mock(side_effect=OSError("disk full")), [0])
        kill.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertIsNotNone(child.poll())


if __name__ == "__main__":
    unittest.main()
