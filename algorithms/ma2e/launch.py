"""Supervise one MA2E seed with a leased GPU and a fresh CPU retry on CUDA OOM."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO
from uuid import uuid4


LEASE_ROOT = Path("/tmp")
STOP_TIMEOUT = 10
OOM_PATTERN = re.compile(
    r"cuda\s*(?:error\s*:)?\s*:?\s*out\s+of\s+memory"
    r"|cublas[_\s]*(?:status[_\s]*)?alloc(?:ation)?[_\s]*(?:failed|failure|error)"
    r"|cudamemory",
    re.IGNORECASE,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_free_memory_mb() -> int | None:
    """Query physical GPU 0 without importing a CUDA runtime."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5, check=False,
        )
        free = int(result.stdout.strip()) if result.returncode == 0 else -1
        return free if free >= 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


@dataclass
class Lease:
    path: Path
    handle: IO[str]

    def release(self) -> None:
        if not self.handle.closed:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()


def acquire_lease(max_workers: int, root: Path = LEASE_ROOT) -> Lease | None:
    """Never wait for a slot and never unlink shared flock files."""
    for slot in range(max_workers):
        path = root / f"heterogrid-ma2e-gpu0.slot_{slot}"
        handle = path.open("a", encoding="ascii")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        except BaseException:
            handle.close()
            raise
        return Lease(path, handle)
    return None


def choose_attempt(requested: str, min_free_mb: int, max_workers: int,
                   lease_root: Path = LEASE_ROOT) -> tuple[str, str, int | None, Lease | None]:
    if requested == "cpu":
        return "cpu", "CPU requested", None, None
    free = gpu_free_memory_mb()
    if free is None:
        return "cpu", "GPU 0 unavailable or nvidia-smi query failed", free, None
    if free < min_free_mb:
        return "cpu", "GPU 0 free memory below threshold", free, None
    lease = acquire_lease(max_workers, lease_root)
    if lease is None:
        return "cpu", "All GPU 0 leases busy; no wait", free, None
    return "cuda", "GPU 0 has sufficient free memory and a lease", free, lease


def is_cuda_oom(log_path: Path) -> bool:
    with log_path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - 4096))
        return bool(OOM_PATTERN.search(log.read().decode("utf-8", errors="replace")))


def reserve_path(path: Path) -> Path:
    """Preserve earlier logs/provenance even when a seed is launched again."""
    while True:
        try:
            with path.open("x", encoding="utf-8"):
                pass
            return path
        except FileExistsError:
            path = path.with_name(f"{path.stem}_{uuid4().hex}{path.suffix}")


def write_decision(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".device_", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def event(name: str, **fields) -> None:
    print(json.dumps({"event": name, **fields}), flush=True)


def signal_child(child: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def run_attempt(command: list[str], log_path: Path, row: dict, persist,
                received_signal: list[int]) -> int:
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    if row["device"] == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True,
        )
        try:
            row.update(pid=child.pid, status="running", started=timestamp())
            persist()
            event("attempt_started", device=row["device"], pid=child.pid, log=str(log_path))
            deadline = None
            while True:
                if received_signal[0] and deadline is None:
                    signal_child(child, received_signal[0])
                    deadline = time.monotonic() + STOP_TIMEOUT
                if deadline is not None and time.monotonic() >= deadline:
                    signal_child(child, signal.SIGKILL)
                try:
                    return child.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # Includes failures while recording provenance, not just normal signals.
            if child.poll() is None:
                signal_child(child, signal.SIGTERM)
                try:
                    child.wait(timeout=STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    signal_child(child, signal.SIGKILL)
                    child.wait()


def archive_failed_output(output: Path) -> Path | None:
    if not (output.exists() or output.is_symlink()):
        return None
    archive = output.parent / "failed_attempts"
    archive.mkdir(exist_ok=True)
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = archive / f"{output.name}_cuda_{label}_{uuid4().hex}"
    output.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-min-free-mb", type=int, default=10000)
    parser.add_argument("--gpu-max-workers", type=int, default=2)
    parser.add_argument("--no-cpu-fallback", action="store_true",
                        help="Refuse CPU execution when GPU admission or CUDA allocation fails")
    args, remaining = parser.parse_known_args(argv)
    if args.gpu_min_free_mb < 0 or args.gpu_max_workers < 1:
        parser.error("GPU minimum free memory must be nonnegative and workers must be positive")
    training = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, default=42)
    run, _ = training.parse_known_args(remaining)
    # Do not resolve the final component: even dangling output symlinks must be refused.
    output = Path(os.path.abspath(run.output))
    if output.exists() or output.is_symlink():
        parser.error(f"Refusing existing output: {output}")
    if output.name in ("logs", "failed_attempts"):
        parser.error("Output name is reserved for launcher logs/failed attempts")
    logs = output.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    decision_path = reserve_path(logs / f"seed_{run.seed}.device.json")
    decision = {
        "schema_version": 1, "wrapper_pid": os.getpid(), "seed": run.seed,
        "output": str(output), "requested_device": args.device,
        "gpu_min_free_mb": args.gpu_min_free_mb, "gpu_max_workers": args.gpu_max_workers,
        "cpu_fallback_enabled": not args.no_cpu_fallback,
        "lease_wait_seconds": 0, "created": timestamp(), "status": "selecting", "attempts": [],
    }

    def persist():
        write_decision(decision_path, decision)

    received_signal = [0]

    def interrupted(signum, frame):
        received_signal[0] = received_signal[0] or signum

    previous_handlers = {}
    lease = None
    active_row = None
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        persist()
        device, reason, free, lease = choose_attempt(
            args.device, args.gpu_min_free_mb, args.gpu_max_workers,
        )
        decision.update(selected_device=device, reason=reason, gpu_free_mb=free,
                        lease=str(lease.path) if lease else None, status="ready")
        persist()
        event("device_selected", device=device, reason=reason, provenance=str(decision_path))
        if args.no_cpu_fallback and device != "cuda":
            decision.update(status="failed", return_code=75, finished=timestamp(),
                            error="GPU-only run refused CPU fallback: " + reason)
            persist()
            return 75
        while not received_signal[0]:
            # The training process owns directory creation; the wrapper only observes it.
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Refusing existing output before attempt: {output}")
            log_path = reserve_path(logs / f"seed_{run.seed}.{device}.log")
            command = [sys.executable, "-u", "-m", "algorithms.ma2e.train", *remaining,
                       "--device", device]
            active_row = {"device": device, "command": command, "log": str(log_path),
                          "pid": None, "status": "starting", "lease": str(lease.path) if lease else None}
            decision["attempts"].append(active_row)
            decision["status"] = "running"
            persist()
            try:
                code = run_attempt(command, log_path, active_row, persist, received_signal)
            finally:
                if lease is not None:
                    lease.release()
                    active_row["lease_released"] = timestamp()
                    lease = None
            active_row.update(return_code=code, finished=timestamp(),
                              status="interrupted" if received_signal[0] else "completed" if code == 0 else "failed")
            oom = device == "cuda" and code != 0 and not received_signal[0] and is_cuda_oom(log_path)
            active_row["cuda_oom"] = bool(oom)
            persist()
            event("attempt_finished", device=device, return_code=code, cuda_oom=bool(oom), log=str(log_path))
            if received_signal[0]:
                break
            if not oom:
                decision.update(status=active_row["status"], return_code=code, finished=timestamp())
                persist()
                return code if code >= 0 else 128 - code
            if args.no_cpu_fallback:
                decision.update(status="failed", return_code=code, finished=timestamp(),
                                error="CUDA OOM; CPU fallback disabled by resource policy")
                persist()
                return code if code >= 0 else 128 - code
            archived = archive_failed_output(output)
            active_row["archived_output"] = str(archived) if archived else None
            decision["status"] = "retrying_cpu"
            persist()
            event("retrying_cpu", reason="CUDA memory allocation failure", archived_output=active_row["archived_output"])
            device = "cpu"
        code = 128 + received_signal[0]
        decision.update(status="interrupted", signal=received_signal[0], return_code=code, finished=timestamp())
        persist()
        return code
    except Exception as error:
        if active_row is not None and active_row["status"] in ("starting", "running"):
            active_row.update(status="failed", error=str(error), finished=timestamp())
        decision.update(status="failed", error=str(error), return_code=1, finished=timestamp())
        persist()
        event("launcher_failed", error=str(error), provenance=str(decision_path))
        return 1
    finally:
        if lease is not None:
            lease.release()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
