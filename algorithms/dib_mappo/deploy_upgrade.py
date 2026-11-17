"""Prepare a new immutable replay snapshot and opt-in audited live migration."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from algorithms.common.experiment_log import write_atomic
from algorithms.dib_mappo.live_upgrade import VERSION, now


def process_start(pid):
    # The process name in stat may contain spaces or parentheses.
    return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]


def prepare(coordinator):
    coordinator = coordinator.resolve()
    manifest_path = coordinator / "optimization_upgrade.json"
    if manifest_path.exists():
        raise FileExistsError("Upgrade is already prepared; never overwrite its snapshot")
    status = json.loads((coordinator / "status.json").read_text())
    if status["status"] != "RUNNING" or any(job["algorithm"] != "dib_mappo" for job in status["jobs"]):
        raise ValueError("Expected a live DIB-only campaign")
    runtime = coordinator / "_runtime_optimized_20261019"
    shutil.copytree(coordinator / "_runtime", runtime,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(Path(__file__).parent, runtime / "algorithms/dib_mappo", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    paths = sorted((runtime / "algorithms/dib_mappo").glob("*.py"))
    hashes = {str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest = {"version": VERSION, "status": "PREPARED", "created": now(), "runtime": str(runtime),
        "original_runtime": str(coordinator / "_runtime"), "source_sha256": hashes,
        "manager_pid": status["manager_pid"],
        "process_start_ticks": {str(pid): process_start(pid) for pid in
            [status["manager_pid"]] + [job["pid"] for job in status["jobs"] if job["status"] == "RUNNING"]},
        "workers": [{key: job.get(key) for key in ("algorithm", "scenario", "seed", "output", "pid", "status")}
                    for job in status["jobs"]],
        "method": "Py_AddPendingCall arms a next-complete-batch change; no restart or checkpoint reload"}
    archive = coordinator / "upgrade_20261019"
    archive.mkdir()
    shutil.copy2(coordinator / "plan.json", archive / "plan.before.json")
    shutil.copy2(coordinator / "status.json", archive / "status.before.json")
    write_atomic(manifest_path, manifest)
    print(json.dumps({"prepared": str(manifest_path), "runtime": str(runtime)}), flush=True)


def inject(coordinator, pid):
    coordinator = coordinator.resolve()
    manifest = json.loads((coordinator / "optimization_upgrade.json").read_text())
    allowed = {manifest["manager_pid"]} | {item["pid"] for item in manifest["workers"] if item["status"] == "RUNNING"}
    if pid not in allowed:
        raise ValueError("PID is not a prepared worker/manager")
    if process_start(pid) != manifest['process_start_ticks'][str(pid)]:
        raise ValueError("PID has been reused since preparation")
    proc = Path("/proc") / str(pid)
    argv = (proc / "cmdline").read_bytes().decode().split("\0")
    if Path(os.readlink(proc / "exe")).resolve() != Path(sys.executable).resolve():
        raise ValueError("Unexpected interpreter")
    if pid == manifest["manager_pid"]:
        if "algorithms.dib_mappo.campaign" not in argv or str(coordinator) not in argv:
            raise ValueError("Unexpected manager command")
    else:
        item = next(item for item in manifest["workers"] if item["pid"] == pid)
        if "algorithms.dib_mappo.train" not in argv or item["output"] not in argv:
            raise ValueError("Unexpected worker command")
    runtime = Path(manifest["runtime"])
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((runtime / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen upgrade source hash mismatch")
    result = subprocess.run(["/usr/bin/gdb", "-q", "-nx", "-batch", "-p", str(pid),
        "-ex", "source " + str(runtime / "algorithms/dib_mappo/gdb_pending.py")],
        env={**os.environ, "DIB_PENDING_PAYLOAD": str(runtime / "algorithms/dib_mappo/live_payload.py")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25)
    log = coordinator / "upgrade_20261019" / ("inject_" + str(pid) + ".log")
    log.write_text(result.stdout)
    if result.returncode or "DIB_PENDING_SCHEDULED=0" not in result.stdout:
        raise RuntimeError("Injection was not acknowledged; inspect " + str(log))
    print(json.dumps({"pid": pid, "scheduled": True, "log": str(log)}), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinator", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--inject", type=int)
    group.add_argument("--inject-workers", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        return prepare(args.coordinator)
    if args.inject:
        return inject(args.coordinator, args.inject)
    manifest = json.loads((args.coordinator / "optimization_upgrade.json").read_text())
    pids = [item["pid"] for item in manifest["workers"] if item["status"] == "RUNNING"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pid: inject(args.coordinator, pid), pids))


if __name__ == "__main__":
    main()
