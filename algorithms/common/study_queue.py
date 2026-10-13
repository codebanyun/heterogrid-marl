"""Detached worker pool with progress tracking, shutdown and per-condition audits."""
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.progress import latest_progress


def jobs_for(manifest, output):
    jobs = [{"id": f"{label}/seed_{seed}", "condition": label, "seed": seed,
             "path": str(output / label / f"seed_{seed}")}
            for seed in manifest["seeds"] for label in manifest["conditions"]]
    # Start expensive teams first to reduce idle cores at the end of the study.
    sizes = manifest.get("team_sizes", {})
    return sorted(jobs, key=lambda job: -sizes.get(job["condition"], 6))


def run_queue(output, training_command, audit, summarize):
    import fcntl
    lock = (output / "manager.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("A manager already owns this experiment")
    manifest = json.loads((output / "experiment.json").read_text())
    jobs = jobs_for(manifest, output)
    status = {"state": "RUNNING", "started": timestamp(), "manager_pid": os.getpid(),
              "workers": manifest["workers"], "runs": {job["id"]: dict(job, state="QUEUED") for job in jobs}}
    active = {}
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                       MPLBACKEND="Agg")

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Supervisor received signal " + str(signum))

    def publish():
        status["heartbeat"] = timestamp()
        status["counts"] = {state: sum(row["state"] == state for row in status["runs"].values())
                            for state in ("QUEUED", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")}
        status["total_environment_steps"] = sum(row.get("environment_steps", 0) for row in status["runs"].values())
        status["disk_free_bytes"] = shutil.disk_usage(output).free
        write_atomic(output / "status.json", status)

    previous_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while jobs or active:
            for key, child in list(active.items()):
                row = status["runs"][key]
                row.update(latest_progress(Path(row["path"])))
                row["elapsed_seconds"] = round(time.monotonic() - row["start_monotonic"], 1)
                code = child.poll()
                if code is not None:
                    row.update(state="COMPLETED" if code == 0 else "FAILED", return_code=code, finished=timestamp())
                    del active[key]
                    print(json.dumps({"event": "finished", "job": key, "return_code": code}), flush=True)
            if shutil.disk_usage(output).free < manifest["minimum_free_disk_bytes"]:
                raise RuntimeError("Disk safety threshold reached; stopping workers to protect experiment records")
            while jobs and len(active) < manifest["workers"]:
                job = jobs.pop(0)
                row = status["runs"][job["id"]]
                command = training_command(manifest, output, job)
                log_path = output / job["condition"] / "logs" / f"seed_{job['seed']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=manifest["repository"], env=environment,
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[job["id"]] = child
                row.update(state="RUNNING", pid=child.pid, started=timestamp(), start_monotonic=time.monotonic(),
                           log=str(log_path), command=command)
                print(json.dumps({"event": "started", "job": job["id"], "pid": child.pid}), flush=True)
            publish()
            if jobs or active:
                time.sleep(10)
        if status["counts"]["FAILED"]:
            raise RuntimeError("Some training jobs failed; inspect condition/logs/seed_N.log")
        status["state"] = "SUMMARIZING"
        publish()
        summaries = {}
        for label in manifest["conditions"]:
            runs = []
            for seed in manifest["seeds"]:
                path = output / label / f"seed_{seed}"
                audit(path, recompute=True)
                config = json.loads((path / "config.json").read_text())
                runs.append({"seed": seed, "path": str(path), "registered_path": config["registered_path"],
                             "summary": json.loads((path / "summary.json").read_text())})
            summarize(output / label, runs)
            summaries[label] = json.loads((output / label / "summary.json").read_text())
            status["summarized_conditions"] = list(summaries)
            publish()
        write_atomic(output / "summary.json", summaries)
        status["state"] = "COMPLETED"
    except BaseException as error:
        status.update(state="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        traceback.print_exc()
        for child in active.values():
            if child.poll() is None:
                child.terminate()
        for key, child in active.items():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            row = status["runs"][key]
            row.update(state=status["state"], return_code=child.returncode, finished=timestamp())
            path = Path(row["path"]) / "run.json"
            if path.exists():
                run = json.loads(path.read_text())
                run.update(status=status["state"], stop_time=timestamp(), return_code=child.returncode)
                write_atomic(path, run)
        for row in status["runs"].values():
            if row["state"] == "QUEUED":
                row["state"] = "INTERRUPTED"
        raise
    finally:
        status["finished"] = timestamp()
        publish()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        lock.close()
