"""Live resource admission and adoption of already-running MA2E launchers."""
import argparse
import fcntl
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
from algorithms.common.study_queue import jobs_for

GIB = 1024 ** 3


def process_token(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def process_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available host RAM")


def device_record(row):
    path = Path(row["path"]).parent / "logs" / f"seed_{row['seed']}.device.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("attempts"):
            attempt = record["attempts"][-1]
            return attempt["device"], attempt.get("pid"), record
    return row.get("requested_device", "auto"), None, None


class AdoptedProcess:
    def __init__(self, row, token, allow_exited=False):
        self.row, self.pid, self.token, self.returncode = row, int(row["pid"]), token, None
        current = process_token(self.pid)
        if current is None and allow_exited:
            return
        if current != token:
            raise RuntimeError("Launcher identity changed before adoption")
        command = [value.decode() for value in Path(f"/proc/{self.pid}/cmdline").read_bytes().split(b"\0") if value]
        module_index = command.index("-m") if "-m" in command else -1
        flags = {"-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO"}
        valid_module = (module_index >= 1 and len(command) > module_index + 1
            and Path(command[0]).name.startswith(("python", "pypy"))
            and all(value in flags for value in command[1:module_index])
            and command[module_index + 1] == "algorithms.ma2e.launch")
        parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
        parser.add_argument("--output")
        try:
            invocation, _ = parser.parse_known_args(command[module_index + 2:]) if valid_module else (None, None)
        except argparse.ArgumentError:
            invocation = None
        if invocation is None or invocation.output != str(Path(row["path"])):
            raise RuntimeError("Refusing to adopt an unrelated process")

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if process_token(self.pid) == self.token:
            return None
        _, _, record = device_record(self.row)
        self.returncode = int(record.get("return_code", 1)) if record else 1
        if self.returncode == 0:
            path = Path(self.row["path"])
            if not (path / "summary.json").exists() or not (path / "run.json").exists():
                self.returncode = 1
            elif json.loads((path / "run.json").read_text())["status"] != "COMPLETED":
                self.returncode = 1
        return self.returncode

    def terminate(self):
        if process_token(self.pid) == self.token:
            os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        if process_token(self.pid) == self.token:
            os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return self.returncode


def peak_ram(info, args, device):
    """Reserve a full raw replay, full float32 latent cache and transient workspace."""
    n, obs, state, actions = (info[k] for k in ("n_agents", "obs_shape", "state_shape", "n_actions"))
    length = info["episode_limit"] + 1
    per_step = 4 * (n * obs + state + 2 * n * actions) + 8 * n + 13
    replay = per_step * length * args["buffer_size"]
    latents = 4 * n * n * args["MT_n_embd"] * length * args["buffer_size"]
    history = 4 * args["MT_batchsize"] * length * n * args["MT_traj_length"] * (obs + actions + n)
    workspace = max(.75 * GIB, 2 * history, (replay + latents) * args["batch_size"] / args["buffer_size"] + .5 * GIB)
    return int(replay + latents + workspace + (2.0 if device in ("cuda", "auto") else 1.25) * GIB)


def capacity(available, active, peaks, floor):
    reserved_growth = sum(max(0, peaks[(row["condition"], device_record(row)[0])] -
        process_rss(device_record(row)[1] or row["pid"])) for row in active)
    return available - reserved_growth - floor


def choose_job(jobs, running, control, headroom, peaks):
    gpu_count = sum(device_record(row)[0] in ("cuda", "auto") for row in running)
    mode = "auto" if gpu_count < control["gpu_max_workers"] else "cpu"
    candidates = jobs if mode == "auto" else sorted(jobs, key=lambda job: (control["team_sizes"][job["condition"]], job["id"]))
    priority = {key: index for index, key in enumerate(control.get("priority_jobs", []))}
    candidates = sorted(candidates, key=lambda job: priority.get(job["id"], len(priority)))
    gpu_only = set(control.get("gpu_only_jobs", []))
    for job in candidates:
        if mode == "cpu" and job["id"] in gpu_only:
            continue
        if mode == "cpu" and control["team_sizes"][job["condition"]] > control["cpu_max_agents"]:
            continue
        selected_mode = "cuda" if job["id"] in gpu_only else mode
        if peaks[(job["condition"], selected_mode)] <= headroom:
            return job, selected_mode
    return None, None


def run(output, adopt=False):
    from algorithms.ma2e.audit import audit
    from algorithms.ma2e.campaign import summarize, training_command, write_results
    from heterogrid import EnvConfig, SMACAdapter
    output = Path(output).resolve()
    lock = (output / "manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((output / "experiment.json").read_text())
    planned = jobs_for(manifest, output)
    active = {}
    if adopt:
        status = json.loads((output / "status.json").read_text())
        handover = json.loads((output / "supervisor_handover.json").read_text())
        if process_token(handover["old_manager_pid"]) == handover["old_manager_token"]:
            raise RuntimeError("Previous manager is still alive")
        if set(status["runs"]) != {job["id"] for job in planned}:
            raise RuntimeError("Saved queue does not match the experiment")
        for key, row in status["runs"].items():
            if row["state"] == "RUNNING":
                active[key] = AdoptedProcess(row, handover["launcher_tokens"][key], allow_exited=True)
    else:
        status = {"state": "RUNNING", "started": timestamp(),
                  "runs": {job["id"]: dict(job, state="QUEUED") for job in planned}}
    status.update(state="RUNNING", manager_pid=os.getpid(), supervisor="ma2e_resource_aware")
    jobs = [job for job in planned if status["runs"][job["id"]]["state"] == "QUEUED"]
    peaks = {}
    for name in manifest["conditions"]:
        env = SMACAdapter(config=EnvConfig.load(output / "configs" / (name + ".json")))
        try:
            info = env.get_env_info()
        finally:
            env.close()
        for device in ("cpu", "cuda", "auto"):
            peaks[(name, device)] = peak_ram(info, manifest["algorithm_args"], device)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")

    def publish():
        status["heartbeat"] = timestamp()
        status["counts"] = {state: sum(row["state"] == state for row in status["runs"].values())
            for state in ("QUEUED", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")}
        status["total_environment_steps"] = sum(row.get("environment_steps", 0) for row in status["runs"].values())
        status["disk_free_bytes"] = shutil.disk_usage(output).free
        write_atomic(output / "status.json", status)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Supervisor signal " + str(signum))

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        publish()
        while jobs or active:
            for key, child in list(active.items()):
                row = status["runs"][key]
                row.update(latest_progress(Path(row["path"])))
                row["elapsed_seconds"] = round(time.monotonic() - row["start_monotonic"], 1)
                device, pid, _ = device_record(row)
                row.update(device=device, training_pid=pid, rss_bytes=process_rss(pid or row["pid"]))
                code = child.poll()
                if code is not None:
                    row.update(state="COMPLETED" if code == 0 else "FAILED", return_code=code, finished=timestamp())
                    del active[key]
                    print(json.dumps({"event": "finished", "job": key, "return_code": code}), flush=True)
            control_path = output / "resource_control.json"
            control = json.loads(control_path.read_text()) if control_path.exists() else {
                "max_workers": manifest["workers"], "gpu_max_workers": manifest["gpu_max_workers"],
                "minimum_available_ram_gib": 6, "cpu_max_agents": 12}
            if min(control["max_workers"], control["gpu_max_workers"], control["cpu_max_agents"]) < 1 or control["minimum_available_ram_gib"] < 4:
                raise ValueError("Invalid live resource limits")
            control["team_sizes"] = manifest["team_sizes"]
            status["workers"] = control["max_workers"]
            if shutil.disk_usage(output).free < manifest["minimum_free_disk_bytes"]:
                raise RuntimeError("Disk safety threshold reached")
            while jobs and len(active) < control["max_workers"]:
                running = [status["runs"][key] for key in active]
                headroom = capacity(memory_available(), running, peaks, control["minimum_available_ram_gib"] * GIB)
                job, mode = choose_job(jobs, running, control, headroom, peaks)
                if job is None:
                    break
                row = status["runs"][job["id"]]
                if Path(job["path"]).exists():
                    raise RuntimeError("Refusing to restart or overwrite an existing queued run: " + job["id"])
                parameters = dict(manifest, device=mode, gpu_max_workers=control["gpu_max_workers"])
                command = training_command(parameters, output, job)
                if job["id"] in control.get("gpu_only_jobs", []):
                    command.append("--no-cpu-fallback")
                log_path = output / job["condition"] / "logs" / f"seed_{job['seed']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=manifest["repository"], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[job["id"]] = child
                jobs.remove(job)
                row.update(state="RUNNING", pid=child.pid, started=timestamp(), start_monotonic=time.monotonic(),
                    log=str(log_path), command=command, requested_device=mode, reserved_peak_ram_bytes=peaks[(job["condition"], mode)])
                print(json.dumps({"event": "started", "job": job["id"], "pid": child.pid, "requested_device": mode}), flush=True)
            running = [status["runs"][key] for key in active]
            status["resource_admission"] = {"available_ram_bytes": memory_available(),
                "unreserved_ram_bytes": capacity(memory_available(), running, peaks, control["minimum_available_ram_gib"] * GIB),
                "minimum_available_ram_gib": control["minimum_available_ram_gib"], "policy": "reserve full replay/cache growth; small CPU rosters first"}
            publish()
            if jobs or active:
                time.sleep(10)
        if status["counts"]["FAILED"]:
            raise RuntimeError("Some training jobs failed")
        status["state"] = "SUMMARIZING"
        publish()
        summaries = {}
        for name in manifest["conditions"]:
            runs = []
            for seed in manifest["seeds"]:
                path = output / name / f"seed_{seed}"
                audit(path, recompute=True)
                config = json.loads((path / "config.json").read_text())
                runs.append({"seed": seed, "path": str(path), "registered_path": config["registered_path"],
                             "summary": json.loads((path / "summary.json").read_text())})
            summarize(output / name, runs)
            summaries[name] = json.loads((output / name / "summary.json").read_text())
            status["summarized_conditions"] = list(summaries)
            publish()
        write_atomic(output / "summary.json", summaries)
        write_results(output)
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
            status["runs"][key].update(state=status["state"], return_code=child.returncode, finished=timestamp())
        raise
    finally:
        status["finished"] = timestamp()
        publish()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adopt", action="store_true")
    cli = parser.parse_args()
    run(cli.output, cli.adopt)
