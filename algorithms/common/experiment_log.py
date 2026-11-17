"""Export local runs in the results tree's Sacred-compatible file layout."""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers can briefly deny replacement while inspecting status.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.02 * (attempt + 1))


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile

    def write(self, value):
        self.logfile.write(value)
        return self.original.write(value)

    def flush(self):
        self.logfile.flush()
        self.original.flush()


class ExperimentRecorder:
    def __init__(self, output, group, config, sources=(), capture=True, *, resume_step=None):
        self.output = Path(output).resolve()
        self.registered_path = self.output
        if group is not None and resume_step is None:
            group = Path(group).resolve()
            group.mkdir(parents=True, exist_ok=True)
            index = 1
            while True:
                self.registered_path = group / str(index)
                try:
                    self.registered_path.symlink_to(self.output, target_is_directory=True)
                    break
                except FileExistsError:
                    index += 1
        self.metrics = {}
        self.info = {}
        self.closed = False
        self.run = {"status": "RUNNING", "start_time": timestamp(), "stop_time": None,
                    "heartbeat": timestamp(), "command": config["name"],
                    "format": "sacred-compatible export (not a Sacred execution)",
                    "host": {"hostname": socket.gethostname(), "python_version": platform.python_version(),
                             "os": platform.platform()},
                    "experiment": {"name": config["name"], "framework": config.get("framework", "onpolicy/MAPPO"), "sources": []},
                    "artifacts": [], "result": None}
        if resume_step is not None:
            previous = json.loads((self.output / "config.json").read_text(encoding="utf-8"))
            self.run = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
            if self.run["command"] != config["name"]:
                raise ValueError("Cannot resume a different experiment")
            self.registered_path = Path(previous["registered_path"])
            metrics = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
            for name, metric in metrics.items():
                retained = [i for i, step in enumerate(metric["steps"]) if step <= resume_step]
                self.metrics[name] = {key: [values[i] for i in retained] for key, values in metric.items()}
                self.info[name] = self.metrics[name]["values"]
                self.info[name + "_T"] = self.metrics[name]["steps"]
            self.run.update(status="RUNNING", stop_time=None, heartbeat=timestamp(), result=None)
            self.run.setdefault("resumptions", []).append({"time": timestamp(), "checkpoint_step": resume_step})
        source_dir = self.output / "_sources"
        source_dir.mkdir(exist_ok=resume_step is not None)
        for source in sources:
            source = Path(source).resolve()
            content = source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            target = source_dir / (digest[:16] + "_" + source.name)
            target.write_bytes(content)
            entry = {"path": str(source), "snapshot": str(target.relative_to(self.output)), "sha256": digest}
            if entry not in self.run["experiment"]["sources"]:
                self.run["experiment"]["sources"].append(entry)
        config = {**config, "registered_path": str(self.registered_path)}
        if resume_step is None:
            write_atomic(self.output / "config.json", config)
        self.flush()
        self.logfile = (self.output / "cout.txt").open("w" if resume_step is None else "a", encoding="utf-8", buffering=1)
        self.streams = (sys.stdout, sys.stderr) if capture else None
        if capture:
            sys.stdout, sys.stderr = Tee(sys.stdout, self.logfile), Tee(sys.stderr, self.logfile)
        atexit.register(self.close)

    def log(self, step, values, *, flush=True):
        now = timestamp()
        for name, value in values.items():
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Non-finite metric: " + name)
            metric = self.metrics.setdefault(name, {"steps": [], "timestamps": [], "values": []})
            metric["steps"].append(int(step))
            metric["timestamps"].append(now)
            metric["values"].append(value)
            self.info[name] = metric["values"]
            self.info[name + "_T"] = metric["steps"]
        self.run["heartbeat"] = now
        if flush:
            self.flush()

    def evaluation(self, step, result, deterministic):
        prefix = "test_" if deterministic else "sampled_test_"
        names = {"win_rate": "battle_won_mean", "mean_return": "return_mean", "mean_length": "ep_length_mean",
                 "timeout_rate": "timeout_mean", "mean_damage": "damage_mean", "mean_kills": "kills_mean",
                 "mean_useful_reports": "useful_reports_mean", "contact_rate": "contact_mean",
                 "mean_first_contact_step": "first_contact_step_mean"}
        values = {prefix + name: result[key] for key, name in names.items()}
        returns = [row["return"] for row in result["results"]]
        values[prefix + "return_std"] = (sum((x - result["mean_return"]) ** 2 for x in returns) / len(returns)) ** 0.5
        self.log(step, values)

    def flush(self):
        write_atomic(self.output / "metrics.json", self.metrics)
        write_atomic(self.output / "info.json", self.info)
        write_atomic(self.output / "run.json", self.run)

    def complete(self, result):
        self.run["status"] = "COMPLETED"
        self.run["result"] = result
        print("Finished Training", flush=True)
        self.close()

    def close(self):
        if self.closed:
            return
        if self.run["status"] == "RUNNING":
            self.run["status"] = "FAILED"
        self.run["stop_time"] = timestamp()
        self.run["heartbeat"] = self.run["stop_time"]
        self.flush()
        if self.streams:
            sys.stdout, sys.stderr = self.streams
        self.logfile.close()
        self.closed = True
        atexit.unregister(self.close)
