"""Frozen, resource-aware SMPE and EPyMARL MAPPO replication queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

from algorithms.common.comm_scale_roster import (CONDITIONS, PROTOCOL, SEEDS, canonical_hash,
    finalize_run, scale_configs, supplied_roster_configs, training_command, validate_config)
from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.progress import latest_progress

GIB = 1024 ** 3
PROJECT = Path(__file__).resolve().parents[2]


def process_rss(pid):
    try:
        return next(int(line.split()[1]) * 1024 for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                    if line.startswith("VmRSS:"))
    except (OSError, StopIteration):
        return 0


def available_ram():
    return next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def gpu_status():
    free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    used = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    processes = {}
    for line in used.splitlines():
        pid, memory = line.split(",")
        processes[int(pid.strip())] = int(memory.strip())
    return int(free.splitlines()[0]), processes


def reservation(job, control):
    size = "9" if job["condition"].split("_", 1)[1] in ("N2", "D2") else "6"
    return control["peak_ram_gib"][job["algorithm"]][job["device"]][size] * GIB


def select_job(queued, active, control, ram, gpu_free, gpu_processes):
    if len(active) >= control["max_workers"]:
        return None
    headroom = ram - control["minimum_available_ram_gib"] * GIB - sum(
        max(0, reservation(job, control) - process_rss(job["pid"])) for job in active)
    gpu_jobs = [job for job in active if job["device"] == "cuda"]
    reserved_vram = sum(max(0, control["gpu_peak_mib"][job["algorithm"]] - gpu_processes.get(job["pid"], 0)) for job in gpu_jobs)
    for device, limit in (("cuda", control["gpu_workers"]), ("cpu", control["cpu_workers"])):
        if sum(job["device"] == device for job in active) >= limit:
            continue
        for job in queued:
            candidate = dict(job, device=device)
            if device == "cuda" and (job["algorithm"] not in control["gpu_algorithms"] or
                    gpu_free - reserved_vram - control["gpu_peak_mib"][job["algorithm"]] < control["minimum_free_gpu_mib"]):
                continue
            if reservation(candidate, control) <= headroom:
                return candidate
    return None


def results(output, rows):
    summary = {}
    for condition in CONDITIONS:
        runs = [row for row in rows if row["condition"] == condition and row["state"] == "COMPLETED"]
        values = [json.loads((Path(row["path"]) / "protocol_metrics.json").read_text()) for row in runs]
        wins = [100 * value["models"]["latest"]["greedy"]["win_rate"] for value in values]
        summary[condition] = {"completed_seeds": [row["seed"] for row in runs], "runs": values,
            "latest_greedy_mean_pct": statistics.mean(wins) if len(wins) == 3 else None,
            "latest_greedy_sem_pct": statistics.stdev(wins) / 3 ** .5 if len(wins) == 3 else None}
    write_atomic(output / "summary.json", summary)
    lines = ["# 实验结果", "", "最终预算模型的独立贪心测试胜率，单位为百分比；仅三个 seed 齐全时报告均值和 SEM。", "",
             "| 配置 | 已完成 seed | 胜率（均值 ± SEM） |", "|---|---|---:|"]
    for condition, item in summary.items():
        cell = ("%.2f ± %.2f" % (item["latest_greedy_mean_pct"], item["latest_greedy_sem_pct"])) if len(item["runs"]) == 3 else "待完成"
        lines.append(f"| {condition} | {', '.join(map(str, item['completed_seeds']))} | {cell} |")
    lines += ["", "N2 不归一化奖励，D2 采用参考归一化；两者差异不能全部解释为密度作用。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def manage(output):
    manifest = json.loads((output / "experiment.json").read_text())
    lock = (output / "manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "status.json").exists():
        raise RuntimeError("Refusing to overwrite an existing queue state")
    jobs = []
    for seed in SEEDS:
        for condition in sorted(CONDITIONS, key=lambda name: name.split("_", 1)[1] not in ("N2", "D2")):
            for algorithm, record in manifest["algorithms"].items():
                comm, scenario = condition.split("_", 1)
                jobs.append({"id": f"{algorithm}/{condition}/seed_{seed}", "algorithm": algorithm,
                    "condition": condition, "seed": seed, "path": str(Path(record["output"]) / comm / scenario / f"seed_{seed}"),
                    "state": "QUEUED"})
    state = {"state": "RUNNING", "started": timestamp(), "manager_pid": os.getpid(), "runs": {job["id"]: job for job in jobs}}
    active = {}
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")

    def publish():
        state["heartbeat"] = timestamp()
        state["counts"] = {name: sum(row["state"] == name for row in jobs) for name in
                           ("QUEUED", "RUNNING", "FINALIZING", "COMPLETED", "FAILED", "INTERRUPTED")}
        state["total_environment_steps"] = sum(row.get("environment_steps", 0) for row in jobs)
        state["available_ram_bytes"] = available_ram()
        state["disk_free_bytes"] = shutil.disk_usage(output).free
        write_atomic(output / "status.json", state)
        for algorithm, record in manifest["algorithms"].items():
            rows = {f"{row['condition']}/seed_{row['seed']}": row for row in jobs if row["algorithm"] == algorithm}
            write_atomic(Path(record["output"]) / "status.json", {**{k: v for k, v in state.items() if k != "runs"},
                "counts": {name: sum(row["state"] == name for row in rows.values()) for name in state["counts"]},
                "total_environment_steps": sum(row.get("environment_steps", 0) for row in rows.values()), "runs": rows})

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Queue received signal " + str(signum))

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        publish()
        while active or any(row["state"] == "QUEUED" for row in jobs):
            for key, child in list(active.items()):
                row = state["runs"][key]
                row.update(latest_progress(Path(row["path"])), rss_bytes=process_rss(child.pid))
                row["elapsed_seconds"] = round(time.monotonic() - row["start_monotonic"], 1)
                code = child.poll()
                if code is None:
                    continue
                del active[key]
                row.update(state="FINALIZING" if code == 0 else "FAILED", return_code=code, finished=timestamp())
                publish()
                if code == 0:
                    try:
                        row["protocol_metrics"] = finalize_run(row["path"])
                        row["state"] = "COMPLETED"
                    except Exception:
                        row.update(state="FAILED", audit_error=traceback.format_exc())
                        traceback.print_exc()
                print(json.dumps({"event": "finished", "job": key, "state": row["state"], "return_code": code}), flush=True)
                record = manifest["algorithms"][row["algorithm"]]
                results(Path(record["output"]), [job for job in jobs if job["algorithm"] == row["algorithm"]])
            control = json.loads((output / "resource_control.json").read_text())
            if min(control["max_workers"], control["cpu_workers"], control["gpu_workers"]) < 0 or control["minimum_available_ram_gib"] < 4:
                raise ValueError("Invalid resource limits")
            if shutil.disk_usage(output).free < control["minimum_free_disk_gib"] * GIB:
                raise RuntimeError("Disk safety threshold reached")
            while True:
                queued = [row for row in jobs if row["state"] == "QUEUED"]
                if not queued or len(active) >= control["max_workers"]:
                    break
                gpu_free, gpu_processes = gpu_status() if control["gpu_workers"] else (0, {})
                selected = select_job(queued, [state["runs"][key] for key in active], control,
                                      available_ram(), gpu_free, gpu_processes)
                if selected is None:
                    break
                row = state["runs"][selected["id"]]
                if Path(row["path"]).exists():
                    raise RuntimeError("Refusing to overwrite run: " + row["path"])
                row.update(device=selected["device"])
                command = training_command(manifest, row)
                log_path = Path(row["path"]).parent / "logs" / f"seed_{row['seed']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=manifest["repository"], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[row["id"]] = child
                row.update(state="RUNNING", pid=child.pid, log=str(log_path), command=command,
                           started=timestamp(), start_monotonic=time.monotonic())
                print(json.dumps({"event": "started", "job": row["id"], "pid": child.pid, "device": row["device"]}), flush=True)
            state["resource_control"] = control
            publish()
            if active or queued:
                time.sleep(10)
        state["state"] = "COMPLETED" if all(row["state"] == "COMPLETED" for row in jobs) else "FAILED"
    except BaseException as error:
        state.update(state="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        traceback.print_exc()
        for key, child in active.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for key, child in active.items():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            state["runs"][key].update(state="INTERRUPTED", return_code=child.returncode, finished=timestamp())
            run_path = Path(state["runs"][key]["path"]) / "run.json"
            if run_path.exists():
                record = json.loads(run_path.read_text())
                record.update(status="INTERRUPTED", stop_time=timestamp(), return_code=child.returncode)
                write_atomic(run_path, record)
        for row in jobs:
            if row["state"] == "QUEUED":
                row["state"] = "INTERRUPTED"
        raise
    finally:
        state["finished"] = timestamp()
        publish()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lock.close()


def prepare(output, smpe_root, epymarl_root):
    from algorithms.epymarl_mappo.backend import resolve_root, settings
    from algorithms.smpe.settings import hyperparameters
    epymarl_root = resolve_root(epymarl_root)
    smpe_root = Path(smpe_root).resolve()
    variants = scale_configs(PROJECT)
    variants.update(supplied_roster_configs(variants["C1S_B0"]))
    for value in variants.values():
        value.update(terminate_on_no_firepower=False, draw_reward=0.0)
    output.mkdir(parents=True, exist_ok=False)
    project = output / "_code/project"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("algorithms", "heterogrid", "configs"):
        shutil.copytree(PROJECT / name, project / name, ignore=ignore)
    for label, root, directories in (
            ("smpe_raw", smpe_root, ("controllers", "learners", "model_learners", "modules", "components", "utils", "config")),
            ("epymarl/src", epymarl_root, ("controllers", "learners", "modules", "components", "utils", "config"))):
        destination = output / "_code" / label
        for name in directories:
            shutil.copytree(root / name, destination / name, ignore=ignore)
        for name in ("run.py", "main.py", "__init__.py", "LICENSE", "requirements.txt"):
            if (root / name).is_file():
                shutil.copy2(root / name, destination / name)
    manifest = {"created": timestamp(), "python": sys.executable, "repository": str(project),
        "protocol": PROTOCOL, "conditions": list(CONDITIONS), "seeds": list(SEEDS), "jobs": 90,
        "total_requested_team_steps": 450_000_000, "algorithms": {},
        "source_repository": str(PROJECT), "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "source_dirty_diff": subprocess.check_output(["git", "diff", "--stat"], cwd=PROJECT, text=True),
        "config_provenance": {"scale": "Existing 6v6 and density_9v9 JSON rosters in their original order; only declared protocol overrides",
                              "role_ratios": "Exact role/platform ordering supplied by user on 2026-11-13"}}
    for algorithm, backend in (("smpe", output / "_code/smpe_raw"), ("epymarl_mappo", output / "_code/epymarl/src")):
        destination = PROJECT / "output" / algorithm / output.name
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "configs").mkdir()
        for name, value in variants.items():
            validate_config(name, value)
            write_atomic(destination / "configs" / (name + ".json"), value)
        parameters = hyperparameters(backend, "cpu") if algorithm == "smpe" else settings(backend)
        record = {"output": str(destination), "backend_root": str(backend), "source_backend_root": str(smpe_root if algorithm == "smpe" else epymarl_root),
                  "algorithm_args_cpu": parameters, "config_hashes": {name: canonical_hash(value) for name, value in variants.items()}}
        manifest["algorithms"][algorithm] = record
        write_atomic(destination / "experiment.json", {**record, "algorithm": algorithm, "conditions": list(CONDITIONS),
            "seeds": list(SEEDS), "jobs": 45, "protocol": PROTOCOL, "coordinator": str(output), "repository": str(project),
            "initialization": "Independent fresh initialization; no reused models", "total_requested_team_steps": 225_000_000})
        (destination / "README.md").write_text(
            "# 通信、规模与角色配比实验\n\n15 个配置，seed 42、43、44，各自从头训练至 5M 团队环境步。\n"
            "训练和测试使用同一通信配置，双方独立链路；R5/P50 固定蓝方，C1S 双方无通信。\n"
            "验证种子 1910042–1910105，独立测试种子 2800042–2800169。\n"
            "周期约 250k 步验证 64 局，最终模型与验证最优模型各做贪心/采样 128 局测试。\n"
            "训练按完整新鲜对局批次更新，到达目标后结束；实际步数最多额外 1999 步，验证测试不计入。\n"
            "训练种子按 1000000 + training_seed * 100000000 + episode_index 分配，索引从 0 开始。\n"
            "无火力不提前终止，平局奖励为 0，保持旧环境语义。\n"
            "N2 不归一化奖励，D2 采用参考人数 6、生命值 480 的归一化，不应把差异全部归因于密度。\n"
            "配置保留提供的角色顺序及平台。源码在 experiment.json 指向的冻结目录，并另存逐运行源码快照。\n"
            "每次完成后自动审计并只保留 latest.pt、best.pt，各一局贪心完整回放及真实动作，逐局日志无损压缩。\n"
            "protocol_metrics.json 记录测试指标、验证曲线面积、首次达到 80% 的步数及训练吞吐；未达到时为 null。\n"
            "metrics.json 保留完整验证曲线；RESULTS.md 和 summary.json 随任务完成自动更新。\n", encoding="utf-8")
    manifest["source_hashes"] = {str(path.relative_to(output / "_code")): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (output / "_code").rglob("*") if path.is_file()}
    write_atomic(output / "experiment.json", manifest)
    control = {"max_workers": 14, "cpu_workers": 10, "gpu_workers": 4, "gpu_algorithms": ["smpe"],
        "minimum_available_ram_gib": 6, "minimum_free_gpu_mib": 4000, "minimum_free_disk_gib": 3,
        "peak_ram_gib": {"smpe": {"cpu": {"6": 3., "9": 4.}, "cuda": {"6": 2., "9": 2.5}},
                         "epymarl_mappo": {"cpu": {"6": 1.2, "9": 1.8}, "cuda": {"6": 1.2, "9": 1.8}}},
        "gpu_peak_mib": {"smpe": 4500, "epymarl_mappo": 2000}}
    write_atomic(output / "resource_control.json", control)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpe-root", type=Path)
    parser.add_argument("--epymarl-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manage", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.manage:
        manage(output)
        return
    if args.smpe_root is None or args.epymarl_root is None:
        parser.error("Both external algorithm roots are required")
    manifest = prepare(output, args.smpe_root, args.epymarl_root)
    if args.prepare_only:
        print(json.dumps({"output": str(output), "jobs": 90, "state": "PREPARED"}))
        return
    command = ["nohup", sys.executable, "-u", "-m", "algorithms.common.replication", "--manage", "--output", str(output)]
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=manifest["repository"], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "manager_pid": process.pid, "jobs": 90}))


if __name__ == "__main__":
    main()
