"""Prepare and supervise a bounded nohup QMIX/QTRAN sensitivity campaign."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, GROUPS, LABELS, scenarios, communication_protocol
from algorithms.common.value_decomposition.report import read_json, summarize


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args):
    coordinator = args.output_root / "campaigns" / (args.name + "_qmix_qtran")
    if coordinator.exists():
        raise FileExistsError("Campaign already exists: " + str(coordinator))
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    runtime.mkdir()
    for name in ("heterogrid", "algorithms", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    backend = coordinator / "_backend"
    for name in ("policy", "network", "common"):
        shutil.copytree(args.marl_root / name, backend / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tasks = scenarios(fixed_blue=args.fixed_blue)
    roots = {}
    for alg in args.algorithms:
        root = args.output_root / alg / args.name
        root.mkdir(parents=True, exist_ok=False)
        (root / "configs").mkdir()
        for name, config in tasks.items():
            write_atomic(root / "configs" / (name + ".json"), config.to_dict())
        experiment = {"algorithm": alg, "variant": "qtran_base" if alg == "qtran" else alg,
            "created": now(), "seeds": args.seeds, "steps": args.steps, "groups": GROUPS, "labels": LABELS,
            "communication_protocol": communication_protocol(args.fixed_blue),
            "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
            "heldout_episodes": args.heldout_episodes, "eval_seed": 100042, "heldout_seed": 200042,
            "save_interval": 2_000_000, "primary_model": "final.pt", "selection": "greedy validation win rate then return",
            "sampled_policy": "epsilon-greedy epsilon=0.05", "num_envs": 4,
            "hyperparameters": {"rnn_hidden_dim": 64, "qmix_hidden_dim": 32, "qtran_hidden_dim": 64,
                "hyper_hidden_dim": 64, "two_hyper_layers": False, "lr": 0.0005, "gamma": 0.99,
                "optimizer": "RMSprop (upstream defaults)", "batch_size": 32, "buffer_episodes": 1000,
                "warmup_episodes": 32, "update_interval_environment_steps": 200, "target_update_cycle_updates": 200,
                "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_anneal_steps": 50000,
                "lambda_opt": 1, "lambda_nopt": 1, "gradient_clip": 10},
            "coordinator": str(coordinator), "runtime_snapshot": str(runtime), "backend_snapshot": str(backend),
            "original_backend": str(args.marl_root), "planned_runs": len(tasks) * len(args.seeds)}
        write_atomic(root / "experiment.json", experiment)
        (root / "README.md").write_text(
            f"# {alg.upper()} 灵敏度实验\n\n13 场景 × {len(args.seeds)} 种子，每次 {args.steps:,} 团队环境步。\n"
            "B0 为基准，M1–M3 改地图，N1–N3 改人数，C1–C6 改通信。完整展开配置在 configs/。\n"
            "双方角色始终1:1:1；规则蓝方、200步上限、当前奖励和自动目标共享。\n"
            "模型、指标、源码与评估按 <场景>/seed_<种子>/ 保存。status.json 与 run_status.csv 显示进度。\n"
            "主结果为固定预算 final.pt；best.pt 仅用验证选择。模型测试128局，贪心为主、ε=0.05采样为辅。\n"
            "最终原始指标见 independent_test_per_seed.csv，聚合见 aggregate.json、RESULTS.md。\n"
            "figures/ 包含地图、人数、通信三组胜率均值±SEM图及源CSV，仅绘制三个种子共有的真实验证点。\n"
            "训练步不乘智能体数量，不含评估；AUC使用含初始点的梯形积分，归一化AUC除以预算。\n"
            "未达到80%的种子标记null/未达到，不用预算值替代。接敌时间以接敌局为条件。\n"
            "算法来自外部 MARL-Algorithms；目标RNN首帧对齐修正写入每次config.json。\n"
            "只从头训练；检查点不含回放，当前不支持精确续训。停止管理器会向工作进程转发TERM并保存中断模型。\n"
            + ("本轮蓝方固定 distance、半径10、失联0；通信对照只改变红方。C5/C6仍是full拓扑叠加节点失联。\n"
               if args.fixed_blue else "本轮旧版共享通信参数同时改变双方。\n"),
            encoding="utf-8")
        roots[alg] = str(root)
    jobs = []
    # Long tasks first reduce the final tail without changing any training parameter.
    task_order = sorted(tasks, key=lambda name: (-len(tasks[name].red), -sum(tasks[name].map_size)))
    for task in task_order:
        for seed in args.seeds:
            for alg in args.algorithms:
                root = Path(roots[alg])
                output = root / task / f"seed_{seed}"
                command = [sys.executable, "-m", f"algorithms.{alg}.train", "--marl-root", str(backend),
                    "--config", str(root / "configs" / (task + ".json")), "--output", str(output),
                    "--seed", str(seed), "--steps", str(args.steps), "--eval-interval", str(args.eval_interval),
                    "--eval-episodes", str(args.eval_episodes), "--heldout-episodes", str(args.heldout_episodes)]
                jobs.append({"algorithm": alg, "scenario": task, "seed": seed, "output": str(output),
                             "command": command, "status": "PENDING", "pid": None})
    plan = {"created": now(), "workers": args.workers, "runtime": str(runtime), "roots": roots,
            "jobs": jobs, "manager_pid": None, "status": "PREPARED", "requested_total_steps": args.steps * len(jobs),
            "disk_free_at_prepare": shutil.disk_usage(coordinator).free,
            "host_cpu_count": os.cpu_count(), "python": sys.executable}
    write_atomic(coordinator / "plan.json", plan)
    write_atomic(coordinator / "status.json", plan)
    for root in roots.values():
        summarize(root)
    print(str(coordinator), flush=True)
    return coordinator


def supervise(coordinator):
    coordinator = Path(coordinator).resolve()
    plan = read_json(coordinator / "plan.json")
    if plan is None:
        raise ValueError("Missing campaign plan")
    status = read_json(coordinator / "status.json")
    if status["status"] != "PREPARED":
        raise ValueError("Only a fresh PREPARED campaign can be started")
    plan.update(manager_pid=os.getpid(), started=now(), status="RUNNING")
    live, stopping = {}, False
    failures = 0

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process, _, _ in live.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "NUMEXPR_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    child_env.pop("PYTHONPATH", None)
    last_report = 0.0

    def persist():
        plan["heartbeat"] = now()
        write_atomic(coordinator / "status.json", plan)
        for alg, root in plan["roots"].items():
            jobs = [j for j in plan["jobs"] if j["algorithm"] == alg]
            write_atomic(Path(root) / "status.json", {"manager_pid": os.getpid(), "status": plan["status"],
                "coordinator": str(coordinator), "heartbeat": plan["heartbeat"], "jobs": jobs,
                "counts": {s: sum(j["status"] == s for j in jobs) for s in
                           ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")}})

    try:
        while True:
            free = shutil.disk_usage(coordinator).free
            if free < 2 * 1024**3 and not stopping:
                print("Disk reserve reached; gracefully stopping campaign", flush=True)
                plan["stop_reason"] = "disk free below 2 GiB"
                stop(None, None)
            for key, (process, logfile, job) in list(live.items()):
                result = process.poll()
                if result is None:
                    job["progress"] = read_json(Path(job["output"]) / "progress.json")
                    continue
                logfile.close()
                job.update(returncode=result, finished=now(), status=("COMPLETED" if result == 0 else
                    "INTERRUPTED" if stopping or result == 130 else "FAILED"))
                job["progress"] = read_json(Path(job["output"]) / "progress.json")
                failures += int(job["status"] == "FAILED")
                print(f"{job['algorithm']} {job['scenario']} seed={job['seed']} {job['status']}", flush=True)
                del live[key]
            if failures >= 3 and not stopping:
                plan["stop_reason"] = "three worker failures; inspect logs before restarting from scratch"
                stop(None, None)
            pending = [j for j in plan["jobs"] if j["status"] == "PENDING"]
            if not stopping:
                for job in pending[:max(0, plan["workers"] - len(live))]:
                    output = Path(job["output"])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    logfile = (output.parent / f"seed_{job['seed']}.log").open("w", buffering=1)
                    process = subprocess.Popen(job["command"], cwd=job.get("runtime", plan["runtime"]), env=child_env,
                        stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    job.update(pid=process.pid, status="RUNNING", started=now())
                    live[process.pid] = (process, logfile, job)
                    print(f"Started {job['algorithm']} {job['scenario']} seed={job['seed']} pid={process.pid}", flush=True)
            persist()
            if time.monotonic() - last_report > 300 or not live:
                for root in plan["roots"].values():
                    try:
                        summarize(root)
                    except Exception as error:
                        print("Report error: " + repr(error), flush=True)
                last_report = time.monotonic()
            if not live and (stopping or not any(j["status"] == "PENDING" for j in plan["jobs"])):
                break
            time.sleep(5)
        plan["status"] = ("INTERRUPTED" if stopping else "FAILED" if failures else "COMPLETED")
        plan["finished"] = now()
        persist()
        return 0 if plan["status"] == "COMPLETED" else 1
    finally:
        if live:
            stop(None, None)
            for process, logfile, job in live.values():
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                logfile.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Run an already prepared campaign under nohup")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261011_sensitivity_5m")
    parser.add_argument("--marl-root", type=Path, default=os.environ.get("MARL_ALGORITHMS_ROOT"))
    parser.add_argument("--algorithms", nargs="+", choices=("qmix", "qtran"), default=["qmix", "qtran"])
    parser.add_argument("--fixed-blue", action="store_true", help="Fix blue distance/radius10/no outages; vary red only")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.marl_root is None:
        parser.error("Set --marl-root or MARL_ALGORITHMS_ROOT")
    args.output_root = args.output_root.resolve()
    args.marl_root = args.marl_root.resolve()
    if min(args.workers, args.steps, args.eval_interval, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("Budgets and workers must be positive")
    if args.steps % 4 or args.eval_interval % 4:
        parser.error("Steps and eval interval must be divisible by four")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.algorithms)) != len(args.algorithms):
        parser.error("Use distinct seeds and algorithms")
    prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
