"""Launch a detached, replicated MAPPO study using nohup and a supervised manifest."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.mappo.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, validate_backend
from heterogrid import EnvConfig


def align_steps(steps, batch=512):
    return ((steps + batch - 1) // batch) * batch


def latest_progress(path):
    try:
        with (path / "metrics.jsonl").open("rb") as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 16384))
            lines = file.read().splitlines()
        # A process may be part-way through appending its newest line.
        for line in reversed(lines):
            try:
                row = json.loads(line)
                return {k: row[k] for k in ("environment_steps", "episodes", "recent_return", "recent_win_rate")}
            except (ValueError, KeyError):
                continue
    except FileNotFoundError:
        pass
    return {"environment_steps": 0, "episodes": 0}


def manage(output):
    manifest = json.loads((output / "experiment.json").read_text())
    repo = Path(manifest["repository"])
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1")
    status = {"state": "RUNNING", "started": timestamp(), "manager_pid": os.getpid(), "runs": {}}
    children = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Supervisor received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for seed in manifest["seeds"]:
            run = output / ("seed_" + str(seed))
            command = [manifest["python"], "-u", "-m", "algorithms.mappo.train", "--config", str(output / "env.json"),
                       "--steps", str(manifest["actual_steps_per_seed"]), "--seed", str(seed), "--output", str(run),
                       "--mappo-root", manifest["mappo_root"], "--results-env-name", output.name,
                       "--device", manifest["device"], "--eval-seed", "100042", "--heldout-seed", "200042"]
            if manifest.get("results_root"):
                command += ["--results-root", manifest["results_root"]]
            for option in ("eval_interval", "save_interval", "trajectory_interval", "log_interval",
                           "eval_episodes", "heldout_episodes", "replay_episodes"):
                command += ["--" + option.replace("_", "-"), str(manifest[option])]
            child = subprocess.Popen(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=sys.stdout, stderr=subprocess.STDOUT)
            children[seed] = child
            status["runs"][str(seed)] = {"pid": child.pid, "state": "RUNNING", "path": str(run), "command": command}
            print(json.dumps({"event": "started", "seed": seed, "pid": child.pid}), flush=True)
        while True:
            for seed, child in children.items():
                row = status["runs"][str(seed)]
                row.update(latest_progress(Path(row["path"])))
                code = child.poll()
                if code is not None and row["state"] == "RUNNING":
                    row["return_code"] = code
                    row["state"] = "COMPLETED" if code == 0 else "FAILED"
                    row["finished"] = timestamp()
                    print(json.dumps({"event": "finished", "seed": seed, "return_code": code}), flush=True)
                    if code != 0:
                        run_path = Path(row["path"]) / "run.json"
                        if run_path.exists():
                            run = json.loads(run_path.read_text())
                            run.update(status="FAILED", stop_time=timestamp(), return_code=code)
                            write_atomic(run_path, run)
            status["heartbeat"] = timestamp()
            write_atomic(output / "status.json", status)
            if all(child.poll() is not None for child in children.values()):
                break
            time.sleep(10)
        if any(child.returncode for child in children.values()):
            raise RuntimeError("One or more seeds failed; inspect nohup.log and per-seed cout.txt")
        status["state"] = "SUMMARIZING"
        write_atomic(output / "status.json", status)
        from algorithms.mappo.study import summarize
        from algorithms.mappo.audit import audit
        runs = []
        for seed in manifest["seeds"]:
            run_path = output / ("seed_" + str(seed))
            audit(run_path, recompute=True)
            config = json.loads((run_path / "config.json").read_text())
            runs.append({"seed": seed, "path": str(run_path), "registered_path": config["registered_path"],
                         "summary": json.loads((run_path / "summary.json").read_text())})
        summarize(output, runs)
        status["state"] = "COMPLETED"
    except BaseException as error:
        status["state"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        status["error"] = str(error)
        traceback.print_exc()
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for seed, child in children.items():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            status["runs"][str(seed)]["return_code"] = child.returncode
            if status["runs"][str(seed)]["state"] == "RUNNING":
                status["runs"][str(seed)]["state"] = status["state"]
                run_path = output / ("seed_" + str(seed)) / "run.json"
                if run_path.exists():
                    run = json.loads(run_path.read_text())
                    run.update(status=status["state"], stop_time=timestamp())
                    write_atomic(run_path, run)
        raise
    finally:
        status["heartbeat"] = timestamp()
        status["finished"] = timestamp()
        write_atomic(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--save-interval", type=int, default=2000000)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--trajectory-interval", type=int, default=2000000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--replay-episodes", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.manage:
        manage(args.output.resolve())
        return
    if min(args.steps, args.eval_interval, args.save_interval, args.log_interval, args.trajectory_interval,
           args.eval_episodes, args.heldout_episodes) < 1 or args.replay_episodes < 1:
        parser.error("Budgets, intervals and episode counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use distinct, nonnegative seeds")
    if args.eval_episodes > 100000:
        parser.error("Validation seeds must not overlap held-out seeds")
    args.mappo_root = validate_backend(parser, args.mappo_root)
    if os.name != "posix":
        parser.error("The nohup launcher requires a POSIX system; use algorithms.mappo.study for foreground runs")
    repo = PROJECT_ROOT
    config_path = args.config or DEFAULT_CONFIG
    config = EnvConfig.load(config_path)
    if config.opponent != "rule" or config.communication != "distance":
        parser.error("This launcher requires a fixed rule opponent and distance-limited communication")
    label = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + config_path.stem
    output = (args.output or repo / "output/mappo" / label).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "algorithm": "mappo", "framework": "onpolicy/MAPPO",
                "policy": "feed-forward shared actor, centralized critic", "repository": str(repo),
                "mappo_root": str(args.mappo_root), "python": sys.executable, "created": timestamp(),
                "results_root": str(args.results_root.resolve()) if args.results_root else None,
                "requested_steps_per_seed": args.steps, "actual_steps_per_seed": align_steps(args.steps),
                "initialization": "from scratch; no warm start or resume",
                "seeds": args.seeds, "device": args.device, "num_envs": 4, "rollout_length": 128,
                "eval_episodes": args.eval_episodes, "heldout_episodes": args.heldout_episodes,
                "replay_episodes": args.replay_episodes, "validation_seed": 100042, "heldout_seed": 200042,
                "trajectory_policy": "sampled first/interval/final rollout; metrics cover every rollout",
                "communication_policy": "automatic one-hop target sharing over distance-limited links"}
    for name in ("eval_interval", "save_interval", "log_interval", "trajectory_interval"):
        manifest[name] = align_steps(getattr(args, name))
    write_atomic(output / "experiment.json", manifest)
    write_atomic(output / "env.json", config.to_dict())
    write_atomic(output / "status.json", {"state": "STARTING", "created": timestamp(), "runs": {}})
    description = ["# MAPPO 实验记录", "", "experiment.json 和 env.json 保存本次实验的完整设置。",
                   "status.json 保存管理进程、训练进程、进度与状态；nohup.log 为合并日志。",
                   "seed_N/ 保存各随机种子的独立记录：models/ 为模型，metrics.jsonl 为逐次 PPO 更新，",
                   "training_episodes.jsonl 为逐局结果，metrics.json / info.json 为兼容格式指标。",
                   "trajectories/ 为抽样训练数组，不是完整数据集；replays/ 为有限数量的完整对局回放。",
                   "evaluations/ 为周期评估统计；所有评估局均计入胜率，不受回放保存数量影响。",
                   "全部训练完成并通过审计后生成 RESULTS.md、summary.json 和 figures/。",
                   "停止整组实验：对 status.json 中的 manager_pid 执行 kill -TERM <PID>。",
                   "模型支持推理与重新评估，不支持精确恢复中断时的训练状态。", ""]
    (output / "README.md").write_text("\n".join(description), encoding="utf-8")
    command = ["nohup", sys.executable, "-u", "-m", "algorithms.mappo.launch", "--manage", "--output", str(output)]
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    manifest["manager_pid"] = process.pid
    manifest["launch_command"] = command
    write_atomic(output / "experiment.json", manifest)
    print(json.dumps({"output": str(output), "manager_pid": process.pid,
                      "actual_steps_per_seed": manifest["actual_steps_per_seed"], "seeds": args.seeds}), flush=True)


if __name__ == "__main__":
    main()
