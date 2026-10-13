"""Run the replicated map-size, team-size and communication sensitivity study."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.progress import align_steps
from algorithms.common.study_queue import jobs_for, run_queue
from algorithms.mappo.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, validate_backend
from heterogrid import EnvConfig
from heterogrid.config import ROLES


def team_roster(base, count):
    if count < 3 or count % 3:
        raise ValueError("Team size must be a positive multiple of three")
    roster = []
    for role in ROLES:
        units = [unit for unit in base if unit.role == role]
        if not units:
            raise ValueError("The baseline must include every role")
        roster.extend(units[index % len(units)] for index in range(count // 3))
    return roster


def conditions(base):
    if base.opponent != "rule" or len(base.red) != 6 or len(base.blue) != 6:
        raise ValueError("The baseline must use a fixed rule opponent and two six-unit teams")
    if base.communication != "distance" or any(sum(unit.role == role for unit in team) != 2
                                               for team in (base.red, base.blue) for role in ROLES):
        raise ValueError("The baseline must use distance communication and equal role proportions")
    result = {"B0_baseline": base}
    for label, size in (("M1_small", (12, 12, 3)), ("M2_large", (36, 36, 9)),
                        ("M3_xlarge", (48, 48, 12))):
        result[label] = replace(base, map_size=size)
    for label, count in (("N1_3v3", 3), ("N2_9v9", 9), ("N3_12v12", 12)):
        result[label] = replace(base, red=team_roster(base.red, count), blue=team_roster(base.blue, count))
    for label, changes in (
        ("C1_none", {"communication": "none"}),
        ("C2_full", {"communication": "full"}),
        ("C3_distance5", {"comm_radius": 5.0}),
        ("C4_distance15", {"comm_radius": 15.0}),
        ("C5_random25", {"communication": "random", "dropout_probability": 0.25}),
        ("C6_random50", {"communication": "random", "dropout_probability": 0.50}),
    ):
        result[label] = replace(base, **changes)
    return result



def training_command(manifest, output, job):
    command = [manifest["python"], "-u", "-m", "algorithms.mappo.train",
               "--mappo-root", manifest["mappo_root"], "--config",
               str(output / "configs" / (job["condition"] + ".json")),
               "--output", job["path"], "--seed", str(job["seed"]), "--device", manifest["device"],
               "--steps", str(manifest["actual_steps_per_seed"]),
               "--eval-seed", "100042", "--heldout-seed", "200042",
               "--no-trajectories", "--no-evaluation-files"]
    for option in ("eval_interval", "save_interval", "log_interval", "eval_episodes",
                   "heldout_episodes", "replay_episodes"):
        command += ["--" + option.replace("_", "-"), str(manifest[option])]
    return command


def manage(output):
    from algorithms.mappo.audit import audit
    from algorithms.mappo.study import summarize
    return run_queue(output, training_command, audit, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    parser.add_argument("--workers", type=int, default=min(12, available_cpus))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--save-interval", type=int, default=2000000)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--replay-episodes", type=int, default=2)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=1.0)
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix" or shutil.which("nohup") is None:
        parser.error("This detached launcher requires POSIX and nohup")
    if args.manage:
        if not args.output:
            parser.error("--manage requires --output")
        manage(args.output.resolve())
        return
    if min(args.steps, args.workers, args.eval_interval, args.save_interval, args.log_interval,
           args.eval_episodes, args.heldout_episodes) <= 0 or args.replay_episodes < 0:
        parser.error("Budgets, workers, intervals and evaluation counts must be positive")
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct, nonnegative seeds")
    if args.eval_episodes > 100000 or args.minimum_free_disk_gb <= 0:
        parser.error("Validation seeds must not overlap held-out seeds; disk threshold must be positive")
    backend = validate_backend(parser, args.mappo_root)
    variants = conditions(EnvConfig.load(args.config))
    label = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sensitivity_5m"
    output = (args.output or PROJECT_ROOT / "output/mappo" / label).resolve()
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < args.minimum_free_disk_gb * 1024 ** 3:
        parser.error("Insufficient free disk space")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    manifest = {"schema_version": 1, "algorithm": "mappo", "created": timestamp(),
                "repository": str(PROJECT_ROOT), "python": sys.executable, "mappo_root": str(backend),
                "requested_steps_per_seed": args.steps, "actual_steps_per_seed": align_steps(args.steps),
                "seeds": args.seeds, "conditions": list(variants), "jobs": len(variants) * len(args.seeds),
                "team_sizes": {name: len(config.red) for name, config in variants.items()},
                "scheduling": "largest teams first; stable seed-interleaved order within each team size",
                "workers": min(args.workers, len(variants) * len(args.seeds)), "device": args.device,
                "num_envs": 4, "rollout_length": 128, "hidden_size": 64, "ppo_epochs": 5, "mini_batches": 4,
                "initialization": "independent from scratch; no reuse of preliminary experiments",
                "primary_result": "latest fixed-budget checkpoint; best validation checkpoint is supplementary",
                "communication_scope": "both teams; random is fully connected online nodes with node dropout, not distance+dropout",
                "validation_seed": 100042, "heldout_seed": 200042,
                "eval_episodes": args.eval_episodes, "heldout_episodes": args.heldout_episodes,
                "replay_episodes": args.replay_episodes, "no_trajectories": True, "no_evaluation_files": True,
                "minimum_free_disk_bytes": int(args.minimum_free_disk_gb * 1024 ** 3)}
    for name in ("eval_interval", "save_interval", "log_interval"):
        manifest[name] = align_steps(getattr(args, name))
    for name, config in variants.items():
        write_atomic(output / "configs" / (name + ".json"), config.to_dict())
    command = ["nohup", sys.executable, "-u", "-m", "algorithms.mappo.matrix", "--manage", "--output", str(output)]
    manifest["launch_command"] = command
    write_atomic(output / "experiment.json", manifest)
    write_atomic(output / "status.json", {"state": "STARTING", "runs": {}})
    description = ["# MAPPO 敏感性实验", "", "13 个环境配置，每个配置使用三个独立训练种子。",
                   "B0：基准；M1–M3：地图大小；N1–N3：队伍规模；C1–C6：通信设置。",
                   "完整配置见 configs/，所有配置使用相同奖励、网络和 PPO 超参数。",
                   "random 通信是节点随机离线，不是距离约束叠加丢包；通信设置同时作用于红蓝双方。",
                   "每次训练从头开始，按固定步数结束，不按胜率提前停止。",
                   "experiment.json：冻结的实验设置；status.json：队列、进度、进程和状态；nohup.log：管理日志。",
                   "每个配置下 seed_N/ 保存模型、逐次 PPO 指标、逐局统计、源码快照及少量回放。",
                   "logs/seed_N.log 保存对应任务的启动、训练和错误日志。",
                   "不保存 trajectories/ 和 evaluations/；周期评估指标仍保留在 metrics.json 和 info.json。",
                   "验证集每次 64 局；最终独立测试集每种决策模式 128 局。具体数量以 experiment.json 为准。",
                   "主要结果使用最终模型，验证集最优模型仅作补充；独立测试集不参与模型选择。",
                   "全部完成后自动审计，按配置生成三种子均值 ±SEM 曲线和 RESULTS.md。",
                   "停止整组实验：向 status.json 中的 manager_pid 发送 SIGTERM，管理进程会终止所有工作进程。",
                   "失败任务不会静默重试；当前不支持精确断点续训。", ""]
    (output / "README.md").write_text("\n".join(description), encoding="utf-8")
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "manager_pid": process.pid, "jobs": manifest["jobs"],
                      "workers": manifest["workers"], "actual_steps_per_seed": manifest["actual_steps_per_seed"]}), flush=True)


if __name__ == "__main__":
    main()
