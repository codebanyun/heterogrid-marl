"""Freeze EPyMARL + environment sources and supervise MAA2C/COMA workers."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from algorithms.common.actor_critic.backend import algorithm_settings
from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, GROUPS, LABELS, scenarios
from algorithms.common.value_decomposition.campaign import now, supervise
from algorithms.common.value_decomposition.report import summarize


def prepare(args):
    coordinator = args.output_root / "campaigns" / (args.name + "_maa2c_coma")
    if coordinator.exists():
        raise FileExistsError("Campaign already exists: " + str(coordinator))
    for algorithm in args.algorithms:
        if (args.output_root / algorithm / args.name).exists():
            raise FileExistsError("Output already exists for " + algorithm)
    # Validate before creating the immutable campaign.
    settings = {algorithm: algorithm_settings(args.epymarl_root, algorithm) for algorithm in args.algorithms}
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    runtime.mkdir()
    for name in ("heterogrid", "algorithms", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    backend = coordinator / "_backend"
    for name in ("learners", "controllers", "modules", "components", "utils", "config"):
        shutil.copytree(args.epymarl_root / name, backend / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("LICENSE", "NOTICE"):
        original = args.epymarl_root.parent / name
        if original.exists():
            shutil.copy2(original, backend / name)
    tasks, roots = scenarios(), {}
    for algorithm in args.algorithms:
        root = args.output_root / algorithm / args.name
        root.mkdir(parents=True)
        (root / "configs").mkdir()
        for name, config in tasks.items():
            write_atomic(root / "configs" / (name + ".json"), config.to_dict())
        experiment = {"algorithm": algorithm, "variant": "centralized V" if algorithm == "maa2c" else "counterfactual Q",
            "created": now(), "seeds": args.seeds, "steps": args.steps, "groups": GROUPS, "labels": LABELS,
            "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
            "heldout_episodes": args.heldout_episodes, "eval_seed": 100042, "heldout_seed": 200042,
            "save_interval": 2_000_000, "primary_model": "final.pt", "selection": "greedy validation win rate then return",
            "sampled_policy": "categorical sampling from masked actor policy", "num_envs": 4,
            "hyperparameters": settings[algorithm], "on_policy": "four fresh episodes, one learner update; no replay",
            "budget_tail": "discard final unfinished collection batch; record unused steps",
            "timeout_handling": "terminal, add_value_last_step=False", "reward_standardisation": False,
            "coordinator": str(coordinator), "runtime_snapshot": str(runtime), "backend_snapshot": str(backend),
            "original_backend": str(args.epymarl_root), "planned_runs": len(tasks) * len(args.seeds)}
        write_atomic(root / "experiment.json", experiment)
        (root / "README.md").write_text(
            f"# {algorithm.upper()} 灵敏度实验\n\n13 场景 × {len(args.seeds)} 种子，每次 {args.steps:,} 团队环境步。\n"
            "B0基准；M1–M3地图；N1–N3人数；C1–C6通信。完整展开配置见configs/。\n"
            "双方角色1:1:1、规则蓝方、200步时限、奖励和自动目标共享不变。\n"
            "使用冻结的EPyMARL原始learner、共享GRU actor、集中式critic。\n"
            "4个环境同步采集4个完整回合后更新一次，数据不回放、不跨策略版本；剩余预算不足整批则丢弃该批并记录。\n"
            "跨场景固定hidden64、4环境、优化器、学习率、n-step、目标更新和熵系数，详见experiment.json。\n"
            "每步团队奖励只取一次，不标准化；超时作为已评分终局，不bootstrap。\n"
            "主模型final.pt，best.pt只用验证胜率和回报选择。验证64局，独立测试128局。\n"
            "贪心为主；补充采样为actor的categorical分布，不是epsilon-greedy。\n"
            "<场景>/seed_<种子>/下保存模型、指标、带SHA256的源码、逐局评估、有限完整回放。\n"
            "最终汇总RESULTS.md、aggregate.json、independent_test_per_seed.csv。\n"
            "figures/为三组胜率均值±SEM曲线及源CSV，只使用三个seed共有的真实验证点，无平滑/插值。\n"
            "AUC含初始验证点，未达80%标记null，不根据SEM重叠判断显著性。\n"
            "status.json与run_status.csv记录进度；管理器接收TERM会转发给worker保存中断模型。\n"
            "从头训练，不自动覆盖输出，不支持精确续训。\n", encoding="utf-8")
        roots[algorithm] = str(root)
    jobs = []
    order = sorted(tasks, key=lambda name: (-len(tasks[name].red), -sum(tasks[name].map_size)))
    for task in order:
        for seed in args.seeds:
            for algorithm in args.algorithms:
                root = Path(roots[algorithm])
                output = root / task / f"seed_{seed}"
                command = [sys.executable, "-m", f"algorithms.{algorithm}.train", "--epymarl-root", str(backend),
                    "--config", str(root / "configs" / (task + ".json")), "--output", str(output),
                    "--seed", str(seed), "--steps", str(args.steps), "--eval-interval", str(args.eval_interval),
                    "--eval-episodes", str(args.eval_episodes), "--heldout-episodes", str(args.heldout_episodes)]
                jobs.append({"algorithm": algorithm, "scenario": task, "seed": seed, "output": str(output),
                             "command": command, "status": "PENDING", "pid": None})
    plan = {"created": now(), "workers": args.workers, "runtime": str(runtime), "roots": roots,
            "jobs": jobs, "manager_pid": None, "status": "PREPARED", "requested_total_steps": args.steps * len(jobs),
            "disk_free_at_prepare": shutil.disk_usage(coordinator).free,
            "host_cpu_count": os.cpu_count(), "python": sys.executable}
    write_atomic(coordinator / "plan.json", plan); write_atomic(coordinator / "status.json", plan)
    for root in roots.values():
        summarize(root)
    print(str(coordinator), flush=True)
    return coordinator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261012_sensitivity_5m")
    parser.add_argument("--epymarl-root", type=Path, default=os.environ.get("EPYMARL_ROOT"))
    parser.add_argument("--algorithms", nargs="+", choices=("maa2c", "coma"), default=["maa2c", "coma"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.epymarl_root is None:
        parser.error("Set --epymarl-root or EPYMARL_ROOT")
    args.output_root, args.epymarl_root = args.output_root.resolve(), args.epymarl_root.resolve()
    if (args.epymarl_root / "src").is_dir():
        args.epymarl_root = args.epymarl_root / "src"
    if min(args.workers, args.steps, args.eval_interval, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("Budgets and workers must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.algorithms)) != len(args.algorithms):
        parser.error("Use distinct seeds and algorithms")
    prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
