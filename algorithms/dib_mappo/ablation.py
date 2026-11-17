"""Isolated 2x2 active-mask/warmup screening; never mutate the formal campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, scenarios, communication_protocol
from algorithms.common.value_decomposition.campaign import now, supervise
from algorithms.dib_mappo.backend import TRAINING_PROFILES, settings
ABLATION_PROFILES = ("legacy", "active_mask", "warmup1m", "active_mask_1m")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def prepare(args):
    root = args.output_root / "dib_mappo" / args.name
    coordinator = args.output_root / "campaigns" / (args.name + "_dib_mappo_ablation")
    if root.exists() or coordinator.exists():
        raise FileExistsError("Choose a fresh name; do not overwrite any training result")
    if shutil.disk_usage(args.output_root.parent).free < 4 * 1024**3:
        raise OSError("Need at least 4 GiB disk headroom before preparing a screening campaign")
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    for name in ("heterogrid", "algorithms", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    backend = coordinator / "_backend"
    for name in ("learners", "controllers", "modules", "components", "utils", "config"):
        shutil.copytree(args.dib_root / name, backend / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    tasks = scenarios(fixed_blue=True)
    # Match the existing DIB experiment's behavior-neutral diagnostics.
    from dataclasses import replace
    for task in args.scenarios:
        write_atomic(root / "configs" / (task + ".json"), replace(tasks[task], diagnostics_enabled=True).to_dict())
    source_hashes = {str(path.relative_to(coordinator)): hashlib.sha256(path.read_bytes()).hexdigest()
        for base in (runtime, backend) for path in sorted(base.rglob("*.py"))}
    common = {"algorithm": "dib_mappo", "created": now(), "seeds": args.seeds,
        "steps": args.steps, "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
        "heldout_episodes": args.heldout_episodes, "eval_seed": 300042, "heldout_seed": 400042,
        "groups": {"communication": args.scenarios}, "labels": {x:x for x in args.scenarios},
        "communication_protocol": communication_protocol(True), "num_envs": 4,
        "screening_only": True, "not_a_5m_formal_result": args.steps != 5000000,
        "selection_rule": "diagnostic validation only; independent test seeds never select configurations",
        "primary_model": "final.pt", "fresh_training": True,
        "runtime": str(runtime), "backend": str(backend), "coordinator": str(coordinator)}
    jobs, roots = [], {}
    for profile in ABLATION_PROFILES:
        output = root / profile
        output.mkdir()
        roots[profile] = str(output)
        write_atomic(output / "experiment.json", {**common, "training_profile": profile,
                     "hyperparameters": settings(backend, profile)})
    # Start all four treatments on C1 together, then C3. Same seeds and exact
    # team-step budget, not the same number of updates or wall-clock duration.
    for task in args.scenarios:
        for seed in args.seeds:
            for profile in ABLATION_PROFILES:
                output = root / profile / task / ("seed_" + str(seed))
                command = [sys.executable, "-m", "algorithms.dib_mappo.train", "--dib-root", str(backend),
                    "--training-profile", profile, "--config", str(root / "configs" / (task + ".json")),
                    "--output", str(output), "--seed", str(seed), "--steps", str(args.steps),
                    "--eval-interval", str(args.eval_interval), "--eval-episodes", str(args.eval_episodes),
                    "--heldout-episodes", str(args.heldout_episodes), "--eval-seed", "300042",
                    "--heldout-seed", "400042", "--replay-episodes", "0"]
                jobs.append({"algorithm": profile, "scenario": task, "seed": seed, "output": str(output),
                    "runtime": str(runtime), "command": command, "status": "PENDING", "pid": None})
    plan = {"created": now(), "workers": args.workers, "runtime": str(runtime), "roots": roots,
        "jobs": jobs, "manager_pid": None, "status": "PREPARED", "report_options_list": [],
        "requested_total_steps": args.steps * len(jobs), "screening_root": str(root),
        "python": sys.executable, "source_sha256": source_hashes}
    write_atomic(coordinator / "plan.json", plan)
    write_atomic(coordinator / "status.json", plan)
    write_atomic(root / "ablation.json", {**common, "profiles": list(ABLATION_PROFILES),
                 "planned_runs": len(jobs), "source_sha256": source_hashes})
    (root / "README.md").write_text(
        "# DIB存活掩码×预热长度：独立筛查\n\n"
        f"场景{args.scenarios}；种子{args.seeds}；每次从头训练{args.steps:,}团队步，共{len(jobs)}次。\n"
        "不是原48次正式实验的一部分，不覆盖其模型、曲线或测试结果。\n"
        "legacy=原掩码/2M预热；active_mask=存活掩码/2M预热；warmup1m=原掩码/1M预热；"
        "active_mask_1m=存活掩码/1M预热。所有处理保留2M渐入、其他网络/损失/PPO/奖励/环境不变。\n"
        "MHF已按原稿eq:dynamics-filter核对，不加入上一时刻后验递推。\n"
        "actor与辅助学习用动作前存活掩码，团队critic保留回合mask；死亡发生当步仍学习。\n"
        f"每{args.eval_interval:,}步验证{args.eval_episodes}局；验证种子300042起，独立测试400042起。\n"
        "所有比较使用共同验证点；单个训练seed不报告SEM，不等于证明普适优势。\n"
        "2.5M只用于预算内筛查，尚未观察新配置3M门控全开后的效果；不能替代5M三seed正式比较。\n"
        "不得按独立测试胜率选配置或修改当前正式队列。\n\n"
        "汇总命令：\n\n```bash\n"
        f"{sys.executable} -m algorithms.dib_mappo.ablation --report {coordinator}\n```\n",
        encoding="utf-8")
    report(coordinator)
    print(str(coordinator), flush=True)
    return coordinator


def report(coordinator):
    coordinator = Path(coordinator).resolve()
    state = read(coordinator / "status.json")
    if state is None:
        raise ValueError("No screening status")
    root = Path(state["screening_root"])
    metadata = read(root / "ablation.json")
    rows, comparison = [], []
    for task in metadata["groups"]["communication"]:
        validations = {}
        for job in (j for j in state["jobs"] if j["scenario"] == task):
            output = Path(job["output"])
            progress = read(output / "progress.json") or {}
            summary = read(output / "summary.json")
            points = {}
            for path in (output / "evaluations").glob("validation_*/summary.json"):
                value = read(path)
                if value:
                    points[int(path.parent.name.split("_")[-1])] = value["win_rate"]
            validations[(job["algorithm"], job["seed"])] = points
            rows.append({"scenario": task, "profile": job["algorithm"], "seed": job["seed"],
                "status": job["status"], "environment_steps": progress.get("environment_steps", 0),
                "updates": progress.get("updates", 0),
                "final_greedy_win_rate": summary["heldout"]["final"]["greedy"]["win_rate"] if summary else None})
        common = sorted(set.intersection(*(set(p) for p in validations.values())))
        if common:
            step = common[-1]
            for (profile, seed), points in validations.items():
                comparison.append({"scenario": task, "profile": profile, "seed": seed,
                                   "step": step, "win_rate": points[step]})
    result = {"updated": now(), "screening_only": True, "jobs": rows,
              "latest_common_validation": comparison, "metadata": str(root / "ablation.json")}
    write_atomic(root / "ablation_summary.json", result)
    with (root / "aligned_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", "profile", "seed", "step", "win_rate"])
        writer.writeheader(); writer.writerows(comparison)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=Path)
    group.add_argument("--report", type=Path)
    parser.add_argument("--dib-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261021_active_mask_warmup_ablation_2p5m")
    parser.add_argument("--steps", type=int, default=2500000)
    parser.add_argument("--scenarios", nargs="+", choices=("C1", "C3"), default=["C1", "C3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.report:
        print(json.dumps(report(args.report), ensure_ascii=False, indent=2))
        return 0
    if args.dib_root is None:
        parser.error("Set --dib-root; the external backend will be frozen separately")
    if min(args.steps, args.workers, args.eval_interval, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("Positive budgets and worker count required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Repeated seeds/scenarios would duplicate jobs")
    args.output_root, args.dib_root = args.output_root.resolve(), args.dib_root.resolve()
    if (args.dib_root / "src").is_dir():
        args.dib_root = args.dib_root / "src"
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
