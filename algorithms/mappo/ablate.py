"""Compare sharing, terminal costs and information rewards with matched training budgets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig
from algorithms.mappo.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, validate_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=100042)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 512 or args.steps % 512 or min(args.workers, args.eval_episodes) < 1:
        parser.error("steps must be a positive multiple of 512; workers and episodes must be positive")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("training seeds must be distinct and nonnegative")
    args.mappo_root = validate_backend(parser, args.mappo_root)
    repo = PROJECT_ROOT
    variants = {name: repo / "configs/ablations" / f"6v6_{name}.json" for name in ("shared", "terminal")}
    variants["shaped"] = DEFAULT_CONFIG
    configs = {name: EnvConfig.load(path).to_dict() for name, path in variants.items()}
    control_fields = set(configs["shared"]) - {
        "timeout_as_loss", "time_penalty", "discovery_reward", "sharing_reward", "information_reward_cap"}
    for config in configs.values():
        if any(config[k] != configs["shared"][k] for k in control_fields):
            parser.error("ablation variants must have identical task, opponent and communication settings")
    output = (args.output or repo / "output/mappo" / ("ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    metadata = {"algorithm": "mappo", "steps_per_run": args.steps, "training_seeds": args.seeds,
                "evaluation_seed": args.eval_seed, "evaluation_episodes": args.eval_episodes,
                "variants": configs, "mappo_root": str(args.mappo_root.resolve())}
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def train(name, seed):
        label = f"{name}_seed{seed}"
        run = output / label
        command = [sys.executable, "-m", "algorithms.mappo.train", "--config", str(variants[name]),
                   "--mappo-root", str(args.mappo_root.resolve()), "--steps", str(args.steps),
                   "--seed", str(seed), "--eval-seed", str(args.eval_seed),
                   "--eval-episodes", str(args.eval_episodes), "--output", str(run)]
        if args.results_root:
            command += ["--results-root", str(args.results_root.resolve())]
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (output / "logs" / f"{label}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        return {"variant": name, "seed": seed, "path": str(run), "summary": result}

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train, name, seed): (name, seed) for name in variants for seed in args.seeds}
        for future in as_completed(futures):
            name, seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                after = result["summary"]["after"]
                print(json.dumps({"completed": len(results), "variant": name, "seed": seed,
                                  "win_rate": after["win_rate"], "timeout_rate": after["timeout_rate"],
                                  "mean_damage": after["mean_damage"]}), flush=True)
            except Exception as error:
                failures.append({"variant": name, "seed": seed, "error": str(error)})
                print(json.dumps({"failed": failures[-1]}), flush=True)
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} runs failed; see {output / 'logs'}")
    fields = ("win_rate", "timeout_rate", "mean_return", "mean_damage", "mean_kills",
              "contact_rate", "mean_first_contact_step", "mean_discovered_targets", "mean_useful_reports")
    aggregate = {}
    for name in variants:
        group = [r for r in results if r["variant"] == name]
        aggregate[name] = {}
        for mode in ("before", "after", "greedy"):
            aggregate[name][mode] = {}
            for field in fields:
                values = [r["summary"][mode][field] for r in group if r["summary"][mode][field] is not None]
                aggregate[name][mode][field] = {"mean": float(np.mean(values)) if values else None,
                                               "std": float(np.std(values)) if values else None}
    summary = {"settings": metadata, "aggregate": aggregate,
               "runs": [{"variant": r["variant"], "seed": r["seed"], "path": r["path"]} for r in results]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# MAPPO 奖励消融", "", f"训练种子：{args.seeds}；每次运行步数：{args.steps}。",
             f"每次运行使用评估种子 {args.eval_seed}-{args.eval_seed + args.eval_episodes - 1}。", "",
             "| 设置 | 采样胜率 | 超时率 | 伤害 | 有效报告 | 塑形回报 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in variants:
        row = aggregate[name]["after"]
        lines.append(f"| {name} | {row['win_rate']['mean']:.3f} | {row['timeout_rate']['mean']:.3f} | "
                     f"{row['mean_damage']['mean']:.2f} | {row['mean_useful_reports']['mean']:.2f} | {row['mean_return']['mean']:.3f} |")
    lines += ["", "各组使用相同的 24x24x6 地图、6v6 阵容、规则对手和距离通信。",
              "shared：对称自动目标共享与基础奖励。",
              "terminal：在 shared 上增加超时失败和每步 0.005 时间成本。",
              "shaped：在 terminal 上增加有上限、按整局去重的发现与有效报告奖励。",
              "", "奖励函数不同，请使用胜率、伤害与超时率比较行为，不直接比较塑形回报。",
              "短期运行用于检查接入和早期学习，不代表已经收敛或优于其他算法。",
              "各运行目录保存模型、训练数组、指标和完整评估回放。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "completed_runs": len(results)}), flush=True)


if __name__ == "__main__":
    main()
