"""Run a replicated MAPPO learning study and export results and learning curves."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.common.experiment_log import write_atomic
from algorithms.mappo.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, validate_backend


def summarize(output, runs):
    if any(Path(r["path"]).resolve() == output.resolve() for r in runs):
        raise ValueError("Study output must not be a single run directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metric_titles = {"test_battle_won_mean": "Validation win rate (greedy)",
                     "test_return_mean": "Validation return (greedy)",
                     "test_damage_mean": "Validation damage (greedy)",
                     "test_timeout_mean": "Validation timeout rate (greedy)",
                     "sampled_test_battle_won_mean": "Validation win rate (sampled)",
                     "sampled_test_damage_mean": "Validation damage (sampled)"}
    for metric, title in metric_titles.items():
        all_metrics = [json.loads((Path(r["path"]) / "metrics.json").read_text())[metric] for r in runs]
        steps = np.array(all_metrics[0]["steps"])
        if any(m["steps"] != steps.tolist() for m in all_metrics):
            raise ValueError("Runs must use matching evaluation steps")
        values = np.array([m["values"] for m in all_metrics])
        means = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(steps))
        sem = std / np.sqrt(len(runs))
        with (figures / (metric + "_summary.csv")).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "mean", "std", "sem", "n"] + ["seed" + str(r["seed"]) for r in runs])
            writer.writerows([int(s), means[i], std[i], sem[i], len(runs), *values[:, i]] for i, s in enumerate(steps))
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(steps, means, color="#009E73", linewidth=1.5, label="MAPPO mean")
        lower, upper = means - sem, means + sem
        if "won" in metric or "timeout" in metric:
            lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        axis.fill_between(steps, lower, upper, color="#009E73", alpha=0.15, label="SEM across training seeds")
        axis.set(xlabel="Environment steps", ylabel=title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(figures / (metric + "." + suffix), dpi=300)
        plt.close(fig)

    fields = ("win_rate", "mean_return", "mean_damage", "mean_kills", "timeout_rate", "mean_useful_reports")
    aggregate = {}
    for phase in ("before", "latest", "best"):
        aggregate[phase] = {}
        for mode in ("sampled", "greedy"):
            results = [r["summary"]["heldout_before"][mode] if phase == "before" else r["summary"]["heldout"][phase][mode] for r in runs]
            aggregate[phase][mode] = {field: {"mean": float(np.mean([r[field] for r in results])),
                "std": float(np.std([r[field] for r in results], ddof=1)) if len(results) > 1 else 0.0} for field in fields}
    summary = {"aggregate_heldout": aggregate, "runs": [{k: r[k] for k in ("seed", "path", "registered_path")} for r in runs]}
    write_atomic(output / "summary.json", summary)
    lines = ["# MAPPO 多种子训练报告", "", "环境、外部算法来源与评估设置见各次运行的 config.json 和源码快照。",
             "实现复用外部 onpolicy/MAPPO，采用共享前馈 actor 和集中式 critic。",
             "最优模型按贪心验证胜率、回报依次选择，独立测试种子不参与模型选择。",
             "各训练种子使用相同的评估场景；以下独立测试统计与验证曲线分开报告。",
             "", "## 独立测试结果", "", "| 阶段 | 策略 | 胜率 | 回报 | 伤害 | 击杀 | 超时率 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for phase, modes in aggregate.items():
        for mode, result in modes.items():
            lines.append(f"| {phase} | {mode} | {result['win_rate']['mean']:.3f} | {result['mean_return']['mean']:.3f} | "
                         f"{result['mean_damage']['mean']:.2f} | {result['mean_kills']['mean']:.2f} | {result['timeout_rate']['mean']:.3f} |")
    lines += ["", "## 各种子结果", "", "| 种子 | 步数 | 局数 | 最优模型步数 | 最终模型贪心胜率 | 最优模型贪心胜率 | 记录位置 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for run in runs:
        s = run["summary"]
        lines.append(f"| {run['seed']} | {s['environment_steps']} | {s['training_episodes']} | {s['best_validation_step']} | "
                     f"{s['heldout']['latest']['greedy']['win_rate']:.3f} | {s['heldout']['best']['greedy']['win_rate']:.3f} | {run['registered_path']} |")
    lines += ["", "曲线和 CSV 位于 figures/。实线为训练种子均值，阴影为样本标准差除以种子数平方根得到的 SEM。",
              "只有一个训练种子时不估计不确定性，阴影宽度置零；这不表示总体方差为零。",
              "本报告用于学习诊断，不单独构成收敛证明或受控奖励消融证据。",
              "回报或伤害增加但没有胜利，只能说明部分行为变化。种子数量有限，评估地图在训练种子之间共享。",
              "不同实验预算和测试种子不一致时，不能将结果差异全部归因于奖励设置。",
              "完整回放：各运行 replays/heldout_{latest,best}_{sampled,greedy}/episode_NNN.json。",
              "训练数组：trajectories/rollout_NNNNN.npz，不是浏览器回放文件。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    repo = PROJECT_ROOT
    output = args.output.resolve()
    if args.summarize_only:
        runs = []
        for seed in args.seeds:
            path = output / f"seed{seed}"
            config = json.loads((path / "config.json").read_text())
            runs.append({"seed": seed, "path": str(path), "registered_path": config["registered_path"],
                         "summary": json.loads((path / "summary.json").read_text())})
        summarize(output, runs)
        return
    args.mappo_root = validate_backend(parser, args.mappo_root)
    if args.steps <= 0 or args.steps % 16384 or args.workers <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive steps must be a multiple of 16384; use distinct seeds and positive workers")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    write_atomic(output / "config.json", {"steps": args.steps, "seeds": args.seeds, "started": datetime.now().isoformat(),
        "validation_episodes": 8, "heldout_episodes": 32, "eval_interval": 16384, "algorithm": "mappo"})
    def train(seed):
        path = output / f"seed{seed}"
        command = [sys.executable, "-m", "algorithms.mappo.train", "--steps", str(args.steps), "--seed", str(seed),
                   "--mappo-root", str(args.mappo_root), "--config", str(args.config.resolve()),
                   "--eval-interval", "16384", "--eval-episodes", "8", "--heldout-episodes", "32",
                   "--results-env-name", args.config.stem + "_study", "--output", str(path)]
        if args.results_root:
            command += ["--results-root", str(args.results_root.resolve())]
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (output / "logs" / f"seed{seed}.log").open("w") as logfile:
            subprocess.run(command, cwd=repo, env=environment, stdout=logfile, stderr=subprocess.STDOUT, check=True)
        config = json.loads((path / "config.json").read_text())
        return {"seed": seed, "path": str(path), "registered_path": config["registered_path"],
                "summary": json.loads((path / "summary.json").read_text())}
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train, seed): seed for seed in args.seeds}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({"completed_seed": result["seed"], "completed_runs": len(results)}), flush=True)
            except Exception as error:
                failures.append({"seed": futures[future], "error": str(error)})
    if failures:
        write_atomic(output / "failures.json", failures)
        raise RuntimeError("Training failed; see logs and failures.json")
    results.sort(key=lambda r: r["seed"])
    summarize(output, results)
    # Keep the numbered-record directory convenient for existing results consumers.
    if args.results_root:
        group = Path(results[0]["registered_path"]).parent
        for name in ("figures", "RESULTS.md", "summary.json"):
            target = group / name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(output / name, target_is_directory=name == "figures")
    print(json.dumps({"output": str(output), "completed_runs": len(results)}), flush=True)


if __name__ == "__main__":
    main()
