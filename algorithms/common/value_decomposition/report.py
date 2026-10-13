"""Measured validation curves and independent-test tables for the sensitivity study.

Figure contract: test how map, roster and communication affect learning, without
assuming a direction. Three quantitative comparisons share a fixed-budget x axis.
Show only common measured checkpoints across all three independent training seeds;
never interpolate/extrapolate unfinished runs. Exports: editable SVG/PDF and PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import GROUPS, LABELS


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    output = Path(output)
    experiment = read_json(output / "experiment.json")
    if experiment is None:
        raise ValueError("Missing experiment.json")
    seeds = experiment["seeds"]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    source_rows, curves, run_rows, complete = [], {}, [], []
    for task in LABELS:
        by_seed = []
        for seed in seeds:
            run = output / task / f"seed_{seed}"
            points = {}
            for path in sorted((run / "evaluations").glob("validation_*/summary.json")):
                result = read_json(path)
                if result:
                    step = int(path.parent.name.split("_")[-1])
                    points[step] = result["win_rate"]
                    source_rows.append({"scenario": task, "seed": seed, "step": step,
                                        "win_rate": result["win_rate"], "episodes": result["episodes"]})
            by_seed.append(points)
            progress = read_json(run / "progress.json") or {}
            summary = read_json(run / "summary.json")
            run_rows.append({"scenario": task, "seed": seed, "status": progress.get("status", "PENDING"),
                "environment_steps": progress.get("environment_steps", 0),
                "updates": progress.get("updates", 0),
                "environment_steps_per_second": progress.get("environment_steps_per_second", 0),
                "training_seconds": progress.get("training_seconds", 0),
                "elapsed_seconds": progress.get("elapsed_seconds", 0)})
            if summary:
                primary = summary["heldout"]["final"]["greedy"]
                complete.append({"scenario": task, "seed": seed,
                    **{key: primary[key] for key in ("win_rate", "timeout_rate", "contact_rate", "mean_first_contact_step",
                       "mean_kill_fraction", "mean_useful_reports", "mean_damage", "mean_return")},
                    "sampled_win_rate": summary["heldout"]["final"]["sampled"]["win_rate"],
                    "best_greedy_win_rate": summary["heldout"]["best"]["greedy"]["win_rate"],
                    "best_sampled_win_rate": summary["heldout"]["best"]["sampled"]["win_rate"],
                    **{key: summary[key] for key in ("validation_auc", "validation_auc_normalized", "first_80_percent_step",
                      "training_seconds", "wall_seconds", "collection_seconds", "learner_seconds", "evaluation_seconds",
                      "environment_steps_per_second", "end_to_end_steps_per_second", "best_validation_step")}})
        # Common measured steps only: unequal learning speeds never change n.
        common = sorted(set.intersection(*(set(p) for p in by_seed)))
        rows = []
        for step in common:
            values = np.asarray([p[step] for p in by_seed])
            rows.append({"scenario": task, "step": step, "mean": float(values.mean()),
                         "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                         "n": len(values), **{"seed_" + str(s): float(v) for s, v in zip(seeds, values)}})
        curves[task] = rows
    write_csv(figures / "validation_source.csv", source_rows, ("scenario", "seed", "step", "win_rate", "episodes"))
    write_csv(output / "run_status.csv", run_rows, tuple(run_rows[0]))
    curve_rows = [row for task in curves.values() for row in task]
    write_csv(figures / "validation_mean_sem.csv", curve_rows,
              ("scenario", "step", "mean", "sem", "n", *("seed_" + str(s) for s in seeds)))
    if complete:
        write_csv(output / "independent_test_per_seed.csv", complete, tuple(complete[0]))
    aggregates = []
    for task in LABELS:
        rows = [r for r in complete if r["scenario"] == task]
        if len(rows) != len(seeds):
            continue
        item = {"scenario": task, "n": len(rows)}
        for key in rows[0]:
            if key in ("scenario", "seed", "first_80_percent_step"):
                continue
            vals = [r[key] for r in rows if r[key] is not None]
            item[key + "_mean"] = float(np.mean(vals)) if vals else None
            item[key + "_sem"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
            if len(vals) != len(rows):
                item[key + "_observed_seeds"] = len(vals)
        # Do not replace never-reached values with the budget or average them away.
        item["first_80_percent_step_by_seed"] = {str(r["seed"]): r["first_80_percent_step"] for r in rows}
        aggregates.append(item)
    write_atomic(output / "aggregate.json", {"completed_runs": len(complete), "planned_runs": len(run_rows),
                 "seeds": seeds, "complete_scenarios": aggregates})
    lines = ["# " + experiment["algorithm"].upper() + " 灵敏度实验进度与结果", "",
             f"已完成 {len(complete)}/{len(run_rows)} 次独立训练。主报告使用固定预算结束模型。", "",
             "配置和任务矩阵见 experiment.json、configs/；各任务按 seed 保存原始记录。",
             "曲线仅包含所有训练种子都完成的真实验证点，不平滑、不插值、不外推。",
             f"每次验证 {experiment['eval_episodes']} 局；实线为 {len(seeds)} 个训练种子的均值，阴影为样本标准差/√n。",
             f"最终独立测试每种模型/策略 {experiment['heldout_episodes']} 局；测试种子不参与选模型。",
             "补充采样策略：" + experiment.get("sampled_policy", "epsilon-greedy epsilon=0.05") +
             "。接敌时间均值以实际接敌局为条件。",
             "SEM 不是地图置信区间；不根据阴影重叠作显著性判断。", "",
             "| 场景 | 固定预算贪心胜率 ±SEM | 超时率 | 击杀比例 | 归一化 AUC | 首达80%（按seed） |",
             "|---|---:|---:|---:|---:|---|"]
    for r in aggregates:
        reached = ", ".join(s + ":" + (str(v) if v is not None else "未达到")
                            for s, v in r["first_80_percent_step_by_seed"].items())
        sem_text = f"{r['win_rate_sem']:.4f}" if r['win_rate_sem'] is not None else "NA (n=1)"
        lines.append(f"| {r['scenario']} | {r['win_rate_mean']:.4f} ± {sem_text} | "
                     f"{r['timeout_rate_mean']:.3f} | {r['mean_kill_fraction_mean']:.3f} | "
                     f"{r['validation_auc_normalized_mean']:.4f} | {reached} |")
    lines += ["", "完整指标：independent_test_per_seed.csv、aggregate.json。运行状态：run_status.csv。",
              "训练吞吐量分母为采集+学习时间；端到端吞吐量还包括验证、测试和保存。",
              "不同人数的绝对回报/伤害不直接排名；首次80%只在离散验证点判断，不代表持续达到。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    plot_curves(figures, curves, experiment)
    return {"completed": len(complete), "planned": len(run_rows)}


def plot_curves(figures, curves, experiment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False})
    colors = ["#444444", "#4477AA", "#66AACC", "#AA3377", "#CCBB44", "#228833", "#EE8866"]
    titles = {"map_size": "Map size", "agent_count": "Team size", "communication": "Communication"}
    for group, tasks in GROUPS.items():
        fig, ax = plt.subplots(figsize=(7.2047244, 3.9370079))
        for i, task in enumerate(tasks):
            rows = curves[task]
            if not rows:
                continue
            x = np.asarray([r["step"] for r in rows]) / 1e6
            mean = np.asarray([r["mean"] for r in rows])
            sem = np.asarray([r["sem"] or 0 for r in rows])
            label = LABELS[task] if task != "B0" else {
                "map_size": "24x24x6", "agent_count": "6v6", "communication": "Radius 10"}[group]
            ax.plot(x, mean, color=colors[i], marker="o", markersize=2, linewidth=1.2,
                    linestyle=["-", "--", "-.", ":", "-", "--", "-."][i],
                    label=task + " " + label)
            if len(experiment["seeds"]) > 1:
                ax.fill_between(x, np.clip(mean - sem, 0, 1), np.clip(mean + sem, 0, 1), color=colors[i], alpha=0.12)
        ax.set(xlabel="Team environment steps (millions)", ylabel="Validation win rate (greedy)",
               xlim=(0, experiment["steps"] / 1e6), ylim=(0, 1.02),
               title=experiment["algorithm"].upper() + " | " + titles[group])
        ax.grid(alpha=0.15)
        if ax.lines:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7)
        else:
            ax.text(0.5, 0.5, "Waiting for all training seeds", transform=ax.transAxes, ha="center")
        fig.tight_layout()
        for suffix in (".png", ".svg", ".pdf", ".tiff"):
            target = figures / (group + suffix)
            temporary = figures / (group + ".tmp" + suffix)
            options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == ".tiff" else {}
            fig.savefig(temporary, dpi=600, bbox_inches="tight", **options)
            temporary.replace(target)
        plt.close(fig)
    (figures / "QA.md").write_text(
        "# Figure contract and QA\n\nQuantitative comparison of map, team size and communication effects; "
        "no directional claim before results. Fixed 183 mm canvas, editable SVG/PDF text, PNG preview.\n"
        "All raw validation observations are in validation_source.csv. Curves use the intersection of measured "
        "checkpoints across all requested training seeds; missing/unstarted seeds are never treated as zero.\n"
        "Shading = sample SD/sqrt(n) across training seeds; clipped to [0,1] for display. "
        "No smoothing, statistical tests, or inference from SEM overlap. Shared fixed validation maps. "
        "Independent test seeds do not enter model selection. No simulated observations.\n"
        "Current plots are progress figures; final figures require visual inspection after all runs finish.\n",
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(summarize(parser.parse_args().output))
