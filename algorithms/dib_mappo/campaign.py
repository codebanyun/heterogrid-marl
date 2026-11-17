"""Freeze and supervise 48 fresh DIB-MAPPO runs; D0 explicitly aliases its own B0."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from algorithms.common.density_campaign import LABELS as D_LABELS, TASK_COUNTS, REPORT_OPTIONS
from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, GROUPS, LABELS, scenarios, communication_protocol
from algorithms.common.value_decomposition.campaign import now, supervise
from algorithms.common.value_decomposition.report import summarize
from heterogrid.scenarios import scaling_scenarios


def matrix():
    tasks = {name: replace(config, diagnostics_enabled=True) for name, config in scenarios(fixed_blue=True).items()}
    baseline = replace(tasks["B0"], reward_normalization="reference")
    density = scaling_scenarios(baseline, mode="constant_density")
    d_tasks = {name: density[f"{count}v{count}"] for name, count in TASK_COUNTS.items()}
    return tasks, d_tasks


def prepare(args):
    from algorithms.dib_mappo.backend import settings

    root = args.output_root / "dib_mappo" / args.name
    coordinator = args.output_root / "campaigns" / (args.name + "_dib_mappo")
    if root.exists() or coordinator.exists():
        raise FileExistsError("Use a new experiment name")
    hyperparameters = settings(args.dib_root, args.training_profile)
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    runtime.mkdir()
    for name in ("heterogrid", "algorithms", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    backend = coordinator / "_backend"
    for name in ("learners", "controllers", "modules", "components", "utils", "config"):
        shutil.copytree(args.dib_root / name, backend / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "LICENSE", "NOTICE"):
        source = args.dib_root.parent / name
        if source.exists():
            shutil.copy2(source, backend / name)
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    tasks, density = matrix()
    for name, config in {**tasks, **density}.items():
        write_atomic(root / "configs" / (name + ".json"), config.to_dict())
    (root / "D0").symlink_to("B0", target_is_directory=True)
    common = {"algorithm": "dib_mappo", "variant": "robust_comm_v1" if args.training_profile == "robust_comm_v1" else "dib_mappo_v2", "created": now(),
        "training_profile": args.training_profile,
        "seeds": [42, 43, 44], "steps": 5_000_000, "eval_interval": 250_000, "eval_episodes": 64,
        "heldout_episodes": 128, "eval_seed": 100042, "heldout_seed": 200042, "save_interval": 2_000_000,
        "primary_model": "final.pt", "selection": "greedy validation win rate then return",
        "sampled_policy": "categorical masked DIB hypothesis mixture", "num_envs": 4,
        "hyperparameters": hyperparameters, "communication_protocol": communication_protocol(True),
        "coordinator": str(coordinator), "runtime_snapshot": str(runtime), "backend_snapshot": str(backend),
        "original_backend": str(args.dib_root), "diagnostics_enabled": True,
        "capacity_note": "Native hidden128, embed128, AHS128, teacher128, z64, K4; not hidden64 capacity-matched",
        "schedule_clock": "collection batch start; fixed gate until all four episodes are learned",
        "sequence_replay": "equivalent replay v3: fused GRUs, shared AHS projections, centre-only teacher, tied cdist adjoint, vector n-step; parity tested",
        "storage": "gzip training episode JSONL; checkpoint networks and gate only; no optimizer or exact resume"}
    write_atomic(root / "experiment.json", {**common, "groups": GROUPS, "labels": LABELS, "planned_runs": 39})
    write_atomic(root / "D_experiment.json", {**common, "groups": {"density_scale": list(TASK_COUNTS)},
        "group_titles": {"density_scale": "Density-controlled team size"}, "labels": D_LABELS,
        "planned_runs": 12, "validation_sources": {"D0": "B0"},
        "report_notes": ["D0复用本算法新训练的B0，不借用其他算法模型，不是额外独立训练重复。",
            "B0已开启诊断；6v6默认能力的归一化因子都是1，所以D0直接引用B0模型、验证和诊断结果。",
            "D0是B0的符号链接，其env.json记录B0的none协议；configs/D0.json记录等效reference协议。",
            "D1–D3从头训练，环境奖励归一化；与旧N组相比地图和奖励协议同时变化。"]})
    write_atomic(root / "D0_provenance.json", {"source": "B0", "type": "same-algorithm baseline alias",
        "fresh_training_runs": 48, "D0_additional_training_runs": 0,
        "equivalence": "default balanced6v6: reference_HP/initial_HP=480/480 and reference_count/count=6/6"})
    (root / "README.md").write_text(
        "# DIB-MAPPO v2：固定蓝方13场景及D组\n\n"
        "16个不同任务×3种子=48次从头5M步训练，共240M步，不继承其他算法或场景模型。\n"
        "D0是本算法B0的别名，不重复训练。原13场景与前四算法配置相同，只启用无行为影响的双方诊断；"
        "D1–D3使用reference环境奖励和近似固定密度地图。\n"
        "原生DIB v2网络hidden128/embed128/z64/K4；PPO4轮、lr3e-4、clip0.2、KL0.03，跨场景固定。\n"
        "统一4环境，4个新完整回合调用一次原生learner，无回放；关闭二次reward/return标准化、末步bootstrap。\n"
        f"训练profile={args.training_profile}；辅助起点{hyperparameters['dib_warmup_steps']:,}步，残差起点{hyperparameters['dib_actor_warmup_steps']:,}步；"
        f"残差/辅助渐入分别{hyperparameters['dib_actor_ramp_steps']:,}/{hyperparameters['dib_belief_ramp_steps']:,}步。\n"
        "批次起点调度确保采集策略与old-MAC门控一致；active_mask系列只屏蔽已阵亡actor/辅助样本，团队critic不屏蔽。\n"
        "训练原生历史窗口按256分块并反向重计算，不切断梯度；原生输出/损失/PPO与优化器时钟有对照测试。\n"
        "等价计算优化：GRU序列核、共享历史Q/K/V投影、教师中心输出、绑定距离梯度与向量化n-step。\n"
        "执行actor只读自身局部历史；其他节点观测/未来teacher仅训练使用，不绕过物理通信。\n"
        "0及每250k验证64局，2M/4M/结束保存模型；验证最优另存best，主报告final。\n"
        "final/best重载后各128局贪心及128局采样独立测试，种子与验证不交叉。\n"
        "原13场景RESULTS.md与figures/；D组D_report/；所有场景逐局双方诊断随测试JSON保存。\n"
        "training_episodes.jsonl.gz为无损压缩原始记录。模型含全部网络/残差门控/配置，不含优化器，不支持精确续训。\n"
        "本算法保留原生128维结构，与此前64维基线并非等参数量比较；应明确披露。\n", encoding="utf-8")
    jobs = []
    new_tasks = {**tasks, **{key: value for key, value in density.items() if key != "D0"}}
    for name in sorted(new_tasks, key=lambda key: (-len(new_tasks[key].red), -sum(new_tasks[key].map_size))):
        for seed in (42, 43, 44):
            output = root / name / f"seed_{seed}"
            command = [sys.executable, "-m", "algorithms.dib_mappo.train", "--dib-root", str(backend),
                "--config", str(root / "configs" / (name + ".json")), "--output", str(output), "--seed", str(seed),
                "--training-profile", args.training_profile]
            jobs.append({"algorithm": "dib_mappo", "scenario": name, "seed": seed, "output": str(output),
                         "command": command, "runtime": str(runtime), "status": "PENDING", "pid": None})
    plan = {"created": now(), "workers": args.workers, "runtime": str(runtime), "roots": {"dib_mappo": str(root)},
            "jobs": jobs, "manager_pid": None, "status": "PREPARED", "requested_total_steps": 240_000_000,
            "report_options_list": [{}, REPORT_OPTIONS], "disk_free_at_prepare": shutil.disk_usage(coordinator).free,
            "host_cpu_count": os.cpu_count(), "python": sys.executable}
    write_atomic(coordinator / "plan.json", plan); write_atomic(coordinator / "status.json", plan)
    write_atomic(root / "status.json", {"status": "PREPARED", "coordinator": str(coordinator), "counts": {"PENDING": 48}})
    summarize(root)
    summarize(root, **REPORT_OPTIONS)
    print(coordinator, flush=True)
    return coordinator


def main():
    from algorithms.dib_mappo.backend import TRAINING_PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--dib-root", type=Path, default=os.environ.get("DIB_MAPPO_ROOT"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261018_fixed_blue_sensitivity_5m")
    parser.add_argument("--workers", type=int, default=min(32, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--training-profile", choices=TRAINING_PROFILES, default="active_mask_1m")
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.dib_root is None or args.workers < 1:
        parser.error("Set --dib-root and positive workers")
    args.output_root, args.dib_root = args.output_root.resolve(), args.dib_root.resolve()
    if (args.dib_root / "src").is_dir():
        args.dib_root = args.dib_root / "src"
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
