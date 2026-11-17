"""Append D0-D3 to completed studies, without touching the original 13 tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from algorithms.common.experiment_log import ExperimentRecorder, write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT
from algorithms.common.value_decomposition.campaign import now, supervise
from algorithms.common.value_decomposition.report import read_json, summarize
from heterogrid import EnvConfig, SMACAdapter
from heterogrid.core import GridCombat
from heterogrid.scenarios import scaling_scenarios

ALGORITHMS = ("qmix", "qtran", "maa2c", "coma")
TASK_COUNTS = {"D0": 6, "D1": 3, "D2": 9, "D3": 12}
LABELS = {"D0": "6v6, 24x24x6 (reused B0)", "D1": "3v3, 19x19x5",
          "D2": "9v9, 27x27x7", "D3": "12v12, 30x30x8"}
REPORT_OPTIONS = {"experiment_file": "D_experiment.json", "report_directory": "D_report"}


def assert_baseline_equivalent(old, new):
    allowed = {"reward_normalization", "reward_reference_team_size", "reward_reference_team_hp", "diagnostics_enabled"}
    a, b = old.to_dict(), new.to_dict()
    changed = {key for key in a if a[key] != b[key]}
    if changed - allowed:
        raise ValueError("D0 changes baseline task fields: " + str(changed - allowed))
    for config in (old, new):
        if any(value != 1.0 for scales in GridCombat(config).reward_scales.values() for value in scales.values()):
            raise ValueError("D0 reuse requires exactly unchanged damage and kill rewards")
    if len(new.red) != 6 or len(new.blue) != 6 or not new.diagnostics_enabled:
        raise ValueError("D0 must be 6v6 with diagnostics enabled")


def training_command(algorithm, metadata, backend, config, output):
    values = {**metadata["cli"], "config": str(config), "output": str(output)}
    values["marl_root" if algorithm in ("qmix", "qtran") else "epymarl_root"] = str(backend)
    command = [sys.executable, "-m", f"algorithms.{algorithm}.train"]
    for key, value in values.items():
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def prepare(args):
    roots = {algorithm: args.output_root / algorithm / args.name for algorithm in ALGORITHMS}
    coordinator = args.output_root / "campaigns" / (args.name + "_density_D")
    if coordinator.exists():
        raise FileExistsError(coordinator)
    studies, configs = {}, {}
    for algorithm, root in roots.items():
        study = read_json(root / "experiment.json")
        status = read_json(root / "status.json")
        if study is None or status is None or status["status"] != "COMPLETED":
            raise ValueError("Original study must be completed: " + str(root))
        if study["seeds"] != [42, 43, 44] or study["steps"] != 5_000_000:
            raise ValueError("Expected the original 3-seed, 5M study")
        for path in [root / "D_experiment.json", root / "D_status.json", root / "D_report"] + [
                root / name for name in TASK_COUNTS] + [root / "configs" / (name + ".json") for name in TASK_COUNTS]:
            if path.exists():
                raise FileExistsError("Refusing to overwrite " + str(path))
        baseline = EnvConfig.load(root / "configs/B0.json")
        new = replace(baseline, reward_normalization="reference", reward_reference_team_size=6,
                      reward_reference_team_hp=480.0, diagnostics_enabled=True)
        assert_baseline_equivalent(baseline, new)
        generated = scaling_scenarios(new, mode="constant_density")
        configs[algorithm] = {name: generated[f"{count}v{count}"] for name, count in TASK_COUNTS.items()}
        for seed in study["seeds"]:
            source = root / "B0" / f"seed_{seed}"
            summary = read_json(source / "summary.json")
            if summary is None or summary["environment_steps"] != study["steps"]:
                raise ValueError("Missing complete B0 run: " + str(source))
            for model in ("final", "best"):
                if not (source / "models" / (model + ".pt")).is_file():
                    raise FileNotFoundError(source / "models" / (model + ".pt"))
        studies[algorithm] = study
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    runtime.mkdir()
    for name in ("heterogrid", "algorithms", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    backends = {}
    for algorithm in ("qmix", "maa2c"):
        source = Path(studies[algorithm]["backend_snapshot"])
        destination = coordinator / ("_backend_value" if algorithm == "qmix" else "_backend_actor")
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for member in (("qmix", "qtran") if algorithm == "qmix" else ("maa2c", "coma")):
            if Path(studies[member]["backend_snapshot"]) != source:
                raise ValueError("Algorithm family backend snapshots do not agree")
            backends[member] = destination
    for algorithm, root in roots.items():
        for name, config in configs[algorithm].items():
            write_atomic(root / "configs" / (name + ".json"), config.to_dict())
        experiment = {**studies[algorithm], "created": now(), "groups": {"density_scale": list(TASK_COUNTS)},
            "labels": LABELS, "group_titles": {"density_scale": "Density-controlled team size"},
            "planned_runs": 12, "new_training_runs": 9, "reused_baseline_runs": 3,
            "validation_sources": {"D0": "B0"}, "coordinator": str(coordinator),
            "runtime_snapshot": str(runtime), "backend_snapshot": str(backends[algorithm]),
            "reward_normalization": "reference", "diagnostics_enabled": True,
            "density_protocol": "isotropic cube-root map scaling, nearest integer; ranges and max_steps unchanged",
            "report_notes": ["D0复用B0的3个训练模型与验证曲线，补做双方诊断测试，不是新增独立训练重复。",
                "D1–D3共9次新训练，固定5M步；D0的训练耗时/吞吐量引用旧B0，不能当成开启诊断后的新训练成本。",
                "所有D场景归一化伤害/击杀奖励，开启诊断；与旧N组还存在奖励协议差异，不能将差异全部归因于密度。",
                "近似固定密度不等于相同行进距离或难度；未运行额外的固定地图归一化对照。"],
            "figure_contract": "Measured validation win-rate vs fixed team-step budget; no directional claim; three training seeds; D0 reused, not an extra replicate."}
        write_atomic(root / "D_experiment.json", experiment)
        (root / "D_README.md").write_text(
            "# D组：密度受控规模实验\n\n" + "\n\n".join(experiment["report_notes"]) +
            "\n\nD0=6v6/24×24×6；D1=3v3/19×19×5；D2=9v9/27×27×7；D3=12v12/30×30×8。\n"
            "双方角色1:1:1、规则蓝方、双方distance半径10无失联、自动报告、200步。\n"
            "其余算法参数与原实验一致，使用原冻结外部算法的副本。\n"
            "D0目录保存新诊断评估和模型符号链接；模型原文件在B0，不修改B0。\n"
            "D1–D3各seed目录为完整新训练。D_status.json只跟踪D组，原status.json保持不变。\n"
            "D_report/包含D组RESULTS.md、aggregate.json、独立测试CSV、双方诊断CSV/JSON和密度组曲线。\n"
            "原13场景的汇总和三组图片保持不变。\n", encoding="utf-8")
    jobs = []
    for task in ("D3", "D2", "D1", "D0"):
        for seed in (42, 43, 44):
            for algorithm, root in roots.items():
                metadata = read_json(root / "B0" / f"seed_{seed}" / "config.json")
                output, config = root / task / f"seed_{seed}", root / "configs" / (task + ".json")
                command = ([sys.executable, "-m", "algorithms.common.density_campaign", "--reevaluate",
                            "--algorithm", algorithm, "--study-root", str(root), "--seed", str(seed),
                            "--backend", str(backends[algorithm])] if task == "D0" else
                           training_command(algorithm, metadata, backends[algorithm], config, output))
                jobs.append({"algorithm": algorithm, "scenario": task, "seed": seed, "output": str(output),
                             "command": command, "runtime": str(runtime), "status": "PENDING", "pid": None,
                             "kind": "baseline_reevaluation" if task == "D0" else "new_training",
                             "new_training_steps": 0 if task == "D0" else studies[algorithm]["steps"]})
    plan = {"created": now(), "status": "PREPARED", "manager_pid": None, "workers": args.workers,
            "runtime": str(runtime), "roots": {key: str(path) for key, path in roots.items()}, "jobs": jobs,
            "requested_total_steps": 180_000_000, "new_training_runs": 36, "baseline_reevaluations": 12,
            "algorithm_status_filename": "D_status.json", "report_options": REPORT_OPTIONS,
            "python": sys.executable, "host_cpu_count": os.cpu_count(),
            "disk_free_at_prepare": shutil.disk_usage(coordinator).free}
    write_atomic(coordinator / "plan.json", plan)
    write_atomic(coordinator / "status.json", plan)
    for root in roots.values():
        write_atomic(root / "D_status.json", {"status": "PREPARED", "coordinator": str(coordinator),
                    "manager_pid": None, "counts": {"PENDING": 12}})
        summarize(root, **REPORT_OPTIONS)
    print(coordinator, flush=True)
    return coordinator


def reevaluate(args):
    import random
    import torch

    source = args.study_root / "B0" / f"seed_{args.seed}"
    output = args.reevaluation_output or args.study_root / "D0" / f"seed_{args.seed}"
    old_summary = read_json(source / "summary.json")
    metadata = read_json(source / "config.json")
    config_path = args.study_root / "configs/D0.json"
    config = EnvConfig.load(config_path)
    assert_baseline_equivalent(EnvConfig.load(source / "env.json"), config)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cli = SimpleNamespace(**{**metadata["cli"], "config": config_path, "output": output,
                            "marl_root": args.backend, "epymarl_root": args.backend})
    if args.audit_episodes is not None:
        if not 0 < args.audit_episodes <= cli.heldout_episodes:
            raise ValueError("Audit episodes must be positive and no larger than the original test budget")
        cli.heldout_episodes = args.audit_episodes
    if args.algorithm in ("qmix", "qtran"):
        from algorithms.common.value_decomposition.backend import build_policy
        from algorithms.common.value_decomposition.train import evaluate, restore_checkpoint
    else:
        from algorithms.common.actor_critic.backend import build_policy
        from algorithms.common.actor_critic.train import evaluate, restore_checkpoint
    env = SMACAdapter(config=config)
    try:
        policy, settings = build_policy(cli, env)
    finally:
        env.close()
    actual = json.loads(json.dumps(vars(settings)))
    expected = metadata["algorithm_args"]
    ignored = {"map", "model_dir"}
    differences = [key for key in expected if key not in ignored and expected[key] != actual.get(key)]
    if differences:
        raise ValueError("D0 algorithm settings differ: " + str(differences))
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    sources = list((PROJECT_ROOT / "heterogrid").glob("*.py")) + [Path(__file__), config_path,
              PROJECT_ROOT / "algorithms/common/diagnostics.py", PROJECT_ROOT / "algorithms/common/value_decomposition/train.py",
              PROJECT_ROOT / "algorithms/common/actor_critic/train.py"]
    recorder = ExperimentRecorder(output, None, {**metadata, "name": args.algorithm + "_D0_reevaluation",
        "environment": config.to_dict(), "source_training_run": str(source), "training_reused": True,
        "training_cli": metadata["cli"],
        "cli": {key: str(value) if isinstance(value, Path) else value for key, value in vars(cli).items()}}, sources=sources)
    started = time.monotonic()
    write_atomic(output / "env.json", config.to_dict())
    write_atomic(output / "progress.json", {"status": "REEVALUATING", "environment_steps": 5_000_000,
                 "new_training_steps": 0, "training_reused": True, "source_training_run": str(source)})
    heldout, hashes = {}, {}
    try:
        for model in ("final", "best"):
            checkpoint = source / "models" / (model + ".pt")
            hashes[model] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (output / "models" / checkpoint.name).symlink_to(os.path.relpath(checkpoint, output / "models"))
            restore_checkpoint(checkpoint, policy)
            heldout[model] = {}
            for mode in ("greedy", "sampled"):
                options = ({"epsilon": 0.0 if mode == "greedy" else cli.sampled_epsilon}
                           if args.algorithm in ("qmix", "qtran") else {"greedy": mode == "greedy"})
                result = evaluate(policy, config, output / "replays" / f"heldout_{model}_{mode}",
                                  cli.heldout_episodes, cli.heldout_seed, replays=cli.replay_episodes, **options)
                previous = old_summary["heldout"][model][mode]["results"][:cli.heldout_episodes]
                if len(result["results"]) != len(previous) or any(
                        any(new[key] != value for key, value in old.items())
                        for old, new in zip(previous, result["results"])):
                    raise ValueError("D0 reevaluation differs from B0 episode outcomes")
                heldout[model][mode] = result
        provenance = {"source_training_run": str(source), "training_reused": True,
            "new_training_steps": 0, "model_sha256": hashes,
            "preflight_only": args.audit_episodes is not None,
            "baseline_equivalence": "unchanged task/algorithm settings; both reward factors exactly 1; all heldout episode outcomes match B0",
            "validation_source": str(source / "evaluations"), "diagnostic_evaluation_seconds": time.monotonic() - started}
        write_atomic(output / "provenance.json", provenance)
        write_atomic(output / "summary.json", {**old_summary, **provenance, "heldout": heldout})
        recorder.complete(provenance)
        write_atomic(output / "progress.json", {"status": "COMPLETED", "environment_steps": 5_000_000,
                     "new_training_steps": 0, "training_reused": True})
    except BaseException:
        write_atomic(output / "progress.json", {"status": "FAILED", "new_training_steps": 0, "training_reused": True})
        raise
    finally:
        recorder.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261013_fixed_blue_sensitivity_5m")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--audit-episodes", type=int, help="Short audit only; requires a separate output path")
    parser.add_argument("--reevaluation-output", type=Path, help="Optional separate preflight output directory")
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.reevaluate:
        if any(value is None for value in (args.algorithm, args.study_root, args.seed, args.backend)):
            parser.error("Reevaluation requires algorithm, study root, seed and backend")
        if args.audit_episodes is not None and args.reevaluation_output is None:
            parser.error("Short audit must use --reevaluation-output outside formal D0")
        return reevaluate(args)
    if args.workers < 1:
        parser.error("Workers must be positive")
    args.output_root = args.output_root.resolve()
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
