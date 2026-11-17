"""Frozen five-algorithm communication/scale/roster campaign; no algorithm changes.

Prepare records all 225 entries, including 42 read-only model reuses. The shared
queue runs 183 fresh learners and re-evaluates reused models on common new seeds.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from algorithms.common.c1_reference_campaign import commands, compact_completed, digest, read
from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT
from heterogrid import CommunicationConfig, EnvConfig, UnitSpec
from heterogrid.scenarios import balanced_roster

ALGORITHMS = ("dib_v3", "qmix", "qtran", "maa2c", "coma")
SEEDS = (42, 43, 44)
BASELINE_STUDY = "20261013_fixed_blue_sensitivity_5m"
GOLDEN = "20261109_c1_symmetric_none_5m_c1_reference"
ROSTERS = {"R_scout": (3, 1, 2), "R_strike": (2, 3, 1), "R_hybrid": (1, 2, 3)}
GIB = 1024**3


def matrix(base):
    base = replace(base, diagnostics_enabled=True, terminate_on_no_firepower=False)
    scales = {
        "B0": base,
        "M2": replace(base, map_size=(36, 36, 9)),
        "N2": replace(base, red=balanced_roster(9), blue=balanced_roster(9)),
        "D2": replace(base, map_size=(27, 27, 7), red=balanced_roster(9),
                      blue=balanced_roster(9), reward_normalization="reference",
                      reward_reference_team_size=6, reward_reference_team_hp=480.0),
    }
    tasks = {}
    for condition in ("R5", "P50", "C1S"):
        red = (CommunicationConfig("distance", radius=5.0) if condition == "R5" else
               CommunicationConfig("full", dropout_probability=.5) if condition == "P50" else
               CommunicationConfig("none"))
        blue = CommunicationConfig("none") if condition == "C1S" else CommunicationConfig("distance", radius=10.0)
        for scale, config in scales.items():
            tasks[condition + "_" + scale] = replace(config, red_communication=red, blue_communication=blue)
    for name, counts in ROSTERS.items():
        platforms = (("uav", "usv"), ("robot", "robot_dog"), ("uav", "robot_dog"))
        roster = [UnitSpec(role, platforms[r][i % 2]) for r, (role, count) in
                  enumerate(zip(("scout", "strike", "hybrid"), counts)) for i in range(count)]
        tasks["C1S_" + name] = replace(tasks["C1S_B0"], red=roster, blue=list(roster))
    return tasks


def equivalent(a, b):
    # Shared legacy none vs explicit team-none is a representation difference.
    ignored = {"communication", "comm_radius", "dropout_probability", "red_communication",
               "blue_communication", "diagnostics_enabled"}
    if any(a.communication_for(team) != b.communication_for(team) for team in ("red", "blue")):
        return False
    x, y = a.to_dict(), b.to_dict()
    return all(x[k] == y[k] for k in x if k not in ignored)


def reuse_source(output, algorithm, scene, seed):
    if scene not in ("R5_B0", "P50_B0", "C1S_B0"):
        return None
    if algorithm == "dib_v3":
        if scene == "R5_B0":
            return None
        if scene == "P50_B0":
            return output / "dib_mappo/20261107_resilient_v3_c1_c6_5m/belief_first/C6" / f"seed_{seed}"
        return output / "dib_mappo/20261109_c1_symmetric_none_5m/dib_v3/C1S" / f"seed_{seed}"
    if scene == "C1S_B0":
        date = "20261011" if algorithm in ("qmix", "qtran") else "20261012"
        return output / algorithm / (date + "_sensitivity_5m") / "C1" / f"seed_{seed}"
    return output / algorithm / BASELINE_STUDY / ("C3" if scene == "R5_B0" else "C6") / f"seed_{seed}"


def verify_reuse(source, config, steps):
    summary = read(source / "summary.json")
    if not summary or summary.get("environment_steps") != steps:
        raise ValueError("Incomplete reuse source: " + str(source))
    if not equivalent(EnvConfig.load(source / "env.json"), config):
        raise ValueError("Reuse task mismatch: " + str(source))
    if not all((source / "models" / (m + ".pt")).is_file() for m in ("final", "best")):
        raise ValueError("Reuse weights missing: " + str(source))
    return {"source_run": str(source), "new_training_steps": 0,
            "historical_training_steps": steps, "summary_sha256": digest(source / "summary.json"),
            "model_sha256": {m: digest(source / "models" / (m + ".pt")) for m in ("final", "best")},
            "equivalence": "all effective task/reward fields equal; diagnostics/communication representation ignored"}


def prepare(args):
    output = args.output_root.resolve()
    coordinator = output / "campaigns" / (args.name + "_all")
    roots = {a: output / ("dib_mappo" if a == "dib_v3" else a) / args.name for a in ALGORITHMS}
    if coordinator.exists() or any(p.exists() for p in roots.values()):
        raise FileExistsError("Fresh campaign name required")
    golden = output / "campaigns" / GOLDEN
    if shutil.disk_usage(output).free < 6 * GIB:
        raise OSError("Need 6 GiB disk headroom")
    base = EnvConfig.load(output / "dib_mappo/20261107_resilient_v3_c1_c6_5m/configs/C6.json")
    tasks = matrix(base)
    references = {}
    for algorithm in ALGORITHMS:
        for scene, config in tasks.items():
            for seed in SEEDS:
                source = reuse_source(output, algorithm, scene, seed)
                if source:
                    references[algorithm, scene, seed] = verify_reuse(source, config, args.steps)
    if len(references) != 42 or len(tasks) != 15:
        raise AssertionError("Expected exactly 15 configurations and 42 reusable models")
    # Preserve the already-tested computation; add only this new orchestrator.
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    shutil.copytree(golden / "_runtime", runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(Path(__file__), runtime / "algorithms/common/comm_scale_campaign.py")
    shutil.copy2(PROJECT_ROOT / "scripts/export_retained_replays.py", runtime / "scripts/export_retained_replays.py")
    backends = {}
    for family in ("dib_v2", "qmix", "maa2c"):
        dest = coordinator / ("_backend_" + family)
        shutil.copytree(golden / ("_backend_" + family), dest,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for algorithm in (("dib_v3",) if family == "dib_v2" else
                          ("qmix", "qtran") if family == "qmix" else ("maa2c", "coma")):
            backends[algorithm] = dest
    jobs = []
    for algorithm, root in roots.items():
        root.mkdir(parents=True); (root / "configs").mkdir()
        template = (None if algorithm == "dib_v3" else
                    read(output / algorithm / BASELINE_STUDY / "C3/seed_42/config.json"))
        for scene, config in tasks.items():
            config_path = root / "configs" / (scene + ".json")
            write_atomic(config_path, config.to_dict())
            condition, scale = scene.split("_", 1)
            for seed in SEEDS:
                out = root / condition / scale / f"seed_{seed}"
                reuse = references.get((algorithm, scene, seed))
                cmd = commands(algorithm, template, backends[algorithm], config_path, out, args, seed)
                jobs.append({"id": algorithm + "/" + scene + "/" + str(seed), "algorithm": algorithm,
                             "scenario": scene, "condition": condition, "scale": scale, "seed": seed,
                             "n_agents": len(config.red), "output": str(out), "config": str(config_path),
                             "backend": str(backends[algorithm]), "command": cmd, "status": "PENDING",
                             "phase": "REEVALUATION" if reuse else "TRAINING", "pid": None,
                             "training_reused": bool(reuse), "new_training_steps": 0 if reuse else args.steps,
                             **({"reuse": reuse} if reuse else {})})
        write_atomic(root / "experiment.json", {
            "algorithm": algorithm, "training_profile": "resilient_v3" if algorithm == "dib_v3" else algorithm,
            "coordinator": str(coordinator), "runtime": str(runtime), "backend": str(backends[algorithm]),
            "seeds": list(SEEDS), "steps": args.steps, "scenarios": list(tasks),
            "new_training_runs": 39 if algorithm == "dib_v3" else 36,
            "reused_training_runs": 6 if algorithm == "dib_v3" else 9,
            "validation_seed": args.eval_seed, "heldout_seed": args.heldout_seed,
            "eval_episodes": args.eval_episodes, "heldout_episodes": args.heldout_episodes,
            "primary_model": "fixed-budget latest/final; best is validation-only supplement",
            "termination_protocol": "legacy_elimination; terminate_on_no_firepower=False",
            "reward_protocol": "M/N/rosters original none; D2 historical reference normalization; N-D not pure density effect",
            "frozen_preset": "belief_first=.30" if algorithm == "dib_v3" else None,
            "reuse_does_not_reselect_models": True, "created": timestamp()})
    state = {"status": "PREPARED", "created": timestamp(), "manager_pid": None,
             "runtime": str(runtime), "roots": {k: str(v) for k, v in roots.items()},
             "backends": {k: str(v) for k, v in backends.items()}, "jobs": jobs,
             "workers": args.workers, "dib_workers_while_baselines_pending": args.dib_workers,
             "seeds": list(SEEDS), "scenarios": list(tasks), "steps": args.steps,
             "requested_total_steps": args.steps * 183, "new_training_runs": 183,
             "reused_training_runs": 42, "validation_seed": args.eval_seed, "heldout_seed": args.heldout_seed,
             "eval_episodes": args.eval_episodes, "heldout_episodes": args.heldout_episodes,
             "memory_reserve_gib": 16, "disk_dispatch_reserve_gib": 4,
             "disk_emergency_reserve_gib": 2, "new_training_is_from_scratch": True}
    hashes = {str(p.relative_to(coordinator)): digest(p) for base in
              [runtime, *set(backends.values())] for p in sorted(base.rglob("*")) if p.is_file()}
    hashes.update({str(p): digest(p) for r in roots.values() for p in (r / "configs").glob("*.json")})
    state["source_sha256"] = hashes
    write_atomic(coordinator / "plan.json", state)
    write_atomic(coordinator / "matrix.json", {"tasks": {k: v.to_dict() for k, v in tasks.items()},
                 "new_runs": 183, "reused_runs": 42, "total_entries": 225})
    with (coordinator / "matrix.csv").open("w", newline="") as stream:
        fields = ("algorithm", "scenario", "condition", "scale", "seed", "n_agents", "training_reused",
                  "new_training_steps", "output", "source_run")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for job in jobs:
            writer.writerow({k: job.get(k, job.get("reuse", {}).get(k)) for k in fields})
    lines = ["# 通信受限 × 规模与角色配比：冻结实验矩阵", "",
             "五算法：DIB-v3、QMIX、QTRAN-base、MAA2C、COMA。每配置 seed42/43/44，各从头5M团队环境步。",
             "共15配置×5算法×3seed=225条记录；复用42次，新增183次，新增915M训练步。",
             "DIB39次；四基线各36次。原始plan.json只读，status.json跟踪阶段，不因胜率提前结束。", "",
             "| 配置 | 地图 | 双方人数/每方角色数量 | 红方通信 | 蓝方通信 | 奖励归一化 |",
             "|---|---|---|---|---|---|"]
    for scene, config in tasks.items():
        counts = [sum(u.role == role for u in config.red) for role in ("scout", "strike", "hybrid")]
        lines.append(f"| {scene} | {'×'.join(map(str, config.map_size))} | {len(config.red)}v{len(config.blue)} / {counts} | "
                     f"{config.communication_for('red')} | {config.communication_for('blue')} | {config.reward_normalization} |")
    lines += ["", "R5红方distance5；P50红方full+每步独立节点失联50%，不是距离+失联；两组蓝方rule/distance10无失联。",
              "C1S双方none，是改变蓝方通信的对称参考，不把组间全部差异归因于红方通信。",
              "角色按侦察/打击/查打一体排列；双方同配比，原平台循环分配规则不变。",
              "D2沿用历史reference归一化；其他场景沿用原奖励。因此N2与D2并非纯密度单因素因果对照。",
              "不改视野/射程/角色能力/动作掩码/自动目标共享/200步上限/终止语义。",
              "训练与测试使用相同通信。actor无集中信息输入；critic/teacher仍按冻结算法训练。",
              f"新训练验证seed={args.eval_seed}起64局/约250k；统一新独立测试seed={args.heldout_seed}起128局。",
              "旧模型不重新挑选best，训练/验证曲线保留原种子来源；所有final/best在新测试集重评，不能拼接旧测试胜率。",
              "DIB固定resilient_v3+belief_first=.30；其他算法完整复制历史CLI与超参数，只有输入输出随环境自动适配。",
              f"总计算工作进程上限{args.workers}（包含旧实验），有基线排队时新DIB上限{args.dib_workers}；每进程单线程。",
              "优先DIB九人任务，基线轮转；内存预留16GiB，磁盘少于4GiB停止派发、少于2GiB安全中止。",
              "完成后仅保留latest/best；final为兼容链接；逐局日志无损gzip，统计、配置、源码快照全部保留。",
              "每个模型保留新测试集第一局贪心完整回放与真实动作，不按胜负挑回放。",
              "失败不自动从头重跑；DIB中断只支持精确Adam/RNG边界续训，基线不做权重冒充续训。", ""]
    (coordinator / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for root in roots.values():
        (root / "README.md").write_text("# 实验记录\n\n完整矩阵和资源策略见 " + str(coordinator / "README.md") + "\n")
    persist(coordinator, state); report(coordinator, state)
    print(coordinator, flush=True)
    return coordinator


def reevaluate(coordinator, job_id):
    import numpy as np
    import torch
    from heterogrid import SMACAdapter
    state = read(coordinator / "plan.json")
    job = next(j for j in state["jobs"] if j["id"] == job_id)
    source = Path(job["reuse"]["source_run"]); out = Path(job["output"])
    cfg = EnvConfig.load(job["config"])
    ref = verify_reuse(source, cfg, state["steps"])
    if ref["model_sha256"] != job["reuse"]["model_sha256"]:
        raise ValueError("Historical weights changed")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(job["seed"]); np.random.seed(job["seed"]); torch.manual_seed(job["seed"])
    metadata = read(source / "config.json")
    cli = SimpleNamespace(**metadata["cli"])
    cli.algorithm = "dib_mappo" if job["algorithm"] == "dib_v3" else job["algorithm"]
    cli.output = out; cli.config = Path(job["config"])
    cli.heldout_seed = state["heldout_seed"]; cli.heldout_episodes = state["heldout_episodes"]
    if job["algorithm"] == "dib_v3":
        from algorithms.dib_mappo.backend import build_policy
        from algorithms.dib_mappo.train import restore_checkpoint, evaluate
        cli.dib_root = Path(job["backend"]); cli.training_profile = "resilient_v3"; cli.tactical_preset = "belief_first"
    elif job["algorithm"] in ("qmix", "qtran"):
        from algorithms.common.value_decomposition.backend import build_policy
        from algorithms.common.value_decomposition.train import restore_checkpoint, evaluate
        cli.marl_root = Path(job["backend"])
    else:
        from algorithms.common.actor_critic.backend import build_policy
        from algorithms.common.actor_critic.train import restore_checkpoint, evaluate
        cli.epymarl_root = Path(job["backend"])
    if out.exists():
        raise FileExistsError("Re-evaluation output already exists; inspect before retry")
    out.mkdir(parents=True); (out / "models").mkdir()
    for name in ("latest", "best"):
        (out / "models" / (name + ".pt")).symlink_to(source / "models" / ("final.pt" if name == "latest" else "best.pt"))
    (out / "models/final.pt").symlink_to("latest.pt")
    write_atomic(out / "env.json", cfg.to_dict())
    write_atomic(out / "config.json", {**metadata, "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                 "environment": cfg.to_dict(), "training_reused": True, "source_run": str(source),
                 "heldout_seed": state["heldout_seed"]})
    env = SMACAdapter(config=cfg)
    try:
        policy, _ = build_policy(cli, env)
    finally:
        env.close()
    heldout = {}
    for model in ("final", "best"):
        restore_checkpoint(out / "models" / (model + ".pt"), policy); heldout[model] = {}
        for mode in ("greedy", "sampled"):
            opts = ({"epsilon": 0 if mode == "greedy" else cli.sampled_epsilon} if
                    job["algorithm"] in ("qmix", "qtran") else {"greedy": mode == "greedy"})
            heldout[model][mode] = evaluate(policy, cfg, out / "evaluations" / ("heldout_" + model + "_" + mode),
                state["heldout_episodes"], state["heldout_seed"], **opts)
    original = read(source / "summary.json")
    write_atomic(out / "summary.json", {**original, "heldout": heldout, "training_reused": True,
                 "source_run": str(source), "historical_validation_seed": metadata.get("validation_seed", metadata["cli"].get("eval_seed")),
                 "new_training_steps": 0, "reevaluation_seconds": sum(v["elapsed_seconds"] for d in heldout.values() for v in d.values())})
    write_atomic(out / "provenance.json", {**job["reuse"], "new_heldout_seed": state["heldout_seed"],
                 "reselected_models": False, "source_unchanged": all(digest(source / "models" / (m + ".pt")) ==
                     job["reuse"]["model_sha256"][m] for m in ("final", "best"))})


def persist(coordinator, state):
    state["heartbeat"] = timestamp()
    write_atomic(coordinator / "status.json", state)
    for algorithm, root in state["roots"].items():
        jobs = [j for j in state["jobs"] if j["algorithm"] == algorithm]
        write_atomic(Path(root) / "status.json", {"status": state["status"], "manager_pid": state["manager_pid"],
                     "heartbeat": state["heartbeat"], "coordinator": str(coordinator),
                     "counts": dict(collections.Counter(j["status"] for j in jobs)), "jobs": jobs})


def report(coordinator, state):
    rows, points = [], []
    for job in state["jobs"]:
        out = Path(job["output"]); summary = read(out / "summary.json")
        source = Path(job["reuse"]["source_run"]) if job["training_reused"] else out
        trained = read(source / "training_summary.json") or read(source / "summary.json")
        validation = (trained or {}).get("validation", [])
        if not validation:
            validation = [{"step": int(p.parent.name.split("_")[-1]), **read(p)} for p in
                          sorted((source / "evaluations").glob("**/validation_*/summary.json"))]
        for value in validation:
            points.append({"algorithm": job["algorithm"], "scenario": job["scenario"], "seed": job["seed"],
                           "step": value["step"], "win_rate": value["win_rate"], "training_reused": job["training_reused"]})
        if not summary or "heldout" not in summary:
            continue
        main = summary["heldout"]["final"]["greedy"]
        if main.get("base_seed") != state["heldout_seed"] or main.get("episodes") != state["heldout_episodes"]:
            raise ValueError("Mixed heldout protocol refused")
        rows.append({"algorithm": job["algorithm"], "scenario": job["scenario"], "seed": job["seed"],
                     "training_reused": job["training_reused"], "heldout_seed": main["base_seed"],
                     "win_rate": main["win_rate"], "timeout_rate": main["timeout_rate"],
                     "contact_rate": main["contact_rate"], "mean_first_contact_step": main["mean_first_contact_step"],
                     "mean_kill_fraction": main["mean_kill_fraction"], "mean_useful_reports": main["mean_useful_reports"],
                     "training_seconds": summary["training_seconds"],
                     "environment_steps_per_second": summary["environment_steps_per_second"],
                     "validation_auc_normalized": summary.get("validation_auc_normalized"),
                     "first_80_percent_step": summary.get("first_80_percent_step"),
                     "sampled_win_rate": summary["heldout"]["final"]["sampled"]["win_rate"],
                     "best_greedy_win_rate": summary["heldout"]["best"]["greedy"]["win_rate"]})
    for name, data in (("independent_test_per_seed.csv", rows), ("validation_per_seed.csv", points)):
        with (coordinator / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]) if data else ["algorithm", "scenario", "seed", "win_rate"])
            writer.writeheader(); writer.writerows(data)
    aggregates = []
    lines = ["# 固定5M结束模型：统一新独立测试", "", "均值±SEM仅汇总完整三个训练seed；尚未完成不填零。",
             "复用模型不重训，保留历史验证集及best选择；全部模型在同一新测试集重评。",
             "C1S为对称无通信参考；N/D仍有历史奖励归一化差异。", "",
             "| 配置 | DIB-v3 | QMIX | QTRAN-base | MAA2C | COMA |", "|---|---:|---:|---:|---:|---:|"]
    for scene in state["scenarios"]:
        values = []
        for algorithm in ALGORITHMS:
            subset = [r for r in rows if r["algorithm"] == algorithm and r["scenario"] == scene]
            if len(subset) != len(SEEDS):
                values.append("待完成"); continue
            wins = [r["win_rate"] for r in subset]; mean = statistics.mean(wins); sem = statistics.stdev(wins) / len(wins)**.5
            aggregates.append({"algorithm": algorithm, "scenario": scene, "mean": mean, "sem": sem, "n": len(wins)})
            values.append(f"{mean*100:.2f} ±{sem*100:.2f}%")
        lines.append("| " + scene + " | " + " | ".join(values) + " |")
    write_atomic(coordinator / "aggregate.json", {"results": aggregates, "per_seed": rows})
    (coordinator / "RESULTS.md").write_text("\n".join(lines) + "\n")
    for root in state["roots"].values():
        for name in ("RESULTS.md", "aggregate.json", "independent_test_per_seed.csv", "validation_per_seed.csv"):
            link = Path(root) / name
            if not link.exists(): link.symlink_to(coordinator / name)


def external_workers(own_pids):
    found = []
    modules = {"algorithms.dib_mappo.tactical_train", "algorithms.qmix.train", "algorithms.qtran.train",
               "algorithms.maa2c.train", "algorithms.coma.train", "algorithms.dib_mappo.matched_eval"}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in own_pids:
            continue
        try:
            args = (p / "cmdline").read_bytes().decode().split("\0")
            if "-m" in args and args[args.index("-m") + 1] in modules:
                found.append(int(p.name))
        except (OSError, UnicodeError, IndexError):
            pass
    return found


def rss_gib(pid):
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / GIB
    except (OSError, IndexError):
        return 0.


def memory_estimate(job):
    if job["phase"] != "TRAINING":
        return 1.5
    return (4. if job["n_agents"] == 9 else 2.5) if job["algorithm"] == "dib_v3" else (6. if job["n_agents"] == 9 else 3.)


def select_pending(state, live_jobs, slots):
    pending = [j for j in state["jobs"] if j["status"] == "PENDING"]
    baseline_waiting = any(j["algorithm"] != "dib_v3" and not j["training_reused"] for j in pending)
    dib_live = sum(j["algorithm"] == "dib_v3" and j["phase"] == "TRAINING" for j in live_jobs)
    def priority(j):
        if j["phase"] in ("EVALUATION", "REPLAYS"):
            return (0, 0, 0, j["seed"])
        if j["algorithm"] == "dib_v3" and not j["training_reused"]:
            return (1, 0 if j["n_agents"] == 9 else 1 if j["scale"] == "M2" else 2, j["scenario"], j["seed"])
        if j["training_reused"]:
            return (3, 0, j["scenario"], j["seed"])
        # Round-robin families within each scene/seed, not 36 serial jobs per algorithm.
        return (2, j["scenario"], j["seed"], ALGORITHMS.index(j["algorithm"]))
    chosen = []
    for job in sorted(pending, key=priority):
        if len(chosen) >= slots:
            break
        is_dib = job["algorithm"] == "dib_v3" and job["phase"] == "TRAINING"
        if is_dib and baseline_waiting and dib_live >= state["dib_workers_while_baselines_pending"]:
            continue
        chosen.append(job)
        if is_dib: dib_live += 1
    return chosen


def supervise(coordinator):
    lock = (coordinator / "manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(coordinator / "status.json")
    if state["status"] != "PREPARED":
        raise ValueError("Only a fresh prepared queue can start; no implicit restarts")
    for relative, sha in state["source_sha256"].items():
        if digest(coordinator / relative) != sha:
            raise ValueError("Frozen source changed: " + relative)
    live = {}; stopping = False; last_report = 0
    def stop(*_):
        nonlocal stopping
        stopping = True
        for proc, _, _ in live.values():
            if proc.poll() is None: proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}; env.pop("PYTHONPATH", None)
    state.update(status="RUNNING", manager_pid=os.getpid(), started=timestamp())
    try:
        while True:
            for pid, (proc, stream, job) in list(live.items()):
                code = proc.poll(); job["progress"] = read(Path(job["output"]) / "progress.json")
                if code is None: continue
                stream.close(); del live[pid]; job.update(returncode=code, finished=timestamp())
                if code:
                    job.update(status="INTERRUPTED" if stopping or code == 130 else "FAILED")
                elif job["phase"] == "TRAINING" and job["algorithm"] == "dib_v3":
                    data = read(Path(job["output"]) / "training_summary.json")
                    if not data or data.get("steps") != state["steps"]:
                        job.update(status="FAILED", error="Training budget not reached")
                    else: job.update(status="PENDING", phase="EVALUATION")
                elif job["phase"] == "REPLAYS":
                    compact_completed(job["output"]); job.update(status="COMPLETED")
                else:
                    if not read(Path(job["output"]) / "summary.json"):
                        job.update(status="FAILED", error="Missing independent summary")
                    else: job.update(status="PENDING", phase="REPLAYS")
                print("FINISHED", job["id"], job["phase"], job["status"], code, flush=True)
            free = shutil.disk_usage(coordinator).free / GIB
            meminfo = dict((line.split(":")[0], int(line.split()[1])) for line in Path("/proc/meminfo").read_text().splitlines())
            available = meminfo["MemAvailable"] * 1024 / GIB
            external = external_workers(set(live))
            failures = sum(j["status"] == "FAILED" for j in state["jobs"])
            if free < state["disk_emergency_reserve_gib"] and not stopping:
                state["stop_reason"] = "2GiB emergency disk reserve"; stop()
            slots = max(0, state["workers"] - len(live) - len(external))
            reserved_growth = sum(max(0, memory_estimate(j) - rss_gib(pid)) for pid, (_, _, j) in live.items())
            allowed = not stopping and failures < 3 and free >= state["disk_dispatch_reserve_gib"]
            for job in select_pending(state, [v[2] for v in live.values()], slots) if allowed else []:
                estimate = memory_estimate(job)
                if available - reserved_growth - estimate < state["memory_reserve_gib"]:
                    continue
                out = Path(job["output"]); out.parent.mkdir(parents=True, exist_ok=True)
                stream = (out.parent / (out.name + "_" + job["phase"].lower() + ".log")).open("a", buffering=1)
                if job["phase"] == "REEVALUATION":
                    command = [sys.executable, "-u", "-m", "algorithms.common.comm_scale_campaign", "--reevaluate", str(coordinator), "--job-id", job["id"]]
                elif job["phase"] == "REPLAYS":
                    command = [sys.executable, "-u", "-c", "import runpy;runpy.run_path('scripts/export_retained_replays.py',run_name='__main__')", "--run", job["output"]]
                else:
                    command = job["command"] + (["--resume", "--evaluate-only"] if job["phase"] == "EVALUATION" else [])
                proc = subprocess.Popen(command, cwd=state["runtime"], env=env, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT)
                job.update(status="RUNNING", pid=proc.pid, started=timestamp())
                live[proc.pid] = (proc, stream, job); reserved_growth += estimate
                print("STARTED", job["id"], job["phase"], proc.pid, flush=True)
            state["resources"] = {"live_workers": len(live), "external_workers": external,
                "total_worker_limit": state["workers"], "mem_available_gib": round(available, 2),
                "reserved_growth_gib": round(reserved_growth, 2), "disk_free_gib": round(free, 2),
                "dispatch_halted": stopping or failures >= 3 or free < state["disk_dispatch_reserve_gib"]}
            persist(coordinator, state)
            if time.monotonic() - last_report > 180:
                report(coordinator, state); last_report = time.monotonic()
            if not live:
                if stopping or failures >= 3 or not any(j["status"] == "PENDING" for j in state["jobs"]): break
            time.sleep(5)
        state["status"] = "INTERRUPTED" if stopping else "FAILED" if failures else "COMPLETED"
    except BaseException as error:
        state.update(status="FAILED", error=repr(error)); stop(); raise
    finally:
        for proc, stream, job in live.values():
            if proc.poll() is None: proc.send_signal(signal.SIGTERM)
        for proc, stream, job in live.values():
            # Learners finish their current complete-rollout boundary; no SIGKILL.
            proc.wait(); stream.close(); job.update(status="INTERRUPTED", returncode=proc.returncode)
        persist(coordinator, state); report(coordinator, state); lock.close()
    return 0 if state["status"] == "COMPLETED" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path); p.add_argument("--reevaluate", type=Path); p.add_argument("--job-id")
    p.add_argument("--name", default="20261113_comm_scale_roster_5m")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    p.add_argument("--workers", type=int, default=48); p.add_argument("--dib-workers", type=int, default=24)
    p.add_argument("--steps", type=int, default=5000000)
    p.add_argument("--eval-interval", type=int, default=250000); p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--heldout-episodes", type=int, default=128)
    p.add_argument("--eval-seed", type=int, default=1910042); p.add_argument("--heldout-seed", type=int, default=2800042)
    args = p.parse_args()
    if args.run: return supervise(args.run.resolve())
    if args.reevaluate: return reevaluate(args.reevaluate.resolve(), args.job_id)
    if args.steps != 5000000 or args.eval_interval != 250000 or args.eval_episodes != 64 or args.heldout_episodes != 128:
        p.error("Formal campaign uses the frozen 5M/250k/64/128 budget")
    if not 1 <= args.dib_workers <= args.workers <= 48: p.error("Expected 1 <= DIB workers <= total workers <= 48")
    prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
