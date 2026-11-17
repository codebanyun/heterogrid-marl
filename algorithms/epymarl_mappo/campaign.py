"""Detached B/C/M/N/D experiment queue for native EPyMARL MAPPO."""
import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.study import summarize_runs
from algorithms.common.study_queue import run_queue
from heterogrid import EnvConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def reference_configs(references):
    variants, sources, protocols = {}, {}, []
    for reference, prefix in zip(references, (("B", "C"), ("M", "N"), ("D",))):
        reference = Path(reference).resolve()
        manifest = json.loads((reference / "experiment.json").read_text())
        status = json.loads((reference / "status.json").read_text())
        if status["state"] != "COMPLETED":
            raise ValueError("Configuration reference must be completed: " + str(reference))
        protocols.append(manifest)
        for name in manifest["conditions"]:
            if not name.startswith(prefix):
                continue
            if Path(name).name != name or name in ("", ".", "..") or name in variants:
                raise ValueError("Invalid or duplicate scenario name")
            value = json.loads((reference / "configs" / (name + ".json")).read_text())
            env = EnvConfig.from_dict(value)
            if prefix == ("B", "C"):
                blue = env.communication_for("blue")
                if env.blue_communication is None or (blue.mode, blue.radius, blue.dropout_probability) != ("distance", 10, 0):
                    raise ValueError("B/C blue communication must be explicitly fixed")
            if prefix == ("D",) and (not env.diagnostics_enabled or env.reward_normalization != "reference"):
                raise ValueError("D group must retain reference rewards and bilateral diagnostics")
            variants[name], sources[name] = value, str(reference)
    if len(variants) != 17:
        raise ValueError("Expected B0, C1-C6, M1-M3, N1-N3 and D0-D3")
    for key in ("validation_seed", "heldout_seed", "eval_episodes", "heldout_episodes"):
        if any(protocol[key] != protocols[0][key] for protocol in protocols[1:]):
            raise ValueError("Reference evaluation protocols differ: " + key)
    return variants, sources, protocols[0]


def training_command(manifest, output, job):
    command = [manifest["python"], "-u", "-m", "algorithms.epymarl_mappo.train",
               "--epymarl-root", manifest["epymarl_root"], "--config", str(output / "configs" / (job["condition"] + ".json")),
               "--output", job["path"], "--steps", str(manifest["target_steps_per_seed"]), "--seed", str(job["seed"]),
               "--device", manifest["device"], "--eval-seed", str(manifest["validation_seed"]),
               "--heldout-seed", str(manifest["heldout_seed"])]
    for key in ("eval_episodes", "heldout_episodes", "eval_interval", "log_interval", "replay_episodes"):
        command += ["--" + key.replace("_", "-"), str(manifest[key])]
    return command


def summarize(path, runs):
    result = summarize_runs(path, runs, "EPyMARL MAPPO", "直接调用 EPyMARL 原生 PPOLearner、BasicMAC 和 mappo.yaml。")
    if runs[0]["summary"]["heldout"]["latest"]["greedy"].get("mean_team_episode_metrics"):
        from algorithms.common.density_study import diagnostic_summary
        diagnostic_summary(path, runs)
    return result


def write_results(output):
    manifest = json.loads((output / "experiment.json").read_text())
    lines = ["# EPyMARL MAPPO 实验结果", "", "使用最终预算模型的贪心独立测试，单位为百分比，报告训练种子均值及 SEM。",
             "不是 on-policy/MAPPO 实现；网络和训练超参数见 experiment.json，各 seed 的 network.json 记录实际结构。", "",
             "| 配置 | EPyMARL MAPPO |", "|---|---:|"]
    for name in manifest["conditions"]:
        wins = [100 * json.loads((output / name / ("seed_" + str(seed)) / "summary.json").read_text())
                ["heldout"]["latest"]["greedy"]["win_rate"] for seed in manifest["seeds"]]
        cell = "%.2f" % statistics.mean(wins)
        cell += " ±%.2f" % (statistics.stdev(wins) / math.sqrt(len(wins))) if len(wins) > 1 else " (n=1)"
        lines.append("| " + name + " | " + cell + " |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epymarl-root", type=Path, default=os.environ.get("EPYMARL_ROOT"))
    parser.add_argument("--communication-reference", type=Path)
    parser.add_argument("--sensitivity-reference", type=Path)
    parser.add_argument("--density-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--steps", type=int, default=5000192)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=250368)
    parser.add_argument("--log-interval", type=int, default=10240)
    parser.add_argument("--replay-episodes", type=int, default=1)
    parser.add_argument("--external-status", type=Path, nargs="*", default=[])
    parser.add_argument("--minimum-free-disk-gb", type=float, default=2.)
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    cli = parser.parse_args()
    output = cli.output.resolve()
    if cli.manage:
        from algorithms.epymarl_mappo.audit import audit
        run_queue(output, training_command, audit, summarize)
        write_results(output)
        return
    if cli.epymarl_root is None or any(path is None for path in
            (cli.communication_reference, cli.sensitivity_reference, cli.density_reference)):
        parser.error("Supply the EPyMARL root and all three completed configuration references")
    if min(cli.steps, cli.workers, cli.eval_episodes, cli.heldout_episodes, cli.eval_interval, cli.log_interval) < 1:
        parser.error("Budgets, workers and evaluation counts must be positive")
    if cli.replay_episodes < 0 or cli.minimum_free_disk_gb <= 0 or not cli.seeds or min(cli.seeds) < 0 or len(cli.seeds) != len(set(cli.seeds)):
        parser.error("Invalid seeds, replay count or disk threshold")
    from algorithms.epymarl_mappo.backend import resolve_root, settings
    root = resolve_root(cli.epymarl_root)
    variants, sources, protocol = reference_configs((cli.communication_reference, cli.sensitivity_reference, cli.density_reference))
    if cli.conditions:
        if len(cli.conditions) != len(set(cli.conditions)) or any(name not in variants for name in cli.conditions):
            parser.error("Use distinct reference condition names")
        variants = {name: variants[name] for name in cli.conditions}
    if protocol["validation_seed"] + cli.eval_episodes > protocol["heldout_seed"]:
        parser.error("Validation and heldout seeds overlap")
    for path in cli.external_status:
        state = json.loads(path.read_text())
        if state["state"] == "STARTING" or any(row["state"] == "QUEUED" for row in state.get("runs", {}).values()):
            parser.error("External experiments must have no queued jobs")
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < cli.minimum_free_disk_gb * 1024 ** 3:
        parser.error("Insufficient disk space")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    for name, value in variants.items():
        write_atomic(output / "configs" / (name + ".json"), value)
    project, external = output / "_code/project", output / "_code/epymarl/src"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("algorithms", "heterogrid", "configs"):
        shutil.copytree(PROJECT_ROOT / name, project / name, ignore=ignore)
    for name in ("controllers", "learners", "modules", "components", "utils", "config"):
        shutil.copytree(root / name, external / name, ignore=ignore)
    manifest = {"algorithm": "epymarl_mappo", "created": timestamp(), "python": sys.executable,
                "repository": str(project), "epymarl_root": str(external), "source_epymarl_root": str(root),
                "conditions": list(variants), "condition_references": {name: sources[name] for name in variants},
                "config_hashes": {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() for name, value in variants.items()},
                "seeds": cli.seeds, "jobs": len(variants) * len(cli.seeds), "workers": min(cli.workers, len(variants) * len(cli.seeds)),
                "team_sizes": {name: len(value["red"]) for name, value in variants.items()},
                "device": cli.device, "target_steps_per_seed": cli.steps, "validation_seed": protocol["validation_seed"],
                "heldout_seed": protocol["heldout_seed"], "eval_episodes": cli.eval_episodes, "heldout_episodes": cli.heldout_episodes,
                "eval_interval": cli.eval_interval, "log_interval": cli.log_interval, "replay_episodes": cli.replay_episodes,
                "algorithm_args": settings(external, cli.device), "external_status_paths": [str(p.resolve()) for p in cli.external_status],
                "environment_source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (project / "heterogrid").glob("*.py")},
                "minimum_free_disk_bytes": int(cli.minimum_free_disk_gb * 1024 ** 3),
                "initialization": "all runs independently initialized from scratch; no pretrained or smoke weights",
                "checkpoint_policy": "initial, best validation, latest; complete networks, optimizers and normalizers",
                "budget_policy": "complete fresh batches, overshoot < 10 * episode_limit; evaluation excluded"}
    command = ["nohup", sys.executable, "-u", "-m", "algorithms.epymarl_mappo.campaign", "--manage", "--output", str(output)]
    manifest["launch_command"] = command
    write_atomic(output / "experiment.json", manifest)
    write_atomic(output / "status.json", {"state": "STARTING", "runs": {}})
    (output / "README.md").write_text("# EPyMARL MAPPO 实验\n\n"
        "本目录独立于 output/mappo；使用 EPyMARL 原生 PPOLearner、BasicMAC、mappo.yaml。\n"
        "B/C 来自固定蓝方配置，M/N 保留 sensitivity，D 保留密度受控、奖励归一化及双方诊断。\n"
        "默认 17 配置、三个 seed、约 5M 步；实际范围和预算见 experiment.json。\n"
        "GRU 128、每批 10 个完整对局、4 轮 PPO、裁剪 0.2、学习率 0.0003、熵系数 0.001。\n"
        "保留原生奖励标准化与五步目标；评分超时为终局，不额外跨终局 bootstrap。\n"
        "仅保存初始、验证最优、最终模型；逐局日志无损 gzip，不生成 trajectories/ 或 evaluations/。\n"
        "_code/ 冻结源码；status.json 为进度，nohup.log 为日志；结束后自动审计并生成均值及 SEM 曲线。\n"
        "external_status_paths 中的已有实验与本队列共享 workers 总上限。\n", encoding="utf-8")
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "manager_pid": process.pid, "jobs": manifest["jobs"],
                      "workers": manifest["workers"], "target_steps_per_seed": cli.steps}), flush=True)


if __name__ == "__main__":
    main()
