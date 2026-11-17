"""Detached B/C/M/N/D campaign for QMIX + MA2E, with frozen sources."""
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


def reference_configs(references):
    from algorithms.epymarl_mappo.campaign import reference_configs as load_references
    return load_references(references)


def training_command(manifest, output, job):
    # Device selection must happen in the worker, not once in the supervisor.
    command = [manifest["python"], "-u", "-m", "algorithms.ma2e.launch",
               "--ma2e-root", manifest["ma2e_root"],
               "--config", str(output / "configs" / (job["condition"] + ".json")),
               "--output", job["path"], "--steps", str(manifest["target_steps_per_seed"]),
               "--seed", str(job["seed"]), "--device", manifest.get("device", "auto"),
               "--gpu-min-free-mb", str(manifest.get("gpu_min_free_mb", 10000)),
               "--gpu-max-workers", str(manifest.get("gpu_max_workers", 2)),
               "--eval-seed", str(manifest["validation_seed"]),
               "--heldout-seed", str(manifest["heldout_seed"])]
    for key in ("eval_episodes", "heldout_episodes", "eval_interval", "log_interval", "replay_episodes"):
        command += ["--" + key.replace("_", "-"), str(manifest[key])]
    return command


def summarize(path, runs):
    result = summarize_runs(path, runs, "QMIX + MA2E",
                            "原生 QMIX 结合 MA2E 掩码轨迹预训练；执行仅使用自身局部历史，不是普通 QMIX。")
    if runs[0]["summary"]["heldout"]["latest"]["greedy"].get("mean_team_episode_metrics"):
        from algorithms.common.density_study import diagnostic_summary
        diagnostic_summary(path, runs)
    return result


def write_results(output):
    manifest = json.loads((output / "experiment.json").read_text())
    lines = ["# QMIX + MA2E 实验结果", "",
             "最终预算模型的贪心独立测试胜率（%），报告训练种子均值及 SEM。",
             "最优检查点仅由验证集选择；独立测试不参与模型选择。",
             "预训练环境交互计入团队步预算，验证和测试交互不计入。", "",
             "| 配置 | QMIX + MA2E | 种子数 |", "|---|---:|---:|"]
    for name in manifest["conditions"]:
        wins = [100 * json.loads((output / name / ("seed_" + str(seed)) / "summary.json").read_text())
                ["heldout"]["latest"]["greedy"]["win_rate"] for seed in manifest["seeds"]]
        if not wins or any(not math.isfinite(win) or not 0 <= win <= 100 for win in wins):
            raise ValueError("Invalid heldout win rates: " + name)
        cell = "%.2f" % statistics.mean(wins)
        cell += " ± %.2f" % (statistics.stdev(wins) / math.sqrt(len(wins))) if len(wins) > 1 else " (SEM 不估计)"
        lines.append("| " + name + " | " + cell + " | " + str(len(wins)) + " |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def git_provenance(root):
    def query(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=False, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        return {"commit": query("rev-parse", "HEAD"), "remote": query("remote", "get-url", "origin"),
                "status": query("status", "--short"), "source_root": str(root)}
    except (OSError, subprocess.TimeoutExpired):
        return {"source_root": str(root), "git_unavailable": True}


def snapshot_sources(project_root, root, output):
    project, external = output / "_code/project", output / "_code/ma2e/src"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git")
    for name in ("algorithms", "heterogrid", "configs"):
        shutil.copytree(project_root / name, project / name, ignore=ignore)
    for name in ("pyproject.toml",):
        if (project_root / name).is_file():
            shutil.copy2(project_root / name, project / name)
    # Copy all src dependencies, including envs.matrix_game imported by learners.
    shutil.copytree(root, external, ignore=ignore)
    provenance = {"project": git_provenance(project_root), "ma2e": git_provenance(root),
                  "project_source_hashes": source_hashes(project), "ma2e_source_hashes": source_hashes(external),
                  "hash_policy": "SHA-256 of frozen file bytes; git status may include local changes"}
    write_atomic(output / "_code/provenance.json", provenance)
    return project, external, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    references = Path(__file__).resolve().parents[2] / "output/mappo"
    parser.add_argument("--ma2e-root", type=Path, default=os.environ.get("MA2E_ROOT"))
    parser.add_argument("--communication-reference", type=Path, default=references / "20261014_blue_fixed_5m")
    parser.add_argument("--sensitivity-reference", type=Path, default=references / "20261010_sensitivity_5m")
    parser.add_argument("--density-reference", type=Path, default=references / "20261018_density_5m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--steps", type=int, default=5000192)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-min-free-mb", type=int, default=10000)
    parser.add_argument("--gpu-max-workers", type=int, choices=(1, 2, 3), default=2)
    for key in ("eval-episodes", "heldout-episodes", "eval-interval", "log-interval"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--replay-episodes", type=int, default=1)
    parser.add_argument("--external-status", type=Path, nargs="*", default=[])
    parser.add_argument("--minimum-free-disk-gb", type=float, default=0.4)
    parser.add_argument("--projected-storage-mb", type=int, default=1500)
    parser.add_argument("--checkpoint-estimate-mb", type=int, default=8)
    parser.add_argument("--logs-replays-estimate-mb", type=int, default=5)
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    cli = parser.parse_args()
    output = cli.output.expanduser().resolve()
    if cli.manage:
        from algorithms.ma2e.audit import audit
        run_queue(output, training_command, audit, summarize)
        write_results(output)
        return
    if cli.ma2e_root is None:
        parser.error("Supply --ma2e-root or MA2E_ROOT")
    if min(cli.steps, cli.workers, cli.gpu_min_free_mb) < 1:
        parser.error("Budgets, workers and GPU memory reservation must be positive")
    if (cli.replay_episodes < 0 or not math.isfinite(cli.minimum_free_disk_gb)
            or cli.minimum_free_disk_gb < 0.4 or cli.projected_storage_mb < 1500
            or cli.checkpoint_estimate_mb < 1 or cli.logs_replays_estimate_mb < 0
            or min(cli.seeds) < 0 or len(cli.seeds) != len(set(cli.seeds))):
        parser.error("Invalid seeds, replay count or disk admission thresholds")
    from algorithms.ma2e.backend import CORRECTIONS, PROJECT_ROOT, resolve_root, settings
    root = resolve_root(cli.ma2e_root)
    for name in ("components", "config", "controllers", "envs", "learners", "modules", "utils"):
        if not (root / name).is_dir():
            parser.error("Incomplete MA2E source directory: " + name)
    for name in ("backend.py", "train.py", "audit.py", "launch.py"):
        if not (PROJECT_ROOT / "algorithms/ma2e" / name).is_file():
            parser.error("MA2E integration is not ready: " + name)
    variants, sources, protocol = reference_configs(
        (cli.communication_reference, cli.sensitivity_reference, cli.density_reference))
    expected = {"B0", *("C" + str(i) for i in range(1, 7)), *("M" + str(i) for i in range(1, 4)),
                *("N" + str(i) for i in range(1, 4)), *("D" + str(i) for i in range(4))}
    if {name.split("_", 1)[0] for name in variants} != expected:
        parser.error("References must contain B0, C1-C6, M1-M3, N1-N3 and D0-D3")
    reference_protocols = [json.loads((path.resolve() / "experiment.json").read_text()) for path in
                           (cli.communication_reference, cli.sensitivity_reference, cli.density_reference)]
    for key in ("eval_episodes", "heldout_episodes", "eval_interval", "log_interval"):
        if any(p[key] != protocol[key] for p in reference_protocols):
            parser.error("Reference protocols differ: " + key)
        if getattr(cli, key) is None:
            setattr(cli, key, protocol[key])
        if getattr(cli, key) < 1:
            parser.error("Evaluation counts and intervals must be positive")
    if cli.conditions:
        if len(cli.conditions) != len(set(cli.conditions)) or any(name not in variants for name in cli.conditions):
            parser.error("Use distinct reference condition names")
        variants = {name: variants[name] for name in cli.conditions}
    if (min(protocol["validation_seed"], protocol["heldout_seed"]) < 0
            or protocol["validation_seed"] + cli.eval_episodes > protocol["heldout_seed"]):
        parser.error("Invalid or overlapping validation and heldout seeds")
    for path in cli.external_status:
        state = json.loads(path.read_text())
        if state["state"] == "STARTING" or any(row["state"] == "QUEUED" for row in state.get("runs", {}).values()):
            parser.error("External experiments must have no queued jobs")
    # Keep output outside source trees so snapshots cannot recursively copy themselves.
    for source in (root, *(PROJECT_ROOT / name for name in ("algorithms", "heterogrid", "configs"))):
        if output == source.resolve() or source.resolve() in output.parents:
            parser.error("Output must not be inside a source directory")
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    minimum_free = int(cli.minimum_free_disk_gb * 1024 ** 3)
    jobs = len(variants) * len(cli.seeds)
    estimated_runs_mb = jobs * (3 * cli.checkpoint_estimate_mb + cli.logs_replays_estimate_mb)
    projected = max(cli.projected_storage_mb, estimated_runs_mb) * 1024 ** 2
    snapshot_bytes = sum(p.stat().st_size for source in (root, *(PROJECT_ROOT / name for name in
                         ("algorithms", "heterogrid", "configs"))) for p in source.rglob("*") if p.is_file())
    if shutil.disk_usage(probe).free < minimum_free + projected + snapshot_bytes:
        parser.error("Insufficient disk: require safety floor + projected storage + source snapshots")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    for name, value in variants.items():
        write_atomic(output / "configs" / (name + ".json"), value)
    project, external, provenance = snapshot_sources(PROJECT_ROOT, root, output)
    if shutil.disk_usage(output).free < minimum_free + projected:
        parser.error("Insufficient disk after freezing sources; existing files have been retained")
    manifest = {"algorithm": "ma2e", "algorithm_label": "QMIX + MA2E", "created": timestamp(),
                "python": sys.executable, "repository": str(project), "ma2e_root": str(external),
                "source_ma2e_root": str(root), "conditions": list(variants),
                "condition_references": {name: sources[name] for name in variants},
                "config_hashes": {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                                  for name, value in variants.items()},
                "seeds": cli.seeds, "jobs": len(variants) * len(cli.seeds),
                "workers": min(cli.workers, len(variants) * len(cli.seeds)),
                "team_sizes": {name: len(value["red"]) for name, value in variants.items()},
                "device": cli.device, "gpu_min_free_mb": cli.gpu_min_free_mb,
                "gpu_max_workers": cli.gpu_max_workers, "target_steps_per_seed": cli.steps,
                "validation_seed": protocol["validation_seed"], "heldout_seed": protocol["heldout_seed"],
                "eval_episodes": cli.eval_episodes, "heldout_episodes": cli.heldout_episodes,
                "eval_interval": cli.eval_interval, "log_interval": cli.log_interval,
                "replay_episodes": cli.replay_episodes, "algorithm_args": settings(external, "cpu" if cli.device == "auto" else cli.device),
                "algorithm_args_device_policy": "Template only; launch selects and records the actual device per attempt",
                "external_status_paths": [str(p.resolve()) for p in cli.external_status],
                "minimum_free_disk_bytes": minimum_free, "projected_storage_bytes": projected,
                "storage_estimate": {"jobs": jobs, "checkpoints_per_run": 3,
                                     "full_checkpoint_mb": cli.checkpoint_estimate_mb,
                                     "sparse_compressed_logs_and_replays_per_run_mb": cli.logs_replays_estimate_mb,
                                     "runs_total_mb": estimated_runs_mb,
                                     "admission_floor_mb": cli.projected_storage_mb,
                                     "policy": "Estimate, not a quota; calibrate with a full-state smoke checkpoint"},
                "source_provenance": "_code/provenance.json",
                "ma2e_source_hashes": provenance["ma2e_source_hashes"],
                "environment_source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (project / "heterogrid").glob("*.py")},
                "protocol_overrides": {key: getattr(cli, key) for key in
                                       ("eval_episodes", "heldout_episodes", "eval_interval", "log_interval")
                                       if getattr(cli, key) != protocol[key]},
                "initialization": "independent fresh seeds; no reference weights or data",
                "execution_policy": "own local observation/action history only; no privileged teammate histories",
                "integration_corrections": CORRECTIONS,
                "exploration_schedule": "native pretraining epsilon=1; RL annealing uses RL steps only, total budget includes pretraining",
                "checkpoint_policy": "initial, best validation, latest; networks, optimizer, counters; no replay/RNG exact resume",
                "budget_policy": "pretraining collection included in team steps; validation/test excluded; finish current collection batch",
                "storage_policy": "no trajectories/ or evaluations/; compressed episode logs; bounded replay count",
                "device_policy": "launch: nvidia-smi probe + atomic shared leases + per-job memory reservation; CUDA OOM restarts fresh on CPU"}
    command = ["nohup", sys.executable, "-u", "-m", "algorithms.ma2e.campaign", "--manage", "--output", str(output)]
    manifest["launch_command"] = command
    write_atomic(output / "experiment.json", manifest)
    write_atomic(output / "status.json", {"state": "STARTING", "runs": {}})
    (output / "README.md").write_text(
        "# QMIX + MA2E 实验\n\n"
        "配置来自已完成的 MAPPO B/C、M/N、D 实验，不加载其权重。\n"
        "实际预算、评估协议和覆盖范围见 experiment.json；预训练交互计入团队步，评估除外。\n"
        "执行仅使用自身局部历史；不是普通 QMIX。仅保留 initial/best/latest 检查点。\n"
        "不生成 trajectories/ 或 evaluations/，逐局统计压缩保存。\n"
        "_code/provenance.json 记录源码来源及逐文件 SHA-256；全部 worker 使用冻结代码。\n"
        "launch 动态选择设备并管理 GPU 租约；CUDA OOM 必须安全地从头 CPU 重启。\n"
        "status.json 为进度，nohup.log 为管理日志；结束后复算审计、汇总曲线和均值及 SEM。\n",
        encoding="utf-8")
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "manager_pid": process.pid, "jobs": manifest["jobs"],
                      "workers": manifest["workers"], "target_steps_per_seed": cli.steps}), flush=True)


if __name__ == "__main__":
    main()
