"""Launch a frozen SMPE replication of an existing MAPPO sensitivity matrix."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.study import summarize_runs
from algorithms.common.study_queue import run_queue
from algorithms.smpe.settings import PROJECT_ROOT, add_backend_argument, hyperparameters, validate_backend
from heterogrid import EnvConfig


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_configs(reference):
    manifest = json.loads((reference / "experiment.json").read_text())
    if manifest["algorithm"] != "mappo" or len(manifest["conditions"]) != 13:
        raise ValueError("Reference must contain the 13-condition MAPPO matrix")
    result = {}
    for name in manifest["conditions"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid condition name")
        value = json.loads((reference / "configs" / (name + ".json")).read_text())
        EnvConfig.from_dict(value)
        result[name] = value
    if len(result) != 13:
        raise ValueError("Condition names must be distinct")
    return manifest, result


def verify_environment_sources(reference, manifest):
    source_run = reference / manifest["conditions"][0] / ("seed_" + str(manifest["seeds"][0])) / "run.json"
    sources = json.loads(source_run.read_text())["experiment"]["sources"]
    checked = {}
    for source in sources:
        original = Path(source["path"])
        if original.parent.name == "heterogrid" and original.suffix == ".py":
            current = PROJECT_ROOT / "heterogrid" / original.name
            digest = hashlib.sha256(current.read_bytes()).hexdigest()
            if digest != source["sha256"]:
                raise ValueError("Environment changed since the MAPPO reference: " + original.name)
            checked[original.name] = digest
    if not {"core.py", "config.py", "env.py", "smac.py"} <= set(checked):
        raise ValueError("Reference lacks the required environment source provenance")
    return checked


def freeze_code(output, backend):
    frozen = output / "_code"
    project = frozen / "project"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("algorithms", "heterogrid", "configs"):
        shutil.copytree(PROJECT_ROOT / name, project / name, ignore=ignore)
    raw = frozen / "smpe_raw"
    for name in ("controllers", "learners", "model_learners", "modules", "components", "utils", "config"):
        shutil.copytree(backend / name, raw / name, ignore=ignore)
    for name in ("run.py", "main.py", "README.md", "LICENSE", "requirements.txt", "__init__.py"):
        source = backend / name
        if source.is_file():
            shutil.copy2(source, raw / name)
    return project, raw


def training_command(manifest, output, job):
    command = [manifest["python"], "-u", "-m", "algorithms.smpe.train", "--smpe-root", manifest["smpe_root"],
               "--config", str(output / "configs" / (job["condition"] + ".json")),
               "--output", job["path"], "--seed", str(job["seed"]), "--device", manifest["device"],
               "--steps", str(manifest["target_steps_per_seed"]),
               "--eval-seed", str(manifest["validation_seed"]), "--heldout-seed", str(manifest["heldout_seed"])]
    for option in ("eval_interval", "save_interval", "log_interval", "eval_episodes", "heldout_episodes", "replay_episodes"):
        command += ["--" + option.replace("_", "-"), str(manifest[option])]
    return command


def summarize(output, runs):
    result = summarize_runs(output, runs, "SMPE",
                            "实现复用外部 SMPE 的循环 actor、集中式 critic、状态 VAE、过滤器和内在探索奖励。")
    path = output / "RESULTS.md"
    with path.open("a", encoding="utf-8") as file:
        file.write("\n曲线横轴为统一的评估预算阈值，实际评估在完整对局批次后进行，偏移小于一个批次的最大步数。\n"
                   "每个 seed 的实际评估步数见 metrics.json 的 test_actual_environment_steps；最终实际步数见 summary.json。\n"
                   "训练及最终回报仅统计环境奖励，未混入 SMPE 内在奖励；内在奖励单列在训练指标中。\n")
    return result


def manage(output):
    from algorithms.smpe.audit import audit
    return run_queue(output, training_command, audit, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--heldout-episodes", type=int)
    parser.add_argument("--replay-episodes", type=int, default=2)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=2.0)
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix" or shutil.which("nohup") is None:
        parser.error("This detached launcher requires POSIX and nohup")
    if args.manage:
        if args.output is None:
            parser.error("--manage requires --output")
        manage(args.output.resolve())
        return
    backend = validate_backend(parser, args.smpe_root)
    if args.reference is None:
        parser.error("--reference must point to a completed MAPPO sensitivity experiment")
    reference = args.reference.resolve()
    try:
        original, variants = reference_configs(reference)
        source_hashes = verify_environment_sources(reference, original)
        if json.loads((reference / "status.json").read_text())["state"] != "COMPLETED":
            raise ValueError("The reference experiment is not complete")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    seeds = args.seeds if args.seeds is not None else original["seeds"]
    target = args.steps if args.steps is not None else original["actual_steps_per_seed"]
    validation = args.eval_episodes if args.eval_episodes is not None else original["eval_episodes"]
    heldout = args.heldout_episodes if args.heldout_episodes is not None else original["heldout_episodes"]
    if min(target, args.workers, validation, heldout) <= 0 or args.replay_episodes < 0 or args.minimum_free_disk_gb <= 0:
        parser.error("Budgets, workers, evaluation counts and disk threshold must be positive")
    if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
        parser.error("Use distinct, nonnegative seeds")
    if validation > original["heldout_seed"] - original["validation_seed"]:
        parser.error("Validation and held-out seeds must not overlap")
    intervals = {name: getattr(args, name) if getattr(args, name) is not None else original[name]
                 for name in ("eval_interval", "save_interval", "log_interval")}
    if min(intervals.values()) <= 0:
        parser.error("Evaluation, save and log intervals must be positive")
    label = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sensitivity_5m"
    output = (args.output or PROJECT_ROOT / "output/smpe" / label).resolve()
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < args.minimum_free_disk_gb * 1024 ** 3:
        parser.error("Insufficient disk space")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    for name, value in variants.items():
        write_atomic(output / "configs" / (name + ".json"), value)
    repository, raw = freeze_code(output, backend)
    settings = hyperparameters(raw, args.device)
    manifest = {"schema_version": 1, "algorithm": "smpe", "created": timestamp(), "reference": str(reference),
                "source_repository": str(PROJECT_ROOT), "source_smpe_root": str(backend),
                "repository": str(repository), "smpe_root": str(raw), "python": sys.executable,
                "conditions": list(variants), "config_hashes": {name: canonical_hash(value) for name, value in variants.items()},
                "environment_source_hashes": source_hashes, "seeds": seeds, "jobs": len(variants) * len(seeds),
                "team_sizes": {name: len(value["red"]) for name, value in variants.items()},
                "workers": min(args.workers, len(variants) * len(seeds)), "device": args.device,
                "target_steps_per_seed": target, "budget_policy": "complete episode batches; actual overshoot is recorded per run",
                "initialization": "independent from scratch; no MAPPO or smoke-check weights",
                "primary_result": "latest fixed-budget checkpoint; best validation model supplementary",
                "validation_seed": original["validation_seed"], "heldout_seed": original["heldout_seed"],
                "eval_episodes": validation, "heldout_episodes": heldout, "replay_episodes": args.replay_episodes,
                **intervals, "smpe_args": settings,
                "minimum_free_disk_bytes": int(args.minimum_free_disk_gb * 1024 ** 3),
                "code_policy": "frozen project and backend for every queued seed; no writes to original SMPE checkout"}
    command = ["nohup", sys.executable, "-u", "-m", "algorithms.smpe.matrix", "--manage", "--output", str(output)]
    manifest["launch_command"] = command
    write_atomic(output / "experiment.json", manifest)
    write_atomic(output / "status.json", {"state": "STARTING", "runs": {}})
    description = ["# SMPE 敏感性实验", "", "本组配置直接复制自 experiment.json 中 reference 指向的 MAPPO 实验。",
                   "共 13 个配置，每个配置默认三个独立训练种子；具体预算和种子以 experiment.json 为准。",
                   "configs/ 保存完整环境配置；地图、阵容、通信、奖励及评估种子与参考实验一致。",
                   "B0：基准；M1–M3：地图；N1–N3：阵容；C1–C6：通信。通信同时影响红蓝双方。",
                   "SMPE 使用外部仓库原始 actor-critic、VAE、过滤器和内在奖励，基于 smpe_lbf.yaml 预设。",
                   "完整状态使用原代码的 sc2 分支；实际环境是 HeteroGrid，不调用 StarCraft。",
                   "策略为隐藏维度 128 的 GRU，潜变量维度 32，10 个对局构成一个更新批次。",
                   "按原生 LBF 更新计数器调度 VAE 和过滤器，不减少模型训练轮数。",
                   "评分后的超时视为终局，不跨终局 bootstrap；内在奖励只用于训练，不计入环境回报。",
                   "每项训练达到预算后完成当前批次，实际步数可能略超预算；最终值见各 seed 的 summary.json。",
                   "曲线按相同评估预算阈值对齐，实际评估步数另存在 test_actual_environment_steps 指标中。",
                   "_code/ 冻结本次项目与算法源码，排队任务不会受到后续代码编辑的影响。",
                   "status.json 保存进度、状态及进程；nohup.log 为管理日志；各配置 logs/ 为任务日志。",
                   "seed_N/ 保存模型、指标、逐局统计、源码快照及少量最终回放。",
                   "模型完整保存 actor、两个 critic 及目标网络、VAE、过滤器、归一化统计和哈希计数。",
                   "不保存 trajectories/ 或 evaluations/。训练完成后自动审计并生成均值 ±SEM 曲线和报告。",
                   "停止整组任务：向 status.json 中 manager_pid 发送 SIGTERM。当前不支持精确断点续训。", ""]
    (output / "README.md").write_text("\n".join(description), encoding="utf-8")
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=repository, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "manager_pid": process.pid, "jobs": manifest["jobs"],
                      "workers": manifest["workers"], "device": args.device, "target_steps_per_seed": target}), flush=True)


if __name__ == "__main__":
    main()
