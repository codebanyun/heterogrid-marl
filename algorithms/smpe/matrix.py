"""Launch a frozen SMPE replication of a completed sensitivity or density study."""
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
    density = manifest.get("study_group") == "density"
    expected = 4 if density else 13
    if manifest["algorithm"] not in ("mappo", "smpe") or len(manifest["conditions"]) != expected:
        raise ValueError("Reference must contain a complete MAPPO sensitivity or MAPPO/SMPE density matrix")
    result = {}
    for name in manifest["conditions"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid condition name")
        value = json.loads((reference / "configs" / (name + ".json")).read_text())
        EnvConfig.from_dict(value)
        result[name] = value
    if len(result) != expected:
        raise ValueError("Condition names must be distinct")
    return manifest, result


def combine_communication_reference(variants, reference):
    manifest = json.loads((reference / "experiment.json").read_text())
    expected = {"B0_baseline", "C1_none", "C2_full", "C3_distance5", "C4_distance15", "C5_random25", "C6_random50"}
    if manifest["algorithm"] != "smpe" or set(manifest["conditions"]) != expected:
        raise ValueError("Communication reference must contain the seven completed SMPE fixed-blue conditions")
    if json.loads((reference / "status.json").read_text())["state"] != "COMPLETED":
        raise ValueError("Communication reference is not complete")
    result = dict(variants)
    for name in manifest["conditions"]:
        value = json.loads((reference / "configs" / (name + ".json")).read_text())
        config = EnvConfig.from_dict(value)
        blue = config.communication_for("blue")
        if (blue.mode, blue.radius, blue.dropout_probability) != ("distance", 10, 0):
            raise ValueError("Blue communication must be fixed distance/radius=10/dropout=0")
        if config.blue_communication is None or config.red_communication is None:
            raise ValueError("Communication controls must explicitly isolate both teams")
        result[name] = value
    return manifest, result


def verify_replay_compatibility(reference, names):
    import numpy as np
    from heterogrid import SMACAdapter
    manifest = json.loads((reference / "experiment.json").read_text())
    checked = []
    for name in names:
        path = reference / name / ("seed_" + str(manifest["seeds"][0]))
        replay = json.loads((path / "replays/heldout_latest_greedy/episode_000.json").read_text())
        env = SMACAdapter(config=EnvConfig.load(reference / "configs" / (name + ".json")))
        try:
            env.reset(seed=replay["evaluation"]["seed"])
            for index, frame in enumerate(replay["frames"]):
                snapshot = env.env.core.snapshot()
                # Historical snapshots lack the newer communication metadata fields.
                assert {key: snapshot[key] for key in frame} == frame, (name, index)
                if index < len(replay["transitions"]):
                    transition = replay["transitions"][index]
                    reward, done, _ = env.step(transition["actions"])
                    np.testing.assert_allclose(reward, transition["team_reward"], atol=1e-7)
                    assert done == transition["done"]
            checked.append(dict(condition=name, frames=len(replay["frames"]), seed=replay["evaluation"]["seed"]))
        finally:
            env.close()
    return checked


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
    if manifest.get("compact_episode_log", False):
        command.append("--compact-episode-log")
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


def write_reference_comparison(output):
    import math
    import statistics
    manifest = json.loads((output / "experiment.json").read_text())
    reference = Path(manifest["reference"])
    old = json.loads((reference / "experiment.json").read_text())
    reference_name = "参考 " + old.get("algorithm", "smpe").upper()
    if old.get("algorithm") == "smpe" and not old.get("learner_correction"):
        reference_name += "（修复前）"
    lines = ["# SMPE 与参考实验对照", "",
             "使用最终预算模型的贪心独立测试胜率，单位为百分比，报告训练种子均值及 SEM。",
             "修复前后配置、预算与评估设置见各自 experiment.json；短程预检不代表学习结论。", "",
             "| 配置 | " + reference_name + " | 本次 SMPE（修复后） |", "|---|---:|---:|"]
    for condition in manifest["conditions"]:
        cells = []
        condition_reference = Path(manifest.get("condition_references", {}).get(condition, reference))
        prior = json.loads((condition_reference / "experiment.json").read_text())
        for root, seeds in ((condition_reference, prior["seeds"]), (output, manifest["seeds"])):
            wins = [100 * json.loads((root / condition / ("seed_" + str(seed)) / "summary.json").read_text())
                    ["heldout"]["latest"]["greedy"]["win_rate"] for seed in seeds]
            cell = "%.2f" % statistics.mean(wins)
            cell += " ±%.2f" % (statistics.stdev(wins) / math.sqrt(len(wins))) if len(wins) > 1 else " (n=1)"
            cells.append(cell)
        lines.append("| " + condition + " | " + " | ".join(cells) + " |")
    lines += ["", "本次启用过滤 critic 目标同步修复；参考算法、预算及参数差异以 experiment.json 为准。",
              "每个 seed 的 network.json 记录网络结构，audit.json 检查同步次数及目标参数变化。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def manage(output):
    from algorithms.smpe.audit import audit
    if json.loads((output / "experiment.json").read_text()).get("study_group") == "density":
        from algorithms.common.density_study import audit_density, summarize_density
        result = run_queue(output, training_command, audit_density, summarize_density)
        write_reference_comparison(output)
        return result
    result = run_queue(output, training_command, audit, summarize)
    if json.loads((output / "experiment.json").read_text()).get("condition_references"):
        write_reference_comparison(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--communication-reference", type=Path, help="Replace B/C with completed SMPE fixed-blue configs")
    parser.add_argument("--allow-compatible-environment", action="store_true",
                        help="Verify historical replay behavior when environment source hashes changed")
    parser.add_argument("--external-status", type=Path, nargs="*", default=[],
                        help="Running experiments with no queued jobs; share the total worker limit")
    parser.add_argument("--compact-episode-log", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", help="Optional subset of reference condition names")
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
        parser.error("--reference must point to a completed sensitivity or density experiment")
    reference = args.reference.resolve()
    communication_reference = args.communication_reference.resolve() if args.communication_reference else None
    compatibility = {}
    try:
        original, variants = reference_configs(reference)
        if json.loads((reference / "status.json").read_text())["state"] != "COMPLETED":
            raise ValueError("The reference experiment is not complete")
        condition_references = {name: str(reference) for name in variants}
        if communication_reference:
            if original.get("study_group") == "density":
                raise ValueError("Communication replacement applies to B/C/M/N, not a density reference")
            communication, variants = combine_communication_reference(variants, communication_reference)
            for key in ("seeds", "validation_seed", "heldout_seed", "eval_episodes", "heldout_episodes"):
                if original[key] != communication[key]:
                    raise ValueError("Reference protocols differ: " + key)
            condition_references.update({name: str(communication_reference) for name in communication["conditions"]})
        references = {str(reference): original}
        if communication_reference:
            references[str(communication_reference)] = communication
        for directory, protocol in references.items():
            try:
                verify_environment_sources(Path(directory), protocol)
            except ValueError:
                if not args.allow_compatible_environment:
                    raise
                names = [name for name, source in condition_references.items() if source == directory]
                compatibility[directory] = verify_replay_compatibility(Path(directory), names)
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT / "heterogrid").glob("*.py")}
        for path in args.external_status:
            state = json.loads(path.read_text())
            if state["state"] == "STARTING" or any(row["state"] == "QUEUED" for row in state.get("runs", {}).values()):
                raise ValueError("An external experiment must have no queued jobs to share worker slots safely")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    seeds = args.seeds if args.seeds is not None else original["seeds"]
    target = args.steps if args.steps is not None else original.get("actual_steps_per_seed", original.get("target_steps_per_seed"))
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
    if args.conditions is not None:
        if len(set(args.conditions)) != len(args.conditions) or any(name not in variants for name in args.conditions):
            parser.error("Conditions must be distinct names from the reference")
        variants = {name: variants[name] for name in args.conditions}
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
    manifest.update(compact_episode_log=args.compact_episode_log,
                    external_status_paths=[str(path.resolve()) for path in args.external_status],
                    reference_compatibility_checks=compatibility)
    if communication_reference:
        manifest.update(study_group="fixed_blue_and_sensitivity", condition_references={
            name: condition_references[name] for name in variants}, communication_reference=str(communication_reference))
    if original.get("study_group") == "density":
        manifest.update(study_group="density", compact_episode_log=True)
    for key, protocol in (("reference", original), ("communication_reference", communication if communication_reference else {})):
        if "smpe_args" in protocol:
            previous = protocol["smpe_args"]
            manifest[key + "_hyperparameter_differences"] = {
                name: {"before": previous.get(name), "after": settings.get(name)}
                for name in previous.keys() | settings.keys() if previous.get(name) != settings.get(name)}
    from algorithms.smpe.settings import TARGET_SYNC_FIX
    manifest["learner_correction"] = TARGET_SYNC_FIX
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
    if original.get("study_group") == "density":
        description = ["# SMPE 过滤 critic 目标同步修复验证", "",
                       "配置直接复制 reference 指向的已完成 D 组；场景、种子及预算以 experiment.json 为准。",
                       "仅修复过滤 critic 的目标网络同步，硬更新与软更新均与普通 critic 同步触发。",
                       "保留原始 GRU 128、潜变量 32、双 critic、VAE、过滤器、探索奖励及原生更新调度。",
                       "每个 seed 的 network.json 记录网络结构、维度和参数量；summary.json 记录同步次数及参数变化。",
                       "双方距离通信半径 10、无失联，参考奖励归一化与双方诊断开启。",
                       "全部训练逐局诊断无损保存在 training_episodes.jsonl.gz；旧实验和外部源码不改动。",
                       "_code/ 冻结源码；status.json 为进度，nohup.log 为管理日志。",
                       "完成后自动审计、复算回放，并生成各配置均值及 SEM 曲线、诊断和结果报告。",
                       "停止整组任务：向 status.json 中 manager_pid 发送 SIGTERM。", ""]
    if communication_reference:
        description = ["# SMPE B/C/M/N 目标同步修复验证", "",
                       "B0、C1–C6 复制自固定蓝方通信实验：蓝方 distance、半径 10、无失联，仅改变红方。",
                       "M1–M3、N1–N3 复制自 sensitivity，保留原地图、阵容、奖励及无通信覆盖时的旧随机流语义。",
                       "各配置来源记录在 experiment.json 的 condition_references；不重复运行旧 sensitivity 的 B/C。",
                       "环境源码变化时，启动前逐帧复算历史对局；reference_compatibility_checks 记录抽查结果。",
                       "使用过滤目标 critic 同步修复，网络与训练超参数保持不变，所有任务从头训练。",
                       "各配置三个 seed、约 5M 步，验证 64 局、独立测试 128 局；实际参数见 experiment.json。",
                       "与 external_status_paths 中已全部启动的实验共享 workers 总并发上限，空闲后自动接续。",
                       "逐局日志无损压缩；network.json 记录网络，audit.json 检查目标同步和回放。",
                       "models/ 保存初始、验证最优、最终模型及按保存间隔产生的快照。",
                       "完成后生成各配置曲线及修复前后对照；历史数据不改动。", ""]
    (output / "README.md").write_text("\n".join(description), encoding="utf-8")
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=repository, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "manager_pid": process.pid, "jobs": manifest["jobs"],
                      "workers": manifest["workers"], "device": args.device, "target_steps_per_seed": target}), flush=True)


if __name__ == "__main__":
    main()
