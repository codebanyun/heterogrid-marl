"""Rerun MAPPO and SMPE communication controls with a fixed blue network."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.progress import align_steps
from algorithms.common.study_queue import run_queue
from algorithms.mappo import matrix as mappo
from algorithms.smpe import matrix as smpe
from heterogrid import CommunicationConfig, EnvConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ("mappo", "smpe")
CONDITION_NAMES = ("B0_baseline", "C1_none", "C2_full", "C3_distance5", "C4_distance15",
                   "C5_random25", "C6_random50")


def conditions(base):
    if base.map_size != (24, 24, 6) or len(base.red) != 6 or len(base.blue) != 6 or base.opponent != "rule":
        raise ValueError("The reference must use the 24x24x6, 6v6 fixed-rule baseline")
    blue = CommunicationConfig("distance", 10.0, 0.0)
    baseline = replace(base, communication="distance", comm_radius=10.0, dropout_probability=0.0,
                       red_communication=blue, blue_communication=blue)
    red_settings = (blue, CommunicationConfig("none", 10.0, 0.0), CommunicationConfig("full", 10.0, 0.0),
                    CommunicationConfig("distance", 5.0, 0.0), CommunicationConfig("distance", 15.0, 0.0),
                    CommunicationConfig("full", 10.0, 0.25), CommunicationConfig("full", 10.0, 0.50))
    return {name: replace(baseline, red_communication=red) for name, red in zip(CONDITION_NAMES, red_settings)}


def training_command(manifest, output, job):
    algorithm, label = job["condition"].split("/")
    root = output / algorithm
    local_job = dict(job, condition=label)
    builder = mappo.training_command if algorithm == "mappo" else smpe.training_command
    settings = manifest["experiments"][algorithm]
    command = builder(settings, root, local_job)
    for option, key in (("--eval-seed", "validation_seed"), ("--heldout-seed", "heldout_seed")):
        command[command.index(option) + 1] = str(settings[key])
    if settings.get("compact_episode_log", False):
        command.append("--compact-episode-log")
    return command


def audit_run(path, recompute=False):
    algorithm = json.loads((path / "config.json").read_text())["name"]
    if algorithm == "mappo":
        from algorithms.mappo.audit import audit
    elif algorithm == "smpe":
        from algorithms.smpe.audit import audit
    else:
        raise ValueError("Unexpected training algorithm: " + algorithm)
    return audit(path, recompute=recompute)


def summarize(path, runs):
    algorithm = path.parent.name
    if algorithm == "mappo":
        from algorithms.mappo.study import summarize as summarize_runs
    else:
        summarize_runs = smpe.summarize
    result = summarize_runs(path, runs)
    with (path / "RESULTS.md").open("a", encoding="utf-8") as file:
        file.write("\n本组只改变红方通信；蓝方固定 distance、半径 10、失联概率 0。"
                   "B0 与其他配置均使用新的独立通信随机流，并从头训练，不混用旧基准。\n")
    return result


def write_comparison(output):
    import math
    import statistics
    manifest = json.loads((output / "experiment.json").read_text())
    density = manifest.get("study_group") == "density"
    labels = manifest.get("experiments", {}).get("mappo", {}).get("conditions", CONDITION_NAMES)
    title = "密度受控规模实验结果" if density else "固定蓝方通信实验结果"
    protocol = ("D0–D3 近似保持人数/体积密度；双方距离半径 10、无失联，开启参考奖励归一化和双方诊断。"
                if density else "仅改变红方通信，蓝方始终为 distance、半径 10、失联概率 0。")
    lines = ["# " + title, "", protocol,
             "胜率单位为百分比，使用最终模型的贪心独立测试结果，报告训练种子的均值 ±SEM。",
             "评估种子和局数见 experiment.json；共同基准从头训练，不合并旧基准。", "",
             "| 配置 | MAPPO | SMPE |", "|---|---:|---:|"]
    for label in labels:
        values = []
        for algorithm in ALGORITHMS:
            root = output / algorithm
            wins = [100 * json.loads((root / label / ("seed_" + str(seed)) / "summary.json").read_text())
                    ["heldout"]["latest"]["greedy"]["win_rate"] for seed in manifest["seeds"]]
            sem = statistics.stdev(wins) / math.sqrt(len(wins)) if len(wins) > 1 else 0.0
            values.append("%.2f ±%.2f" % (statistics.mean(wins), sem))
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += ["", "SEM 为训练种子间样本标准差除以种子数平方根，不是测试局数上的置信区间。",
              ("地图取整会带来密度误差，距离、边界和 200 步时限仍影响难度；不能直接与旧 N 组作纯密度归因。"
               if density else "C5/C6 为红方在线节点全连接并随机离线，不是距离通信叠加失联。"), ""]
    text = "\n".join(lines)
    (output / "RESULTS.md").write_text(text, encoding="utf-8")
    all_summaries = json.loads((output / "summary.json").read_text())
    for algorithm in ALGORITHMS:
        root = output / algorithm
        write_atomic(root / "summary.json", {label: all_summaries[algorithm + "/" + label] for label in labels})
        (root / "RESULTS.md").write_text(text, encoding="utf-8")


def manage(output):
    manifest = json.loads((output / "experiment.json").read_text())
    if manifest.get("study_group") == "density":
        from algorithms.common.density_study import audit_density, summarize_density
        run_queue(output, training_command, audit_density, summarize_density)
    else:
        run_queue(output, training_command, audit_run, summarize)
    write_comparison(output)


def launch(args, parser):
    density = args.profile == "density"
    reference = args.reference.resolve()
    try:
        original = json.loads((reference / "experiment.json").read_text())
        if original["algorithm"] != "mappo" or json.loads((reference / "status.json").read_text())["state"] != "COMPLETED":
            raise ValueError("Reference must be a completed MAPPO experiment")
        base = EnvConfig.load(reference / "configs/B0_baseline.json")
        if density:
            from algorithms.common.density_study import conditions as density_conditions
            variants = density_conditions(base)
        else:
            variants = conditions(base)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if max(original["validation_seed"], original["heldout_seed"]) < min(
            original["validation_seed"] + args.eval_episodes, original["heldout_seed"] + args.heldout_episodes):
        parser.error("Validation and independent test seeds must not overlap")
    backend_mappo = mappo.validate_backend(parser, args.mappo_root)
    backend_smpe = smpe.validate_backend(parser, args.smpe_root)
    budget = align_steps(args.steps)
    options = {name: align_steps(getattr(args, name)) for name in ("eval_interval", "save_interval", "log_interval")}
    roots = {name: args.output_root.resolve() / name / args.name for name in ALGORITHMS}
    output = args.output_root.resolve() / args.profile / args.name
    if any(path.exists() for path in [output, *roots.values()]):
        parser.error("Experiment paths must not already exist")
    probe = args.output_root.resolve()
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < args.minimum_free_disk_gb * 1024 ** 3:
        parser.error("Insufficient free disk space")
    output.mkdir(parents=True)
    project, raw_smpe = smpe.freeze_code(output, backend_smpe)
    raw_mappo = output / "_code/mappo"
    shutil.copytree(backend_mappo / "onpolicy", raw_mappo / "onpolicy",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (project / "heterogrid").glob("*.py")}
    common = {"schema_version": 1, "created": timestamp(), "repository": str(project), "python": sys.executable,
              "reference": str(reference), "source_repository": str(PROJECT_ROOT), "seeds": args.seeds,
              "conditions": list(variants), "jobs": len(variants) * len(args.seeds), "workers": args.workers,
              "device": "cpu", "actual_steps_per_seed": budget, "target_steps_per_seed": budget,
              "requested_steps_per_seed": args.steps, "validation_seed": original["validation_seed"],
              "heldout_seed": original["heldout_seed"], "eval_episodes": args.eval_episodes,
              "heldout_episodes": args.heldout_episodes, "replay_episodes": args.replay_episodes, **options,
              "mappo_root": str(raw_mappo), "smpe_root": str(raw_smpe),
              "source_mappo_root": str(backend_mappo), "source_smpe_root": str(backend_smpe),
              "environment_source_hashes": source_hashes,
              "config_hashes": {name: smpe.canonical_hash(config.to_dict()) for name, config in variants.items()},
              "communication_scope": "red only; blue fixed distance/radius=10/dropout=0; independent team RNGs",
              "blue_communication": vars(CommunicationConfig()),
              "baseline_policy": "fresh B0 under independent team communication RNGs; never reuse legacy B0",
              "initialization": "independent from scratch; no old checkpoints or smoke weights",
              "primary_result": "latest fixed-budget model; best validation model supplementary",
              "coordinator": str(output), "status_scope": "shared queue covering both algorithms",
              "minimum_free_disk_bytes": int(args.minimum_free_disk_gb * 1024 ** 3),
              "no_trajectories": True, "no_evaluation_files": True}
    common["study_group"] = args.profile
    if density:
        common.update(communication_scope="both fixed distance/radius=10/dropout=0; independent team RNGs",
                      baseline_policy="fresh D0 with reference reward normalization and bilateral diagnostics",
                      compact_episode_log=True, reward_normalization="reference", diagnostics_enabled=True,
                      density_definition="initial team size / product(map dimensions); approximately constant",
                      ranges_policy="vision, range, radius, movement and horizon remain unchanged")
    experiments = {}
    for algorithm, root in roots.items():
        (root / "configs").mkdir(parents=True)
        for name, config in variants.items():
            write_atomic(root / "configs" / (name + ".json"), config.to_dict())
        experiments[algorithm] = dict(common, algorithm=algorithm)
        if algorithm == "smpe":
            experiments[algorithm]["smpe_args"] = smpe.hyperparameters(raw_smpe, "cpu")
            experiments[algorithm]["budget_policy"] = "complete episode batches; overshoot recorded per seed"
        else:
            experiments[algorithm].update(num_envs=4, rollout_length=128, hidden_size=64, ppo_epochs=5, mini_batches=4)
        write_atomic(root / "experiment.json", experiments[algorithm])
        (output / algorithm).symlink_to(os.path.relpath(root, output), target_is_directory=True)
        for filename in ("status.json", "nohup.log"):
            (root / filename).symlink_to(os.path.relpath(output / filename, root))
        description = ["# " + algorithm.upper() + " 固定蓝方通信实验", "",
                       "本次重跑 B0 和 C1–C6，每个配置使用三个独立训练种子；实际预算和种子以 experiment.json 为准。",
                       "蓝方固定规则策略，通信固定 distance、半径 10、失联概率 0。仅改变红方通信配置。",
                       "双方均显式设置通信，使用独立随机流；B0 从头重跑，不复用旧实验基准。",
                       "C1：无通信；C2：全通信；C3/C4：距离半径 5/15；C5/C6：全连接在线节点、失联概率 25%/50%。",
                       "地图、阵容、奖励及其他环境设置取自参考实验 B0，两个算法使用完全相同的新配置。",
                       "各算法保持原实验训练参数；每次约 5M 环境步，验证 64 局，独立测试 128 局。",
                       "以上为正式实验默认值，短程检查的实际参数以 experiment.json 为准。",
                       "configs/ 保存完整配置；各配置 seed_N/ 保存指标、模型、源码快照和少量最终回放。",
                       "不保存 trajectories/ 和 evaluations/。完成后生成均值 ±SEM 曲线、审计及 RESULTS.md。",
                       "status.json 和 nohup.log 为共享队列的链接，统计两个算法合计任务，不是单算法任务数。",
                       "源码冻结在 experiment.json 的 repository、mappo_root 和 smpe_root 指向的目录。",
                       "停止全部任务：向共享 status.json 中 manager_pid 发送 SIGTERM。", ""]
        if density:
            description = ["# " + algorithm.upper() + " 密度受控规模实验", "",
                           "D0：6v6、24×24×6；D1：3v3、19×19×5；D2：9v9、27×27×7；D3：12v12、30×30×8。",
                           "双方角色比例 1:1:1，规则蓝方，200 步；双方 distance、半径 10、无失联，通信随机流独立。",
                           "参考奖励归一化仅缩放有效伤害与击杀：参考人数 6、总 HP 480；其余奖励不缩放。",
                           "启用双方诊断，仅用于记录，不能输入执行 actor；视野、射程、移动和通信半径不随地图缩放。",
                           "每个算法默认三个 seed、每次约 5M 环境步，验证 64 局、独立测试 128 局；实际设置见 experiment.json。",
                           "D0 从头训练，不复用旧基准；两个算法保持各自网络、优化器与更新设置。",
                           "training_episodes.jsonl.gz 无损保存全部逐局统计和双方诊断，不再保存重复的整数组。",
                           "各配置 diagnostics.csv/diagnostics.json 汇总最终贪心独立测试的双方诊断与有效样本数。",
                           "各配置 figures/ 为训练种子均值 ±SEM 曲线，RESULTS.md 为结果报告。",
                           "status.json 与 nohup.log 链接到两个算法共享队列；_code/ 冻结源码位于 coordinator 指向的目录。",
                           "不保存 trajectories/ 和 evaluations/，仅保留少量最终回放和若干模型。",
                           "密度仅近似一致，初始距离与边界效应仍可能变化，不与旧 N 组直接作纯密度归因。",
                           "停止全部任务：向共享 status.json 中 manager_pid 发送 SIGTERM。", ""]
        (root / "README.md").write_text("\n".join(description), encoding="utf-8")
    groups = [algorithm + "/" + label for label in variants for algorithm in ALGORITHMS]
    manifest = dict(common, algorithm="mappo+smpe", conditions=groups, jobs=len(groups) * len(args.seeds),
                    workers=min(args.workers, len(groups) * len(args.seeds)), experiments=experiments)
    if density:
        manifest["team_sizes"] = {algorithm + "/" + name: len(config.red)
                                  for name, config in variants.items() for algorithm in ALGORITHMS}
    module = "algorithms.common.density_study" if density else "algorithms.common.communication_study"
    command = ["nohup", sys.executable, "-u", "-m", module,
               "--manage", "--output", str(output)]
    manifest["launch_command"] = command
    write_atomic(output / "experiment.json", manifest)
    write_atomic(output / "status.json", {"state": "STARTING", "runs": {}})
    heading = "# 密度受控规模实验共享队列" if density else "# 固定蓝方通信实验共享队列"
    purpose = ("本目录管理 MAPPO 与 SMPE 的 D0–D3 实验，共享并发资源。" if density else
               "本目录管理 MAPPO 与 SMPE 的 B0、C1–C6 重跑，共享并发资源。")
    (output / "README.md").write_text(heading + "\n\n" + purpose + "\n"
                                     "mappo/ 和 smpe/ 链接到各算法实际数据目录。\n"
                                     "双方通信随机流独立，蓝方固定 distance、半径 10、失联概率 0。\n"
                                     "experiment.json 为冻结协议，status.json 为共享进度，_code/ 为冻结源码。\n"
                                     "完成后 RESULTS.md 汇总新基准下的对比，不合并旧基准。\n", encoding="utf-8")
    with (output / "nohup.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"output": str(output), "algorithm_outputs": {k: str(v) for k, v in roots.items()},
                      "manager_pid": process.pid, "jobs": manifest["jobs"], "workers": manifest["workers"],
                      "target_steps_per_seed": budget}), flush=True)


def main(profile="communication"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mappo-root", type=Path, default=os.environ.get("MAPPO_ROOT"))
    parser.add_argument("--smpe-root", type=Path, default=os.environ.get("SMPE_ROOT"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    suffix = "_density_5m" if profile == "density" else "_blue_fixed_5m"
    parser.add_argument("--name", default=datetime.now().strftime("%Y%m%d_%H%M%S") + suffix)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--replay-episodes", type=int, default=1 if profile == "density" else 2)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--save-interval", type=int, default=2000000)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=2.0)
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.profile = profile
    if os.name != "posix" or shutil.which("nohup") is None:
        parser.error("This shared detached queue requires POSIX, symlinks and nohup")
    if args.manage:
        if args.output is None:
            parser.error("--manage requires --output")
        manage(args.output.resolve())
        return
    if args.reference is None:
        parser.error("--reference must point to the completed MAPPO sensitivity study")
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        parser.error("--name must be a single directory name")
    if min(args.steps, args.workers, args.eval_episodes, args.heldout_episodes,
           args.eval_interval, args.save_interval, args.log_interval) <= 0 or args.replay_episodes < 0:
        parser.error("Budgets, workers and evaluation counts must be positive")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or args.minimum_free_disk_gb <= 0:
        parser.error("Use distinct nonnegative seeds and a positive disk threshold")
    launch(args, parser)


if __name__ == "__main__":
    main()
