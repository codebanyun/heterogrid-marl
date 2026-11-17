"""Fresh symmetric-no-communication references without replacing historical C1.

Default: C1S only, legacy termination, reuse completed baseline training. An explicit
--early-termination --scenarios C1 C1S prepares a separate matched new-objective
study; never resume historical weights or silently change an active campaign.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from algorithms.common.density_campaign import training_command
from algorithms.common.experiment_log import timestamp, write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, c1_reference_scenarios
from heterogrid import EnvConfig

ALGORITHMS = ("qmix", "qtran", "maa2c", "coma", "dib_v2", "dib_v3")
BASELINE_STUDY = "20261013_fixed_blue_sensitivity_5m"
DIB_STUDY = "20261107_resilient_v3_c1_c6_5m"
COMPUTE_FILES = ("backend.py", "tactical.py", "robust.py", "sequence.py", "auxiliary.py",
                 "alive.py", "returns.py", "train.py", "resilient.py")


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_reference(output, algorithm, scene, seed, config, steps):
    """Reuse complete historical both-none C1, never an early-terminal task.

    Normalize communication representation, not actual settings. The original
    shared `none` and explicit per-team `none` have identical transport semantics;
    diagnostic logging does not affect dynamics, observations or rewards.
    """
    if algorithm not in ("qmix", "qtran", "maa2c", "coma") or scene != "C1S":
        return None
    date = "20261011" if algorithm in ("qmix", "qtran") else "20261012"
    run = Path(output) / algorithm / (date + "_sensitivity_5m") / "C1" / ("seed_" + str(seed))
    summary = read(run / "summary.json")
    if summary is None or summary.get("environment_steps") != steps:
        return None
    if not all((run / "models" / (model + ".pt")).is_file() for model in ("final", "best")):
        return None
    old = EnvConfig.load(run / "env.json")
    ignored = {"communication", "comm_radius", "dropout_probability", "red_communication",
               "blue_communication", "diagnostics_enabled"}
    if any(old.communication_for(team) != config.communication_for(team) for team in ("red", "blue")):
        return None
    a, b = old.to_dict(), config.to_dict()
    if any(a[key] != b[key] for key in a if key not in ignored):
        return None
    for model in ("final", "best"):
        for mode in ("greedy", "sampled"):
            stats = summary.get("heldout", {}).get(model, {}).get(mode, {})
            if stats.get("episodes", 0) < 128:
                return None
    return {"source_run": str(run.resolve()), "source_scene": "C1",
            "new_training_steps": 0, "historical_training_steps": steps,
            "summary_sha256": digest(run / "summary.json"),
            "model_sha256": {model: digest(run / "models" / (model + ".pt")) for model in ("final", "best")},
            "heldout_seed": summary["heldout"]["final"]["greedy"]["base_seed"],
            "equivalence": "same effective both-none communication, task and reward; logging / representation differences only"}


def commands(algorithm, metadata, backend, config, output, args, seed):
    if algorithm in ("dib_v2", "dib_v3"):
        command = [sys.executable, "-u", "-m", "algorithms.dib_mappo.tactical_train",
                   "--dib-root", str(backend), "--config", str(config), "--output", str(output),
                   "--seed", str(seed), "--training-profile",
                   "tactical_v2" if algorithm == "dib_v2" else "resilient_v3",
                   "--tactical-preset", "belief_first", "--storage-policy", "compact"]
    else:
        # Copy the original training CLI, including every non-default hyperparameter.
        metadata = {**metadata, "cli": {**metadata["cli"], "seed": seed}}
        command = training_command(algorithm, metadata, backend, config, output)
    values = {"steps": args.steps, "eval_interval": args.eval_interval,
              "eval_episodes": args.eval_episodes, "heldout_episodes": args.heldout_episodes,
              "eval_seed": args.eval_seed, "heldout_seed": args.heldout_seed}
    if algorithm not in ("dib_v2", "dib_v3"):
        values["replay_episodes"] = 1
    for name, value in values.items():
        flag = "--" + name.replace("_", "-")
        if flag in command:
            command[command.index(flag) + 1] = str(value)
        else:
            command.extend([flag, str(value)])
    return command


def prepare(args):
    output = args.output_root.resolve()
    coordinator = output / "campaigns" / (args.name + "_c1_reference")
    roots = {algo: output / ("dib_mappo" if algo.startswith("dib_") else algo) /
             args.name / (algo if algo.startswith("dib_") else "") for algo in args.algorithms}
    if coordinator.exists() or any(path.exists() for path in roots.values()):
        raise FileExistsError("Fresh study name required; never overwrite existing runs")
    if shutil.disk_usage(output).free < 4 * 1024**3:
        raise OSError("Need at least 4 GiB of disk headroom")
    sources, metadata = {}, {}
    for algo in args.algorithms:
        if algo.startswith("dib_"):
            old = read(output / "dib_mappo" / DIB_STUDY / "experiment.json")
            source_runtime = Path(old["runtime"])
            for name in COMPUTE_FILES:
                if digest(PROJECT_ROOT / "algorithms/dib_mappo" / name) != digest(
                        source_runtime / "algorithms/dib_mappo" / name):
                    raise ValueError("DIB computation differs from frozen v2/v3: " + name)
            source = Path(old["backend"])
        else:
            source = Path(read(output / algo / BASELINE_STUDY / "experiment.json")["backend_snapshot"])
            metadata[algo] = read(output / algo / BASELINE_STUDY / "C1/seed_42/config.json")
        if not source.is_dir():
            raise FileNotFoundError(source)
        sources[algo] = source
    coordinator.mkdir(parents=True)
    runtime = coordinator / "_runtime"
    for name in ("heterogrid", "algorithms", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "scripts").mkdir()
    shutil.copy2(PROJECT_ROOT / "scripts/export_retained_replays.py",
                 runtime / "scripts/export_retained_replays.py")
    backends, copied = {}, {}
    for algo, source in sources.items():
        if source not in copied:
            dest = coordinator / ("_backend_" + algo)
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            copied[source] = dest
        backends[algo] = copied[source]
    tasks = c1_reference_scenarios(early_termination=args.early_termination)
    jobs = []
    for algo, root in roots.items():
        root.mkdir(parents=True)
        (root / "configs").mkdir()
        for scene in args.scenarios:
            write_atomic(root / "configs" / (scene + ".json"),
                         replace(tasks[scene], diagnostics_enabled=True).to_dict())
        write_atomic(root / "experiment.json", {
            "algorithm": algo, "created": timestamp(), "seeds": args.seeds,
            "steps": args.steps, "scenarios": args.scenarios, "fresh_training": True,
            "training_definition": "frozen previous learner; no new hyperparameter selection",
            "tactical_preset": "belief_first .30" if algo.startswith("dib_") else None,
            "termination_protocol": "firepower_v1" if args.early_termination else "legacy_elimination",
            "communication_protocol": {"C1": "red none, blue distance10",
                                       "C1S": "red none, blue none"},
            "validation_seed": args.eval_seed, "heldout_seed": args.heldout_seed,
            "eval_episodes": args.eval_episodes, "heldout_episodes": args.heldout_episodes,
            "primary_model": "latest.pt (fixed budget; final.pt compatibility link)",
            "best_model": "validation-only supplement", "runtime": str(runtime),
            "backend": str(backends[algo]), "coordinator": str(coordinator),
            "original_backend": str(sources[algo]), "previous_C1_not_modified": True,
            "new_cost_reported_separately": True})
    # Long DIB runs first; two algorithms share a parent but have distinct outputs.
    order = [x for x in ("dib_v3", "dib_v2", "qmix", "qtran", "maa2c", "coma") if x in roots]
    for algo in order:
        for scene in args.scenarios:
            for seed in args.seeds:
                root = roots[algo]
                out = root / scene / ("seed_" + str(seed))
                command = commands(algo, metadata.get(algo), backends[algo],
                                   root / "configs" / (scene + ".json"), out, args, seed)
                reused = completed_reference(output, algo, scene, seed, tasks[scene], args.steps)
                jobs.append({"algorithm": algo, "scenario": scene, "seed": seed,
                             "output": str(out), "command": command, "phase": "TRAINING",
                             "status": "REUSED" if reused else "PENDING", "pid": None,
                             **({"reuse": reused} if reused else {})})
    for algo, root in roots.items():
        metadata = read(root / "experiment.json")
        refs = [job["reuse"] for job in jobs if job["algorithm"] == algo and job.get("reuse")]
        metadata.update(fresh_training=not refs, reused_runs=refs,
                        new_training_runs=sum(job["algorithm"] == algo and job["status"] != "REUSED" for job in jobs))
        write_atomic(root / "experiment.json", metadata)
    hashes = {str(path.relative_to(coordinator)): digest(path)
              for base in [runtime, *copied.values()] for path in sorted(base.rglob("*")) if path.is_file()}
    for root in roots.values():
        for path in (root / "configs").glob("*.json"):
            hashes[str(path)] = digest(path)
    state = {"status": "PREPARED", "created": timestamp(), "manager_pid": None,
             "workers": args.workers, "runtime": str(runtime), "jobs": jobs,
             "roots": {algo: str(root) for algo, root in roots.items()},
             "source_sha256": hashes, "seeds": args.seeds, "scenarios": args.scenarios,
             "steps": args.steps, "requested_total_steps": args.steps * sum(j["status"] != "REUSED" for j in jobs),
             "reused_training_runs": sum(j["status"] == "REUSED" for j in jobs),
             "termination_protocol": "firepower_v1" if args.early_termination else "legacy_elimination",
             "validation_seed": args.eval_seed, "heldout_seed": args.heldout_seed,
             "disk_free_at_prepare": shutil.disk_usage(output).free}
    write_atomic(coordinator / "plan.json", state)
    write_atomic(coordinator / "status.json", state)
    (coordinator / "README.md").write_text(
        "# C1S：双方均无通信参考实验\n\n"
        f"共{len(jobs)}个模型记录，其中{state['reused_training_runs']}个复用旧训练，其余从头训练；每次{args.steps:,}步，seed={args.seeds}。\n"
        f"终止协议：{state['termination_protocol']}；原C1与正在运行的实验不改动。\n"
        "默认只新增C1S，不重跑旧C1；只有蓝方通信与旧C1不同，双方仍各自局部感知。\n"
        "DIB-v2/v3各自保持既有网络、更新和辅助参数；belief_first固定，不重新搜索。\n"
        "旧C1若使用不同验证/测试地图，不直接作配对检验；旧结果均保留原目录。\n"
        "四基线旧双方none实验若完整且任务等价则自动复用，绝不因测试种子不同而重复5M训练。\n"
        f"统一验证{args.eval_seed}起、独立测试{args.heldout_seed}起；测试不选模型。\n"
        "workers为额外并行上限，单进程单线程；磁盘保留2GiB，三个异常停止派发。\n"
        "DIB每个run完成后用精确边界状态进入独立测试；无权重初始化或跨场景续训。\n"
        "RESULTS.md/aggregate.json/validation_per_seed.csv定期更新，不绘制未测点。\n"
        "完成后仅保留latest/best权重，final为兼容链接；逐局日志无损gzip，指标/配置/源码不删。\n"
        "每个模型保留首个独立测试种子的完整贪心回放及真实动作；不是补充统计或按胜负挑选。\n"
        "本管理器不自动重启失败基线（它们不支持精确续训）；需人工审计后用新目录重跑。\n",
        encoding="utf-8")
    persist(coordinator, state)
    report(coordinator, state)
    print(coordinator, flush=True)
    return coordinator


def persist(coordinator, state):
    state["heartbeat"] = timestamp()
    write_atomic(coordinator / "status.json", state)
    for algo, root in state["roots"].items():
        jobs = [job for job in state["jobs"] if job["algorithm"] == algo]
        write_atomic(Path(root) / "status.json", {"status": "REUSED" if jobs and all(j["status"] == "REUSED" for j in jobs) else state["status"],
                     "coordinator": str(coordinator), "manager_pid": state["manager_pid"],
                     "heartbeat": state["heartbeat"], "jobs": jobs,
                     "counts": dict(collections.Counter(job["status"] for job in jobs))})


def report(coordinator, state):
    rows, points = [], []
    for job in state["jobs"]:
        out = Path(job.get("reuse", {}).get("source_run", job["output"]))
        summary = read(out / "summary.json")
        trained = read(out / "training_summary.json")
        if job["algorithm"].startswith("dib_"):
            validation = (trained or {}).get("validation", [])
            if not validation:
                validation = [{"step": int(p.parent.name.split("_")[-1]), **read(p)}
                              for p in sorted((out / "evaluations").glob("attempt_*/validation_*/summary.json"))]
        else:
            validation = [{"step": int(p.parent.name.split("_")[-1]), **read(p)}
                          for p in sorted((out / "evaluations").glob("validation_*/summary.json"))]
        for value in validation:
            points.append({"algorithm": job["algorithm"], "scenario": job["scenario"],
                           "seed": job["seed"], "step": value["step"], "win_rate": value["win_rate"]})
        if summary and "heldout" in summary:
            main = summary["heldout"]["final"]["greedy"]
            rows.append({"algorithm": job["algorithm"], "scenario": job["scenario"], "seed": job["seed"],
                         "training_reused": bool(job.get("reuse")), "source_run": str(out),
                         "heldout_seed": main.get("base_seed"),
                         "win_rate": main["win_rate"], "timeout_rate": main["timeout_rate"],
                         "mean_length": main["mean_length"], "training_seconds": summary["training_seconds"],
                         "environment_steps_per_second": summary["environment_steps_per_second"],
                         "sampled_win_rate": summary["heldout"]["final"]["sampled"]["win_rate"],
                         "best_greedy_win_rate": summary["heldout"]["best"]["greedy"]["win_rate"]})
    for name, data, fields in (("independent_test_per_seed.csv", rows,
            ("algorithm", "scenario", "seed", "training_reused", "source_run", "heldout_seed", "win_rate", "timeout_rate", "mean_length", "training_seconds",
             "environment_steps_per_second", "sampled_win_rate", "best_greedy_win_rate")),
            ("validation_per_seed.csv", points, ("algorithm", "scenario", "seed", "step", "win_rate"))):
        with (coordinator / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    aggregate = []
    lines = ["# C1S independent tests", "", "Fixed-budget final model, mean ± SEM across training seeds.",
             "Only complete three-seed results are aggregated; no win-based early stopping.",
             "Reused baseline tests retain their original seeds (200042); new DIB tests use 1800042. Not paired episode comparisons.",
             "Termination protocol: " + state["termination_protocol"], "",
             "| Algorithm | Scene | Complete seeds | Greedy win rate |", "|---|---|---:|---:|"]
    for algo in state["roots"]:
        for scene in state["scenarios"]:
            selected = [row for row in rows if row["algorithm"] == algo and row["scenario"] == scene]
            value = "pending"
            if len(selected) == len(state["seeds"]):
                wins = [row["win_rate"] for row in selected]
                mean = statistics.mean(wins)
                sem = statistics.stdev(wins) / len(wins)**0.5 if len(wins) > 1 else None
                aggregate.append({"algorithm": algo, "scenario": scene, "n": len(wins), "mean": mean, "sem": sem})
                value = f"{100*mean:.2f} ± {100*sem:.2f}%" if sem is not None else f"{100*mean:.2f}% (n=1)"
            lines.append(f"| {algo} | {scene} | {len(selected)} | {value} |")
    write_atomic(coordinator / "aggregate.json", {"results": aggregate, "per_seed": rows})
    (coordinator / "RESULTS.md").write_text("\n".join(lines) + "\n")


def compact_completed(run):
    """Only completed runs: keep exact final/best bytes and all analysis data."""
    run = Path(run)
    if not read(run / "summary.json") or not (run / "models/best.pt").is_file():
        raise ValueError("Cannot compact an incomplete run")
    models = run / "models"
    final, latest = models / "final.pt", models / "latest.pt"
    if not latest.exists():
        final.rename(latest)
        final.symlink_to("latest.pt")
    for path in [models / "initial.pt", *models.glob("step_*.pt")]:
        if path.is_file():
            path.unlink()
    retained = run / "replays/retained"
    if all((retained / kind / filename).is_file() for kind in ("latest", "best")
           for filename in ("replay.json", "actions.jsonl.gz", "provenance.json")):
        for path in (run / "replays").glob("heldout_*/episode_*.json"):
            path.unlink()
        for filename in ("resume.pt", "resume_previous.pt"):
            path = run / filename
            if path.is_file():
                path.unlink()
    source = run / "training_episodes.jsonl"
    if source.exists():
        target = source.with_suffix(".jsonl.gz")
        if target.exists():
            raise FileExistsError(target)
        with source.open("rb") as src, gzip.open(target, "wb", compresslevel=3) as dst:
            shutil.copyfileobj(src, dst)
        with gzip.open(target, "rb") as verify:
            decoded = hashlib.sha256()
            for block in iter(lambda: verify.read(1024*1024), b""):
                decoded.update(block)
        if decoded.hexdigest() != digest(source):
            raise ValueError("Log compression verification failed")
        source.unlink()


def supervise(coordinator, resuming=False):
    coordinator = coordinator.resolve()
    lock = (coordinator / "manager.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(coordinator / "status.json")
    if not resuming and state["status"] != "PREPARED":
        raise ValueError("Not a fresh prepared study; duplicate / weights-only restart refused")
    if resuming:
        if state["status"] != "INTERRUPTED":
            raise ValueError("Only an interrupted campaign can be resumed")
        for job in state["jobs"]:
            if job["status"] in ("REUSED", "COMPLETED"):
                continue
            pid = job.get("pid")
            proc = Path("/proc/" + str(pid) + "/cmdline")
            if proc.exists() and job["output"].encode() in proc.read_bytes():
                raise RuntimeError("Worker already alive; duplicate continuation refused")
            if job["status"] == "INTERRUPTED":
                if not job["algorithm"].startswith("dib_") or not (Path(job["output"]) / "resume.pt").is_file():
                    raise ValueError("Interrupted learner has no exact continuation; refuse weights-only restart")
                if job["phase"] == "TRAINING" and "--resume" not in job["command"]:
                    job["command"].append("--resume")
                job["status"] = "PENDING"
    for relative, sha in state["source_sha256"].items():
        if digest(coordinator / relative) != sha:
            raise ValueError("Frozen source/config changed: " + relative)
    live, stopping = {}, False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for process, _, _ in live.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    child_env.pop("PYTHONPATH", None)
    state.update(status="RUNNING", manager_pid=os.getpid(), started=timestamp())
    last_report = 0
    try:
        while True:
            if shutil.disk_usage(coordinator).free < 2 * 1024**3:
                state["stop_reason"] = "2 GiB disk reserve"; stop()
            for pid, (process, stream, job) in list(live.items()):
                code = process.poll()
                job["progress"] = read(Path(job["output"]) / "progress.json")
                if code is None:
                    continue
                stream.close(); del live[pid]
                job.update(returncode=code, finished=timestamp())
                if code:
                    job["status"] = "INTERRUPTED" if stopping or code == 130 else "FAILED"
                elif job["algorithm"].startswith("dib_") and job["phase"] == "TRAINING":
                    trained = read(Path(job["output"]) / "training_summary.json")
                    if trained is None or trained["steps"] != state["steps"]:
                        raise ValueError("DIB training did not reach its fixed budget")
                    job.update(status="PENDING", phase="EVALUATION")
                elif job["phase"] == "REPLAYS":
                    compact_completed(job["output"])
                    job["status"] = "COMPLETED"
                else:
                    job.update(status="PENDING", phase="REPLAYS")
                print(job["algorithm"], job["scenario"], job["seed"], job["phase"], job["status"], flush=True)
            if sum(job["status"] == "FAILED" for job in state["jobs"]) >= 3:
                state["stop_reason"] = "three worker failures"; stop()
            if not stopping:
                pending = [job for job in state["jobs"] if job["status"] == "PENDING"]
                for job in pending[:max(0, state["workers"] - len(live))]:
                    out = Path(job["output"])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    stream = (out.parent / (out.name + "_" + job["phase"].lower() + ".log")).open("w", buffering=1)
                    if job["phase"] == "REPLAYS":
                        command = [sys.executable, "-c",
                                   "import runpy; runpy.run_path('scripts/export_retained_replays.py', run_name='__main__')",
                                   "--run", job["output"]]
                    else:
                        command = job["command"] + (["--resume", "--evaluate-only"] if job["phase"] == "EVALUATION" else [])
                    process = subprocess.Popen(command, cwd=state["runtime"], env=child_env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    job.update(status="RUNNING", pid=process.pid, started=timestamp())
                    live[process.pid] = (process, stream, job)
                    print("Started", job["algorithm"], job["scenario"], job["seed"], process.pid, flush=True)
            persist(coordinator, state)
            if time.monotonic() - last_report > 120 or not live:
                report(coordinator, state); last_report = time.monotonic()
            if not live and (stopping or not any(job["status"] == "PENDING" for job in state["jobs"])):
                break
            time.sleep(5)
        state["status"] = "INTERRUPTED" if stopping else "FAILED" if any(
            job["status"] == "FAILED" for job in state["jobs"]) else "COMPLETED"
    except BaseException as error:
        state.update(status="FAILED", error=repr(error)); stop(); raise
    finally:
        if live:
            stop()
            for process, stream, job in live.values():
                try:
                    process.wait(timeout=50)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                stream.close(); job["status"] = "INTERRUPTED"
        persist(coordinator, state); report(coordinator, state); lock.close()
    return 0 if state["status"] == "COMPLETED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261109_c1_symmetric_none_5m")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--scenarios", nargs="+", choices=("C1", "C1S"), default=["C1S"])
    parser.add_argument("--early-termination", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=910042)
    parser.add_argument("--heldout-seed", type=int, default=1800042)
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.resume_run:
        return supervise(args.resume_run, resuming=True)
    if min(args.workers, args.steps, args.eval_interval, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("Positive budgets / workers required")
    for values in (args.algorithms, args.scenarios, args.seeds):
        if len(values) != len(set(values)):
            parser.error("Duplicate algorithms / scenarios / seeds")
    if args.steps % 4 or args.eval_interval % 4:
        parser.error("Baseline budgets must be divisible by four")
    if max(args.eval_seed, args.heldout_seed) < min(args.eval_seed + args.eval_episodes,
                                                args.heldout_seed + args.heldout_episodes):
        parser.error("Validation / heldout seed ranges overlap")
    prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
