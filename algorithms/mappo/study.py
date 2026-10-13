"""Run a replicated MAPPO learning study and export results and learning curves."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.common.experiment_log import write_atomic
from algorithms.common.study import summarize_runs
from algorithms.mappo.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, validate_backend


def summarize(output, runs):
    return summarize_runs(output, runs, "MAPPO",
                          "实现复用外部 onpolicy/MAPPO，采用共享前馈 actor 和集中式 critic。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    repo = PROJECT_ROOT
    output = args.output.resolve()
    if args.summarize_only:
        runs = []
        for seed in args.seeds:
            path = output / f"seed{seed}"
            config = json.loads((path / "config.json").read_text())
            runs.append({"seed": seed, "path": str(path), "registered_path": config["registered_path"],
                         "summary": json.loads((path / "summary.json").read_text())})
        summarize(output, runs)
        return
    args.mappo_root = validate_backend(parser, args.mappo_root)
    if args.steps <= 0 or args.steps % 16384 or args.workers <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive steps must be a multiple of 16384; use distinct seeds and positive workers")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    write_atomic(output / "config.json", {"steps": args.steps, "seeds": args.seeds, "started": datetime.now().isoformat(),
        "validation_episodes": 8, "heldout_episodes": 32, "eval_interval": 16384, "algorithm": "mappo"})
    def train(seed):
        path = output / f"seed{seed}"
        command = [sys.executable, "-m", "algorithms.mappo.train", "--steps", str(args.steps), "--seed", str(seed),
                   "--mappo-root", str(args.mappo_root), "--config", str(args.config.resolve()),
                   "--eval-interval", "16384", "--eval-episodes", "8", "--heldout-episodes", "32",
                   "--results-env-name", args.config.stem + "_study", "--output", str(path)]
        if args.results_root:
            command += ["--results-root", str(args.results_root.resolve())]
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (output / "logs" / f"seed{seed}.log").open("w") as logfile:
            subprocess.run(command, cwd=repo, env=environment, stdout=logfile, stderr=subprocess.STDOUT, check=True)
        config = json.loads((path / "config.json").read_text())
        return {"seed": seed, "path": str(path), "registered_path": config["registered_path"],
                "summary": json.loads((path / "summary.json").read_text())}
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train, seed): seed for seed in args.seeds}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({"completed_seed": result["seed"], "completed_runs": len(results)}), flush=True)
            except Exception as error:
                failures.append({"seed": futures[future], "error": str(error)})
    if failures:
        write_atomic(output / "failures.json", failures)
        raise RuntimeError("Training failed; see logs and failures.json")
    results.sort(key=lambda r: r["seed"])
    summarize(output, results)
    # Keep the numbered-record directory convenient for existing results consumers.
    if args.results_root:
        group = Path(results[0]["registered_path"]).parent
        for name in ("figures", "RESULTS.md", "summary.json"):
            target = group / name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(output / name, target_is_directory=name == "figures")
    print(json.dumps({"output": str(output), "completed_runs": len(results)}), flush=True)


if __name__ == "__main__":
    main()
