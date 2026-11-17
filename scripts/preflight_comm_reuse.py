"""Verify each reusable algorithm/communication combination on isolated seeds."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", type=Path, required=True)
    args = parser.parse_args(); coordinator = args.coordinator.resolve()
    original = json.loads((coordinator / "plan.json").read_text())
    root = coordinator / "preflight_reuse"; root.mkdir(exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    env.pop("PYTHONPATH", None)
    jobs = []
    for job in original["jobs"]:
        if not job["training_reused"] or job["seed"] != 42: continue
        identity = job["algorithm"] + "_" + job["scenario"]
        folder = root / identity; folder.mkdir()
        copied = {**job, "output": str(folder / "run")}
        plan = {**original, "jobs": [copied], "heldout_seed": 4800042, "heldout_episodes": 1}
        (folder / "plan.json").write_text(json.dumps(plan) + "\n")
        stream = (folder / "worker.log").open("w")
        cmd = [sys.executable, "-u", "-m", "algorithms.common.comm_scale_campaign", "--reevaluate", str(folder), "--job-id", copied["id"]]
        proc = subprocess.Popen(cmd, cwd=original["runtime"], env=env, stdout=stream, stderr=subprocess.STDOUT)
        jobs.append((proc, stream, folder, copied)); print("REUSE_CHECK", identity, proc.pid, flush=True)
    rows = []
    for proc, stream, folder, job in jobs:
        proc.wait(); stream.close()
        summary = folder / "run/summary.json"
        row = {"algorithm": job["algorithm"], "scenario": job["scenario"], "returncode": proc.returncode,
               "summary_present": summary.exists(), "formal_training": False}
        if not proc.returncode:
            with (folder / "replay_export.log").open("w") as stream:
                code = subprocess.call([sys.executable, "-u", "-c", "import runpy;runpy.run_path('scripts/export_retained_replays.py',run_name='__main__')",
                    "--run", str(folder / "run")], cwd=original["runtime"], env=env, stdout=stream, stderr=subprocess.STDOUT)
            row["replay_returncode"] = code
        rows.append(row); print(json.dumps(row), flush=True)
    (root / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    if len(rows) != 14 or any(r["returncode"] or not r["summary_present"] or r.get("replay_returncode") for r in rows):
        raise RuntimeError("Reuse/export check failed")


if __name__ == "__main__":
    main()
