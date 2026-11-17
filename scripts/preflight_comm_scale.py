"""Small isolated train/eval checks; never enters the formal 225-entry matrix."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def set_flag(command, flag, value):
    if flag in command: command[command.index(flag) + 1] = str(value)
    else: command.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", type=Path, required=True)
    args = parser.parse_args(); coordinator = args.coordinator.resolve()
    plan = json.loads((coordinator / "plan.json").read_text())
    root = coordinator / "preflight"; root.mkdir(exist_ok=False)
    live = []
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    env.pop("PYTHONPATH", None)
    for algorithm, scene in [(a, "P50_N2") for a in ("dib_v3", "qmix", "qtran", "maa2c", "coma")] + [("dib_v3", "C1S_R_hybrid")]:
        job = next(j for j in plan["jobs"] if j["algorithm"] == algorithm and j["scenario"] == scene and j["seed"] == 42)
        command = list(job["command"]); output = root / (algorithm + "_" + scene)
        for flag, value in [("--output", output), ("--seed", 142), ("--steps", 3200 if algorithm == "dib_v3" else 8000),
                ("--eval-interval", 1600 if algorithm == "dib_v3" else 4000), ("--eval-episodes", 1), ("--heldout-episodes", 1),
                ("--eval-seed", 3710042), ("--heldout-seed", 4800042)]:
            set_flag(command, flag, value)
        if algorithm != "dib_v3":
            set_flag(command, "--save-interval", 4000); set_flag(command, "--log-interval", 4000); set_flag(command, "--replay-episodes", 0)
        stream = (root / (output.name + ".log")).open("w")
        proc = subprocess.Popen(command, cwd=plan["runtime"], env=env, stdout=stream, stderr=subprocess.STDOUT)
        live.append((proc, stream, algorithm, scene, output))
        print("PREFLIGHT", algorithm, scene, proc.pid, flush=True)
    rows = []
    for proc, stream, algorithm, scene, output in live:
        proc.wait(); stream.close()
        p = output / ("training_summary.json" if algorithm == "dib_v3" else "summary.json")
        data = json.loads(p.read_text()) if p.exists() else {}
        row = {"algorithm": algorithm, "scenario": scene, "returncode": proc.returncode,
               "steps": data.get("steps", data.get("environment_steps")), "updates": data.get("updates"),
               "formal_training": False, "performance_claim": False}
        rows.append(row); print(json.dumps(row), flush=True)
    (root / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    if any(r["returncode"] or not r["updates"] for r in rows): raise RuntimeError("Preflight failed; do not launch formal queue")


if __name__ == "__main__":
    main()
