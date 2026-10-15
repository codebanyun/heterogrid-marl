"""One bounded queue for four algorithms with blue communication held fixed."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from algorithms.common.actor_critic.campaign import prepare as prepare_actor_critic
from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, communication_protocol
from algorithms.common.value_decomposition.campaign import now, prepare as prepare_value, supervise
from algorithms.common.value_decomposition.report import read_json


def combine_plans(plans, workers, coordinator):
    """Interleave algorithm families without changing their frozen commands."""
    jobs = []
    for index in range(max(len(plan["jobs"]) for plan in plans)):
        for plan in plans:
            if index < len(plan["jobs"]):
                jobs.append({**plan["jobs"][index], "runtime": plan["runtime"]})
    return {"created": now(), "workers": workers, "runtime": plans[0]["runtime"],
            "roots": {key: value for plan in plans for key, value in plan["roots"].items()},
            "jobs": jobs, "manager_pid": None, "status": "PREPARED",
            "requested_total_steps": sum(plan["requested_total_steps"] for plan in plans),
            "communication_protocol": communication_protocol(True),
            "coordinator": str(coordinator), "python": sys.executable,
            "host_cpu_count": os.cpu_count()}


def prepare(args):
    coordinator = args.output_root / "campaigns" / (args.name + "_all")
    candidates = [coordinator] + [args.output_root / "campaigns" / (args.name + suffix)
                                 for suffix in ("_qmix_qtran", "_maa2c_coma")]
    candidates += [args.output_root / algorithm / args.name for algorithm in ("qmix", "qtran", "maa2c", "coma")]
    for path in candidates:
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
    args.fixed_blue = True
    value_args = argparse.Namespace(**{**vars(args), "algorithms": ["qmix", "qtran"]})
    actor_args = argparse.Namespace(**{**vars(args), "algorithms": ["maa2c", "coma"]})
    family_roots = [prepare_value(value_args), prepare_actor_critic(actor_args)]
    plans = [read_json(root / "plan.json") for root in family_roots]
    coordinator.mkdir(parents=True)
    plan = combine_plans(plans, args.workers, coordinator)
    plan.update(source_campaigns=[str(root) for root in family_roots],
                disk_free_at_prepare=shutil.disk_usage(coordinator).free)
    write_atomic(coordinator / "plan.json", plan)
    write_atomic(coordinator / "status.json", plan)
    # Family directories hold frozen sources/plans, not additional managers.
    for root, family_plan in zip(family_roots, plans):
        write_atomic(root / "status.json", {**family_plan, "status": "DELEGATED",
                                           "coordinator": str(coordinator)})
    for root in plan["roots"].values():
        experiment_path = Path(root) / "experiment.json"
        experiment = read_json(experiment_path)
        experiment["coordinator"] = str(coordinator)
        write_atomic(experiment_path, experiment)
        write_atomic(Path(root) / "status.json", {"status": "PREPARED", "coordinator": str(coordinator),
                    "manager_pid": None, "counts": {"PENDING": experiment["planned_runs"]}})
    print(str(coordinator), flush=True)
    return coordinator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Supervise an already prepared unified queue")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--name", default="20261013_fixed_blue_sensitivity_5m")
    parser.add_argument("--marl-root", type=Path, default=os.environ.get("MARL_ALGORITHMS_ROOT"))
    parser.add_argument("--epymarl-root", type=Path, default=os.environ.get("EPYMARL_ROOT"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.run:
        return supervise(args.run)
    if args.marl_root is None or args.epymarl_root is None:
        parser.error("Set --marl-root and --epymarl-root")
    args.output_root, args.marl_root, args.epymarl_root = (
        args.output_root.resolve(), args.marl_root.resolve(), args.epymarl_root.resolve())
    if (args.epymarl_root / "src").is_dir():
        args.epymarl_root = args.epymarl_root / "src"
    if min(args.workers, args.steps, args.eval_interval, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("Budgets and workers must be positive")
    if args.steps % 4 or args.eval_interval % 4:
        parser.error("Steps and eval interval must be divisible by four")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds")
    prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
