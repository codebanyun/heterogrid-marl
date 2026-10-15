"""Explicit fixed-map or approximately constant-density roster experiments."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from .config import EnvConfig, ROLES, UnitSpec


def balanced_roster(count):
    if type(count) is not int or count < 3 or count % 3:
        raise ValueError("Roster size must be a positive multiple of three")
    platforms = (("uav", "usv"), ("robot", "robot_dog"), ("uav", "robot_dog"))
    return [UnitSpec(role, platforms[r][i % 2]) for r, role in enumerate(ROLES)
            for i in range(count // 3)]


def scaling_scenarios(baseline, counts=(3, 6, 9, 12), *, mode="fixed_map"):
    """Keep all task settings except equal balanced rosters and optional map size.

    Density uses counts of grid cells, not coordinate extents (size - 1).
    Radius, range, vision, movement and time limit are never rescaled.
    """
    if mode not in ("fixed_map", "constant_density"):
        raise ValueError("mode must be fixed_map or constant_density")
    reference = len(baseline.red)
    if reference != len(baseline.blue) or reference % 3 or any(
            sum(unit.role == role for unit in roster) != reference // 3
            for roster in (baseline.red, baseline.blue) for role in ROLES):
        raise ValueError("Baseline must have equal balanced 1:1:1 rosters")
    counts = tuple(counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("Use nonempty, distinct roster sizes")
    result = {}
    for count in counts:
        roster = balanced_roster(count)
        scale = (count / reference) ** (1 / 3) if mode == "constant_density" else 1.0
        size = tuple(max(1, math.floor(dimension * scale + 0.5)) for dimension in baseline.map_size)
        result[f"{count}v{count}"] = replace(baseline, map_size=size, red=roster, blue=list(roster))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Standalone EnvConfig JSON to preserve")
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--mode", choices=("fixed_map", "constant_density"), default="constant_density")
    parser.add_argument("--counts", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument("--reward-normalization", choices=("none", "reference"))
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    baseline = EnvConfig.load(args.base)
    if args.reward_normalization is not None:
        baseline = replace(baseline, reward_normalization=args.reward_normalization)
    if args.diagnostics:
        baseline = replace(baseline, diagnostics_enabled=True)
    tasks = scaling_scenarios(baseline, args.counts, mode=args.mode)
    args.output.mkdir(parents=True, exist_ok=False)
    reference_density = len(baseline.red) / math.prod(baseline.map_size)
    lines = ["# Generated scaling scenarios", "", f"Mode: {args.mode}; source: {args.base.resolve()}", "",
             "Only rosters and (constant_density mode) map size change relative to the expanded base.",
             "Ranges, communication, max_steps and opponent policy are NOT rescaled.",
             "Integer rounding means approximate density; travel distance remains a confound.", "",
             "| Task | Map | Density/reference |", "|---|---|---:|"]
    for name, config in tasks.items():
        path = args.output / (name + ".json")
        path.write_text(json.dumps(config.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        ratio = (len(config.red) / math.prod(config.map_size)) / reference_density
        lines.append(f"| {name} | {config.map_size} | {ratio:.6f} |")
        print(path)
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
