"""Generate explicit source/validation/test suites without editing old configs."""
from __future__ import annotations

import argparse
from dataclasses import replace

from .config import EnvConfig, UnitSpec
from .scenarios import scaling_scenarios
from .transfer import TaskSuite


def make_transfer_suite(base, family="scale_fixed"):
    tasks, splits = {}, {"train": [], "validation": [], "test": []}
    def add(split, name, config):
        task_id = split + "_" + name
        tasks[task_id] = config
        splits[split].append(task_id)
    if family in ("scale_fixed", "scale_density"):
        scenarios = scaling_scenarios(base, mode="fixed_map" if family=="scale_fixed" else "constant_density")
        for name in ("3v3", "6v6"):
            add("train", name, scenarios[name])
            # Same configuration, independent evaluation seeds: ID validation.
            add("validation", name, scenarios[name])
        for name in ("9v9", "12v12"):
            add("test", name, scenarios[name])
    elif family == "map":
        for split, widths in (("train", (20,24)), ("validation", (22,)), ("test", (28,32))):
            for width in widths:
                add(split, f"map_{width}", replace(base,map_size=(width,width,base.map_size[2])))
    elif family == "composition":
        if len(base.red)!=6 or len(base.blue)!=6:
            raise ValueError("Composition suite requires a 6v6 baseline")
        for split, mixes in (("train", ((2,2,2),(1,2,3),(3,2,1))),
                             ("validation", ((2,1,3),)), ("test", ((1,3,2),(3,1,2)))):
            for counts in mixes:
                roster=[UnitSpec(role) for role,count in zip(("scout","strike","hybrid"),counts) for _ in range(count)]
                add(split,"red_"+"_".join(map(str,counts)),replace(base,red=roster))
    else:
        raise ValueError("Unknown transfer family")
    return TaskSuite(tasks,splits)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True)
    parser.add_argument("--family",choices=("scale_fixed","scale_density","map","composition"),default="scale_fixed")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    suite=make_transfer_suite(EnvConfig.load(args.base),args.family)
    suite.save(args.output)
    print(args.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
