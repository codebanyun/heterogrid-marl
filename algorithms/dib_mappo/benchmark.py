"""Isolated full-PPO speed checks, never an experiment or a live-worker patch."""
import argparse
import gc
import hashlib
import importlib.util
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.common.value_decomposition.train import observe, new_history
from algorithms.dib_mappo.backend import build_policy, episode_batch
from heterogrid import EnvConfig, SMACAdapter
from heterogrid.scenarios import balanced_roster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("frozen", "optimized", "native"), required=True)
    parser.add_argument("--frozen-runtime", type=Path)
    parser.add_argument("--dib-root", type=Path, required=True)
    parser.add_argument("--agents", type=int, choices=(3, 6, 9, 12), default=6)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--clock", type=int, default=4000000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-frozen", action="store_true",
                        help="Also check a full identical-batch PPO update against frozen actor/critic/Adam state")
    cli = parser.parse_args()
    if min(cli.length, cli.repeats) < 1:
        parser.error("Positive length and repeats required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference_module = None
    if cli.compare_frozen:
        if cli.implementation != "optimized" or cli.frozen_runtime is None:
            parser.error("--compare-frozen requires optimized and --frozen-runtime")
        reference_spec = importlib.util.spec_from_file_location("reference_dib_sequence",
            cli.frozen_runtime / "algorithms/dib_mappo/sequence.py")
        reference_module = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference_module)
    if cli.implementation == "frozen":
        if cli.frozen_runtime is None:
            parser.error("Frozen implementation requires --frozen-runtime")
        spec = importlib.util.spec_from_file_location("frozen_dib_sequence",
            cli.frozen_runtime / "algorithms/dib_mappo/sequence.py")
        sequence = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sequence)
    else:
        from algorithms.dib_mappo import sequence
    sources = {"sequence": Path(sequence.__file__)}
    if reference_module:
        sources["reference_sequence"] = Path(reference_module.__file__)
    if cli.implementation == "optimized":
        sources.update({name: Path(__file__).parent / (name + ".py") for name in ("backend", "auxiliary", "returns")})
    source_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    results = []
    for repeat in range(cli.repeats):
        roster = balanced_roster(cli.agents)
        env = SMACAdapter(config=EnvConfig(red=roster, blue=roster, max_steps=cli.length))
        env.reset(42)
        torch.manual_seed(42 + repeat)
        policy, _ = build_policy(SimpleNamespace(dib_root=cli.dib_root,
            sequence_acceleration=cli.implementation == "optimized"), env)
        if cli.implementation == "frozen":
            sequence.install(policy.mac)
            sequence.install(policy.learner.old_mac)
        obs, state, available = observe(env)
        history = new_history(obs, state, available)
        # Fixed measured shapes, observations and legal WAIT actions. This is a
        # compute microbenchmark, not an artificial learning-performance result.
        for t in range(cli.length):
            history["u"].append(np.zeros(cli.agents, dtype=np.int64))
            history["r"].append(-10.005 if t == cli.length - 1 else -.005)
            history["o"].append(obs); history["s"].append(state); history["avail"].append(available)
        batch = episode_batch(policy, [history] * 4)
        policy.mac.set_residual_actor_gate_for_t(cli.clock)
        reference_policy = None
        reference_seconds = None
        if reference_module:
            torch.manual_seed(42 + repeat)
            reference_policy, _ = build_policy(SimpleNamespace(dib_root=cli.dib_root,
                sequence_acceleration=False), env)
            reference_module.install(reference_policy.mac)
            reference_module.install(reference_policy.learner.old_mac)
            reference_policy.mac.set_residual_actor_gate_for_t(cli.clock)
            reference_batch = episode_batch(reference_policy, [history] * 4)
            if repeat % 2 == 0:
                reference_start = time.perf_counter()
                reference_module.learn(reference_policy, reference_batch, cli.clock, 4)
                reference_seconds = time.perf_counter() - reference_start
        start = time.perf_counter()
        sequence.learn(policy, batch, cli.clock, 4)
        elapsed = time.perf_counter() - start
        if reference_policy and reference_seconds is None:
            reference_start = time.perf_counter()
            reference_module.learn(reference_policy, reference_batch, cli.clock, 4)
            reference_seconds = time.perf_counter() - reference_start
        result = {"repeat": repeat, "seconds": elapsed, "team_steps": cli.length * 4,
                  "ppo_epochs": policy.logger.values["ppo_epochs"],
                  "updates_skipped": policy.logger.values["agent_update_skipped"],
                  "maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
        if reference_policy:
            differences = {}
            largest_keys = {}
            for name, original, optimized in (
                ("actor", reference_policy.mac.agent, policy.mac.agent),
                ("critic", reference_policy.learner.critic, policy.learner.critic),
                ("target_critic", reference_policy.learner.target_critic, policy.learner.target_critic)):
                differences[name] = 0.
                for key, value in original.state_dict().items():
                    other = optimized.state_dict()[key]
                    torch.testing.assert_close(value, other, atol=3e-5, rtol=3e-4)
                    error = (value - other).abs().max().item()
                    if error > differences[name]:
                        differences[name], largest_keys[name] = error, key
            for name in ("ppo_epochs", "dib_stage", "dib_actor_residual_gate", "agent_update_skipped"):
                assert reference_policy.logger.values[name] == policy.logger.values[name], name
            for name in ("agent_optimiser", "critic_optimiser"):
                a, b = getattr(reference_policy.learner, name), getattr(policy.learner, name)
                for group_a, group_b in zip(a.param_groups, b.param_groups):
                    for p, q in zip(group_a["params"], group_b["params"]):
                        state_a, state_b = a.state[p], b.state[q]
                        assert state_a.keys() == state_b.keys()
                        if state_a:
                            assert float(state_a["step"]) == float(state_b["step"])
                            for key in ("exp_avg", "exp_avg_sq"):
                                torch.testing.assert_close(state_a[key], state_b[key], atol=3e-6, rtol=3e-4)
            result.update(reference_seconds=reference_seconds, max_parameter_absolute_error=differences,
                          largest_error_keys=largest_keys,
                          order="reference_then_optimized" if repeat % 2 == 0 else "optimized_then_reference",
                          parity="actor/critic/target/Adam and PPO schedule passed")
            del reference_batch
        results.append(result)
        print(json.dumps({"implementation": cli.implementation, "agents": cli.agents, **result}), flush=True)
        env.close()
        del policy, batch, reference_policy
        gc.collect()
    summary = {"purpose": "compute microbenchmark only", "implementation": cli.implementation,
               "agents": cli.agents, "length": cli.length, "clock": cli.clock,
               "threads": 1, "hidden_dim": 128, "environments": 4, "epochs_configured": 4,
               "results": results, "median_seconds": float(np.median([r["seconds"] for r in results])),
               "torch_version": torch.__version__}
    summary["source_sha256"] = source_hashes
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
