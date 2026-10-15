"""Fresh-batch on-policy MAA2C/COMA, with exact fixed-budget evaluation barriers."""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.common.actor_critic.backend import act, build_policy, episode_batch
from algorithms.common.experiment_log import ExperimentRecorder, write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT
from algorithms.common.value_decomposition.train import observe, episode_row, aggregate, new_history
from heterogrid import EnvConfig, SMACAdapter
from heterogrid.core import REWARD_COMPONENTS


def save_checkpoint(path, policy, cli, config, steps, updates):
    learner = policy.learner
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"networks": {"actor": policy.mac.agent.state_dict(),
                            "critic": learner.critic.state_dict(),
                            "target_critic": learner.target_critic.state_dict()},
                "optimizers": {"actor": learner.agent_optimiser.state_dict(),
                               "critic": learner.critic_optimiser.state_dict()},
                "critic_training_steps": learner.critic_training_steps,
                "last_target_update_step": learner.last_target_update_step,
                "algorithm_args": vars(policy.args), "env_config": config.to_dict(),
                "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                "environment_steps": steps, "updates": updates,
                "exact_training_resume_supported": False}, temporary)
    temporary.replace(path)


def restore_checkpoint(path, policy):
    checkpoint = torch.load(path, map_location="cpu")
    for key, model in (("actor", policy.mac.agent), ("critic", policy.learner.critic),
                       ("target_critic", policy.learner.target_critic)):
        model.load_state_dict(checkpoint["networks"][key])


def evaluate(policy, config, output, episodes, base_seed, greedy=True, replays=0):
    output.mkdir(parents=True, exist_ok=False)
    env = SMACAdapter(config=config)
    rng = np.random.default_rng(base_seed)
    rows, started = [], time.monotonic()
    try:
        for i in range(episodes):
            env.env.record_replay = i < replays
            env.reset(seed=base_seed + i)
            hidden = np.zeros((1, env.n_agents, policy.args.hidden_dim), dtype=np.float32)
            done, total = False, 0.0
            components = dict.fromkeys(REWARD_COMPONENTS, 0.0)
            while not done:
                obs, _, available = observe(env)
                actions, hidden = act(policy, obs[None], available[None], hidden, rng, greedy)
                reward, done, info = env.step(actions[0].tolist())
                if any(n.startswith("red_") for n in info["invalid_actions"]):
                    raise RuntimeError("Invalid red action in evaluation")
                total += reward
                for key in components:
                    components[key] += info["reward_components"][key]
            rows.append(episode_row(env, base_seed + i, total, components))
            if i < replays:
                env.save_replay(output / f"episode_{i:03d}.json")
    finally:
        env.close()
    result = {**aggregate(rows), "deterministic": greedy,
              "sampling_policy": "argmax" if greedy else "categorical masked policy",
              "base_seed": base_seed, "elapsed_seconds": time.monotonic() - started}
    write_atomic(output / "summary.json", result)
    print(json.dumps({"evaluation": output.name, "win_rate": result["win_rate"],
                      "timeout_rate": result["timeout_rate"], "episodes": episodes}), flush=True)
    return result


def main(algorithm=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("maa2c", "coma"), default=algorithm or "maa2c")
    parser.add_argument("--epymarl-root", type=Path, default=os.environ.get("EPYMARL_ROOT"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, choices=(4,), default=4)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=100042)
    parser.add_argument("--heldout-seed", type=int, default=200042)
    parser.add_argument("--save-interval", type=int, default=2_000_000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--replay-episodes", type=int, default=2)
    cli = parser.parse_args()
    if algorithm and algorithm != cli.algorithm:
        parser.error("Use the matching algorithm entry point")
    if cli.epymarl_root is None:
        parser.error("Set --epymarl-root to EPyMARL/src, or EPYMARL_ROOT")
    for key in ("steps", "eval_interval", "eval_episodes", "heldout_episodes", "save_interval", "log_interval"):
        if getattr(cli, key) < 1:
            parser.error(key + " must be positive")
    if cli.replay_episodes < 0:
        parser.error("Replay count must not be negative")
    if max(cli.eval_seed, cli.heldout_seed) < min(cli.eval_seed + cli.eval_episodes,
                                                 cli.heldout_seed + cli.heldout_episodes):
        parser.error("Validation and independent test seeds overlap")
    cli.output, cli.epymarl_root = cli.output.resolve(), cli.epymarl_root.resolve()
    if (cli.epymarl_root / "src").is_dir():
        cli.epymarl_root = cli.epymarl_root / "src"
    random.seed(cli.seed); np.random.seed(cli.seed); torch.manual_seed(cli.seed)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = EnvConfig.load(cli.config)
    envs = [SMACAdapter(config=config) for _ in range(cli.num_envs)]
    policy, args = build_policy(cli, envs[0])
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "models").mkdir()
    sources = list((PROJECT_ROOT / "heterogrid").glob("*.py"))
    sources += list(Path(__file__).parent.glob("*.py"))
    sources += [PROJECT_ROOT / p for p in ("algorithms/common/sensitivity.py",
                "algorithms/common/experiment_log.py", "algorithms/common/diagnostics.py",
                "algorithms/common/value_decomposition/train.py")]
    sources += [cli.config]
    for name in ("learners", "controllers", "modules", "components", "utils"):
        sources += list((cli.epymarl_root / name).rglob("*.py"))
    sources += [cli.epymarl_root / "config/default.yaml", cli.epymarl_root / f"config/algs/{cli.algorithm}.yaml"]
    metadata = {"name": cli.algorithm, "framework": "EPyMARL external learner",
                "algorithm": cli.algorithm, "seed": cli.seed, "environment": config.to_dict(),
                "algorithm_args": vars(args),
                "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
                "on_policy": "four fresh complete episodes, one update, no replay or policy lag",
                "budget_tail": "discard final incomplete collection batch; never fake a terminal",
                "timeout_handling": "scored finite-horizon terminal; add_value_last_step=False",
                "reward_handling": "unmodified shared team reward once; no standardisation",
                "sampling_policy": "categorical masked actor policy (not epsilon-greedy)",
                "communication": "configured automatic one-hop reports; no learned message head",
                "training_seed_rule": "1000000 + training_seed * 100000000 + episode_index",
                "step_count": "sum of team environment steps, excluding evaluation; not agent steps"}
    recorder = ExperimentRecorder(cli.output, None, metadata, sources=sources)
    write_atomic(cli.output / "env.json", config.to_dict())
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    rng = np.random.default_rng(cli.seed)
    observations, available, histories, totals, components, episode_seeds = [], [], [], [], [], []
    active = set()
    hidden = np.zeros((4, args.n_agents, args.hidden_dim), dtype=np.float32)
    steps = updates = completed = episode_index = learned_steps = 0
    train_seconds = collect_seconds = learn_seconds = eval_seconds = 0.0
    validation, best_score, best_step, phase = [], (-1.0, -float("inf")), 0, "INITIALIZING"
    wall_start = time.monotonic()
    episode_file = (cli.output / "training_episodes.jsonl").open("w", buffering=1)
    update_file = (cli.output / "metrics.jsonl").open("w", buffering=1)

    def reset_batch():
        nonlocal episode_index
        observations.clear(); available.clear(); histories.clear(); totals.clear()
        components.clear(); episode_seeds.clear(); active.clear(); hidden.fill(0)
        for i, env in enumerate(envs):
            seed = 1_000_000 + cli.seed * 100_000_000 + episode_index
            episode_index += 1
            env.reset(seed=seed)
            obs, state, avail = observe(env)
            observations.append(obs); available.append(avail); histories.append(new_history(obs, state, avail))
            totals.append(0.0); components.append(dict.fromkeys(REWARD_COMPONENTS, 0.0)); episode_seeds.append(seed)
            active.add(i)

    def progress(status="RUNNING"):
        write_atomic(cli.output / "progress.json", {"status": status, "pid": os.getpid(), "seed": cli.seed,
            "phase": phase, "environment_steps": steps, "target_steps": cli.steps, "updates": updates,
            "training_episodes": completed, "learned_environment_steps": learned_steps,
            "elapsed_seconds": time.monotonic() - wall_start, "training_seconds": train_seconds,
            "evaluation_seconds": eval_seconds, "environment_steps_per_second": steps / max(train_seconds, 1e-9)})

    def validate():
        nonlocal best_score, best_step, eval_seconds, phase
        phase = "VALIDATION"; progress()
        start = time.monotonic()
        result = evaluate(policy, config, cli.output / "evaluations" / f"validation_{steps:09d}",
                          cli.eval_episodes, cli.eval_seed)
        eval_seconds += time.monotonic() - start
        validation.append({"step": steps, "win_rate": result["win_rate"]})
        recorder.evaluation(steps, result, True)
        score = (result["win_rate"], result["mean_return"])
        if score > best_score:
            best_score, best_step = score, steps
            save_checkpoint(cli.output / "models/best.pt", policy, cli, config, steps, updates)
        phase = "TRAINING"; progress()

    def next_multiple(interval):
        return (steps // interval + 1) * interval

    try:
        reset_batch()
        save_checkpoint(cli.output / "models/initial.pt", policy, cli, config, 0, 0)
        validate()
        while steps < cli.steps and not interrupted:
            loop_start = time.monotonic()
            # Split a vector step at barriers; do not advance hidden states for unstepped slots.
            barrier = min(cli.steps, next_multiple(cli.eval_interval), next_multiple(cli.save_interval),
                          next_multiple(cli.log_interval), next_multiple(1000))
            slots = sorted(active)[:barrier - steps]
            start = time.monotonic()
            actions, next_hidden = act(policy, np.asarray(observations)[slots], np.asarray(available)[slots],
                                       hidden[slots], rng)
            hidden[slots] = next_hidden
            for position, i in enumerate(slots):
                env = envs[i]
                reward, done, info = env.step(actions[position].tolist())
                if any(n.startswith("red_") for n in info["invalid_actions"]):
                    raise RuntimeError("Invalid red action in training")
                totals[i] += reward
                for key in components[i]:
                    components[i][key] += info["reward_components"][key]
                obs, state, avail = observe(env)
                h = histories[i]
                h["u"].append(actions[position]); h["r"].append(reward)
                h["o"].append(obs); h["s"].append(state); h["avail"].append(avail)
                observations[i], available[i] = obs, avail
                if done:
                    active.remove(i); completed += 1
                    row = {**episode_row(env, episode_seeds[i], totals[i], components[i]),
                           "environment_steps": steps + position + 1, "episode": completed}
                    episode_file.write(json.dumps(row) + "\n")
            steps += len(slots)
            collect_seconds += time.monotonic() - start
            if not active:
                start = time.monotonic()
                batch = episode_batch(policy, histories)
                policy.learner.train(batch, steps, completed)
                updates += 1
                learned_steps += sum(len(h["u"]) for h in histories)
                del batch
                learn_seconds += time.monotonic() - start
                if steps < cli.steps:
                    reset_batch()
            train_seconds += time.monotonic() - loop_start
            if steps % 1000 == 0:
                progress()
            if steps % cli.log_interval == 0 or steps == cli.steps:
                values = {**policy.logger.values, "updates": updates, "training_episodes": completed,
                          "learned_environment_steps": learned_steps,
                          "environment_steps_per_second": steps / train_seconds,
                          "collection_seconds": collect_seconds, "learner_seconds": learn_seconds,
                          "training_seconds": train_seconds, "evaluation_seconds": eval_seconds}
                recorder.log(steps, values)
                update_file.write(json.dumps({"step": steps, **values}) + "\n")
                print(json.dumps({"step": steps, **values}), flush=True); progress()
            if steps % cli.save_interval == 0:
                save_checkpoint(cli.output / "models" / f"step_{steps:09d}.pt", policy, cli, config, steps, updates)
            if steps % cli.eval_interval == 0 or steps == cli.steps:
                validate()
        if interrupted:
            save_checkpoint(cli.output / "models/interrupted.pt", policy, cli, config, steps, updates)
            recorder.run["status"] = "INTERRUPTED"; progress("INTERRUPTED")
            return 130
        save_checkpoint(cli.output / "models/final.pt", policy, cli, config, steps, updates)
        heldout = {}
        phase = "HELDOUT"; progress()
        start = time.monotonic()
        for model in ("final", "best"):
            restore_checkpoint(cli.output / "models" / (model + ".pt"), policy)
            heldout[model] = {}
            for mode in ("greedy", "sampled"):
                heldout[model][mode] = evaluate(policy, config, cli.output / "replays" / f"heldout_{model}_{mode}",
                    cli.heldout_episodes, cli.heldout_seed, mode == "greedy", cli.replay_episodes)
        eval_seconds += time.monotonic() - start
        xs = np.asarray([v["step"] for v in validation], dtype=np.float64)
        ys = np.asarray([v["win_rate"] for v in validation])
        reached = [v["step"] for v in validation if v["win_rate"] >= 0.8]
        summary = {"algorithm": cli.algorithm, "seed": cli.seed, "environment_steps": steps,
                   "updates": updates, "training_episodes": completed,
                   "learned_environment_steps": learned_steps,
                   "unused_environment_steps_at_budget": steps - learned_steps,
                   "partial_episodes_at_budget": sum(bool(h["u"]) for i, h in enumerate(histories) if i in active),
                   "best_validation_step": best_step, "validation": validation,
                   "validation_auc": float(np.trapz(ys, xs)),
                   "validation_auc_normalized": float(np.trapz(ys, xs) / cli.steps),
                   "first_80_percent_step": min(reached) if reached else None,
                   "training_seconds": train_seconds, "collection_seconds": collect_seconds,
                   "learner_seconds": learn_seconds, "evaluation_seconds": eval_seconds,
                   "wall_seconds": time.monotonic() - wall_start,
                   "environment_steps_per_second": steps / train_seconds,
                   "end_to_end_steps_per_second": steps / (time.monotonic() - wall_start), "heldout": heldout}
        write_atomic(cli.output / "summary.json", summary)
        phase = "FINISHED"
        recorder.complete({k: v for k, v in summary.items() if k != "heldout"}); progress("COMPLETED")
        return 0
    except BaseException:
        progress("FAILED")
        raise
    finally:
        episode_file.close(); update_file.close()
        for env in envs:
            env.close()
        recorder.close()


if __name__ == "__main__":
    sys.exit(main())
