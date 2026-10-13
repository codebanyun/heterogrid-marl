"""Train external QMIX or QTRAN-base on fixed-slot HeteroGrid environments."""
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

from algorithms.common.experiment_log import ExperimentRecorder, write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT
from algorithms.common.value_decomposition.backend import act, build_policy, modules
from algorithms.common.value_decomposition.replay import EpisodeReplay
from heterogrid import EnvConfig, SMACAdapter
from heterogrid.core import REWARD_COMPONENTS
from heterogrid.smac import flatten_observation


def observe(env):
    cached = env.env._last_observations
    observations = [cached[n] if n in cached else env.env.core.observation(n)
                    for n in env.env.possible_agents]
    return (np.asarray([flatten_observation(o) for o in observations], dtype=np.float32),
            env.get_state(), np.asarray([o["action_mask"] for o in observations], dtype=np.uint8))


def episode_row(env, seed, total, components):
    core = env.env.core
    info = core.episode_metrics["red"]
    return {"seed": int(seed), "return": total, "length": core.step_count,
            "winner": core.winner, "reason": core.reason, **info,
            "kill_fraction": info["kills"] / len(core.names["blue"]),
            "reward_components": dict(components)}


def aggregate(rows):
    contacts = [r["first_contact_step"] for r in rows if r["first_contact_step"] is not None]
    return {"episodes": len(rows), "win_rate": sum(r["winner"] == "red" for r in rows) / len(rows),
            "timeout_rate": sum(r["reason"] == "timeout" for r in rows) / len(rows),
            "contact_rate": len(contacts) / len(rows),
            "mean_first_contact_step": float(np.mean(contacts)) if contacts else None,
            **{"mean_" + key: float(np.mean([r[key] for r in rows])) for key in
               ("return", "length", "damage", "kills", "kill_fraction", "useful_reports", "discovered_targets")},
            "mean_reward_components": {k: float(np.mean([r["reward_components"][k] for r in rows]))
                                       for k in REWARD_COMPONENTS}, "results": rows}


def evaluate(policy, config, output, episodes, base_seed, epsilon=0.0, replays=0):
    """Evaluation uses its own RNG and hidden state; it never enters the replay."""
    output.mkdir(parents=True, exist_ok=False)
    env = SMACAdapter(config=config)
    rng = np.random.default_rng(base_seed)
    rows = []
    started = time.monotonic()
    try:
        for i in range(episodes):
            env.env.record_replay = i < replays
            env.reset(seed=base_seed + i)
            hidden = np.zeros((1, env.n_agents, policy.args.rnn_hidden_dim), dtype=np.float32)
            previous = np.zeros((1, env.n_agents, policy.n_actions), dtype=np.float32)
            done, total = False, 0.0
            components = dict.fromkeys(REWARD_COMPONENTS, 0.0)
            while not done:
                obs, _, avail = observe(env)
                actions, hidden, previous = act(policy, obs[None], avail[None], hidden, previous, rng, epsilon)
                reward, done, info = env.step(actions[0].tolist())
                if any(n.startswith("red_") for n in info["invalid_actions"]):
                    raise RuntimeError("Invalid red action in evaluation")
                total += reward
                for k in components:
                    components[k] += info["reward_components"][k]
            rows.append(episode_row(env, base_seed + i, total, components))
            if i < replays:
                env.save_replay(output / f"episode_{i:03d}.json")
    finally:
        env.close()
    result = {**aggregate(rows), "epsilon": epsilon, "deterministic": epsilon == 0,
              "base_seed": base_seed, "elapsed_seconds": time.monotonic() - started}
    write_atomic(output / "summary.json", result)
    print(json.dumps({"evaluation": output.name, "win_rate": result["win_rate"],
                      "timeout_rate": result["timeout_rate"], "episodes": episodes}), flush=True)
    return result


def save_checkpoint(path, policy, args, cli, config, steps, updates):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"networks": {k: getattr(policy, k).state_dict() for k in modules(policy)},
                "optimizer": policy.optimizer.state_dict(), "algorithm_args": vars(args),
                "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                "env_config": config.to_dict(), "environment_steps": steps, "updates": updates,
                "replay_included": False, "exact_training_resume_supported": False}, temporary)
    temporary.replace(path)


def restore_checkpoint(path, policy):
    saved = torch.load(path, map_location="cpu")
    for k, state in saved["networks"].items():
        getattr(policy, k).load_state_dict(state)
    return saved


def finish_episode(history):
    return {k: np.asarray(v, dtype=(np.uint8 if k in ("u", "avail") else np.float32))
            for k, v in history.items()}


def new_history(obs, state, avail):
    return {"o": [obs], "s": [state], "avail": [avail], "u": [], "r": []}


def parser_for(algorithm=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("qmix", "qtran"), default=algorithm or "qmix")
    parser.add_argument("--marl-root", type=Path, default=os.environ.get("MARL_ALGORITHMS_ROOT"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-episodes", type=int, default=32)
    parser.add_argument("--update-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=100042)
    parser.add_argument("--heldout-seed", type=int, default=200042)
    parser.add_argument("--save-interval", type=int, default=2_000_000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--replay-episodes", type=int, default=2)
    parser.add_argument("--sampled-epsilon", type=float, default=0.05)
    return parser


def main(algorithm=None):
    parser = parser_for(algorithm)
    cli = parser.parse_args()
    if algorithm and cli.algorithm != algorithm:
        parser.error("Use the matching algorithm entry point")
    if cli.marl_root is None:
        parser.error("Set --marl-root or MARL_ALGORITHMS_ROOT")
    for name in ("steps", "num_envs", "buffer_size", "batch_size", "warmup_episodes", "update_interval",
                 "eval_interval", "eval_episodes", "heldout_episodes", "save_interval", "log_interval"):
        if getattr(cli, name) < 1:
            parser.error(name + " must be positive")
    for name in ("steps", "update_interval", "eval_interval", "save_interval", "log_interval"):
        if getattr(cli, name) % cli.num_envs:
            parser.error(name + " must be divisible by num_envs")
    if cli.buffer_size < max(cli.batch_size, cli.warmup_episodes):
        parser.error("Replay capacity must cover minibatch and warmup")
    if cli.replay_episodes < 0 or not 0 <= cli.sampled_epsilon <= 1:
        parser.error("Invalid replay count or sampled epsilon")
    if max(cli.eval_seed, cli.heldout_seed) < min(cli.eval_seed + cli.eval_episodes,
                                                 cli.heldout_seed + cli.heldout_episodes):
        parser.error("Validation and independent test seeds must not overlap")
    cli.output = cli.output.resolve()
    cli.marl_root = cli.marl_root.resolve()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = EnvConfig.load(cli.config)
    envs = [SMACAdapter(config=config) for _ in range(cli.num_envs)]
    policy, args = build_policy(cli, envs[0])
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "models").mkdir()
    sources = list((PROJECT_ROOT / "heterogrid").glob("*.py"))
    sources += list(Path(__file__).parent.glob("*.py"))
    sources += [PROJECT_ROOT / "algorithms/common/sensitivity.py", PROJECT_ROOT / "algorithms/common/experiment_log.py", cli.config]
    sources += [cli.marl_root / p for p in ("policy/qmix.py", "policy/qtran_base.py", "network/base_net.py",
                "network/qmix_net.py", "network/qtran_net.py", "common/arguments.py")]
    metadata = {"name": cli.algorithm, "framework": "MARL-Algorithms external learner",
                "algorithm": cli.algorithm, "variant": args.alg, "seed": cli.seed,
                "environment": config.to_dict(), "algorithm_args": vars(args),
                "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
                "target_history": "both networks process o_0; QMIX upstream offset repaired",
                "timeout_handling": "scored finite-horizon terminal, no bootstrap",
                "communication": "configured automatic one-hop target reports; no learned message head",
                "sampling_policy": "epsilon-greedy with epsilon=" + str(cli.sampled_epsilon),
                "training_seed_rule": "1000000 + training_seed * 100000000 + episode_index",
                "step_count": "sum of team environment steps, excluding evaluation; not agent steps"}
    recorder = ExperimentRecorder(cli.output, None, metadata, sources=sources)
    write_atomic(cli.output / "env.json", config.to_dict())
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    actor_rng = np.random.default_rng(cli.seed)
    replay = EpisodeReplay(cli.buffer_size, cli.seed + 30000)
    observations, states, available, histories, totals, components, episode_seeds = [], [], [], [], [], [], []
    episode_index = 0

    def reset_slot(i):
        nonlocal episode_index
        seed = 1_000_000 + cli.seed * 100_000_000 + episode_index
        episode_index += 1
        envs[i].reset(seed=seed)
        obs, state, avail = observe(envs[i])
        if i >= len(observations):
            observations.append(obs); states.append(state); available.append(avail)
            histories.append(new_history(obs, state, avail)); totals.append(0.0)
            components.append(dict.fromkeys(REWARD_COMPONENTS, 0.0)); episode_seeds.append(seed)
        else:
            observations[i], states[i], available[i] = obs, state, avail
            histories[i], totals[i], components[i], episode_seeds[i] = (
                new_history(obs, state, avail), 0.0, dict.fromkeys(REWARD_COMPONENTS, 0.0), seed)

    for i in range(cli.num_envs):
        reset_slot(i)
    hidden = np.zeros((cli.num_envs, args.n_agents, args.rnn_hidden_dim), dtype=np.float32)
    previous = np.zeros((cli.num_envs, args.n_agents, args.n_actions), dtype=np.float32)
    updates, completed, steps = 0, 0, 0
    train_seconds, eval_seconds, collect_seconds, learn_seconds = 0.0, 0.0, 0.0, 0.0
    validation = []
    best_score, best_step = (-1.0, -float("inf")), 0
    phase = "INITIALIZING"
    last_gradient_norm = 0.0
    original_step = policy.optimizer.step

    def checked_step(*step_args, **step_kwargs):
        nonlocal last_gradient_norm
        squared = 0.0
        for p in policy.eval_parameters:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Non-finite learner gradient")
                squared += float(p.grad.detach().square().sum())
        last_gradient_norm = squared ** 0.5
        result = original_step(*step_args, **step_kwargs)
        if any(not torch.isfinite(p).all() for p in policy.eval_parameters):
            raise FloatingPointError("Non-finite learned parameter")
        return result

    policy.optimizer.step = checked_step
    wall_start = time.monotonic()
    episode_file = (cli.output / "training_episodes.jsonl").open("w", buffering=1)
    update_file = (cli.output / "metrics.jsonl").open("w", buffering=1)

    def progress(status="RUNNING"):
        write_atomic(cli.output / "progress.json", {"status": status, "pid": os.getpid(), "seed": cli.seed,
            "phase": phase,
            "environment_steps": steps, "target_steps": cli.steps, "updates": updates,
            "training_episodes": completed, "replay_episodes": len(replay),
            "elapsed_seconds": time.monotonic() - wall_start, "training_seconds": train_seconds,
            "evaluation_seconds": eval_seconds, "environment_steps_per_second": steps / max(train_seconds, 1e-9)})

    def validate():
        nonlocal best_score, best_step, eval_seconds, phase
        phase = "VALIDATION"
        progress()
        start = time.monotonic()
        result = evaluate(policy, config, cli.output / "evaluations" / f"validation_{steps:09d}",
                          cli.eval_episodes, cli.eval_seed)
        eval_seconds += time.monotonic() - start
        validation.append({"step": steps, "win_rate": result["win_rate"]})
        recorder.evaluation(steps, result, True)
        score = (result["win_rate"], result["mean_return"])
        if score > best_score:
            best_score, best_step = score, steps
            save_checkpoint(cli.output / "models/best.pt", policy, args, cli, config, steps, updates)
        phase = "TRAINING"
        progress()

    try:
        save_checkpoint(cli.output / "models/initial.pt", policy, args, cli, config, 0, 0)
        validate()
        while steps < cli.steps and not interrupted:
            loop_start = time.monotonic()
            epsilon = max(args.min_epsilon, args.epsilon - steps * args.anneal_epsilon)
            start = time.monotonic()
            actions, hidden, previous = act(policy, np.asarray(observations), np.asarray(available),
                                            hidden, previous, actor_rng, epsilon)
            for i, env in enumerate(envs):
                reward, done, info = env.step(actions[i].tolist())
                if any(n.startswith("red_") for n in info["invalid_actions"]):
                    raise RuntimeError("Invalid red action in training")
                totals[i] += reward
                for k in components[i]:
                    components[i][k] += info["reward_components"][k]
                obs, state, avail = observe(env)
                h = histories[i]
                h["u"].append(actions[i].astype(np.uint8)); h["r"].append(reward)
                h["o"].append(obs); h["s"].append(state); h["avail"].append(avail)
                observations[i], states[i], available[i] = obs, state, avail
                if done:
                    replay.add(finish_episode(h))
                    completed += 1
                    row = {**episode_row(env, episode_seeds[i], totals[i], components[i]),
                           "environment_steps": steps + i + 1, "episode": completed, "epsilon": epsilon}
                    episode_file.write(json.dumps(row) + "\n")
                    reset_slot(i)
                    hidden[i].fill(0); previous[i].fill(0)
            steps += cli.num_envs
            collect_seconds += time.monotonic() - start
            if steps % cli.update_interval == 0 and len(replay) >= cli.warmup_episodes:
                start = time.monotonic()
                batch, length = replay.sample(cli.batch_size, args.n_actions)
                policy.learn(batch, length, updates, epsilon)
                updates += 1
                learn_seconds += time.monotonic() - start
            train_seconds += time.monotonic() - loop_start
            if steps % 1000 == 0:
                progress()
            if steps % cli.log_interval == 0 or steps == cli.steps:
                values = {"epsilon": epsilon, "updates": updates, "gradient_norm": last_gradient_norm,
                          "training_episodes": completed, "environment_steps_per_second": steps / train_seconds,
                          "collection_seconds": collect_seconds, "learner_seconds": learn_seconds,
                          "training_seconds": train_seconds, "evaluation_seconds": eval_seconds}
                recorder.log(steps, values)
                update_file.write(json.dumps({"step": steps, **values}) + "\n")
                print(json.dumps({"step": steps, **values}), flush=True)
                progress()
            if steps % cli.save_interval == 0:
                save_checkpoint(cli.output / "models" / f"step_{steps:09d}.pt", policy, args, cli, config, steps, updates)
            if steps % cli.eval_interval == 0 or steps == cli.steps:
                validate()
        if interrupted:
            save_checkpoint(cli.output / "models/interrupted.pt", policy, args, cli, config, steps, updates)
            recorder.run["status"] = "INTERRUPTED"
            progress("INTERRUPTED")
            return 130
        save_checkpoint(cli.output / "models/final.pt", policy, args, cli, config, steps, updates)
        # Test both reloaded models. Only validation ever selected best.pt.
        heldout = {}
        phase = "HELDOUT"
        progress()
        start = time.monotonic()
        for model in ("final", "best"):
            restore_checkpoint(cli.output / "models" / (model + ".pt"), policy)
            heldout[model] = {}
            for mode, epsilon in (("greedy", 0.0), ("sampled", cli.sampled_epsilon)):
                heldout[model][mode] = evaluate(policy, config, cli.output / "replays" / f"heldout_{model}_{mode}",
                    cli.heldout_episodes, cli.heldout_seed, epsilon, cli.replay_episodes)
        eval_seconds += time.monotonic() - start
        xs = np.asarray([v["step"] for v in validation], dtype=np.float64)
        ys = np.asarray([v["win_rate"] for v in validation])
        reached = [v["step"] for v in validation if v["win_rate"] >= 0.8]
        summary = {"algorithm": cli.algorithm, "seed": cli.seed, "environment_steps": steps,
                   "updates": updates, "training_episodes": completed, "partial_episodes_at_budget":
                   sum(bool(h["u"]) for h in histories), "best_validation_step": best_step,
                   "validation": validation, "validation_auc": float(np.trapz(ys, xs)),
                   "validation_auc_normalized": float(np.trapz(ys, xs) / cli.steps),
                   "first_80_percent_step": min(reached) if reached else None,
                   "training_seconds": train_seconds, "collection_seconds": collect_seconds,
                   "learner_seconds": learn_seconds, "evaluation_seconds": eval_seconds,
                   "wall_seconds": time.monotonic() - wall_start,
                   "environment_steps_per_second": steps / train_seconds,
                   "end_to_end_steps_per_second": steps / (time.monotonic() - wall_start),
                   "heldout": heldout}
        write_atomic(cli.output / "summary.json", summary)
        phase = "FINISHED"
        recorder.complete({k: v for k, v in summary.items() if k != "heldout"})
        progress("COMPLETED")
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
