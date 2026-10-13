"""Train SMPE with the actor, critic, VAE and exploration code of an external checkout."""
import argparse
import json
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.common.evaluation import evaluate, write_json
from algorithms.common.experiment_log import ExperimentRecorder
from algorithms.smpe.backend import (EvaluationPolicy, LearnerLogger, assert_finite, build, checkpoint,
                                     collect, modules, parameter_deltas, restore, running_stats, snapshot)
from algorithms.smpe.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, hyperparameters, validate_backend
from heterogrid import EnvConfig, SMACAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=5000192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=100042)
    parser.add_argument("--heldout-seed", type=int, default=200042)
    parser.add_argument("--eval-interval", type=int, default=250368)
    parser.add_argument("--save-interval", type=int, default=2000384)
    parser.add_argument("--log-interval", type=int, default=10240)
    parser.add_argument("--replay-episodes", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    cli = parser.parse_args()
    root = validate_backend(parser, cli.smpe_root)
    if min(cli.steps, cli.eval_episodes, cli.save_interval, cli.log_interval) <= 0:
        parser.error("Training budget, evaluation count and save/log intervals must be positive")
    if min(cli.seed, cli.eval_seed, cli.heldout_seed, cli.eval_interval, cli.heldout_episodes, cli.replay_episodes) < 0:
        parser.error("Seeds, intervals and counts must be nonnegative")
    if cli.heldout_episodes and max(cli.eval_seed, cli.heldout_seed) < min(cli.eval_seed + cli.eval_episodes,
                                                                       cli.heldout_seed + cli.heldout_episodes):
        parser.error("Validation and held-out seed ranges must not overlap")
    if cli.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if cli.device == "cuda":
        torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config = EnvConfig.load(cli.config)
    settings = hyperparameters(root, cli.device)
    saved = None
    if cli.checkpoint:
        saved = torch.load(cli.checkpoint, map_location="cpu")
        config = EnvConfig.from_dict(saved["env_config"])
        settings = {**saved["smpe_args"], "device": cli.device, "use_cuda": cli.device == "cuda"}
    output = (cli.output or PROJECT_ROOT / "output/smpe" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    sources = list((PROJECT_ROOT / "algorithms/smpe").glob("*.py"))
    sources += list((PROJECT_ROOT / "algorithms/common").glob("*.py"))
    sources += list((PROJECT_ROOT / "heterogrid").glob("*.py")) + [cli.config]
    sources += [root / "config/default.yaml", root / "config/algs/smpe_lbf.yaml", root / "run.py"]
    for name in ("controllers", "learners", "model_learners", "modules", "components", "utils"):
        sources += list((root / name).rglob("*.py"))
    metadata = {"name": "smpe", "framework": "external SMPE actor-critic + VAE + intrinsic exploration",
                "algorithm": "smpe", "env": "heterogrid", "seed": cli.seed, "t_max": cli.steps,
                "environment": config.to_dict(), "env_args": config.to_dict(), "smpe_args": settings,
                "smpe_root": str(root), "device": cli.device, "torch_version": torch.__version__,
                "numpy_version": np.__version__, "preset": "smpe_lbf.yaml + structured global-state branch",
                "timeout_handling": "finite-horizon terminal; no value bootstrap at scored game end",
                "model_update_schedule": "native LBF counter += 250 per policy update; VAE every 2000 counter units",
                "budget_policy": "complete the first episode batch reaching the target; overshoot < batch_size_run * max_steps",
                "evaluation_schedule": "nominal thresholds shared with MAPPO; actual evaluation steps recorded separately",
                "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}}
    recorder = ExperimentRecorder(output, None, metadata, sources=sources)
    logger = LearnerLogger()
    envs = [SMACAdapter(config=config) for _ in range(settings["batch_size_run"])]
    backend = build(root, settings, envs[0].get_env_info(), logger)
    policy = EvaluationPolicy(backend)
    eval_args = SimpleNamespace(algorithm_name="smpe", seed=cli.seed, recurrent_N=1, hidden_size=backend.args.hidden_dim)
    print(json.dumps({"output": str(output), "device": cli.device, "env_info": envs[0].get_env_info(),
                      "batch_size_run": backend.args.batch_size_run, "preset": "smpe_lbf"}), flush=True)

    def score(label, count, seed, deterministic, final=False):
        return evaluate(policy, eval_args, config, output / "replays" / label, count, seed, deterministic,
                        save_replays=final and cli.replay_episodes > 0, replay_limit=cli.replay_episodes,
                        save_summary=final)

    try:
        if cli.checkpoint:
            restore(cli.checkpoint, backend)
            for deterministic in (False, True):
                score("greedy" if deterministic else "sampled", cli.eval_episodes, cli.eval_seed, deterministic, True)
            recorder.complete({"evaluation_only": True, "checkpoint_steps": saved["environment_steps"]})
            return
        initial = snapshot(backend)
        checkpoint(output / "models/initial.pt", backend, config, 0)
        before = score("before", cli.eval_episodes, cli.eval_seed, False, True)
        before_greedy = score("before_greedy", cli.eval_episodes, cli.eval_seed, True)
        recorder.evaluation(0, before, False)
        recorder.evaluation(0, before_greedy, True)
        recorder.log(0, {"test_actual_environment_steps": 0})
        best_score, best_step = (before_greedy["win_rate"], before_greedy["mean_return"]), 0
        checkpoint(output / "models/best.pt", backend, config, 0)
        heldout_before = {}
        if cli.heldout_episodes:
            for mode, deterministic in (("sampled", False), ("greedy", True)):
                heldout_before[mode] = score("heldout_before_" + mode, cli.heldout_episodes, cli.heldout_seed, deterministic)

        def validation(scheduled_step, actual_step, sampled, greedy):
            nonlocal best_score, best_step
            recorder.evaluation(scheduled_step, sampled, False)
            recorder.evaluation(scheduled_step, greedy, True)
            recorder.log(scheduled_step, {"test_actual_environment_steps": actual_step})
            rank = (greedy["win_rate"], greedy["mean_return"])
            if rank > best_score:
                best_score, best_step = rank, actual_step
                checkpoint(output / "models/best.pt", backend, config, actual_step)

        recent = deque(maxlen=32)
        reset_random = np.random.default_rng(cli.seed)
        steps = episodes = updates = model_updates = 0
        next_eval, next_save, next_log = cli.eval_interval, cli.save_interval, cli.log_interval
        start = time.perf_counter()
        with (output / "metrics.jsonl").open("w", buffering=1) as log, \
                (output / "training_episodes.jsonl").open("w", buffering=1) as episode_log:
            while steps < cli.steps:
                batch, collected, rows = collect(backend, envs, steps, reset_random)
                steps += collected
                for row in rows:
                    episodes += 1
                    row.update(episode=episodes, environment_steps=steps)
                    episode_log.write(json.dumps(row, allow_nan=False) + "\n")
                    recent.append(row)
                backend.buffer.insert_episode_batch(batch)
                backend.vae.addBatch(batch)
                backend.vae.update_stats(batch)
                sample = backend.buffer.sample(backend.args.batch_size)
                sample = sample[:, :int(sample.max_t_filled().item())]
                if sample.device != backend.args.device:
                    sample.to(backend.args.device)
                updates += 1
                counter = updates * 250
                if counter % backend.args.target_update_filter == 0:
                    backend.vae.update_filters_targets()
                if counter % backend.args.agent_vae_update_period == 0 and steps <= backend.args.stop_time_vae:
                    stats = backend.vae.train_agent_vaes(counter)
                    model_updates += 1
                    for name, value in stats.items():
                        logger.log_stat(name, value, steps)
                rewards = backend.vae.add_intrinsic_rewards(sample)
                valid = sample["filled"].float()
                valid[:, 1:] *= 1 - sample["terminated"][:, :-1].float()
                bonus = rewards - sample["reward"].expand_as(rewards)
                intrinsic_mean = float((bonus * valid).sum().item() / (valid.sum().item() * backend.args.n_agents))
                backend.mac.agent.train()
                backend.learner.train(sample, steps, episodes, rewards)
                assert_finite(backend)
                row = {"seed": cli.seed, "update": updates, "model_updates": model_updates,
                       "environment_steps": steps, "episodes": episodes,
                       "recent_return": float(np.mean([item["return"] for item in recent])),
                       "recent_win_rate": sum(item["winner"] == "red" for item in recent) / len(recent),
                       "recent_timeout_rate": sum(item["reason"] == "timeout" for item in recent) / len(recent),
                       "intrinsic_reward_mean": intrinsic_mean, "invalid_red_actions": 0,
                       "learner_logged_at": logger.step, **logger.values}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if steps >= next_log or updates == 1 or steps >= cli.steps:
                    recorder.log(steps, {"return_mean": row["recent_return"], "battle_won_mean": row["recent_win_rate"],
                                        "timeout_mean": row["recent_timeout_rate"], "episode": episodes,
                                        "model_updates": model_updates, "intrinsic_reward_mean": intrinsic_mean,
                                        **logger.values})
                    print(json.dumps(row, allow_nan=False), flush=True)
                    next_log = (steps // cli.log_interval + 1) * cli.log_interval
                if steps >= next_save and steps < cli.steps:
                    checkpoint(output / "models/latest.pt", backend, config, steps)
                    checkpoint(output / "models" / f"step_{steps:08d}.pt", backend, config, steps)
                    next_save = (steps // cli.save_interval + 1) * cli.save_interval
                while cli.eval_interval and steps >= next_eval and next_eval < cli.steps:
                    sampled = score("step_" + str(next_eval) + "_sampled", cli.eval_episodes, cli.eval_seed, False)
                    greedy = score("step_" + str(next_eval) + "_greedy", cli.eval_episodes, cli.eval_seed, True)
                    validation(next_eval, steps, sampled, greedy)
                    next_eval += cli.eval_interval
        elapsed = time.perf_counter() - start
        deltas = parameter_deltas(backend, initial)
        for name in ("actor", "critic", "critic_w") + (("agent_models", "filters") if model_updates else ()):
            if deltas[name] <= 0:
                raise RuntimeError("SMPE component did not update: " + name)
        checkpoint(output / "models/latest.pt", backend, config, steps)
        expected = snapshot(backend)
        expected_stats = {name: (stats.mean.clone(), stats.var.clone(), float(stats.count))
                          for name, stats in running_stats(backend).items()}
        restore(output / "models/latest.pt", backend)
        equal = all(torch.equal(tensor.detach().cpu(), expected[name][key])
                    for name, module in modules(backend).items() for key, tensor in module.state_dict().items())
        equal = equal and all(torch.equal(stats.mean, expected_stats[name][0]) and torch.equal(stats.var, expected_stats[name][1])
                              and float(stats.count) == expected_stats[name][2] for name, stats in running_stats(backend).items())
        if not equal:
            raise RuntimeError("SMPE checkpoint round trip changed policy or normalization state")
        after = score("after", cli.eval_episodes, cli.eval_seed, False, True)
        greedy = score("greedy", cli.eval_episodes, cli.eval_seed, True, True)
        validation(cli.steps, steps, after, greedy)
        heldout = {}
        if cli.heldout_episodes:
            for label in ("latest", "best"):
                restore(output / "models" / (label + ".pt"), backend)
                heldout[label] = {}
                for mode, deterministic in (("sampled", False), ("greedy", True)):
                    heldout[label][mode] = score("heldout_" + label + "_" + mode, cli.heldout_episodes,
                                                cli.heldout_seed, deterministic, True)
        summary = {"algorithm": "smpe", "requested_environment_steps": cli.steps, "environment_steps": steps,
                   "updates": updates, "model_updates": model_updates, "training_episodes": episodes,
                   "training_seconds": elapsed, "environment_steps_per_second": steps / elapsed,
                   "parameter_deltas_l2": deltas, "checkpoint_reload_equal": equal, "invalid_red_actions": 0,
                   "best_validation_step": best_step, "best_validation_score": best_score,
                   "before": before, "after": after, "greedy": greedy,
                   "heldout_before": heldout_before, "heldout": heldout}
        write_json(output / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key not in
                          ("before", "after", "greedy", "heldout_before", "heldout")}), flush=True)
        recorder.complete({"environment_steps": steps, "best_validation_step": best_step,
                           "final_validation_win_rate": greedy["win_rate"]})
    finally:
        for env in envs:
            env.close()
        recorder.close()


if __name__ == "__main__":
    main()
