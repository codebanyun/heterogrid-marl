"""Train external EPyMARL MAPPO without using the on-policy/MAPPO integration."""
import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.common.episode_log import open_episode_log
from algorithms.common.training_seeds import EpisodeSeedSequence
from algorithms.common.evaluation import evaluate
from algorithms.common.experiment_log import ExperimentRecorder, write_atomic
from algorithms.epymarl_mappo import backend as api
from heterogrid import EnvConfig, SMACAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epymarl-root", type=Path, default=os.environ.get("EPYMARL_ROOT"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=100042)
    parser.add_argument("--heldout-seed", type=int, default=200042)
    parser.add_argument("--eval-interval", type=int, default=250368)
    parser.add_argument("--log-interval", type=int, default=10240)
    parser.add_argument("--replay-episodes", type=int, default=1)
    parser.add_argument("--environment-seed-mode", choices=("legacy", "episode-index"), default="legacy")
    parser.add_argument("--greedy-validation-only", action="store_true")
    parser.add_argument("--skip-initial-heldout", action="store_true")
    cli = parser.parse_args()
    if cli.epymarl_root is None:
        parser.error("Set EPYMARL_ROOT or --epymarl-root")
    if min(cli.steps, cli.eval_episodes, cli.heldout_episodes, cli.eval_interval, cli.log_interval) <= 0:
        parser.error("Budgets, evaluation counts and intervals must be positive")
    if min(cli.seed, cli.eval_seed, cli.heldout_seed, cli.replay_episodes) < 0:
        parser.error("Seeds and replay count must be nonnegative")
    if max(cli.eval_seed, cli.heldout_seed) < min(cli.eval_seed + cli.eval_episodes, cli.heldout_seed + cli.heldout_episodes):
        parser.error("Validation and heldout seeds overlap")
    if cli.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    root = api.resolve_root(cli.epymarl_root)
    torch.set_num_threads(1)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if cli.device == "cuda":
        torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if cli.environment_seed_mode == "episode-index":
        from algorithms.common.comm_scale_roster import environment_config
        config = environment_config(json.loads(cli.config.read_text()))
    else:
        config = EnvConfig.load(cli.config)
    parameters = api.settings(root, cli.device)
    envs = [SMACAdapter(config=config) for _ in range(parameters["batch_size_run"])]
    backend = api.build(root, parameters, envs[0].get_env_info())
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    sources = list((api.PROJECT_ROOT / "algorithms/epymarl_mappo").glob("*.py"))
    sources += list((api.PROJECT_ROOT / "algorithms/common").glob("*.py"))
    sources += [api.PROJECT_ROOT / "algorithms/common/actor_critic/backend.py", cli.config]
    sources += list((api.PROJECT_ROOT / "heterogrid").glob("*.py"))
    for name in ("learners", "controllers", "modules", "components", "utils"):
        sources += list((root / name).rglob("*.py"))
    sources += [root / "config/default.yaml", root / "config/algs/mappo.yaml"]
    metadata = {"name": "epymarl_mappo", "algorithm": "epymarl_mappo", "framework": "EPyMARL PPOLearner + BasicMAC",
                "environment": config.to_dict(), "algorithm_args": vars(backend.args), "seed": cli.seed,
                "t_max": cli.steps, "epymarl_root": str(root),
                "timeout_handling": "scored finite-horizon terminal; add_value_last_step=False",
                "budget_policy": "finish the first complete fresh batch reaching the target; overshoot < batch_size_run * max_steps",
                "native_preset": "default.yaml + algs/mappo.yaml; unchanged PPO, critic and reward-normalization formulas",
                "cli": {key: str(value) if isinstance(value, Path) else value for key, value in vars(cli).items()}}
    recorder = ExperimentRecorder(output, None, metadata, sources=sources)
    write_atomic(output / "network.json", api.network_configuration(backend))
    policy = api.EvaluationPolicy(backend)
    evaluation_args = SimpleNamespace(algorithm_name="epymarl_mappo", seed=cli.seed, recurrent_N=1,
                                      hidden_size=backend.args.hidden_dim)

    def score(label, count, seed, greedy, final=False):
        return evaluate(policy, evaluation_args, config, output / "replays" / label, count, seed, greedy,
                        save_replays=final and cli.replay_episodes > 0, replay_limit=cli.replay_episodes,
                        save_summary=final)

    best_score, best_step = None, 0

    def validation(scheduled, actual, sampled, greedy):
        nonlocal best_score, best_step
        if sampled is not None:
            recorder.evaluation(scheduled, sampled, False)
        recorder.evaluation(scheduled, greedy, True)
        recorder.log(scheduled, {"test_actual_environment_steps": actual})
        rank = (greedy["win_rate"], greedy["mean_return"])
        if best_score is None or rank > best_score:
            best_score, best_step = rank, actual
            api.checkpoint(output / "models/best.pt", backend, config, actual)

    try:
        initial = api.snapshot(backend)
        api.checkpoint(output / "models/initial.pt", backend, config, 0)
        before = None if cli.greedy_validation_only else score("before", cli.eval_episodes, cli.eval_seed, False, True)
        validation(0, 0, before, score("before_greedy", cli.eval_episodes, cli.eval_seed, True))
        heldout_before = {} if cli.skip_initial_heldout else {
            mode: score("heldout_before_" + mode, cli.heldout_episodes, cli.heldout_seed, greedy)
            for mode, greedy in (("sampled", False), ("greedy", True))}
        reset_rng = (EpisodeSeedSequence(cli.seed) if cli.environment_seed_mode == "episode-index"
                     else np.random.default_rng(cli.seed))
        recent = deque(maxlen=32)
        steps = episodes = updates = 0
        next_eval, next_log = cli.eval_interval, cli.log_interval
        started = time.perf_counter()
        with (output / "metrics.jsonl").open("w", buffering=1) as log, \
                open_episode_log(output, compressed=True) as episode_log:
            while steps < cli.steps:
                batch, collected, rows = api.collect(backend, envs, steps, reset_rng)
                steps += collected
                for row in rows:
                    episodes += 1
                    if cli.environment_seed_mode == "episode-index":
                        row["environment_episode_index"] = row["seed"] - reset_rng.base
                    row.update(episode=episodes, environment_steps=steps)
                    episode_log.write(json.dumps(row, allow_nan=False) + "\n")
                    recent.append(row)
                episode_log.flush()
                backend.buffer.insert_episode_batch(batch)
                sample = backend.buffer.sample(backend.args.batch_size)
                sample = sample[:, :int(sample.max_t_filled().item())]
                if sample.device != backend.args.device:
                    sample.to(backend.args.device)
                backend.mac.agent.train()
                backend.learner.train(sample, steps, episodes)
                updates += 1
                for module in api.modules(backend).values():
                    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
                        raise FloatingPointError("Non-finite EPyMARL MAPPO parameter")
                for module in (backend.mac.agent, backend.learner.critic):
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                        raise FloatingPointError("Non-finite EPyMARL MAPPO gradient")
                row = {"environment_steps": steps, "episodes": episodes, "update": updates,
                       "ppo_optimizer_steps": updates * backend.args.epochs,
                       "recent_return": float(np.mean([item["return"] for item in recent])),
                       "recent_win_rate": sum(item["winner"] == "red" for item in recent) / len(recent),
                       "recent_timeout_rate": sum(item["reason"] == "timeout" for item in recent) / len(recent),
                       "invalid_red_actions": 0, **backend.logger.values}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if steps >= next_log or updates == 1 or steps >= cli.steps:
                    recorder.log(steps, {"return_mean": row["recent_return"], "battle_won_mean": row["recent_win_rate"],
                                        "episode": episodes, "updates": updates, "ppo_optimizer_steps": updates * backend.args.epochs,
                                        **backend.logger.values})
                    print(json.dumps(row), flush=True)
                    next_log = (steps // cli.log_interval + 1) * cli.log_interval
                while steps >= next_eval and next_eval < cli.steps:
                    sampled = None if cli.greedy_validation_only else score("validation_sampled", cli.eval_episodes, cli.eval_seed, False)
                    greedy = score("validation_greedy", cli.eval_episodes, cli.eval_seed, True)
                    validation(next_eval, steps, sampled, greedy)
                    next_eval += cli.eval_interval
        elapsed = time.perf_counter() - started
        final_state = api.snapshot(backend)
        deltas = {name: sum(float((value - initial[name][key]).square().sum()) for key, value in values.items()) ** .5
                  for name, values in final_state.items()}
        if any(value <= 0 for value in deltas.values()):
            raise RuntimeError("An EPyMARL MAPPO network did not update")
        if any(not torch.equal(value, final_state["old_actor"][key]) for key, value in final_state["actor"].items()):
            raise RuntimeError("EPyMARL old policy was not synchronized after PPO")
        api.checkpoint(output / "models/latest.pt", backend, config, steps)
        saved = api.restore(output / "models/latest.pt", backend)
        equal = all(torch.equal(value.detach().cpu(), final_state[name][key])
                    for name, module in api.modules(backend).items() for key, value in module.state_dict().items())
        for key, normalizer in saved["normalizers"].items():
            actual = getattr(backend.learner, "rew_ms" if key == "reward" else "ret_ms")
            equal = equal and torch.equal(actual.mean, normalizer["mean"]) and torch.equal(actual.var, normalizer["var"])
        if not equal:
            raise RuntimeError("EPyMARL MAPPO checkpoint round trip changed the policy")
        after = None if cli.greedy_validation_only else score("after", cli.eval_episodes, cli.eval_seed, False, True)
        greedy = score("greedy", cli.eval_episodes, cli.eval_seed, True, True)
        validation(cli.steps, steps, after, greedy)
        heldout = {}
        for model in ("latest", "best"):
            api.restore(output / "models" / (model + ".pt"), backend)
            heldout[model] = {mode: score("heldout_" + model + "_" + mode, cli.heldout_episodes, cli.heldout_seed,
                                         deterministic, True)
                              for mode, deterministic in (("sampled", False), ("greedy", True))}
        summary = {"algorithm": "epymarl_mappo", "seed": cli.seed, "environment_steps": steps,
                   "requested_environment_steps": cli.steps, "training_episodes": episodes, "updates": updates,
                   "ppo_optimizer_steps": updates * backend.args.epochs, "parameter_deltas_l2": deltas,
                   "training_seconds": elapsed, "environment_steps_per_second": steps / elapsed,
                   "checkpoint_reload_equal": equal, "invalid_red_actions": 0, "best_validation_step": best_step,
                   "before": before, "after": after, "greedy": greedy, "heldout_before": heldout_before, "heldout": heldout}
        write_atomic(output / "summary.json", summary)
        recorder.complete({"environment_steps": steps, "best_validation_step": best_step,
                           "final_validation_win_rate": greedy["win_rate"]})
    finally:
        for env in envs:
            env.close()
        recorder.close()


if __name__ == "__main__":
    main()
