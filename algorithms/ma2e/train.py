"""Train native QMIX+MA2E with a total interaction budget including MAE pretraining."""
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
from algorithms.common.evaluation import evaluate
from algorithms.common.experiment_log import ExperimentRecorder, write_atomic
from algorithms.ma2e import backend as api
from heterogrid import EnvConfig, SMACAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma2e-root", type=Path, default=os.environ.get("MA2E_ROOT"))
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
    parser.add_argument("--settings", type=Path, help="Explicit parameter overrides, recorded in config.json")
    cli = parser.parse_args()
    if cli.ma2e_root is None:
        parser.error("Supply --ma2e-root or MA2E_ROOT")
    if min(cli.steps, cli.eval_episodes, cli.heldout_episodes, cli.eval_interval, cli.log_interval) <= 0:
        parser.error("Budgets and evaluation intervals must be positive")
    if min(cli.seed, cli.eval_seed, cli.heldout_seed, cli.replay_episodes) < 0:
        parser.error("Seeds and replay count must be nonnegative")
    if max(cli.eval_seed, cli.heldout_seed) < min(cli.eval_seed + cli.eval_episodes, cli.heldout_seed + cli.heldout_episodes):
        parser.error("Validation and heldout seeds overlap")
    if cli.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    root = api.resolve_root(cli.ma2e_root)
    torch.set_num_threads(1)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if cli.device == "cuda":
        torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config = EnvConfig.load(cli.config)
    parameters = api.settings(root, cli.device)
    if cli.settings:
        overrides = json.loads(cli.settings.read_text())
        if any(key not in parameters for key in overrides):
            parser.error("Unknown algorithm override")
        parameters.update(overrides)
    if parameters["batch_size"] > parameters["buffer_size"] or parameters["MT_batchsize"] > parameters["buffer_size"]:
        parser.error("Replay capacity must cover both learner batches")
    envs = [SMACAdapter(config=config) for _ in range(parameters["batch_size_run"])]
    backend = api.build(root, parameters, envs[0].get_env_info())
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    sources = list((api.PROJECT_ROOT / "algorithms/ma2e").glob("*.py"))
    sources += list((api.PROJECT_ROOT / "algorithms/common").glob("*.py"))
    sources += [api.PROJECT_ROOT / "algorithms/common/actor_critic/backend.py", cli.config]
    sources += list((api.PROJECT_ROOT / "heterogrid").glob("*.py"))
    for name in ("learners", "controllers", "modules", "components", "utils"):
        sources += list((root / name).rglob("*.py"))
    sources += [root / "config/default.yaml", root / "config/algs/qmix.yaml"]
    if cli.settings:
        sources.append(cli.settings)
    metadata = {"name": "ma2e", "algorithm": "ma2e", "framework": "Native MA2E NQLearner + QMIX + MAE",
        "environment": config.to_dict(), "algorithm_args": vars(backend.args), "seed": cli.seed,
        "t_max": cli.steps, "ma2e_root": str(root), "corrections": api.CORRECTIONS,
        "timeout_handling": "scored finite-horizon terminal, no bootstrap across timeout",
        "budget_policy": "MAE pretraining and RL interactions both count; evaluation excluded; finish complete batches",
        "sampled_evaluation": "epsilon-greedy with final exploration epsilon, not a stochastic PPO policy",
        "cli": {key: str(value) if isinstance(value, Path) else value for key, value in vars(cli).items()}}
    recorder = ExperimentRecorder(output, None, metadata, sources=sources)
    write_atomic(output / "network.json", api.network_configuration(backend))
    policy = api.EvaluationPolicy(backend)
    evaluation_args = SimpleNamespace(algorithm_name="ma2e", seed=cli.seed, recurrent_N=1,
                                      hidden_size=backend.args.rnn_hidden_dim)
    def score(label, count, seed, greedy, final=False):
        return evaluate(policy, evaluation_args, config, output / "replays" / label, count, seed, greedy,
            save_replays=final and cli.replay_episodes > 0, replay_limit=cli.replay_episodes, save_summary=final)
    best_score, best_step = None, 0
    counters = dict(episodes=0, updates=0, mae_updates=0, pretraining_environment_steps=0, pretraining_episodes=0)
    def save(name, steps):
        api.checkpoint(output / "models" / (name + ".pt"), backend, config, steps, dict(counters))
    def validation(scheduled, actual, sampled, greedy):
        nonlocal best_score, best_step
        recorder.evaluation(scheduled, sampled, False)
        recorder.evaluation(scheduled, greedy, True)
        recorder.log(scheduled, {"test_actual_environment_steps": actual})
        rank = (greedy["win_rate"], greedy["mean_return"])
        if best_score is None or rank > best_score:
            best_score, best_step = rank, actual
            save("best", actual)
    try:
        initial = api.snapshot(backend)
        save("initial", 0)
        before = score("before", cli.eval_episodes, cli.eval_seed, False, True)
        validation(0, 0, before, score("before_greedy", cli.eval_episodes, cli.eval_seed, True))
        heldout_before = {mode: score("heldout_before_" + mode, cli.heldout_episodes, cli.heldout_seed, greedy)
            for mode, greedy in (("sampled", False), ("greedy", True))}
        rng, recent = np.random.default_rng(cli.seed), deque(maxlen=32)
        steps, pretrain_batches, fine_tune_count, fine_tune_step = 0, 0, 0, 0
        next_eval, next_log = cli.eval_interval, cli.log_interval
        pretrain_total = backend.args.MT_max_pretraining_episode // backend.args.batch_size_run + 1
        started = time.perf_counter()
        with (output / "metrics.jsonl").open("w", buffering=1) as log, open_episode_log(output, compressed=True) as episodes_log:
            while steps < cli.steps:
                pretraining = pretrain_batches < pretrain_total
                exploration_steps = 0 if pretraining else steps - counters["pretraining_environment_steps"]
                batch, collected, rows = api.collect(backend, envs, exploration_steps, rng)
                steps += collected
                if pretraining:
                    counters["pretraining_environment_steps"] += collected
                    counters["pretraining_episodes"] += len(rows)
                for row in rows:
                    counters["episodes"] += 1
                    row.update(episode=counters["episodes"], environment_steps=steps,
                               phase="mae_pretraining" if pretraining else "rl")
                    episodes_log.write(json.dumps(row, allow_nan=False) + "\n")
                    recent.append(row)
                episodes_log.flush()
                backend.buffer.insert_episode_batch(batch)
                mae_loss = None
                if pretraining:
                    if backend.buffer.can_sample(backend.args.MT_batchsize):
                        mae_loss = api.train_mae(backend, api.sample(backend, backend.args.MT_batchsize, mae=True), steps,
                            counters["episodes"], backend.args.MT_train_n_repeat)
                        counters["mae_updates"] += 1
                    pretrain_batches += 1
                    if pretrain_batches == pretrain_total:
                        backend.buffer.episodes_in_buffer = 0
                        backend.buffer.buffer_index = 0
                        backend.buffer.latent_cache.clear()
                else:
                    if backend.buffer.can_sample(backend.args.batch_size):
                        completed_rl_episodes = counters["episodes"] - counters["pretraining_episodes"] - len(rows)
                        backend.learner.train(api.sample(backend, backend.args.batch_size), steps, completed_rl_episodes)
                        counters["updates"] += 1
                    rl_steps = steps - counters["pretraining_environment_steps"]
                    if backend.args.MT_finetune and fine_tune_count <= backend.args.MT_fine_tune_total and fine_tune_step <= rl_steps:
                        fine_tune_step += backend.args.MT_fine_tune_per_step
                        fine_tune_count += 1
                        latest = batch[:, :int(batch.max_t_filled().item())]
                        mae_loss = api.train_mae(backend, latest, steps, counters["episodes"], 2)
                        counters["mae_updates"] += 1
                for module in api.modules(backend).values():
                    if any(not torch.isfinite(p).all() for p in module.parameters()):
                        raise FloatingPointError("Non-finite MA2E parameter")
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                        raise FloatingPointError("Non-finite MA2E gradient")
                row = {"environment_steps": steps, "episodes": counters["episodes"], "update": counters["updates"],
                    "mae_updates": counters["mae_updates"], "pretraining_environment_steps": counters["pretraining_environment_steps"],
                    "phase": "mae_pretraining" if pretraining else "rl", "mae_loss": mae_loss,
                    "recent_return": float(np.mean([r["return"] for r in recent])),
                    "recent_win_rate": sum(r["winner"] == "red" for r in recent) / len(recent),
                    "recent_timeout_rate": sum(r["reason"] == "timeout" for r in recent) / len(recent),
                    "invalid_red_actions": 0, **backend.logger.values}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if steps >= next_log or steps == collected or steps >= cli.steps:
                    recorder.log(steps, {"return_mean": row["recent_return"], "battle_won_mean": row["recent_win_rate"],
                        "episode": counters["episodes"], "updates": counters["updates"], "mae_updates": counters["mae_updates"],
                        **backend.logger.values})
                    print(json.dumps(row), flush=True)
                    next_log = (steps // cli.log_interval + 1) * cli.log_interval
                while steps >= next_eval and next_eval < cli.steps:
                    validation(next_eval, steps, score("validation_sampled", cli.eval_episodes, cli.eval_seed, False),
                               score("validation_greedy", cli.eval_episodes, cli.eval_seed, True))
                    next_eval += cli.eval_interval
        elapsed = time.perf_counter() - started
        final = api.snapshot(backend)
        deltas = {name: sum(float((v - initial[name][key]).square().sum()) for key, v in values.items()) ** .5
                  for name, values in final.items()}
        mae_delta = sum(float((v - initial["actor"][key]).square().sum()) for key, v in final["actor"].items()
                        if key.startswith("mae.")) ** .5
        if any(v <= 0 for v in deltas.values()) or mae_delta <= 0 or counters["updates"] == 0:
            raise RuntimeError("MA2E did not update all required networks; check warmup and budget")
        save("latest", steps)
        api.restore(output / "models/latest.pt", backend)
        equal = all(torch.equal(v.detach().cpu(), final[name][key]) for name, module in api.modules(backend).items()
                    for key, v in module.state_dict().items())
        if not equal:
            raise RuntimeError("MA2E checkpoint round trip changed the networks")
        after, greedy = score("after", cli.eval_episodes, cli.eval_seed, False, True), score("greedy", cli.eval_episodes, cli.eval_seed, True, True)
        validation(cli.steps, steps, after, greedy)
        heldout = {}
        for model in ("latest", "best"):
            api.restore(output / "models" / (model + ".pt"), backend)
            heldout[model] = {mode: score("heldout_" + model + "_" + mode, cli.heldout_episodes, cli.heldout_seed, deterministic, True)
                for mode, deterministic in (("sampled", False), ("greedy", True))}
        summary = {"algorithm": "ma2e", "seed": cli.seed, "environment_steps": steps,
            "requested_environment_steps": cli.steps, "training_episodes": counters["episodes"], **counters,
            "parameter_deltas_l2": deltas, "mae_parameter_delta_l2": mae_delta,
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
