"""Run the native recurrent EPyMARL MAPPO on a HeteroGrid task suite."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from algorithms.common.actor_critic.backend import MetricLogger, external_module
from algorithms.common.experiment_log import ExperimentRecorder, timestamp, write_atomic
from heterogrid import TaskSuite, TransferSMACAdapter


def epymarl_source(path):
    for candidate in (path, path / "src", path / "epymarl" / "src"):
        if (candidate / "learners" / "ppo_learner.py").is_file():
            return candidate.resolve()
    raise ValueError("EPyMARL path does not contain src/learners/ppo_learner.py")


def build_policy(root, info, device, algorithm="mappo"):
    sys.path.insert(0, str(root))
    batch_class = importlib.import_module("components.episode_buffer").EpisodeBatch
    one_hot = importlib.import_module("components.transforms").OneHot
    mac_class = external_module(root, "controllers/basic_controller.py").BasicMAC
    learner_file, learner_name = {
        "mappo": ("ppo_learner", "PPOLearner"),
        "maa2c": ("actor_critic_learner", "ActorCriticLearner"),
        "coma": ("coma_learner", "COMALearner"),
    }[algorithm]
    learner_class = getattr(external_module(root, f"learners/{learner_file}.py"), learner_name)
    settings = yaml.safe_load((root / "config/default.yaml").read_text(encoding="utf-8"))
    settings.update(yaml.safe_load((root / f"config/algs/{algorithm}.yaml").read_text(encoding="utf-8")))
    settings.update(info, use_cuda=device.type == "cuda", common_reward=True,
                    add_value_last_step=False, learner_log_interval=1)
    args = SimpleNamespace(**settings)
    scheme = {
        "state": {"vshape": args.state_shape},
        "obs": {"vshape": args.obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [one_hot(out_dim=args.n_actions)])}

    def new_batch(count, length):
        return batch_class(scheme.copy(), groups, count, length, preprocess=preprocess, device=str(device))

    mac = mac_class(new_batch(1, 1).scheme, groups, args)
    logger = MetricLogger()
    learner = learner_class(mac, new_batch(1, 1).scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    return SimpleNamespace(args=args, mac=mac, learner=learner, logger=logger,
                           new_batch=new_batch, device=device)


def put(batch, key, indices, step, values):
    target = batch[key]
    target[indices, step] = torch.as_tensor(np.asarray(values), dtype=target.dtype, device=batch.device)


@torch.no_grad()
def collect(policy, jobs, steps, *, training):
    batch = policy.new_batch(len(jobs), policy.args.episode_limit + 1)
    # The native controller requires at least one legal action for every padded time step.
    batch["avail_actions"][..., 0] = 1
    obs, returns, rows = [], np.zeros(len(jobs)), [None] * len(jobs)
    for i, (env, task, seed) in enumerate(jobs):
        first_obs, state = env.reset(seed=seed, options={"task_id": task})
        obs.append(first_obs)
        if training:
            put(batch, "state", [i], 0, [state])
    policy.mac.init_hidden(len(jobs))
    running = list(range(len(jobs)))
    t = 0
    while running:
        available = np.asarray([jobs[i][0].get_avail_actions() for i in running])
        put(batch, "obs", running, t, [obs[i] for i in running])
        put(batch, "avail_actions", running, t, available)
        if training:
            action = policy.mac.select_actions(batch, t, steps, bs=running)
        else:
            # Match the greedy test protocol of the other baselines.
            action = policy.mac.forward(batch, t, test_mode=True)[running].argmax(-1)
        selected = action.cpu().numpy()
        if not np.take_along_axis(available, selected[..., None], axis=-1).all():
            raise RuntimeError("EPyMARL selected an unavailable action")
        rewards, done_flags, next_states, remaining = [], [], [], []
        for i, actions in zip(running, selected):
            env, task, seed = jobs[i]
            reward, done, info = env.step(actions.tolist())
            returns[i] += reward
            rewards.append([reward])
            done_flags.append([done])
            if training:
                next_states.append(env.get_state())
            if done:
                rows[i] = {"task": task, "seed": seed, "length": t + 1,
                           "return": float(returns[i]), "won": bool(info["battle_won"]),
                           "reason": info["termination_reason"], **info["episode_metrics"]}
            else:
                obs[i] = env.get_obs()
                remaining.append(i)
        if training:
            # Use EpisodeBatch.update so the native OneHot preprocessing also
            # populates actions_onehot, required by COMA's critic.
            batch.update({"actions": action.unsqueeze(-1)}, bs=running, ts=t, mark_filled=False)
            put(batch, "reward", running, t, rewards)
            put(batch, "terminated", running, t, done_flags)
            put(batch, "state", running, t + 1, next_states)
            batch["filled"][running, t] = 1
        t += 1
        running = remaining
    return rows, batch[:, :t + 1] if training else None


def evaluate(policy, suite, split, count, seed, batch_size):
    tasks = suite.ids(split)
    envs = [TransferSMACAdapter(suite, task_ids=tasks) for _ in range(min(count, batch_size))]
    results = {}
    try:
        for task in tasks:
            rows = []
            for start in range(0, count, len(envs)):
                jobs = [(env, task, seed + index) for env, index in
                        zip(envs, range(start, min(start + len(envs), count)))]
                episodes, _ = collect(policy, jobs, 0, training=False)
                rows.extend(episodes)
            contacts = [row["first_contact_step"] for row in rows if row["first_contact_step"] is not None]
            results[task] = {
                "episodes": count, "win_rate": sum(row["won"] for row in rows) / count,
                "mean_return": float(np.mean([row["return"] for row in rows])),
                "timeout_rate": sum(row["reason"] == "timeout" for row in rows) / count,
                "contact_rate": len(contacts) / count,
                "mean_first_contact_step": float(np.mean(contacts)) if contacts else None,
                "results": rows,
            }
    finally:
        for env in envs:
            env.close()
    return results


def checkpoint(path, policy, metadata, steps, update, *, counts, completed, rng):
    learner = policy.learner
    reward_stats = (vars(learner.rew_ms) if policy.args.standardise_rewards else None)
    temp = path.with_suffix(".tmp")
    torch.save({
        "schema": "heterogrid-epymarl-mappo-v1", "environment_steps": steps, "update": update,
        "metadata": metadata, "actor": policy.mac.agent.state_dict(),
        "critic": learner.critic.state_dict(), "target_critic": learner.target_critic.state_dict(),
        "actor_optimizer": learner.agent_optimiser.state_dict(),
        "critic_optimizer": learner.critic_optimiser.state_dict(), "reward_stats": reward_stats,
        "training_state": {
            "task_environment_steps": dict(counts), "completed_episodes": completed,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "task_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if policy.device.type == "cuda" else None,
            "critic_training_steps": learner.critic_training_steps,
            "last_target_update_step": learner.last_target_update_step,
            "log_stats_t": learner.log_stats_t,
            "return_stats": vars(learner.ret_ms) if policy.args.standardise_returns else None,
        },
    }, temp)
    temp.replace(path)


def restore_checkpoint(saved, policy, output, tasks, episodes_per_update, rng):
    learner = policy.learner
    policy.mac.agent.load_state_dict(saved["actor"])
    learner.old_mac.load_state(policy.mac)
    learner.critic.load_state_dict(saved["critic"])
    learner.target_critic.load_state_dict(saved["target_critic"])
    learner.agent_optimiser.load_state_dict(saved["actor_optimizer"])
    learner.critic_optimiser.load_state_dict(saved["critic_optimizer"])
    if policy.args.standardise_rewards:
        vars(learner.rew_ms).update(saved["reward_stats"])
    state = saved.get("training_state")
    if state is None:
        # Legacy checkpoints lack RNG states and counters. The complete-episode
        # ledger recovers task counts; advancing the generator retains task seeds.
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        record = next(row for row in reversed(records) if row["environment_steps"] == saved["environment_steps"])
        if record["update"] != saved["update"]:
            raise ValueError("Checkpoint and metric ledger updates differ")
        counts = record["task_environment_steps"]
        completed = saved["update"] * episodes_per_update
        rng.integers(0, 2**31, size=completed)
        learner.critic_training_steps = saved["update"]
        interval = policy.args.target_update_interval_or_tau
        learner.last_target_update_step = int(saved["update"] // interval * interval) if interval > 1 else 0
        learner.log_stats_t = saved["environment_steps"]
    else:
        counts, completed = state["task_environment_steps"], state["completed_episodes"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        rng.bit_generator.state = state["task_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if policy.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        for key in ("critic_training_steps", "last_target_update_step", "log_stats_t"):
            setattr(learner, key, state[key])
        if policy.args.standardise_returns:
            vars(learner.ret_ms).update(state["return_stats"])
    if set(counts) != set(tasks) or sum(counts.values()) != saved["environment_steps"]:
        raise ValueError("Checkpoint task counts do not match the training suite or saved steps")
    return dict(counts), completed, state is not None


def resume_logs(output, steps, completed):
    """Archive results newer than the checkpoint and keep one monotone history."""
    archive = output / "interrupted" / f"{steps:09d}-{timestamp().replace(':', '-')}"
    archive.mkdir(parents=True)
    for name in ("run.json", "metrics.json", "info.json"):
        shutil.copy2(output / name, archive / name)
    for name in ("metrics.jsonl", "training_episodes.jsonl"):
        path = output / name
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        valid = []
        for index, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                break
            valid.append(line)
        retained = ([line for line in valid if json.loads(line)["environment_steps"] <= steps]
                    if name == "metrics.jsonl" else valid[:completed])
        if name == "training_episodes.jsonl" and len(retained) != completed:
            raise ValueError("Episode ledger is shorter than the saved training history")
        if retained != lines:
            shutil.copy2(path, archive / name)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text("".join(retained), encoding="utf-8")
            temporary.replace(path)
    for pattern in ("validation_*.json", "test_*.json"):
        for path in output.glob(pattern):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["environment_steps"] > steps:
                path.replace(archive / path.name)
    return str(archive.relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epymarl-root", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="continue from OUTPUT/models/latest.pt")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--checkpoint-minutes", type=float, default=30)
    parser.add_argument("--eval-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=50001)
    parser.add_argument("--test-seed", type=int, default=100001)
    cli = parser.parse_args()
    if min(cli.steps, cli.threads, cli.episodes_per_task, cli.eval_episodes, cli.test_episodes) < 1:
        parser.error("Steps, threads and episode counts must be positive")
    if not np.isfinite(cli.checkpoint_minutes) or cli.checkpoint_minutes <= 0 or cli.eval_interval < 0:
        parser.error("Invalid checkpoint or evaluation interval")
    saved = None
    if cli.resume:
        saved = torch.load(cli.output / "models/latest.pt", map_location=cli.device, weights_only=False)
        if saved.get("schema") != "heterogrid-epymarl-mappo-v1":
            parser.error("Expected an EPyMARL MAPPO checkpoint")
        if saved["environment_steps"] >= cli.steps:
            parser.error("The checkpoint has already reached the cumulative step budget")
        for key in ("seed", "episodes_per_task", "eval_interval", "eval_episodes", "eval_seed", "test_episodes", "test_seed"):
            if saved["metadata"]["cli"][key] != getattr(cli, key):
                parser.error("Cannot resume with changed " + key)
    elif cli.output.exists():
        parser.error("Output exists; use --resume to continue its checkpoint")
    root, suite = epymarl_source(cli.epymarl_root), TaskSuite.load(cli.suite)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(cli.threads)
    device = torch.device(cli.device)
    rng = np.random.default_rng(cli.seed)
    tasks = suite.ids("train")
    schedule = [task for task in tasks for _ in range(cli.episodes_per_task)]
    envs = [TransferSMACAdapter(suite, task_ids=[task]) for task in schedule]
    policy = build_policy(root, envs[0].get_env_info(), device)
    if saved is not None:
        if saved["metadata"]["suite"] != suite.to_dict() or saved["metadata"]["algorithm_args"] != vars(policy.args):
            raise ValueError("Cannot resume with a changed suite or EPyMARL configuration")
    else:
        cli.output.mkdir(parents=True, exist_ok=False)
        (cli.output / "models").mkdir()
        suite.save(cli.output / "suite.json")
    metadata = {
        "name": "epymarl_mappo_transfer", "framework": "EPyMARL native recurrent MAPPO",
        "algorithm_args": vars(policy.args), "suite": suite.to_dict(),
        "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        "collection": "equal complete episode counts by source task; fresh on-policy batch",
        "padding": "unavailable/dead slots wait; learner uses native team-time filled mask",
        "termination": "configured time limit is a scored terminal",
    }
    external_files = ["config/default.yaml", "config/algs/mappo.yaml", "learners/ppo_learner.py",
                      "controllers/basic_controller.py", "modules/agents/rnn_agent.py", "modules/critics/centralV.py",
                      "components/episode_buffer.py", "components/action_selectors.py", "components/standarize_stream.py"]
    counts = dict.fromkeys(tasks, 0)
    steps = update = completed = 0
    resume_info = None
    if saved is not None:
        counts, completed, rng_restored = restore_checkpoint(saved, policy, cli.output, tasks, len(schedule), rng)
        steps, update = saved["environment_steps"], saved["update"]
        archive = resume_logs(cli.output, steps, completed)
        resume_info = {"environment_steps": steps, "update": update, "rng_restored": rng_restored,
                       "archived_interrupted_results": archive}
    recorder = ExperimentRecorder(cli.output, None, metadata,
                                  sources=[Path(__file__), cli.suite] + [root / p for p in external_files],
                                  resume_step=steps if saved is not None else None)
    if resume_info is not None:
        recorder.run["resumptions"][-1].update(resume_info)
        recorder.flush()
        print(json.dumps({"event": "resumed", **resume_info}), flush=True)
    starting_update = update
    last_save = perf_counter()
    next_eval = (steps // cli.eval_interval + 1) * cli.eval_interval if cli.eval_interval else 0

    def run_evaluation(split, episode_count, seed):
        results = evaluate(policy, suite, split, episode_count, seed, len(envs))
        path = cli.output / f"{split}_{steps:09d}.json"
        write_atomic(path, {"environment_steps": steps, "split": split, "tasks": results})
        rates = {task: stats["win_rate"] for task, stats in results.items()}
        recorder.log(steps, {f"{split}_win_rate": float(np.mean(list(rates.values())))})
        print(json.dumps({"environment_steps": steps, "split": split, "win_rates": rates}), flush=True)
        return path.name

    try:
        mode = "a" if saved is not None else "w"
        with (cli.output / "metrics.jsonl").open(mode, encoding="utf-8", buffering=1 << 20) as metrics, \
             (cli.output / "training_episodes.jsonl").open(mode, encoding="utf-8", buffering=1 << 20) as episodes:
            while steps < cli.steps:
                started = perf_counter()
                jobs = [(env, task, int(rng.integers(0, 2**31))) for env, task in zip(envs, schedule)]
                rows, batch = collect(policy, jobs, steps, training=True)
                for row in rows:
                    counts[row["task"]] += row["length"]
                    episodes.write(json.dumps(row, allow_nan=False) + "\n")
                steps, update, completed = sum(counts.values()), update + 1, completed + len(rows)
                policy.learner.train(batch, steps, completed)
                losses = dict(policy.logger.values)
                record = {"environment_steps": steps, "update": update, "task_environment_steps": dict(counts),
                          "training_win_rate": sum(row["won"] for row in rows) / len(rows),
                          "update_seconds": perf_counter() - started, **losses}
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                save = steps >= cli.steps or perf_counter() - last_save >= cli.checkpoint_minutes * 60
                if save:
                    checkpoint(cli.output / "models/latest.pt", policy, metadata, steps, update,
                               counts=counts, completed=completed, rng=rng)
                    last_save = perf_counter()
                flush = save or update == starting_update + 1
                recorder.log(steps, losses, flush=flush)
                if flush:
                    metrics.flush()
                    episodes.flush()
                    print(json.dumps(record), flush=True)
                if next_eval and steps >= next_eval and steps < cli.steps:
                    run_evaluation("validation", cli.eval_episodes, cli.eval_seed)
                    next_eval = (steps // cli.eval_interval + 1) * cli.eval_interval
            validation = run_evaluation("validation", cli.eval_episodes, cli.eval_seed)
            test = run_evaluation("test", cli.test_episodes, cli.test_seed)
        recorder.complete({"environment_steps": steps, "validation": validation, "test": test,
                           "checkpoint": "models/latest.pt"})
    finally:
        for env in envs:
            env.close()
        recorder.close()


if __name__ == "__main__":
    main()
