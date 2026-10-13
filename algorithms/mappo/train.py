"""Train with the policy, PPO trainer and buffer from an existing MAPPO checkout."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch

from heterogrid import EnvConfig, SMACAdapter
from heterogrid.core import REWARD_COMPONENTS
from algorithms.common.evaluation import evaluate, observe, write_json
from algorithms.common.experiment_log import ExperimentRecorder
from algorithms.mappo.settings import DEFAULT_CONFIG, PROJECT_ROOT, add_backend_argument, validate_backend



def numpy_batch(tensor, count):
    return np.asarray(np.split(tensor.detach().cpu().numpy(), count))


def checkpoint(path, policy, trainer, args, config, steps):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"actor": policy.actor.state_dict(), "critic": policy.critic.state_dict(),
                "actor_optimizer": policy.actor_optimizer.state_dict(),
                "critic_optimizer": policy.critic_optimizer.state_dict(),
                "value_normalizer": trainer.value_normalizer.state_dict(),
                "mappo_args": vars(args), "env_config": config.to_dict(),
                "environment_steps": steps}, temporary)
    temporary.replace(path)


def restore(path, policy, trainer, device):
    saved = torch.load(path, map_location=device)
    policy.actor.load_state_dict(saved["actor"])
    policy.critic.load_state_dict(saved["critic"])
    trainer.value_normalizer.load_state_dict(saved["value_normalizer"])
    return saved



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--algorithm", choices=["mappo", "rmappo"], default="mappo")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=100042)
    parser.add_argument("--eval-interval", type=int, default=0, help="Environment steps between validation evaluations; 0 disables")
    parser.add_argument("--save-interval", type=int, help="Environment steps between model snapshots; defaults to 25 rollouts")
    parser.add_argument("--trajectory-interval", type=int, default=0, help="Save sampled rollouts at this interval; 0 saves every rollout")
    parser.add_argument("--no-trajectories", action="store_true", help="Do not retain training rollout arrays")
    parser.add_argument("--no-evaluation-files", action="store_true", help="Keep periodic evaluation metrics without separate directories")
    parser.add_argument("--log-interval", type=int, default=0, help="Export metric series at this interval; 0 exports every rollout")
    parser.add_argument("--replay-episodes", type=int, help="Maximum full replays per final evaluation group; all episodes are scored")
    parser.add_argument("--heldout-episodes", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=200042)
    parser.add_argument("--results-root", type=Path, help="Optional numbered-results index; otherwise records stay in --output")
    parser.add_argument("--results-env-name", help="Optional results-index task name; defaults to the config filename")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a saved checkpoint without training")
    cli = parser.parse_args()
    if cli.results_env_name is None:
        cli.results_env_name = cli.config.stem
    batch_size = cli.num_envs * cli.rollout_length
    if cli.save_interval is None:
        cli.save_interval = 25 * batch_size
    if min(cli.steps, cli.num_envs, cli.rollout_length, cli.ppo_epochs,
           cli.mini_batches, cli.hidden_size, cli.eval_episodes) < 1:
        parser.error("Training sizes and episode counts must be positive")
    if cli.steps % (cli.num_envs * cli.rollout_length):
        parser.error("--steps must be a multiple of --num-envs * --rollout-length")
    if cli.eval_interval < 0 or cli.eval_interval % (cli.num_envs * cli.rollout_length):
        parser.error("--eval-interval must be zero or a multiple of the rollout batch size")
    if cli.save_interval <= 0 or cli.save_interval % batch_size:
        parser.error("--save-interval must be a positive multiple of the rollout batch size")
    if any(v < 0 or v % batch_size for v in (cli.trajectory_interval, cli.log_interval)):
        parser.error("Trajectory/log intervals must be zero or positive multiples of the rollout batch size")
    if cli.replay_episodes is not None and cli.replay_episodes < 0:
        parser.error("--replay-episodes must be nonnegative")
    if cli.heldout_episodes < 0:
        parser.error("--heldout-episodes must be nonnegative")
    if cli.heldout_episodes and max(cli.eval_seed, cli.heldout_seed) < min(cli.eval_seed + cli.eval_episodes, cli.heldout_seed + cli.heldout_episodes):
        parser.error("Validation and held-out seed ranges must not overlap")
    if cli.mini_batches > cli.num_envs * cli.rollout_length:
        parser.error("Too many mini-batches")
    if cli.algorithm == "rmappo" and cli.rollout_length % 16:
        parser.error("R-MAPPO requires --rollout-length to be a multiple of 16")
    cli.mappo_root = validate_backend(parser, cli.mappo_root)
    sys.path.insert(0, str(cli.mappo_root.resolve()))
    from onpolicy.config import get_config
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
    from onpolicy.utils.shared_buffer import SharedReplayBuffer

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(1)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config = EnvConfig.load(cli.config) if cli.config else EnvConfig()
    args = get_config().parse_args([])
    args.algorithm_name = cli.algorithm
    args.env_name = "HeteroGrid"
    args.use_recurrent_policy = cli.algorithm == "rmappo"
    args.data_chunk_length = 16
    args.use_naive_recurrent_policy = False
    args.use_wandb = False
    args.use_centralized_V = True
    args.use_policy_active_masks = True
    args.use_value_active_masks = True
    # HeteroGrid scores the game at its horizon; timeout is a finite-horizon terminal.
    args.use_proper_time_limits = False
    args.n_rollout_threads = cli.num_envs
    args.episode_length = cli.rollout_length
    args.hidden_size = cli.hidden_size
    args.ppo_epoch = cli.ppo_epochs
    args.num_mini_batch = cli.mini_batches
    args.seed = cli.seed
    args.num_env_steps = cli.steps
    if cli.checkpoint:
        saved = torch.load(cli.checkpoint, map_location="cpu")
        config = EnvConfig.from_dict(saved["env_config"])
        args = argparse.Namespace(**saved["mappo_args"])
    default_output = Path("output") / args.algorithm_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = cli.output or default_output
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "replays") + (() if cli.no_trajectories else ("trajectories",)):
        (output / name).mkdir()
    metadata = {"algorithm": args.algorithm_name, "environment": config.to_dict(),
               "mappo_args": vars(args), "device": str(device), "mappo_root": str(cli.mappo_root.resolve()),
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "evaluation_seed": cli.eval_seed,
               "timeout_handling": "finite-horizon terminal; no value bootstrap at game end",
               "communication": "environment configuration; no learned message head",
               "name": args.algorithm_name, "seed": cli.seed, "t_max": cli.steps,
               "env": "heterogrid", "env_args": config.to_dict(), "cli": vars(cli).copy(),
               "results_group": "heterogrid", "results_env_name": cli.results_env_name}
    metadata["cli"] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata["cli"].items()}
    sources = [Path(__file__)] + list((PROJECT_ROOT / "algorithms/common").glob("*.py"))
    sources += [Path(__file__).with_name(name) for name in ("launch.py", "matrix.py", "study.py", "ablate.py", "audit.py", "settings.py")]
    sources += list((PROJECT_ROOT / "heterogrid").glob("*.py"))
    sources += [cli.config, cli.mappo_root / "onpolicy/config.py"]
    sources += list((cli.mappo_root / "onpolicy/algorithms").rglob("*.py"))
    sources += list((cli.mappo_root / "onpolicy/utils").glob("*.py"))
    group = cli.results_root / args.algorithm_name / "heterogrid" / cli.results_env_name if cli.results_root else None
    recorder = ExperimentRecorder(output, group,
                                  metadata, sources=[p for p in sources if p is not None])

    envs = [SMACAdapter(config=config) for _ in range(cli.num_envs)]
    env_info = envs[0].get_env_info()
    n_agents = env_info["n_agents"]
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(env_info["obs_shape"],), dtype=np.float32)
    state_space = gym.spaces.Box(-np.inf, np.inf, shape=(env_info["state_shape"],), dtype=np.float32)
    action_space = gym.spaces.Discrete(env_info["n_actions"])
    policy = R_MAPPOPolicy(args, obs_space, state_space, action_space, device=device)
    trainer = R_MAPPO(args, policy, device=device)
    print(json.dumps({"output": str(output.resolve()), "device": str(device), "env_info": env_info}), flush=True)
    def final_evaluation(directory, episodes, seed, deterministic, evaluated_policy=None):
        return evaluate(evaluated_policy or policy, args, config, directory, episodes, seed, deterministic,
                        replay_limit=cli.replay_episodes)
    if cli.checkpoint:
        restore(cli.checkpoint, policy, trainer, device)
        try:
            for deterministic in (False, True):
                final_evaluation(output / "replays" / ("sampled" if not deterministic else "greedy"),
                                 cli.eval_episodes, cli.eval_seed, deterministic)
        finally:
            for env in envs:
                env.close()
        recorder.complete({"evaluation_only": True})
        return

    initial_actor = {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}
    initial_critic = {k: v.detach().cpu().clone() for k, v in policy.critic.state_dict().items()}
    checkpoint(output / "models" / "initial.pt", policy, trainer, args, config, 0)
    before = final_evaluation(output / "replays" / "before", cli.eval_episodes, cli.eval_seed, False)
    recorder.evaluation(0, before, False)
    best_score, best_step = None, 0
    def validation(step, sampled, greedy):
        nonlocal best_score, best_step
        recorder.evaluation(step, sampled, False)
        recorder.evaluation(step, greedy, True)
        # Use validation only; held-out seeds never participate in checkpoint selection.
        score = (greedy["win_rate"], greedy["mean_return"])
        if best_score is None or score > best_score:
            best_score, best_step = score, step
            checkpoint(output / "models" / "best.pt", policy, trainer, args, config, step)
    if cli.eval_interval:
        initial_greedy = evaluate(policy, args, config, output / "evaluations" / "step_000000_greedy",
                                  cli.eval_episodes, cli.eval_seed, True, save_replays=False,
                                  save_summary=not cli.no_evaluation_files)
        # Sampled baseline has already been recorded.
        recorder.evaluation(0, initial_greedy, True)
        best_score = (initial_greedy["win_rate"], initial_greedy["mean_return"])
        checkpoint(output / "models" / "best.pt", policy, trainer, args, config, 0)
    heldout_before = {}
    if cli.heldout_episodes:
        for deterministic in (False, True):
            mode = "greedy" if deterministic else "sampled"
            heldout_before[mode] = evaluate(policy, args, config, output / "evaluations" / ("heldout_before_" + mode),
                cli.heldout_episodes, cli.heldout_seed, deterministic, save_replays=False,
                save_summary=not cli.no_evaluation_files)
    buffer = SharedReplayBuffer(args, n_agents, obs_space, state_space, action_space)
    for i, env in enumerate(envs):
        env.reset(seed=cli.seed + i)
        obs, state, available, active = observe(env)
        buffer.obs[0, i], buffer.share_obs[0, i] = obs, state
        buffer.available_actions[0, i], buffer.active_masks[0, i] = available, active
    updates = cli.steps // (cli.num_envs * cli.rollout_length)
    episode_returns = np.zeros(cli.num_envs)
    episode_components = {k: np.zeros(cli.num_envs) for k in REWARD_COMPONENTS}
    episodes = []
    invalid_red_actions = 0
    saved_rollouts = 0
    start = time.perf_counter()
    metrics_path = output / "metrics.jsonl"
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics, \
                (output / "training_episodes.jsonl").open("w", encoding="utf-8") as episode_log:
            for update in range(1, updates + 1):
                trainer.prep_rollout()
                terminal_flags = np.zeros((cli.rollout_length, cli.num_envs), dtype=bool)
                reward_components = {k: np.zeros((cli.rollout_length, cli.num_envs), dtype=np.float32) for k in REWARD_COMPONENTS}
                for step in range(cli.rollout_length):
                    with torch.no_grad():
                        tensors = policy.get_actions(np.concatenate(buffer.share_obs[step]),
                            np.concatenate(buffer.obs[step]), np.concatenate(buffer.rnn_states[step]),
                            np.concatenate(buffer.rnn_states_critic[step]), np.concatenate(buffer.masks[step]),
                            np.concatenate(buffer.available_actions[step]))
                    values, actions, log_probs, rnn, critic_rnn = [numpy_batch(t, cli.num_envs) for t in tensors]
                    next_obs, next_state, next_available, next_active = [], [], [], []
                    rewards = np.zeros((cli.num_envs, n_agents, 1), dtype=np.float32)
                    masks = np.ones_like(rewards)
                    for i, env in enumerate(envs):
                        selected = actions[i, :, 0].astype(np.int64)
                        if np.any(buffer.available_actions[step, i, np.arange(n_agents), selected] != 1):
                            raise RuntimeError("Policy selected a masked action")
                        reward, done, info = env.step(selected.tolist())
                        for key in REWARD_COMPONENTS:
                            reward_components[key][step, i] = info["reward_components"][key]
                            episode_components[key][i] += info["reward_components"][key]
                        invalid_red_actions += sum(n.startswith("red_") for n in info["invalid_actions"])
                        rewards[i, :, 0] = reward
                        episode_returns[i] += reward
                        terminal_flags[step, i] = done
                        if done:
                            row = {"env": i, "return": float(episode_returns[i]), "winner": info["winner"],
                                   "length": env.env.core.step_count, "reason": info["termination_reason"],
                                   "reward_components": {k: float(episode_components[k][i]) for k in REWARD_COMPONENTS},
                                   **info["episode_metrics"]}
                            episodes.append(row)
                            episode_log.write(json.dumps(row, allow_nan=False) + "\n")
                            episode_returns[i] = 0
                            for key in REWARD_COMPONENTS:
                                episode_components[key][i] = 0
                            env.reset()
                            masks[i] = 0
                            rnn[i] = 0
                            critic_rnn[i] = 0
                        obs, state, available, active = observe(env)
                        next_obs.append(obs)
                        next_state.append(state)
                        next_available.append(available)
                        next_active.append(active)
                    buffer.insert(np.asarray(next_state), np.asarray(next_obs), rnn, critic_rnn,
                                  actions, log_probs, values, rewards, masks,
                                  active_masks=np.asarray(next_active), available_actions=np.asarray(next_available))
                with torch.no_grad():
                    next_values = policy.get_values(np.concatenate(buffer.share_obs[-1]),
                        np.concatenate(buffer.rnn_states_critic[-1]), np.concatenate(buffer.masks[-1]))
                buffer.compute_returns(numpy_batch(next_values, cli.num_envs), trainer.value_normalizer)
                trainer.prep_training()
                losses = {k: float(v) for k, v in trainer.train(buffer).items()}
                if not all(np.isfinite(v) for v in losses.values()):
                    raise RuntimeError(f"Non-finite training metrics: {losses}")
                total_steps = update * cli.num_envs * cli.rollout_length
                episode_log.flush()
                if not cli.no_trajectories and (not cli.trajectory_interval or total_steps % cli.trajectory_interval == 0 or update in (1, updates)):
                    # T+1 observations retain the next state; masks distinguish resets from transitions.
                    np.savez_compressed(output / "trajectories" / f"rollout_{update:05d}.npz",
                        obs=buffer.obs, state=buffer.share_obs, available_actions=buffer.available_actions,
                        actions=buffer.actions, rewards=buffer.rewards, masks=buffer.masks,
                        active_masks=buffer.active_masks, values=buffer.value_preds, returns=buffer.returns,
                        action_log_probs=buffer.action_log_probs, environment_done=terminal_flags,
                        **{f"reward_{k}": v for k, v in reward_components.items()})
                    saved_rollouts += 1
                recent = episodes[-32:]
                row = {"seed": cli.seed, "update": update, "environment_steps": total_steps, "episodes": len(episodes),
                       "recent_return": float(np.mean([r["return"] for r in recent])) if recent else None,
                       "recent_win_rate": sum(r["winner"] == "red" for r in recent) / len(recent) if recent else None,
                       "recent_timeout_rate": sum(r["reason"] == "timeout" for r in recent) / len(recent) if recent else None,
                       "combat_zero_reward_fraction": float(np.mean((reward_components["damage"] + reward_components["kill"]) == 0)),
                       "mean_reward_components": {k: float(v.mean()) for k, v in reward_components.items()},
                       "invalid_red_actions": invalid_red_actions, **losses}
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                metrics.flush()
                log_due = not cli.log_interval or total_steps % cli.log_interval == 0 or update in (1, updates)
                if log_due:
                    recorder.log(total_steps, {"episode": len(episodes), "return_mean": row["recent_return"],
                        "battle_won_mean": row["recent_win_rate"], "timeout_mean": row["recent_timeout_rate"],
                        "ep_length_mean": float(np.mean([r["length"] for r in recent])) if recent else None,
                        "combat_zero_reward_fraction": row["combat_zero_reward_fraction"], **losses})
                if (cli.log_interval and log_due) or update == 1 or (not cli.log_interval and update % 10 == 0) or update == updates:
                    print(json.dumps(row), flush=True)
                if total_steps % cli.save_interval == 0 or update == updates:
                    checkpoint(output / "models" / "latest.pt", policy, trainer, args, config, total_steps)
                    if update != updates:
                        checkpoint(output / "models" / f"step_{total_steps:08d}.pt", policy, trainer, args, config, total_steps)
                if cli.eval_interval and total_steps % cli.eval_interval == 0 and update != updates:
                    sampled_eval = evaluate(policy, args, config, output / "evaluations" / f"step_{total_steps:06d}_sampled",
                                            cli.eval_episodes, cli.eval_seed, False, save_replays=False,
                                            save_summary=not cli.no_evaluation_files)
                    greedy_eval = evaluate(policy, args, config, output / "evaluations" / f"step_{total_steps:06d}_greedy",
                                           cli.eval_episodes, cli.eval_seed, True, save_replays=False,
                                           save_summary=not cli.no_evaluation_files)
                    validation(total_steps, sampled_eval, greedy_eval)
                buffer.after_update()
    finally:
        for env in envs:
            env.close()
    elapsed = time.perf_counter() - start
    actor_delta = sum(float((v.detach().cpu() - initial_actor[k]).square().sum()) for k, v in policy.actor.state_dict().items()) ** 0.5
    critic_delta = sum(float((v.detach().cpu() - initial_critic[k]).square().sum()) for k, v in policy.critic.state_dict().items()) ** 0.5
    if actor_delta == 0 or critic_delta == 0:
        raise RuntimeError("Actor or critic parameters did not update")
    # Evaluate the reloaded artifact, not just the in-memory training policy.
    reloaded = R_MAPPOPolicy(args, obs_space, state_space, action_space, device=device)
    reloaded_trainer = R_MAPPO(args, reloaded, device=device)
    restore(output / "models" / "latest.pt", reloaded, reloaded_trainer, device)
    reload_equal = all(torch.equal(v, reloaded.actor.state_dict()[k]) for k, v in policy.actor.state_dict().items())
    if not reload_equal:
        raise RuntimeError("Checkpoint round trip changed actor parameters")
    after = final_evaluation(output / "replays" / "after", cli.eval_episodes, cli.eval_seed, False, reloaded)
    greedy = final_evaluation(output / "replays" / "greedy", cli.eval_episodes, cli.eval_seed, True, reloaded)
    validation(cli.steps, after, greedy)
    heldout = {}
    if cli.heldout_episodes:
        for label, model in (("latest", "latest.pt"), ("best", "best.pt")):
            restore(output / "models" / model, reloaded, reloaded_trainer, device)
            heldout[label] = {}
            for deterministic in (False, True):
                mode = "greedy" if deterministic else "sampled"
                heldout[label][mode] = final_evaluation(output / "replays" / ("heldout_" + label + "_" + mode),
                                                      cli.heldout_episodes, cli.heldout_seed, deterministic, reloaded)
    summary = {"algorithm": args.algorithm_name, "environment_steps": cli.steps, "updates": updates,
               "training_episodes": len(episodes), "training_seconds": elapsed,
               "environment_steps_per_second": cli.steps / elapsed,
               "actor_parameter_delta_l2": actor_delta, "critic_parameter_delta_l2": critic_delta,
               "checkpoint_reload_equal": reload_equal, "invalid_red_actions": invalid_red_actions,
               "saved_rollouts": saved_rollouts, "trajectory_interval": cli.trajectory_interval,
               "before": before, "after": after, "greedy": greedy,
               "best_validation_step": best_step, "best_validation_score": best_score,
               "heldout_before": heldout_before, "heldout": heldout}
    write_json(output / "summary.json", summary)
    write_json(output / "training_episodes.json", episodes)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("before", "after", "greedy", "heldout", "heldout_before")}), flush=True)
    recorder.complete({"environment_steps": cli.steps, "best_validation_step": best_step,
                       "final_validation_win_rate": greedy["win_rate"]})


if __name__ == "__main__":
    main()
