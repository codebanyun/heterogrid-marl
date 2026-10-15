"""Shared policy evaluation, environment metrics and viewer replay recording."""
from pathlib import Path
import json

import numpy as np
import torch

from heterogrid import SMACAdapter
from heterogrid.core import REWARD_COMPONENTS
from heterogrid.smac import flatten_observation
from algorithms.common.diagnostics import summarize_team_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def observe(env):
    # The PettingZoo step already computed live observations and action masks.
    observations = [env.env._last_observations[n] if n in env.env._last_observations
                    else env.env.core.observation(n) for n in env.env.possible_agents]
    obs = np.asarray([flatten_observation(o) for o in observations], dtype=np.float32)
    state = np.repeat(env.get_state()[None], env.n_agents, axis=0)
    available = np.asarray([o["action_mask"] for o in observations], dtype=np.float32)
    active = np.asarray([[env.env.core.units[n].alive] for n in env.env.possible_agents], dtype=np.float32)
    return obs, state, available, active


@torch.no_grad()
def evaluate(policy, args, config, directory, episodes, seed, deterministic, save_replays=True, replay_limit=None,
             save_summary=True):
    if save_summary or save_replays:
        directory.mkdir(parents=True, exist_ok=True)
    env = SMACAdapter(config=config, record_replay=save_replays)
    rows = []
    trainer_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng = np.random.get_state()
    policy.actor.eval()
    try:
        for episode in range(episodes):
            record_episode = save_replays and (replay_limit is None or episode < replay_limit)
            env.env.record_replay = record_episode
            episode_seed = seed + episode
            torch.manual_seed(episode_seed)
            env.reset(seed=episode_seed)
            rnn = np.zeros((env.n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((env.n_agents, 1), dtype=np.float32)
            transitions = []
            total_reward = 0.0
            component_totals = dict.fromkeys(REWARD_COMPONENTS, 0.0)
            invalid = 0
            done = False
            while not done:
                obs, state, available, active = observe(env)
                action, rnn_tensor = policy.act(obs, rnn, masks, available, deterministic=deterministic)
                actions = action.cpu().numpy().reshape(env.n_agents).astype(np.int64)
                if np.any(available[np.arange(env.n_agents), actions] != 1):
                    raise RuntimeError("Policy selected a masked action")
                reward, done, info = env.step(actions.tolist())
                for key in REWARD_COMPONENTS:
                    component_totals[key] += info["reward_components"][key]
                invalid += sum(n.startswith("red_") for n in info["invalid_actions"])
                transitions.append({"step": env.env.core.step_count, "actions": actions.tolist(),
                                    "team_reward": reward, "reward_components": info["reward_components"], "done": done})
                total_reward += reward
                rnn = rnn_tensor.cpu().numpy()
            row = {"episode": episode, "seed": episode_seed, "return": total_reward,
                   "length": env.env.core.step_count, "winner": info["winner"],
                   "reason": info["termination_reason"], "invalid_red_actions": invalid,
                   "reward_components": component_totals, **info["episode_metrics"]}
            if "team_episode_metrics" in info:
                row["team_episode_metrics"] = info["team_episode_metrics"]
            rows.append(row)
            if record_episode:
                replay = Path(env.save_replay(directory / f"episode_{episode:03d}.json"))
                data = json.loads(replay.read_text(encoding="utf-8"))
                data.update(algorithm=args.algorithm_name, evaluation=row, transitions=transitions)
                write_json(replay, data)
                write_json(directory / f"actions_{episode:03d}.json", {"episode": row, "transitions": transitions})
    finally:
        env.close()
        torch.random.set_rng_state(trainer_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
    contacts = [r["first_contact_step"] for r in rows if r["first_contact_step"] is not None]
    result = {"episodes": episodes, "deterministic": deterministic,
              "win_rate": sum(r["winner"] == "red" for r in rows) / episodes,
              "mean_return": float(np.mean([r["return"] for r in rows])),
              "mean_length": float(np.mean([r["length"] for r in rows])),
              "timeout_rate": sum(r["reason"] == "timeout" for r in rows) / episodes,
              "contact_rate": len(contacts) / episodes,
              "mean_first_contact_step": float(np.mean(contacts)) if contacts else None,
              "mean_damage": float(np.mean([r["damage"] for r in rows])),
              "mean_kills": float(np.mean([r["kills"] for r in rows])),
              "mean_discovered_targets": float(np.mean([r["discovered_targets"] for r in rows])),
              "mean_useful_reports": float(np.mean([r["useful_reports"] for r in rows])),
              "mean_reward_components": {k: float(np.mean([r["reward_components"][k] for r in rows])) for k in REWARD_COMPONENTS},
              "invalid_red_actions": sum(r["invalid_red_actions"] for r in rows),
              **summarize_team_metrics(rows), "results": rows}
    if save_summary:
        write_json(directory / "summary.json", result)
    print(json.dumps({"training_seed": args.seed, "evaluation": directory.name,
                      **{k: v for k, v in result.items() if k != "results"}}), flush=True)
    return result
