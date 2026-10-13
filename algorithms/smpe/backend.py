"""Episode collection and complete artifacts around the unchanged SMPE learner."""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.common.evaluation import observe
from heterogrid.core import REWARD_COMPONENTS


class LearnerLogger:
    def __init__(self):
        self.console_logger = logging.getLogger("smpe")
        self.values = {}
        self.step = 0

    def log_stat(self, name, value, step):
        value = float(value)
        if not np.isfinite(value):
            raise RuntimeError("Non-finite SMPE metric: " + name)
        self.values[name] = value
        self.step = int(step)


def build(root, settings, env_info, logger):
    sys.path.insert(0, str(root))
    from components.episode_buffer import EpisodeBatch, ReplayBuffer
    from components.transforms import OneHot
    from controllers.basic_controller import BasicMAC
    from learners.actor_critic_learner import ActorCriticVLearner
    from model_learners.vae_controller import VAEController

    args = SimpleNamespace(**settings)
    args.n_agents, args.n_actions = env_info["n_agents"], env_info["n_actions"]
    args.state_shape = args.state_dim = env_info["state_shape"]
    scheme = {"state": {"vshape": env_info["state_shape"]},
              "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu")
    vae = VAEController(buffer.scheme, args)
    mac = BasicMAC(buffer.scheme, groups, args, vae)
    learner = ActorCriticVLearner(mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    return SimpleNamespace(args=args, scheme=scheme, groups=groups, preprocess=preprocess,
                           EpisodeBatch=EpisodeBatch, buffer=buffer, mac=mac, vae=vae, learner=learner)


def new_batch(backend, size, length):
    return backend.EpisodeBatch(backend.scheme, backend.groups, size, length,
                               preprocess=backend.preprocess, device=backend.args.device)


def transition_data(envs):
    observations = [observe(env) for env in envs]
    return {"obs": np.asarray([row[0] for row in observations]),
            "state": np.asarray([row[1][0] for row in observations]),
            "avail_actions": np.asarray([row[2] for row in observations])}


@torch.no_grad()
def collect(backend, envs, t_env, random):
    batch = new_batch(backend, len(envs), envs[0].episode_limit + 1)
    # Padded slots must have a defined categorical distribution, but are never executed.
    batch["avail_actions"][..., 0] = 1
    seeds = [int(random.integers(0, 2 ** 31 - 1)) for _ in envs]
    for env, seed in zip(envs, seeds):
        env.reset(seed=seed)
    batch.update(transition_data(envs), ts=0)
    backend.mac.agent.eval()
    backend.mac.init_hidden(len(envs))
    done = [False] * len(envs)
    totals = [0.0] * len(envs)
    components = [dict.fromkeys(REWARD_COMPONENTS, 0.0) for _ in envs]
    rows, steps = [], 0
    for time in range(envs[0].episode_limit):
        active = [index for index, finished in enumerate(done) if not finished]
        actions = backend.mac.select_actions(batch, time, t_env, bs=active, test_mode=False)
        chosen = actions.cpu().numpy().astype(np.int64)
        available = batch["avail_actions"][active, time].cpu().numpy()
        if not np.take_along_axis(available, chosen[..., None], axis=-1).all():
            raise RuntimeError("SMPE selected a masked action")
        batch.update({"actions": chosen[..., None]}, bs=active, ts=time, mark_filled=False)
        rewards, terminations = [], []
        for local, index in enumerate(active):
            env = envs[index]
            reward, done[index], info = env.step(chosen[local].tolist())
            invalid = sum(name.startswith("red_") for name in info["invalid_actions"])
            if invalid:
                raise RuntimeError("SMPE executed an illegal action")
            totals[index] += reward
            for name in REWARD_COMPONENTS:
                components[index][name] += info["reward_components"][name]
            rewards.append([reward])
            # A scored timeout is a terminal game, not a bootstrap boundary.
            terminations.append([done[index]])
            steps += 1
            if done[index]:
                rows.append({"seed": seeds[index], "return": totals[index], "length": time + 1,
                             "winner": info["winner"], "reason": info["termination_reason"],
                             "invalid_red_actions": invalid, "reward_components": components[index],
                             **info["episode_metrics"]})
        batch.update({"reward": rewards, "terminated": terminations}, bs=active, ts=time, mark_filled=False)
        batch.update(transition_data([envs[index] for index in active]), bs=active, ts=time + 1)
        if all(done):
            break
    if not all(done):
        raise RuntimeError("Environment exceeded its episode limit")
    return batch, steps, rows


class EvaluationPolicy:
    def __init__(self, backend):
        self.backend = backend
        self.actor = backend.mac.agent

    @torch.no_grad()
    def act(self, obs, rnn, masks, available, deterministic=False):
        backend = self.backend
        batch = new_batch(backend, 1, 1)
        batch.update({"obs": np.asarray(obs)[None], "avail_actions": np.asarray(available)[None]}, ts=0)
        backend.mac.hidden_states = torch.as_tensor(rnn, device=backend.args.device).reshape(1, backend.args.n_agents, -1)
        probabilities = backend.mac.forward(batch, 0)
        actions = probabilities.argmax(dim=-1) if deterministic else torch.distributions.Categorical(probabilities).sample()
        hidden = backend.mac.hidden_states.reshape(backend.args.n_agents, 1, backend.args.hidden_dim)
        return actions.reshape(-1, 1), hidden


def modules(backend):
    learner = backend.learner
    return {"actor": backend.mac.agent, "critic": learner.critic, "critic_w": learner.critic_w,
            "target_critic": learner.target_critic, "target_critic_w": learner.target_critic_w,
            "agent_models": backend.vae.agent_models, "filters": backend.vae.filters,
            "target_filters": backend.vae.target_filters}


def running_stats(backend):
    result = {"obs": backend.vae.obs_ms, "state": backend.vae.state_ms, "reward": backend.vae.rew_ms}
    if backend.args.standardise_rewards:
        result["learner_reward"] = backend.learner.rew_ms
    if backend.args.standardise_returns:
        result["learner_return"] = backend.learner.ret_ms
    return result


def snapshot(backend):
    return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules(backend).items()}


def parameter_deltas(backend, before):
    return {name: sum(float((value.detach().cpu() - before[name][key]).square().sum())
                      for key, value in module.state_dict().items()) ** 0.5
            for name, module in modules(backend).items()}


def assert_finite(backend):
    for name, module in modules(backend).items():
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise RuntimeError("Non-finite SMPE parameters: " + name)


def checkpoint(path, backend, config, steps):
    learner, vae = backend.learner, backend.vae
    optimizers = {"actor": learner.agent_optimiser.state_dict(), "critic": learner.critic_optimiser.state_dict(),
                  "critic_w": learner.critic_optimiser_w.state_dict(),
                  "agent_models": [optimizer.state_dict() for optimizer in vae.agent_optimizers],
                  "filters": [optimizer.state_dict() for optimizer in vae.filter_optimizers]}
    counters = {name: [{"tables": torch.from_numpy(hasher.tables.copy()),
                       "projection_matrix": torch.from_numpy(hasher.projection_matrix.copy())} for hasher in hashers]
                for name, hashers in (("obs", vae.hash_obs), ("z", vae.hash_z))}
    value = {"schema_version": 1, "algorithm": "smpe", "environment_steps": steps,
             "env_config": config.to_dict(), "smpe_args": vars(backend.args),
             "modules": {name: module.state_dict() for name, module in modules(backend).items()},
             "normalizers": {name: {"mean": stats.mean, "var": stats.var, "count": float(stats.count)}
                             for name, stats in running_stats(backend).items()},
             "optimizers": optimizers, "hash_counts": counters,
             "learner_counters": {name: getattr(learner, name) for name in
                                  ("last_target_update_step", "critic_training_steps", "log_stats_t")}}
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def restore(path, backend):
    saved = torch.load(path, map_location=backend.args.device)
    for name, module in modules(backend).items():
        module.load_state_dict(saved["modules"][name])
    for name, stats in running_stats(backend).items():
        stats.mean = saved["normalizers"][name]["mean"]
        stats.var = saved["normalizers"][name]["var"]
        stats.count = saved["normalizers"][name]["count"]
    for name, hashers in (("obs", backend.vae.hash_obs), ("z", backend.vae.hash_z)):
        for hasher, state in zip(hashers, saved["hash_counts"][name]):
            hasher.tables = state["tables"].cpu().numpy().copy()
            hasher.projection_matrix = state["projection_matrix"].cpu().numpy().copy()
    return saved
