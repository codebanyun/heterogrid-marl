"""Use the external EPyMARL BasicMAC, PPOLearner and native MAPPO preset."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from algorithms.common.actor_critic.backend import MetricLogger, external_module
from algorithms.common.evaluation import observe
from heterogrid.core import REWARD_COMPONENTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_root(root):
    root = Path(root).expanduser().resolve()
    if (root / "src").is_dir():
        root = root / "src"
    for name in ("learners/ppo_learner.py", "controllers/basic_controller.py", "config/algs/mappo.yaml"):
        if not (root / name).is_file():
            raise ValueError("Incomplete EPyMARL src directory: " + name)
    return root


def settings(root, device="cpu"):
    with (root / "config/default.yaml").open() as file:
        result = yaml.safe_load(file)
    with (root / "config/algs/mappo.yaml").open() as file:
        result.update(yaml.safe_load(file))
    result.update(device=device, use_cuda=device == "cuda", common_reward=True,
                  add_value_last_step=False, use_tensorboard=False, use_wandb=False)
    if result["batch_size_run"] != result["batch_size"] or result["buffer_size"] != result["batch_size"]:
        raise ValueError("EPyMARL MAPPO must use fresh complete on-policy batches")
    return result


def build(root, parameters, env_info):
    sys.path.insert(0, str(root))
    episode_module = importlib.import_module("components.episode_buffer")
    onehot = importlib.import_module("components.transforms").OneHot
    mac_class = external_module(root, "controllers/basic_controller.py").BasicMAC
    learner_class = external_module(root, "learners/ppo_learner.py").PPOLearner
    args = SimpleNamespace(**{**parameters, **env_info})
    scheme = {"state": {"vshape": args.state_shape}, "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [onehot(out_dim=args.n_actions)])}
    buffer = episode_module.ReplayBuffer(scheme, groups, args.buffer_size, args.episode_limit + 1,
                                         preprocess=preprocess, device="cpu")
    mac = mac_class(buffer.scheme, groups, args)
    logger = MetricLogger()
    learner = learner_class(mac, buffer.scheme, logger, args)
    learner.old_mac.agent.requires_grad_(False)
    learner.target_critic.requires_grad_(False)
    if args.use_cuda:
        learner.cuda()
    return SimpleNamespace(args=args, mac=mac, learner=learner, logger=logger, buffer=buffer,
                           scheme=scheme, groups=groups, preprocess=preprocess,
                           EpisodeBatch=episode_module.EpisodeBatch)


def new_batch(backend, count, length):
    return backend.EpisodeBatch(backend.scheme, backend.groups, count, length,
                               preprocess=backend.preprocess, device=backend.args.device)


def update_batch(batch, values, **kwargs):
    # EPyMARL's EpisodeBatch expects tensors; unlike SMPE it does not coerce arrays.
    tensors = {key: torch.as_tensor(np.asarray(value), dtype=batch[key].dtype, device=batch.device)
               for key, value in values.items()}
    batch.update(tensors, **kwargs)


def transition_data(envs):
    data = [observe(env) for env in envs]
    return {"obs": np.asarray([row[0] for row in data]), "state": np.asarray([row[1][0] for row in data]),
            "avail_actions": np.asarray([row[2] for row in data])}


@torch.no_grad()
def collect(backend, envs, steps, rng):
    from algorithms.common.training_seeds import episode_seeds
    batch = new_batch(backend, len(envs), envs[0].episode_limit + 1)
    batch["avail_actions"][..., 0] = 1
    seeds = episode_seeds(rng, len(envs))
    for env, seed in zip(envs, seeds):
        env.reset(seed=seed)
    update_batch(batch, transition_data(envs), ts=0)
    backend.mac.agent.eval()
    backend.mac.init_hidden(len(envs))
    done, totals = [False] * len(envs), [0.] * len(envs)
    components = [dict.fromkeys(REWARD_COMPONENTS, 0.) for _ in envs]
    rows, collected = [], 0
    for time in range(envs[0].episode_limit):
        active = [i for i, value in enumerate(done) if not value]
        chosen = backend.mac.select_actions(batch, time, steps, bs=active, test_mode=False).cpu().numpy()
        available = batch["avail_actions"][active, time].cpu().numpy()
        if not np.take_along_axis(available, chosen[..., None], axis=-1).all():
            raise RuntimeError("EPyMARL MAPPO selected an illegal action")
        update_batch(batch, {"actions": chosen[..., None]}, bs=active, ts=time, mark_filled=False)
        rewards, terminal = [], []
        for local, index in enumerate(active):
            env = envs[index]
            reward, done[index], info = env.step(chosen[local].tolist())
            if any(name.startswith("red_") for name in info["invalid_actions"]):
                raise RuntimeError("EPyMARL MAPPO executed an illegal action")
            collected += 1
            totals[index] += reward
            for name in REWARD_COMPONENTS:
                components[index][name] += info["reward_components"][name]
            rewards.append([reward])
            terminal.append([done[index]])
            if done[index]:
                row = dict(seed=seeds[index], length=time + 1, **{"return": totals[index]},
                           winner=info["winner"], reason=info["termination_reason"], invalid_red_actions=0,
                           reward_components=components[index], **info["episode_metrics"])
                if "team_episode_metrics" in info:
                    row["team_episode_metrics"] = info["team_episode_metrics"]
                rows.append(row)
        update_batch(batch, {"reward": rewards, "terminated": terminal}, bs=active, ts=time, mark_filled=False)
        update_batch(batch, transition_data([envs[i] for i in active]), bs=active, ts=time + 1)
        if all(done):
            break
    if not all(done):
        raise RuntimeError("EPyMARL collection exceeded the episode limit")
    return batch, collected, rows


class EvaluationPolicy:
    def __init__(self, backend):
        self.backend, self.actor = backend, backend.mac.agent

    @torch.no_grad()
    def act(self, obs, hidden, masks, available, deterministic=False):
        backend = self.backend
        batch = new_batch(backend, 1, 1)
        update_batch(batch, {"obs": np.asarray(obs)[None], "avail_actions": np.asarray(available)[None]}, ts=0)
        backend.mac.hidden_states = torch.as_tensor(hidden, device=backend.args.device).reshape(1, backend.args.n_agents, -1)
        probabilities = backend.mac.forward(batch, 0, test_mode=True)
        actions = probabilities.argmax(-1) if deterministic else torch.distributions.Categorical(probabilities).sample()
        hidden = backend.mac.hidden_states.reshape(backend.args.n_agents, 1, backend.args.hidden_dim)
        return actions.reshape(-1, 1), hidden


def modules(backend):
    return {"actor": backend.mac.agent, "old_actor": backend.learner.old_mac.agent,
            "critic": backend.learner.critic, "target_critic": backend.learner.target_critic}


def snapshot(backend):
    return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules(backend).items()}


def checkpoint(path, backend, config, steps):
    learner = backend.learner
    normalizers = {name: {"mean": value.mean, "var": value.var, "count": float(value.count)}
                   for name, value in (("reward", getattr(learner, "rew_ms", None)),
                                       ("return", getattr(learner, "ret_ms", None))) if value is not None}
    value = {"algorithm": "epymarl_mappo", "environment_steps": steps, "env_config": config.to_dict(),
             "algorithm_args": vars(backend.args), "modules": {k: v.state_dict() for k, v in modules(backend).items()},
             "optimizers": {"actor": learner.agent_optimiser.state_dict(), "critic": learner.critic_optimiser.state_dict()},
             "normalizers": normalizers, "learner_counters": {name: getattr(learner, name) for name in
             ("last_target_update_step", "critic_training_steps", "log_stats_t")}}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def restore(path, backend):
    saved = torch.load(path, map_location=backend.args.device)
    for name, module in modules(backend).items():
        module.load_state_dict(saved["modules"][name])
    for name, value in saved["normalizers"].items():
        normalizer = getattr(backend.learner, "rew_ms" if name == "reward" else "ret_ms")
        normalizer.mean, normalizer.var, normalizer.count = value["mean"], value["var"], value["count"]
    for name, value in saved["learner_counters"].items():
        setattr(backend.learner, name, value)
    return saved


def network_configuration(backend):
    args = backend.args
    assert not args.obs_last_action and args.obs_agent_id and not args.obs_individual_obs
    assert backend.mac.agent.fc1.in_features == args.obs_shape + args.n_agents
    assert backend.learner.critic.fc1.in_features == args.state_shape + args.n_agents
    return {"actor_uses_global_state": False, "actor_input": args.obs_shape + args.n_agents,
            "critic_input": args.state_shape + args.n_agents, "hidden_dim": args.hidden_dim,
            "use_rnn": args.use_rnn, "ppo_epochs": args.epochs, "clip": args.eps_clip,
            "lr": args.lr, "entropy_coef": args.entropy_coef, "batch_size": args.batch_size,
            "q_nstep": args.q_nstep, "standardise_rewards": args.standardise_rewards,
            "modules": {name: {"architecture": str(module), "parameters": sum(p.numel() for p in module.parameters())}
                        for name, module in modules(backend).items()}}
