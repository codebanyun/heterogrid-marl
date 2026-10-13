"""Call EPyMARL's original MAA2C/COMA learners, controller and critics."""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


def external_module(root, relative):
    # Avoid learners/__init__.py: it imports unrelated PAC algorithms/dependencies.
    name = "heterogrid_epymarl_" + relative.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def algorithm_settings(root, algorithm):
    root = Path(root)
    with (root / "config/default.yaml").open() as handle:
        settings = yaml.safe_load(handle)
    with (root / f"config/algs/{algorithm}.yaml").open() as handle:
        settings.update(yaml.safe_load(handle))
    settings.update(use_cuda=False, common_reward=True, hidden_dim=64, batch_size_run=4,
                    batch_size=4, buffer_size=4, standardise_rewards=False,
                    standardise_returns=False, add_value_last_step=False,
                    learner_log_interval=1, obs_last_action=False, obs_agent_id=True,
                    obs_individual_obs=False, test_greedy=True)
    return settings


class MetricLogger:
    def __init__(self):
        self.values = {}
        self.console_logger = logging.getLogger("epymarl")

    def log_stat(self, name, value, step):
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite learner metric: " + name)
        self.values[name] = value


def build_policy(cli, env):
    root = Path(cli.epymarl_root)
    required = ("learners/actor_critic_learner.py", "learners/coma_learner.py",
                "controllers/basic_controller.py", "components/episode_buffer.py",
                "config/default.yaml", "config/algs/maa2c.yaml", "config/algs/coma.yaml")
    for name in required:
        if not (root / name).is_file():
            raise ValueError("Invalid EPyMARL src directory: missing " + name)
    sys.path.insert(0, str(root))
    episode_class = importlib.import_module("components.episode_buffer").EpisodeBatch
    onehot = importlib.import_module("components.transforms").OneHot
    controller = external_module(root, "controllers/basic_controller.py").BasicMAC
    filename, classname = (("actor_critic_learner", "ActorCriticLearner") if cli.algorithm == "maa2c"
                           else ("coma_learner", "COMALearner"))
    learner_class = getattr(external_module(root, "learners/" + filename + ".py"), classname)
    args = SimpleNamespace(**{**algorithm_settings(root, cli.algorithm), **env.get_env_info()})
    args.rnn_hidden_dim = args.hidden_dim
    scheme = {"state": {"vshape": args.state_shape},
              "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [onehot(out_dim=args.n_actions)])}
    template = episode_class(scheme, groups, 1, 1, preprocess=preprocess)
    mac = controller(template.scheme, groups, args)
    logger = MetricLogger()
    learner = learner_class(mac, template.scheme, logger, args)
    learner.target_critic.requires_grad_(False)
    policy = SimpleNamespace(args=args, mac=mac, learner=learner, logger=logger,
                             scheme=scheme, groups=groups, preprocess=preprocess,
                             episode_class=episode_class, n_actions=args.n_actions)
    for optimizer, parameters in ((learner.agent_optimiser, learner.agent_params),
                                  (learner.critic_optimiser, learner.critic_params)):
        original = optimizer.step

        def checked_step(*a, _original=original, _parameters=parameters, **kw):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in _parameters):
                raise FloatingPointError("Non-finite actor/critic gradient")
            result = _original(*a, **kw)
            if any(not torch.isfinite(p).all() for p in _parameters):
                raise FloatingPointError("Non-finite actor/critic parameter")
            return result

        optimizer.step = checked_step
    return policy, args


def episode_batch(policy, histories):
    """One fresh synchronous collection; each transition is learned exactly once."""
    length = max(len(h["u"]) for h in histories)
    batch = policy.episode_class(policy.scheme, policy.groups, len(histories), length + 1,
                                 preprocess=policy.preprocess)
    # Padding must have a legal action so that masked softmax is well-defined.
    batch["avail_actions"][..., 0] = 1
    for i, h in enumerate(histories):
        n = len(h["u"])
        if not n:
            raise ValueError("Cannot learn an empty episode")
        for key, source in (("obs", "o"), ("state", "s"), ("avail_actions", "avail")):
            value = torch.as_tensor(np.asarray(h[source]), dtype=batch[key].dtype).unsqueeze(0)
            batch.update({key: value}, bs=slice(i, i + 1), ts=slice(0, n + 1), mark_filled=False)
        batch.update({"actions": torch.as_tensor(np.asarray(h["u"]), dtype=torch.long)[None, ..., None],
                      "reward": torch.as_tensor(h["r"], dtype=torch.float32)[None, :, None]},
                     bs=slice(i, i + 1), ts=slice(0, n))
        batch["terminated"][i, n - 1, 0] = 1
    return batch


@torch.no_grad()
def act(policy, observations, available, hidden, rng, greedy=False):
    """Same masked-softmax recurrent policy as BasicMAC, using a separate NumPy RNG."""
    k, n, _ = observations.shape
    ids = np.broadcast_to(np.eye(n, dtype=np.float32), (k, n, n))
    inputs = np.concatenate((observations, ids), axis=-1).reshape(k * n, -1)
    logits, next_hidden = policy.mac.agent(torch.from_numpy(inputs),
                                         torch.from_numpy(hidden.reshape(k * n, -1)))
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite actor logits")
    logits[torch.from_numpy(available.reshape(k * n, -1)) == 0] = -1e10
    probabilities = torch.softmax(logits, dim=-1).numpy().reshape(k, n, -1)
    if greedy:
        actions = probabilities.argmax(axis=-1)
    else:
        # Categorical policy sampling, not epsilon-greedy as in value decomposition.
        cumulative = probabilities.astype(np.float64).cumsum(axis=-1)
        cumulative /= cumulative[..., -1:]
        draws = rng.random((k, n, 1))
        actions = (draws >= cumulative).sum(axis=-1)
    if not np.all(np.take_along_axis(available, actions[..., None], axis=-1)):
        raise RuntimeError("Actor selected an illegal action")
    return actions, next_hidden.numpy().reshape(k, n, -1)
