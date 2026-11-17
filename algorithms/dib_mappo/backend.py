"""External DIB modules with isolated imports and equivalence-tested replay kernels."""
from __future__ import annotations

import importlib
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from algorithms.common.actor_critic.backend import episode_batch


@contextmanager
def external_imports(root):
    """Do not reuse another EPyMARL/SMPE checkout's top-level Python modules."""
    names = {"components", "modules", "controllers", "learners", "utils"}
    saved = {key: value for key, value in sys.modules.items() if key.split(".")[0] in names}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        # Avoid registry __init__ imports of unrelated optional algorithms.
        for name in ("learners", "controllers"):
            package = types.ModuleType(name)
            package.__path__ = [str(root / name)]
            package.__file__ = str(root / name / "__init__.py")
            sys.modules[name] = package
        yield
    finally:
        for key in list(sys.modules):
            if key.split(".")[0] in names:
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path.remove(str(root))


TRAINING_PROFILES = ("legacy", "active_mask", "warmup1m", "active_mask_1m", "robust_comm_v1", "tactical_v2", "resilient_v3")


def settings(root, training_profile="legacy"):
    if training_profile not in TRAINING_PROFILES:
        raise ValueError("Unknown DIB training profile: " + str(training_profile))
    root = Path(root)
    if (root / "src").is_dir():
        root = root / "src"
    with (root / "config/default.yaml").open() as file:
        result = yaml.safe_load(file)
    with (root / "config/algs/dib_mappo_v2.yaml").open() as file:
        result.update(yaml.safe_load(file))
    result.update(use_cuda=False, device="cpu", common_reward=True, batch_size_run=4,
                  batch_size=4, buffer_size=4, add_value_last_step=False,
                  standardise_rewards=False, standardise_returns=False,
                  learner_log_interval=1, test_greedy=True, env="heterogrid", t_max=5_000_000)
    result.update(dib_training_profile=training_profile,
                  dib_use_alive_masks=training_profile in ("active_mask", "active_mask_1m", "robust_comm_v1", "tactical_v2", "resilient_v3"))
    if training_profile in ("warmup1m", "active_mask_1m"):
        # Keep the upstream 2M ramp. The actor gate and auxiliary schedule
        # start together, and both are frozen at the collection-batch clock.
        result.update(dib_warmup_steps=1_000_000, dib_actor_warmup_steps=1_000_000)
    if training_profile in ("robust_comm_v1", "tactical_v2", "resilient_v3"):
        result.update(dib_robust_entities=True, dib_warmup_steps=1_000_000,
            dib_actor_warmup_steps=1_500_000, dib_actor_ramp_steps=1_500_000,
            dib_belief_ramp_steps=500_000, dib_belief_decay_steps=3_500_000,
            dib_ahs_len_coef=0., dib_ahs_sharpness_coef=.001,
            dib_residual_actor_scale=.5, dib_residual_logit_bound=2.,
            dib_tau_d=.25, dib_target_sigma=1., dib_obs_sigma=1., dib_sep_margin=.15,
            dib_aux_to_pg_ratio=0., lambda_fwd=.1, entropy_coef_final=.003)
    if training_profile in ("tactical_v2", "resilient_v3"):
        result.update(dib_tactical=True,tactical_aux_weight=.1,tactical_preset='balanced',gae_lambda=.95,
            lr=.0005,dib_warmup_steps=50000,dib_belief_ramp_steps=200000,
            dib_actor_warmup_steps=250000,dib_actor_ramp_steps=750000,
            dib_belief_decay_steps=4750000,entropy_anneal_steps=3000000)
    if training_profile == "resilient_v3":
        result.update(dib_resilient=True,gamma=.997,gae_lambda=.99,lr=.0003,
            batch_size_run=8,batch_size=8,buffer_size=8,
            entropy_coef_start=.02,entropy_coef_final=.005,entropy_anneal_steps=4000000)
    return result


class Logger:
    def __init__(self):
        self.values = {}
        self.console_logger = logging.getLogger("dib_mappo")

    def log_stat(self, key, value, step):
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite DIB learner metric: " + key)
        if key == "agent_update_skipped" and value:
            raise FloatingPointError("Native DIB learner skipped a non-finite update")
        self.values[key] = value


def build_policy(cli, env):
    root = Path(cli.dib_root).resolve()
    if (root / "src").is_dir():
        root = root / "src"
    required = ("controllers/dib_controller.py", "learners/dib_ppo_learner.py",
                "learners/ppo_learner.py", "modules/agents/dib_recurrent_agent.py")
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Invalid DIB-MAPPO src directory")
    args = SimpleNamespace(**{**settings(root, getattr(cli, "training_profile", "legacy")), **env.get_env_info()})
    if getattr(args,'dib_tactical',False):
        from algorithms.dib_mappo.tactical import PRESETS
        args.tactical_preset=getattr(cli,'tactical_preset','balanced')
        args.tactical_aux_weight=PRESETS[args.tactical_preset]
    if getattr(args,'dib_resilient',False):
        args.resilient_coordinate_scale=[max(1,x-1) for x in env.env.config.map_size]
    scheme = {"state": {"vshape": args.state_shape}, "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    groups = {"agents": args.n_agents}
    with external_imports(root):
        episode_class = importlib.import_module("components.episode_buffer").EpisodeBatch
        onehot = importlib.import_module("components.transforms").OneHot
        controller = importlib.import_module("controllers.dib_controller").DIBMAC
        learner_class = importlib.import_module("learners.dib_ppo_learner").DIBPPOLearner
        preprocess = {"actions": ("actions_onehot", [onehot(out_dim=args.n_actions)])}
        template = episode_class(scheme, groups, 1, 1, preprocess=preprocess)
        mac = controller(template.scheme, groups, args)
        if getattr(args, "dib_tactical", False):
            from algorithms.dib_mappo.tactical import install_agent
            install_agent(mac.agent)
            if getattr(args,'dib_resilient',False):
                from algorithms.dib_mappo.resilient import install_agent as install_resilient
                install_resilient(mac.agent)
        elif getattr(args, "dib_robust_entities", False):
            from algorithms.dib_mappo.robust import install as install_robust
            install_robust(mac.agent)
        logger = Logger()
        learner = learner_class(mac, template.scheme, logger, args)
    learner.target_critic.requires_grad_(False)
    learner.old_mac.agent.requires_grad_(False)
    if getattr(cli, "sequence_acceleration", True):
        from algorithms.dib_mappo.sequence import install
        from algorithms.dib_mappo.auxiliary import install as install_auxiliary
        from algorithms.dib_mappo.returns import nstep_returns
        install(mac)
        install(learner.old_mac)
        if not getattr(args, "dib_robust_entities", False):
            install_auxiliary(mac.agent)
        learner.nstep_returns = lambda rewards, mask, values, nsteps: nstep_returns(
            args, rewards, mask, values, nsteps)
    if args.dib_use_alive_masks:
        from algorithms.dib_mappo.alive import install as install_alive_masks
        install_alive_masks(learner)
    if getattr(args,'dib_tactical',False):
        from algorithms.dib_mappo.tactical import install_learner
        install_learner(learner)
    policy = SimpleNamespace(args=args, mac=mac, learner=learner, logger=logger, scheme=scheme,
        groups=groups, preprocess=preprocess, episode_class=episode_class, n_actions=args.n_actions,
        batch_start_step=0, source_root=root)
    for optimizer, parameters in ((learner.agent_optimiser, learner.agent_params),
                                  (learner.critic_optimiser, learner.critic_params)):
        original = optimizer.step
        def checked_step(*a, _original=original, _parameters=parameters, **kw):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in _parameters):
                raise FloatingPointError("Non-finite DIB gradient")
            result = _original(*a, **kw)
            if any(not torch.isfinite(p).all() for p in _parameters):
                raise FloatingPointError("Non-finite DIB parameter")
            return result
        optimizer.step = checked_step
    return policy, args


def initial_memory(policy, batch_size):
    memory = {key: value.numpy().copy() for key, value in policy.mac.agent.init_hidden(batch_size).items()}
    memory["last_action"] = np.zeros((batch_size, policy.args.n_agents, policy.n_actions), np.float32)
    return memory


def select_memory(memory, slots):
    return {key: value[slots] for key, value in memory.items()}


def assign_memory(memory, slots, values):
    for key in memory:
        memory[key][slots] = values[key]


@torch.no_grad()
def act(policy, observations, available, memory, rng, greedy=False):
    hidden = {key: torch.from_numpy(value) for key, value in memory.items() if key != "last_action"}
    agent=policy.mac.agent
    if getattr(policy.args,'dib_tactical',False) and agent.residual_actor_gate==0:
        # A zero-gated hypothesis mixture equals the base policy. Each new
        # rollout resets memory, so unused belief state need not be advanced.
        b,n,o=observations.shape
        obs=torch.from_numpy(observations).reshape(b*n,o)
        previous=torch.from_numpy(memory['last_action']).reshape(b*n,policy.n_actions)
        ids=torch.eye(n).unsqueeze(0).expand(b,-1,-1).reshape(b*n,n)
        base=agent._finite(agent.base_policy_gru(agent.encode_local(obs,previous,ids),hidden['base_policy_h'].reshape(b*n,agent.hidden_dim)))
        logits=agent._finite(agent.action_logits(base,obs),clip=policy.args.dib_logit_clip).reshape(b,n,1,policy.n_actions)
        output={'logits_k':logits,'weights':logits.new_ones(b,n,1)}
        next_hidden={**hidden,'base_policy_h':base.reshape(b,n,agent.hidden_dim)}
    else:
        output, next_hidden = agent(torch.from_numpy(observations),
            torch.from_numpy(memory["last_action"]), hidden)
    # Exactly DIBMAC.forward's masked mixture of hypothesis-conditioned policies.
    logits = torch.nan_to_num(output["logits_k"], nan=0., posinf=1e10, neginf=-1e10)
    avail = torch.from_numpy(available).float()
    probabilities_k = torch.softmax(logits.masked_fill(avail.unsqueeze(2) == 0, -1e10), dim=-1)
    probabilities_k = torch.nan_to_num(probabilities_k, nan=0., posinf=1., neginf=0.)
    probabilities = (output["weights"].unsqueeze(-1) * probabilities_k).sum(dim=2) * avail
    probabilities = torch.nan_to_num(probabilities, nan=0., posinf=1., neginf=0.)
    mass = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(mass > 1e-8, probabilities / mass.clamp_min(1e-8),
                                avail / avail.sum(dim=-1, keepdim=True).clamp_min(1.0)).numpy()
    if greedy:
        actions = probabilities.argmax(axis=-1)
    else:
        cumulative = probabilities.astype(np.float64).cumsum(axis=-1)
        cumulative /= cumulative[..., -1:]
        actions = (rng.random((*observations.shape[:2], 1)) >= cumulative).sum(axis=-1)
    if not np.all(np.take_along_axis(available, actions[..., None], axis=-1)):
        raise RuntimeError("DIB actor selected a masked action")
    result = {key: value.numpy().copy() for key, value in next_hidden.items()}
    result["last_action"] = np.eye(policy.n_actions, dtype=np.float32)[actions]
    return actions, result
