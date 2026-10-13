"""Reuse external learners, with batched local execution and aligned RNN history."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch


def load_backend(root):
    root = Path(root).resolve()
    for name in ("policy/qmix.py", "policy/qtran_base.py", "network/base_net.py", "common/arguments.py"):
        if not (root / name).is_file():
            raise ValueError("Invalid MARL-Algorithms checkout: missing " + name)
    # Its top-level 'common' is unrelated to algorithms.common.
    sys.path.insert(0, str(root))
    arguments = importlib.import_module("common.arguments")
    qmix = importlib.import_module("policy.qmix").QMIX
    qtran = importlib.import_module("policy.qtran_base").QtranBase
    return arguments, qmix, qtran


def make_args(cli, env, arguments):
    info = env.get_env_info()
    args = arguments.get_mixer_args(argparse.Namespace(
        alg="qmix" if cli.algorithm == "qmix" else "qtran_base", map=cli.config.stem,
        gamma=0.99, optimizer="RMS", last_action=True, reuse_network=True,
        cuda=False, load_model=False, model_dir=str(cli.output / "models"),
        n_actions=info["n_actions"], n_agents=info["n_agents"],
        obs_shape=info["obs_shape"], state_shape=info["state_shape"], episode_limit=info["episode_limit"]))
    args.buffer_size = cli.buffer_size
    args.batch_size = cli.batch_size
    return args


def recurrent_values(policy, batch, length):
    """Both networks process o_0 before producing values for o_1.

    This repairs upstream QMIX's target-history offset; QTRAN already warms up.
    Targets are detached, leaving the external objectives/optimizers unchanged.
    """
    b, _, n, _ = batch["o"].shape
    ids = torch.eye(n).unsqueeze(0).expand(b, -1, -1)
    eval_hidden = torch.zeros(b * n, policy.args.rnn_hidden_dim)
    target_hidden = torch.zeros_like(eval_hidden)
    eval_values, target_values, eval_states, target_states = [], [], [], []
    for t in range(length):
        previous = torch.zeros_like(batch["u_onehot"][:, t]) if t == 0 else batch["u_onehot"][:, t - 1]
        inputs = torch.cat((batch["o"][:, t], previous, ids), dim=-1).reshape(b * n, -1)
        inputs_next = torch.cat((batch["o_next"][:, t], batch["u_onehot"][:, t], ids), dim=-1).reshape(b * n, -1)
        if t == 0:
            with torch.no_grad():
                _, target_hidden = policy.target_rnn(inputs, target_hidden)
        q, eval_hidden = policy.eval_rnn(inputs, eval_hidden)
        with torch.no_grad():
            target, target_hidden = policy.target_rnn(inputs_next, target_hidden)
        eval_values.append(q.reshape(b, n, -1))
        target_values.append(target.reshape(b, n, -1))
        eval_states.append(eval_hidden.reshape(b, n, -1))
        target_states.append(target_hidden.reshape(b, n, -1))
    return tuple(torch.stack(values, dim=1) for values in (eval_values, target_values, eval_states, target_states))


def build_policy(cli, env):
    arguments, qmix_class, qtran_class = load_backend(cli.marl_root)
    args = make_args(cli, env, arguments)

    class AlignedQMIX(qmix_class):
        def get_q_values(self, batch, max_episode_len):
            return recurrent_values(self, batch, max_episode_len)[:2]

    class AlignedQTRAN(qtran_class):
        def _get_individual_q(self, batch, max_episode_len):
            return recurrent_values(self, batch, max_episode_len)

    policy = (AlignedQMIX if cli.algorithm == "qmix" else AlignedQTRAN)(args)
    for name in target_modules(policy):
        getattr(policy, name).requires_grad_(False)
    return policy, args


def target_modules(policy):
    return ("target_rnn", "target_qmix_net") if policy.args.alg == "qmix" else ("target_rnn", "target_joint_q")


def modules(policy):
    return ("eval_rnn", "target_rnn", "eval_qmix_net", "target_qmix_net") if policy.args.alg == "qmix" else (
        "eval_rnn", "target_rnn", "eval_joint_q", "target_joint_q", "v")


@torch.no_grad()
def act(policy, observations, available, hidden, previous, rng, epsilon):
    """Actor sees only flattened local observations, previous action and agent ID."""
    k, n, _ = observations.shape
    ids = np.broadcast_to(np.eye(n, dtype=np.float32), (k, n, n))
    inputs = np.concatenate((observations, previous, ids), axis=-1)
    values, next_hidden = policy.eval_rnn(torch.from_numpy(inputs.reshape(k * n, -1)),
                                        torch.from_numpy(hidden.reshape(k * n, -1)))
    values = values.numpy().reshape(k, n, -1)
    if not np.isfinite(values).all():
        raise FloatingPointError("Actor produced non-finite Q values")
    values[available == 0] = -np.inf
    actions = values.argmax(axis=-1)
    explore = rng.random((k, n)) < epsilon
    for e, a in np.argwhere(explore):
        actions[e, a] = rng.choice(np.flatnonzero(available[e, a]))
    if not np.all(np.take_along_axis(available, actions[..., None], axis=-1)):
        raise RuntimeError("Actor selected an illegal action")
    onehot = np.eye(policy.n_actions, dtype=np.float32)[actions]
    return actions, next_hidden.numpy().reshape(k, n, -1), onehot
