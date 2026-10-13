"""Local EPyMARL integration contracts; optional external backend."""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:
    torch = None

BACKEND = Path(os.environ.get("EPYMARL_ROOT", "/root/research/epymarl/src"))
if (BACKEND / "src").is_dir():
    BACKEND = BACKEND / "src"


@unittest.skipUnless(torch is not None and (BACKEND / "learners/coma_learner.py").exists(),
                     "Optional torch/EPyMARL checkout not available")
class ActorCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def policy(self, algorithm, config=None):
        from algorithms.common.actor_critic.backend import build_policy
        from algorithms.common.sensitivity import scenarios
        from heterogrid import SMACAdapter
        env = SMACAdapter(config=config or scenarios()["B0"])
        self.addCleanup(env.close)
        return build_policy(SimpleNamespace(algorithm=algorithm, epymarl_root=BACKEND), env)[0]

    def history(self, policy, length):
        n, a = policy.args.n_agents, policy.args.n_actions
        return {"o": np.zeros((length + 1, n, policy.args.obs_shape), dtype=np.float32),
                "s": np.ones((length + 1, policy.args.state_shape), dtype=np.float32),
                "avail": np.ones((length + 1, n, a), dtype=np.uint8),
                "u": np.zeros((length, n), dtype=np.int64), "r": np.ones(length, dtype=np.float32)}

    def test_all_scenarios_keep_fixed_hyperparameters(self):
        from algorithms.common.sensitivity import scenarios
        for algorithm in ("maa2c", "coma"):
            invariant = None
            for config in scenarios().values():
                policy = self.policy(algorithm, config)
                args = policy.args
                values = (args.hidden_dim, args.lr, args.q_nstep, args.entropy_coef,
                          args.target_update_interval_or_tau, args.batch_size_run, args.batch_size)
                invariant = values if invariant is None else invariant
                self.assertEqual(values, invariant)
                self.assertFalse(args.add_value_last_step)
                self.assertFalse(args.standardise_rewards)
                self.assertEqual(args.n_agents, len(config.red))

    def test_terminal_padding_and_nstep_no_bootstrap_after_terminal(self):
        from algorithms.common.actor_critic.backend import episode_batch
        for algorithm in ("maa2c", "coma"):
            p = self.policy(algorithm)
            batch = episode_batch(p, [self.history(p, 2), self.history(p, 7)])
            self.assertEqual(batch.max_seq_length, 8)
            self.assertEqual(batch["filled"][0].sum(), 2)
            self.assertEqual(batch["terminated"][0, 1, 0], 1)
            self.assertEqual(batch["terminated"][1, 6, 0], 1)
            self.assertTrue(torch.all(batch["avail_actions"][0, 3:, :, 0] == 1))
            rewards = batch["reward"][:, :-1].expand(-1, -1, p.args.n_agents)
            mask = batch["filled"][:, :-1].expand_as(rewards).float()
            values = torch.full((2, 8, p.args.n_agents), 1000.)
            returns = p.learner.nstep_returns(rewards, mask, values, 10)
            self.assertAlmostEqual(float(returns[0, 0, 0]), 1 + p.args.gamma, places=5)
            self.assertEqual(float(returns[0, 1, 0]), 1.)
            self.assertEqual(float(returns[1, 6, 0]), 1.)
            self.assertEqual(float(returns[0, 2:].sum()), 0.)

    def test_act_matches_external_mac_and_masks_and_does_not_touch_torch_rng(self):
        from algorithms.common.actor_critic.backend import act, episode_batch
        for algorithm in ("maa2c", "coma"):
            p = self.policy(algorithm)
            h = self.history(p, 3)
            h["o"][:] = np.random.default_rng(3).normal(size=h["o"].shape)
            h["avail"][:] = 0; h["avail"][..., 0] = 1; h["avail"][..., 2] = 1
            batch = episode_batch(p, [h])
            p.mac.init_hidden(1)
            hidden = np.zeros((1, p.args.n_agents, p.args.hidden_dim), dtype=np.float32)
            rng = np.random.default_rng(42)
            state = torch.get_rng_state().clone()
            for t in range(3):
                with torch.no_grad():
                    expected = p.mac.forward(batch, t)
                actions, hidden = act(p, h["o"][None, t], h["avail"][None, t], hidden, rng, True)
                np.testing.assert_array_equal(actions, expected.argmax(-1).numpy())
                np.testing.assert_allclose(hidden.reshape(-1, p.args.hidden_dim),
                                           p.mac.hidden_states.numpy().reshape(-1, p.args.hidden_dim), atol=1e-6)
                actions, _ = act(p, h["o"][None, t], h["avail"][None, t], hidden, rng)
                self.assertTrue(np.all(np.isin(actions, [0, 2])))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_coma_critic_excludes_own_current_action(self):
        from algorithms.common.actor_critic.backend import episode_batch
        p = self.policy("coma")
        batch = episode_batch(p, [self.history(p, 2)])
        before = p.learner.critic._build_inputs(batch).clone()
        batch["actions_onehot"][:, :, 0] = 0
        batch["actions_onehot"][:, :, 0, 2] = 1
        after = p.learner.critic._build_inputs(batch)
        self.assertTrue(torch.equal(before[:, :, 0], after[:, :, 0]))
        self.assertFalse(torch.equal(before[:, :, 1], after[:, :, 1]))

    def test_target_updates_and_checkpoint_actor_roundtrip(self):
        from algorithms.common.actor_critic.backend import episode_batch
        from algorithms.common.actor_critic.train import save_checkpoint, restore_checkpoint
        from algorithms.common.sensitivity import scenarios
        for algorithm in ("maa2c", "coma"):
            p = self.policy(algorithm)
            target_before = [v.clone() for v in p.learner.target_critic.parameters()]
            if algorithm == "coma":
                p.learner.critic_training_steps = 199
            p.learner.train(episode_batch(p, [self.history(p, 3)] * 4), 12, 4)
            tau = .01 if algorithm == "maa2c" else 1.
            for before, critic, target in zip(target_before, p.learner.critic.parameters(),
                                               p.learner.target_critic.parameters()):
                torch.testing.assert_close(target, before * (1 - tau) + critic * tau)
            with tempfile.TemporaryDirectory() as directory:
                cli = SimpleNamespace(algorithm=algorithm, output=Path(directory))
                filename = Path(directory) / "checkpoint.pt"
                before = {k: v.clone() for k, v in p.mac.agent.state_dict().items()}
                save_checkpoint(filename, p, cli, scenarios()["B0"], 12, 1)
                with torch.no_grad():
                    for value in p.mac.parameters():
                        value.add_(1.)
                restore_checkpoint(filename, p)
                self.assertTrue(all(torch.equal(value, p.mac.agent.state_dict()[k]) for k, value in before.items()))

    def test_real_fresh_batch_updates_actor_and_critic(self):
        from algorithms.common.actor_critic.backend import act, episode_batch
        from algorithms.common.sensitivity import scenarios
        from algorithms.common.value_decomposition.train import observe, new_history
        from heterogrid import SMACAdapter
        for algorithm in ("maa2c", "coma"):
            config = replace(scenarios()["M1"], max_steps=6)
            p = self.policy(algorithm, config)
            histories = []
            for seed in range(4):
                env = SMACAdapter(config=config); env.reset(seed=seed)
                obs, state, avail = observe(env)
                h = new_history(obs, state, avail)
                hidden = np.zeros((1, p.args.n_agents, p.args.hidden_dim), dtype=np.float32)
                done = False; rng = np.random.default_rng(seed)
                while not done:
                    actions, hidden = act(p, obs[None], avail[None], hidden, rng)
                    reward, done, info = env.step(actions[0].tolist())
                    self.assertFalse(any(n.startswith("red_") for n in info["invalid_actions"]))
                    obs, state, avail = observe(env)
                    h["u"].append(actions[0]); h["r"].append(reward)
                    h["o"].append(obs); h["s"].append(state); h["avail"].append(avail)
                env.close(); histories.append(h)
            actor = [v.detach().clone() for v in p.mac.parameters()]
            critic = [v.detach().clone() for v in p.learner.critic.parameters()]
            p.learner.train(episode_batch(p, histories), 24, 4)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor, p.mac.parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic, p.learner.critic.parameters())))
            self.assertTrue(all(np.isfinite(v) for v in p.logger.values.values()))


if __name__ == "__main__":
    unittest.main()
