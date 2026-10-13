import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, SMACAdapter

BACKEND = os.environ.get("SMPE_ROOT")


@unittest.skipUnless(importlib.util.find_spec("torch") and BACKEND and Path(BACKEND).is_dir(),
                     "Optional SMPE tests require torch and SMPE_ROOT")
class SMPEBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from algorithms.smpe import backend
        from algorithms.smpe.settings import hyperparameters
        cls.torch, cls.api = torch, backend
        torch.set_num_threads(1)
        settings = hyperparameters(Path(BACKEND), "cpu")
        settings.update(batch_size_run=2, batch_size=2, buffer_size=2, agent_epochs=1, agent_vae_batch_size=2)
        cls.config = EnvConfig(max_steps=3, timeout_as_loss=True)
        cls.envs = [SMACAdapter(config=cls.config) for _ in range(2)]
        cls.backend = backend.build(Path(BACKEND), settings, cls.envs[0].get_env_info(), backend.LearnerLogger())

    @classmethod
    def tearDownClass(cls):
        for env in cls.envs:
            env.close()

    def test_collection_marks_scored_timeouts_terminal_and_masks_actions(self):
        batch, steps, rows = self.api.collect(self.backend, self.envs, 0, np.random.default_rng(42))
        self.assertEqual(steps, sum(row["length"] for row in rows))
        self.assertEqual(len(rows), 2)
        for index, row in enumerate(rows):
            self.assertEqual(row["reason"], "timeout")
            self.assertEqual(batch["terminated"][index, row["length"] - 1, 0].item(), 1)
            actions = batch["actions"][index, :row["length"]].long()
            legal = self.torch.gather(batch["avail_actions"][index, :row["length"]], -1, actions)
            self.assertTrue(legal.all())
        self.assertFalse(self.backend.args.add_value_last_step)
        self.assertEqual(self.backend.vae.target_state_dim, self.envs[0].get_state_size())

    def test_native_models_intrinsic_reward_and_policy_update(self):
        before = self.api.snapshot(self.backend)
        batch, steps, _ = self.api.collect(self.backend, self.envs, 0, np.random.default_rng(43))
        self.backend.vae.addBatch(batch)
        self.backend.vae.update_stats(batch)
        stats = self.backend.vae.train_agent_vaes(2000)
        self.assertTrue(np.isfinite(stats["agent_vae_loss"]))
        rewards = self.backend.vae.add_intrinsic_rewards(batch)
        self.assertEqual(tuple(rewards.shape), (2, 4, self.envs[0].n_agents))
        self.assertTrue((rewards - batch["reward"].expand_as(rewards) > 0).all())
        self.backend.learner.train(batch, steps, 2, rewards)
        self.api.assert_finite(self.backend)
        deltas = self.api.parameter_deltas(self.backend, before)
        for name in ("actor", "critic", "critic_w", "agent_models", "filters"):
            self.assertGreater(deltas[name], 0)

    def test_complete_checkpoint_restores_models_normalizers_and_hashes(self):
        before = self.api.snapshot(self.backend)
        mean = self.backend.vae.obs_ms.mean.clone()
        hashes = self.backend.vae.hash_z[0].tables.copy()
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "policy.pt"
            self.api.checkpoint(file, self.backend, self.config, 123)
            with self.torch.no_grad():
                next(self.backend.mac.agent.parameters()).add_(1)
                next(self.backend.vae.agent_models.parameters()).add_(1)
                self.backend.vae.obs_ms.mean.add_(5)
                self.backend.vae.hash_z[0].tables += 10
            saved = self.api.restore(file, self.backend)
            self.assertEqual(saved["environment_steps"], 123)
            self.assertIn("critic_w", saved["optimizers"])
            for name, module in self.api.modules(self.backend).items():
                for key, value in module.state_dict().items():
                    self.assertTrue(self.torch.equal(value.cpu(), before[name][key]))
            self.assertTrue(self.torch.equal(self.backend.vae.obs_ms.mean, mean))
            np.testing.assert_array_equal(self.backend.vae.hash_z[0].tables, hashes)

    def test_evaluation_bridge_respects_masks_and_returns_recurrent_state(self):
        self.envs[0].reset(seed=44)
        obs = np.asarray(self.envs[0].get_obs())
        available = np.asarray(self.envs[0].get_avail_actions())
        available[0] = 0
        available[0, 0] = 1
        hidden = np.zeros((self.envs[0].n_agents, 1, self.backend.args.hidden_dim), dtype=np.float32)
        actions, state = self.api.EvaluationPolicy(self.backend).act(obs, hidden, None, available, True)
        chosen = actions.cpu().numpy()
        self.assertTrue(np.take_along_axis(available, chosen, axis=-1).all())
        self.assertEqual(chosen[0, 0], 0)
        self.assertEqual(tuple(state.shape), tuple(hidden.shape))


if __name__ == "__main__":
    unittest.main()
