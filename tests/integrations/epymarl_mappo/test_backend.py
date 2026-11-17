import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, SMACAdapter

ROOT = os.environ.get("EPYMARL_ROOT")


@unittest.skipUnless(importlib.util.find_spec("torch") and ROOT, "Optional native MAPPO tests require EPYMARL_ROOT and torch")
class NativeMAPPOTests(unittest.TestCase):
    def setUp(self):
        import torch
        from algorithms.epymarl_mappo import backend
        self.torch, self.api = torch, backend
        torch.set_num_threads(1)
        self.root = backend.resolve_root(ROOT)
        parameters = backend.settings(self.root)
        parameters.update(batch_size_run=2, batch_size=2, buffer_size=2)
        self.config = EnvConfig(max_steps=3, timeout_as_loss=True)
        self.envs = [SMACAdapter(config=self.config) for _ in range(2)]
        self.backend = backend.build(self.root, parameters, self.envs[0].get_env_info())

    def tearDown(self):
        for env in self.envs:
            env.close()

    def test_native_preset_and_network_shapes(self):
        native = self.api.settings(self.root)
        self.assertEqual((native["hidden_dim"], native["batch_size_run"], native["epochs"]), (128, 10, 4))
        self.assertEqual(native["lr"], .0003)
        self.assertEqual(native["entropy_coef"], .001)
        self.assertEqual(native["eps_clip"], .2)
        self.assertTrue(native["standardise_rewards"])
        self.assertFalse(native["add_value_last_step"])
        self.assertEqual(self.backend.learner.__class__.__name__, "PPOLearner")
        self.assertFalse(self.api.network_configuration(self.backend)["actor_uses_global_state"])

    def test_collection_terminal_masks_and_shared_reward(self):
        batch, steps, rows = self.api.collect(self.backend, self.envs, 0, np.random.default_rng(42))
        self.assertEqual(steps, 6)
        self.assertEqual(steps, sum(row["length"] for row in rows))
        self.assertEqual(tuple(batch["reward"].shape), (2, 4, 1))
        for i, row in enumerate(rows):
            self.assertEqual(batch["terminated"][i, row["length"] - 1, 0].item(), 1)
            self.assertAlmostEqual(row["return"], sum(row["reward_components"].values()))
            self.assertTrue(self.torch.gather(batch["avail_actions"][i, :row["length"]], -1,
                                             batch["actions"][i, :row["length"]]).all())

    def test_episode_index_environment_seeds_exceed_uint32_without_wrapping(self):
        from algorithms.common.training_seeds import EpisodeSeedSequence
        stream = EpisodeSeedSequence(44)
        _, _, rows = self.api.collect(self.backend, self.envs, 0, stream)
        self.assertEqual(sorted(row["seed"] for row in rows), [4_401_000_000, 4_401_000_001])

    def test_four_native_ppo_epochs_update_and_resynchronize_old_policy(self):
        before = self.api.snapshot(self.backend)
        batch, steps, _ = self.api.collect(self.backend, self.envs, 0, np.random.default_rng(43))
        self.backend.learner.train(batch, steps, 2)
        after = self.api.snapshot(self.backend)
        for name, values in after.items():
            delta = sum(float((value - before[name][key]).square().sum()) for key, value in values.items())
            self.assertGreater(delta, 0)
        for key, value in after["actor"].items():
            self.assertTrue(self.torch.equal(value, after["old_actor"][key]))
        for optimizer in (self.backend.learner.agent_optimiser, self.backend.learner.critic_optimiser):
            for state in optimizer.state.values():
                self.assertEqual(int(state["step"]), 4)
        self.assertEqual(self.backend.learner.critic_training_steps, 1)

    def test_complete_checkpoint_restores_old_policy_and_reward_normalizer(self):
        batch, steps, _ = self.api.collect(self.backend, self.envs, 0, np.random.default_rng(44))
        self.backend.learner.train(batch, steps, 2)
        before = self.api.snapshot(self.backend)
        mean = self.backend.learner.rew_ms.mean.clone()
        count = self.backend.learner.rew_ms.count
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            self.api.checkpoint(path, self.backend, self.config, steps)
            with self.torch.no_grad():
                for module in self.api.modules(self.backend).values():
                    next(module.parameters()).add_(1)
                self.backend.learner.rew_ms.mean.add_(1)
            saved = self.api.restore(path, self.backend)
            for name, values in self.api.snapshot(self.backend).items():
                for key, value in values.items():
                    self.assertTrue(self.torch.equal(value, before[name][key]))
            self.assertTrue(self.torch.equal(self.backend.learner.rew_ms.mean, mean))
            self.assertEqual(self.backend.learner.rew_ms.count, count)
            self.assertEqual(saved["algorithm"], "epymarl_mappo")

    def test_evaluation_bridge_matches_original_mac(self):
        self.envs[0].reset(seed=45)
        obs = np.asarray(self.envs[0].get_obs())
        available = np.asarray(self.envs[0].get_avail_actions())
        hidden = np.zeros((self.envs[0].n_agents, 1, 128), dtype=np.float32)
        actions, state = self.api.EvaluationPolicy(self.backend).act(obs, hidden, None, available, True)
        batch = self.api.new_batch(self.backend, 1, 1)
        self.api.update_batch(batch, {"obs": obs[None], "avail_actions": available[None]}, ts=0)
        self.backend.mac.init_hidden(1)
        probabilities = self.backend.mac.forward(batch, 0, test_mode=True)
        self.assertTrue(self.torch.equal(actions.reshape(-1), probabilities.argmax(-1).reshape(-1)))
        self.assertTrue(self.torch.equal(state.reshape(-1, 128), self.backend.mac.hidden_states.reshape(-1, 128)))

    def test_all_density_roster_sizes_keep_actor_local(self):
        from algorithms.common.density_study import conditions
        from dataclasses import replace
        base = EnvConfig.load(self.api.PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json")
        for name, config in conditions(base).items():
            with self.subTest(condition=name):
                env = SMACAdapter(config=replace(config, max_steps=3))
                try:
                    model = self.api.build(self.root, self.api.settings(self.root), env.get_env_info())
                    env.reset(seed=42)
                    obs = np.asarray(env.get_obs())
                    avail = np.asarray(env.get_avail_actions())
                    hidden = np.zeros((env.n_agents, 1, 128), dtype=np.float32)
                    actions, state = self.api.EvaluationPolicy(model).act(obs, hidden, None, avail, True)
                    self.assertTrue(np.take_along_axis(avail, actions.cpu().numpy(), axis=-1).all())
                    self.assertEqual(tuple(state.shape), tuple(hidden.shape))
                    self.assertFalse(self.api.network_configuration(model)["actor_uses_global_state"])
                finally:
                    env.close()

    def test_bilateral_diagnostics_do_not_change_episode_batch_inputs(self):
        from dataclasses import replace
        from heterogrid import CommunicationConfig
        config = replace(self.config, diagnostics_enabled=True, reward_normalization="reference",
                         red_communication=CommunicationConfig(), blue_communication=CommunicationConfig())
        envs = [SMACAdapter(config=config) for _ in range(2)]
        try:
            batch, _, rows = self.api.collect(self.backend, envs, 0, np.random.default_rng(46))
            self.assertNotIn("team_episode_metrics", batch.scheme)
            self.assertTrue(all(set(row["team_episode_metrics"]) == {"red", "blue"} for row in rows))
        finally:
            for env in envs:
                env.close()


if __name__ == "__main__":
    unittest.main()
