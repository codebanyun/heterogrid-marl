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

    def test_episode_index_environment_seeds_exceed_uint32_without_wrapping(self):
        from algorithms.common.training_seeds import EpisodeSeedSequence
        stream = EpisodeSeedSequence(44)
        _, _, rows = self.api.collect(self.backend, self.envs, 0, stream)
        self.assertEqual(sorted(row["seed"] for row in rows), [4_401_000_000, 4_401_000_001])
        _, _, rows = self.api.collect(self.backend, self.envs, 0, stream)
        self.assertEqual(sorted(row["seed"] for row in rows), [4_401_000_002, 4_401_000_003])

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
        for name in ("actor", "critic", "critic_w", "target_critic_w", "agent_models", "filters"):
            self.assertGreater(deltas[name], 0)

    def test_filtered_target_hard_and_soft_updates_match_native_schedule(self):
        learner = self.backend.learner
        state = self.api.snapshot(self.backend)
        counter = learner.filtered_target_updates
        try:
            for tau in (0.0, 0.01, 1.0):
                with self.torch.no_grad():
                    for source, target in ((learner.critic, learner.target_critic),
                                           (learner.critic_w, learner.target_critic_w)):
                        for value in source.parameters():
                            value.fill_(4)
                        for value in target.parameters():
                            value.fill_(2)
                learner._update_targets_soft(tau)
                for target in (learner.target_critic, learner.target_critic_w):
                    for value in target.parameters():
                        self.torch.testing.assert_close(value, self.torch.full_like(value, 2 + 2 * tau))
            for name in ("_update_targets_hard", "_update_targets"):
                with self.torch.no_grad():
                    for value in learner.critic_w.parameters():
                        value.add_(1)
                getattr(learner, name)()
                for source, target in ((learner.critic, learner.target_critic),
                                       (learner.critic_w, learner.target_critic_w)):
                    for value, copied in zip(source.parameters(), target.parameters()):
                        self.assertTrue(self.torch.equal(value, copied))
            self.assertEqual(learner.filtered_target_updates, counter + 5)
        finally:
            for name, module in self.api.modules(self.backend).items():
                module.load_state_dict(state[name])
            learner.filtered_target_updates = counter

    def test_native_hard_target_interval_also_updates_filtered_target(self):
        learner = self.backend.learner
        interval = learner.args.target_update_interval_or_tau
        counters = (learner.critic_training_steps, learner.last_target_update_step, learner.filtered_target_updates)
        state = self.api.snapshot(self.backend)
        try:
            learner.args.target_update_interval_or_tau = 2
            learner.critic_training_steps = learner.last_target_update_step = 0
            learner.filtered_target_updates = 0
            batch, steps, _ = self.api.collect(self.backend, self.envs, 0, np.random.default_rng(46))
            learner.train(batch, steps, 2)
            self.assertEqual(learner.filtered_target_updates, 0)
            learner.train(batch, 2 * steps, 4)
            self.assertEqual(learner.filtered_target_updates, 1)
            for value, target in zip(learner.critic_w.parameters(), learner.target_critic_w.parameters()):
                self.assertTrue(self.torch.equal(value, target))
        finally:
            learner.args.target_update_interval_or_tau = interval
            learner.critic_training_steps, learner.last_target_update_step, learner.filtered_target_updates = counters
            for name, module in self.api.modules(self.backend).items():
                module.load_state_dict(state[name])

    def test_network_dimensions_match_observations_and_structured_state(self):
        report = self.api.network_configuration(self.backend)
        self.assertEqual(report["actor_input"], report["local_observation_size"] + 32 + report["n_agents"])
        self.assertEqual(report["critic_input"], report["state_size"] + report["n_agents"])
        self.assertEqual(report["hidden_dim"], 128)
        self.assertTrue(report["use_rnn"])
        self.assertFalse(report["actor_uses_global_state"])
        self.assertGreater(report["modules"]["actor"]["parameters"], 0)

    def test_density_networks_execute_with_each_roster_size(self):
        from algorithms.common.density_study import conditions
        from algorithms.smpe.settings import DEFAULT_CONFIG, hyperparameters
        for name, config in conditions(EnvConfig.load(DEFAULT_CONFIG)).items():
            with self.subTest(condition=name):
                env = SMACAdapter(config=config)
                try:
                    backend = self.api.build(Path(BACKEND), hyperparameters(Path(BACKEND), "cpu"),
                                             env.get_env_info(), self.api.LearnerLogger())
                    report = self.api.network_configuration(backend)
                    env.reset(seed=42)
                    obs = np.asarray(env.get_obs())
                    available = np.asarray(env.get_avail_actions())
                    hidden = np.zeros((env.n_agents, 1, 128), dtype=np.float32)
                    actions, state = self.api.EvaluationPolicy(backend).act(obs, hidden, None, available, True)
                    self.assertEqual(tuple(state.shape), tuple(hidden.shape))
                    self.assertTrue(np.take_along_axis(available, actions.cpu().numpy(), axis=-1).all())
                    self.assertEqual(report["n_agents"], env.n_agents)
                    inputs = self.torch.zeros(1, 1, env.n_agents, report["critic_input"])
                    for critic in (backend.learner.critic, backend.learner.critic_w):
                        self.assertEqual(tuple(critic(inputs).shape), (1, 1, env.n_agents, 1))
                finally:
                    env.close()

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
            self.assertEqual(self.backend.learner.filtered_target_updates,
                             saved["learner_counters"]["filtered_target_updates"])
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

    def test_collection_preserves_bilateral_diagnostics_without_changing_policy_inputs(self):
        from dataclasses import replace
        from heterogrid import CommunicationConfig
        config = replace(self.config, diagnostics_enabled=True, reward_normalization="reference",
                         red_communication=CommunicationConfig(), blue_communication=CommunicationConfig())
        envs = [SMACAdapter(config=config) for _ in range(2)]
        try:
            batch, _, rows = self.api.collect(self.backend, envs, 0, np.random.default_rng(45))
            self.assertNotIn("team_episode_metrics", batch.scheme)
            for row in rows:
                self.assertEqual(set(row["team_episode_metrics"]), {"red", "blue"})
                for team in ("red", "blue"):
                    self.assertEqual(row["team_episode_metrics"][team]["diagnostics"]["initial_agents"], envs[0].n_agents)
        finally:
            for env in envs:
                env.close()


if __name__ == "__main__":
    unittest.main()
