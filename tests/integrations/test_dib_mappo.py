"""Native DIB parity, causal execution, joint objectives and stage-correct checkpoints."""
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from heterogrid import EnvConfig, SMACAdapter
from heterogrid.scenarios import balanced_roster
from algorithms.dib_mappo.campaign import matrix

try:
    import torch
    from algorithms.dib_mappo.backend import build_policy, initial_memory, act, episode_batch, settings
    from algorithms.dib_mappo.sequence import replay_sequence, learn
except ImportError:
    torch = None

ROOT = Path(os.environ.get("DIB_MAPPO_ROOT", "/root/research/DIB-MAPPO/src"))
if (ROOT / "src").is_dir():
    ROOT = ROOT / "src"


class DIBMatrixTests(unittest.TestCase):
    def test_13_legacy_tasks_and_3_new_density_tasks_have_same_protocols(self):
        from algorithms.common.sensitivity import scenarios
        old, density = matrix()
        self.assertEqual(len(old), 13)
        self.assertEqual(len(density), 4)
        for name, config in scenarios(fixed_blue=True).items():
            self.assertEqual(old[name], replace(config, diagnostics_enabled=True))
        self.assertEqual(density["D0"], replace(old["B0"], reward_normalization="reference"))
        for name, count in (("D1", 3), ("D2", 9), ("D3", 12)):
            self.assertEqual(len(density[name].red), count)
            self.assertEqual(density[name].reward_normalization, "reference")


@unittest.skipIf(torch is None or not (ROOT / "learners/dib_ppo_learner.py").is_file(), "Optional DIB-MAPPO backend unavailable")
class DIBBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def policy(self, native=False, max_steps=6, count=3):
        roster = balanced_roster(count)
        config = EnvConfig(red=roster, blue=roster, max_steps=max_steps, opponent="dummy", timeout_as_loss=True,
                           time_penalty=.005, target_sharing="auto")
        env = SMACAdapter(config=config)
        env.reset(42)
        torch.manual_seed(42)
        policy, _ = build_policy(SimpleNamespace(dib_root=ROOT, sequence_acceleration=not native), env)
        return policy, env

    def batch(self, policy, env):
        from algorithms.common.value_decomposition.train import observe, new_history
        memory = initial_memory(policy, 1)
        obs, state, available = observe(env)
        history = new_history(obs, state, available)
        rng = np.random.default_rng(42)
        while True:
            actions, memory = act(policy, obs[None], available[None], memory, rng)
            reward, done, _ = env.step(actions[0].tolist())
            obs, state, available = observe(env)
            history["u"].append(actions[0]); history["r"].append(reward)
            history["o"].append(obs); history["s"].append(state); history["avail"].append(available)
            if done:
                break
        return episode_batch(policy, [history] * 4)

    def test_native_configuration_and_batching_overrides(self):
        args = settings(ROOT)
        for name, value in {"hidden_dim": 128, "epochs": 4, "lr": .0003, "dib_warmup_steps": 2000000,
                            "dib_use_ahs": True, "dib_use_bbd": True, "dib_use_mhf": True,
                            "batch_size_run": 4, "standardise_rewards": False, "add_value_last_step": False}.items():
            self.assertEqual(args[name], value)

    def test_full_sequence_outputs_and_auxiliary_losses_match_native(self):
        policy, env = self.policy()
        try:
            policy.mac.set_residual_actor_gate_for_t(4000000)
            batch = self.batch(policy, env)
            policy.mac._learning_clock = 4000000
            probability, auxiliary = replay_sequence(policy.mac, batch, True)
            policy.mac.init_hidden(batch.batch_size)
            native_p, native_aux = [], []
            for t in range(batch.max_seq_length - 1):
                p, aux = policy.mac.forward(batch, t, return_aux=True)
                native_p.append(p); native_aux.append(aux)
            torch.testing.assert_close(probability, torch.stack(native_p, 1), atol=2e-6, rtol=2e-5)
            for key, value in auxiliary.items():
                expected = torch.stack([step[key] for step in native_aux], 1)
                torch.testing.assert_close(value, expected, atol=3e-6, rtol=3e-5)
            fast_steps = [{key: value[:, t] for key, value in auxiliary.items()} for t in range(batch.max_seq_length - 1)]
            mask = batch["filled"][:, :-1].float().expand(-1, -1, policy.args.n_agents)
            a = policy.mac.compute_auxiliary_losses(native_aux, batch, mask)
            b = policy.mac.compute_auxiliary_losses(fast_steps, batch, mask)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-5)
        finally:
            env.close()

    def test_native_and_vectorized_ppo_updates_match(self):
        native, env_a = self.policy(native=True)
        fast, env_b = self.policy()
        try:
            native.mac.set_residual_actor_gate_for_t(4000000)
            fast.mac.set_residual_actor_gate_for_t(4000000)
            batch = self.batch(native, env_a)
            learn(native, batch, 4000000, 4)
            learn(fast, batch, 4000000, 4)
            for name, parameter in native.mac.agent.named_parameters():
                torch.testing.assert_close(parameter, dict(fast.mac.agent.named_parameters())[name], atol=3e-5, rtol=3e-4)
            self.assertGreater(fast.logger.values["belief_loss"], 0)
            self.assertEqual(fast.logger.values["dib_actor_residual_gate"], 1)
            self.assertEqual(fast.logger.values["agent_update_skipped"], 0)
        finally:
            env_a.close(); env_b.close()

    def test_warmup_does_not_train_teacher_and_first_ppo_ratio_is_one(self):
        policy, env = self.policy()
        try:
            before = policy.mac.agent.teacher_head.weight.detach().clone()
            batch = self.batch(policy, env)
            learn(policy, batch, 0, 4)
            torch.testing.assert_close(before, policy.mac.agent.teacher_head.weight, atol=0, rtol=0)
            self.assertEqual(policy.logger.values["dib_stage"], 0)
            self.assertEqual(policy.logger.values["belief_loss"], 0)
            self.assertLess(abs(policy.logger.values["ppo_ratio_mean"] - 1), .05)
        finally:
            env.close()

    def test_warmup_preserves_native_adam_clocks_and_transition_to_joint(self):
        native, env_a = self.policy(native=True)
        fast, env_b = self.policy()
        try:
            batch = self.batch(native, env_a)
            for clock in (0, 800, 2000800):
                learn(native, batch, clock, 4)
                learn(fast, batch, clock, 4)
                for name, parameter in native.mac.agent.named_parameters():
                    other = dict(fast.mac.agent.named_parameters())[name]
                    a = native.learner.agent_optimiser.state[parameter]
                    b = fast.learner.agent_optimiser.state[other]
                    self.assertEqual(set(a), set(b), name)
                    if a:
                        self.assertEqual(float(a["step"]), float(b["step"]), name)
                    torch.testing.assert_close(parameter, other, atol=3e-5, rtol=3e-4)
        finally:
            env_a.close(); env_b.close()

    def test_checkpointed_multitile_history_keeps_native_joint_gradients(self):
        native, env_a = self.policy(native=True, max_steps=30)
        fast, env_b = self.policy(max_steps=30)
        try:
            batch = self.batch(native, env_a)
            self.assertGreater((batch.max_seq_length - 1) * 4 * 3, 256)
            for policy in (native, fast):
                policy.mac.set_residual_actor_gate_for_t(4000000)
                learn(policy, batch, 4000000, 4)
            for name, parameter in native.mac.agent.named_parameters():
                torch.testing.assert_close(parameter, dict(fast.mac.agent.named_parameters())[name],
                                           atol=3e-5, rtol=3e-4)
        finally:
            env_a.close(); env_b.close()

    def test_teacher_centre_matches_outputs_and_all_input_parameter_gradients(self):
        from algorithms.dib_mappo.auxiliary import teacher_centre
        policy, env = self.policy()
        try:
            agent = policy.mac.agent
            kb, kf = policy.args.dib_teacher_kb, policy.args.dib_teacher_kf
            for length in (1, 5, 23):
                tokens = (torch.randn(2, length, 3, agent.embed_dim) * .1).requires_grad_()
                sequence = tokens.permute(0, 2, 1, 3).reshape(6, length, agent.embed_dim)
                ids = torch.arange(length)
                windows = torch.stack([sequence[:, (ids + offset).clamp(0, length - 1)]
                    for offset in range(-kb, kf + 1)], 2).reshape(6 * length, kb + kf + 1, agent.embed_dim)
                expected = agent.teacher_gru(windows)[0][:, kb]
                actual = teacher_centre(agent, tokens)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                coefficient = torch.randn_like(expected)
                parameters = (tokens,) + tuple(agent.teacher_gru.parameters())
                native_grad = torch.autograd.grad((expected * coefficient).sum(), parameters)
                optimized_grad = torch.autograd.grad((actual * coefficient).sum(), parameters)
                for a, b in zip(native_grad, optimized_grad):
                    torch.testing.assert_close(a, b, atol=5e-6, rtol=5e-5)
        finally:
            env.close()

    def test_fused_backbone_retains_native_per_step_nonfinite_recovery(self):
        policy, env = self.policy()
        try:
            with torch.no_grad():
                policy.mac.agent.base_policy_encoder.weight[0, 0] = float("nan")
                batch = self.batch(policy, env)
                policy.mac.set_residual_actor_gate_for_t(4000000)
                policy.mac._learning_clock = 4000000
                actual, _ = replay_sequence(policy.mac, batch, True)
                policy.mac.init_hidden(batch.batch_size)
                expected = torch.stack([policy.mac.forward(batch, t)
                    for t in range(batch.max_seq_length - 1)], 1)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        finally:
            env.close()

    def test_vector_returns_preserve_native_masking_tail_and_bootstrap(self):
        from algorithms.dib_mappo.returns import nstep_returns
        policy, env = self.policy(native=True)
        try:
            for length in (1, 4, 10):
                rewards = torch.randn(4, length, 3)
                values = torch.randn(4, length + 1, 3)
                mask = torch.zeros_like(rewards)
                for row, count in enumerate((length, max(0, length - 1), 1, 0)):
                    mask[row, :count] = 1
                for bootstrap in (False, True):
                    policy.args.add_value_last_step = bootstrap
                    for n in (1, 5, 12):
                        actual = nstep_returns(policy.args, rewards, mask, values, n)
                        expected = policy.learner.nstep_returns(rewards, mask, values, n)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        finally:
            env.close()

    def test_tied_distance_forward_backward_and_zero_distance_match_native(self):
        from algorithms.dib_mappo.auxiliary import SelfDistances
        for identical in (False, True):
            values = torch.randn(3, 4, 9, dtype=torch.float64)
            if identical:
                values[:] = values[:, :1].clone()
            values.requires_grad_()
            expected = torch.cdist(values, values, p=2)
            actual = SelfDistances.apply(values)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            coefficient = torch.randn_like(expected)
            a, = torch.autograd.grad((expected * coefficient).sum(), values)
            b, = torch.autograd.grad((actual * coefficient).sum(), values)
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        values = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(SelfDistances.apply, (values,)))

    def test_large_rosters_padding_and_ramp_preserve_four_epoch_updates(self):
        for count, clock in ((6, 2400000), (9, 4000000), (12, 2400000)):
            with self.subTest(count=count, clock=clock):
                native, env_a = self.policy(native=True, max_steps=8, count=count)
                fast, env_b = self.policy(max_steps=8, count=count)
                try:
                    batch = self.batch(native, env_a)
                    batch["filled"][2, 3:] = 0
                    batch["terminated"][2, 2] = 1
                    for policy in (native, fast):
                        policy.mac.set_residual_actor_gate_for_t(clock)
                        learn(policy, batch, clock, 4)
                    for name in ("ppo_epochs", "dib_stage", "dib_actor_residual_gate", "agent_update_skipped"):
                        self.assertEqual(native.logger.values[name], fast.logger.values[name])
                    for a, b in ((native.mac.agent, fast.mac.agent),
                                 (native.learner.critic, fast.learner.critic),
                                 (native.learner.target_critic, fast.learner.target_critic)):
                        for key, value in a.state_dict().items():
                            torch.testing.assert_close(value, b.state_dict()[key], atol=3e-5, rtol=3e-4)
                finally:
                    env_a.close(); env_b.close()

    def test_actor_cannot_see_other_agents_or_future_observations(self):
        policy, env = self.policy()
        try:
            obs = np.asarray(env.get_obs(), np.float32)[None]
            available = np.asarray(env.get_avail_actions(), np.uint8)[None]
            memory = initial_memory(policy, 1)
            p, _ = policy.mac.agent(torch.from_numpy(obs), torch.from_numpy(memory["last_action"]),
                {key: torch.from_numpy(value) for key, value in memory.items() if key != "last_action"})
            changed = obs.copy(); changed[:, 1:] = 100
            q, _ = policy.mac.agent(torch.from_numpy(changed), torch.from_numpy(memory["last_action"]),
                {key: torch.from_numpy(value) for key, value in memory.items() if key != "last_action"})
            torch.testing.assert_close(p["logits_k"][:, 0], q["logits_k"][:, 0], atol=0, rtol=0)
            batch = self.batch(policy, env)
            policy.mac.set_residual_actor_gate_for_t(4000000);policy.mac._learning_clock = 4000000
            a, _ = replay_sequence(policy.mac, batch, True)
            batch["obs"][:, 3:] += 100
            b, _ = replay_sequence(policy.mac, batch, True)
            torch.testing.assert_close(a[:, :3], b[:, :3], atol=0, rtol=0)
        finally:
            env.close()

    def test_checkpoint_restores_residual_gate_as_well_as_weights(self):
        from algorithms.dib_mappo.train import save_checkpoint, restore_checkpoint
        policy, env = self.policy()
        try:
            policy.mac.set_residual_actor_gate(.75); policy.batch_start_step = 3500000
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                save_checkpoint(path, policy, SimpleNamespace(seed=42), env.env.config, 3500100, 1)
                policy.mac.set_residual_actor_gate(0)
                restore_checkpoint(path, policy)
                self.assertEqual(policy.mac.agent.residual_actor_gate, .75)
                self.assertEqual(policy.batch_start_step, 3500000)
                self.assertFalse(torch.load(path)["optimizer_included"])
        finally:
            env.close()
