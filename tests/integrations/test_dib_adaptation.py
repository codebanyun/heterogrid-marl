"""Death-mask semantics, 1M scheduling, upstream MHF definition and PPO parity."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from heterogrid import EnvConfig, SMACAdapter
from heterogrid.scenarios import balanced_roster

try:
    import torch
    from algorithms.dib_mappo.backend import build_policy, initial_memory, act, episode_batch, settings
    from algorithms.dib_mappo.alive import alive_at_action
    from algorithms.dib_mappo.sequence import learn, replay_sequence
except ImportError:
    torch = None

ROOT = Path(os.environ.get("DIB_MAPPO_ROOT", "/root/research/DIB-MAPPO/src"))
if (ROOT / "src").is_dir():
    ROOT = ROOT / "src"


@unittest.skipIf(torch is None or not (ROOT / "learners/dib_ppo_learner.py").exists(),
                 "Optional DIB backend unavailable")
class DIBAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def policy(self, profile="active_mask_1m", native=False):
        roster = balanced_roster(3)
        env = SMACAdapter(config=EnvConfig(red=roster, blue=roster, max_steps=6,
            opponent="dummy", target_sharing="auto", timeout_as_loss=True))
        env.reset(42)
        torch.manual_seed(42)
        p, _ = build_policy(SimpleNamespace(dib_root=ROOT, training_profile=profile,
                                            sequence_acceleration=not native), env)
        self.addCleanup(env.close)
        return p, env

    def batch(self, p, env):
        from algorithms.common.value_decomposition.train import observe, new_history
        obs, state, avail = observe(env)
        h = new_history(obs, state, avail)
        memory, rng = initial_memory(p, 1), np.random.default_rng(42)
        while True:
            actions, memory = act(p, obs[None], avail[None], memory, rng)
            reward, done, _ = env.step(actions[0].tolist())
            obs, state, avail = observe(env)
            h["u"].append(actions[0]); h["r"].append(reward)
            h["o"].append(obs); h["s"].append(state); h["avail"].append(avail)
            if done:
                return episode_batch(p, [h] * 4)

    def assert_models_close(self, a, b):
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=3e-5, rtol=3e-4)

    def test_profiles_only_change_requested_masks_and_warmup(self):
        baseline = settings(ROOT)
        permitted = {"dib_training_profile", "dib_use_alive_masks", "dib_warmup_steps", "dib_actor_warmup_steps"}
        for profile, masks, warmup in (("legacy", False, 2000000), ("active_mask", True, 2000000),
                                      ("warmup1m", False, 1000000), ("active_mask_1m", True, 1000000)):
            a = settings(ROOT, profile)
            self.assertEqual(a["dib_use_alive_masks"], masks)
            self.assertEqual(a["dib_warmup_steps"], warmup)
            self.assertEqual(a["dib_actor_warmup_steps"], warmup)
            self.assertEqual(a["dib_actor_ramp_steps"], 2000000)
            self.assertEqual({k for k in baseline.keys() | a.keys() if baseline.get(k) != a.get(k)} - permitted, set())
        with self.assertRaises(ValueError):
            settings(ROOT, "unknown")

    def test_mask_uses_pre_action_hp_not_future_survival_or_message_contents(self):
        p, env = self.policy()
        batch = self.batch(p, env)
        batch["obs"][:, 3:, 1, 3] = 0
        # Nonzero other fields must not make a dead actor active again.
        batch["obs"][:, 3:, 1, -1] = 1
        mask = alive_at_action(batch, batch.max_seq_length - 1)
        self.assertTrue(bool((mask[:, :3, 1] == 1).all()))
        self.assertTrue(bool((mask[:, 3:, 1] == 0).all()))
        self.assertTrue(bool((mask[:, :, 0] == 1).all()))
        # t=2 is learned even though its post-action observation at t=3 is dead.
        self.assertEqual(mask[0, 2, 1].item(), 1)

    def test_actual_update_masks_actor_aux_and_padding_but_not_team_critic(self):
        p, env = self.policy()
        batch = self.batch(p, env)
        batch["obs"][:, 3:, 1, 3] = 0
        batch["filled"][2, 4:] = 0
        batch["terminated"][2, 3] = 1
        actor_masks, critic_masks = [], []
        original_aux = p.mac.compute_auxiliary_losses
        original_critic = p.learner.train_critic_sequential
        def auxiliary(steps, data, mask):
            actor_masks.append(mask.detach().clone())
            return original_aux(steps, data, mask)
        def critic(critic, target, data, rewards, mask):
            critic_masks.append(mask.detach().clone())
            return original_critic(critic, target, data, rewards, mask)
        p.mac.compute_auxiliary_losses = auxiliary
        p.learner.train_critic_sequential = critic
        p.mac.set_residual_actor_gate_for_t(3000800)
        learn(p, batch, 3000800, 4)
        self.assertEqual(len(actor_masks), 4)
        for a, c in zip(actor_masks, critic_masks):
            self.assertTrue(bool((a[:, 3:, 1] == 0).all()))
            self.assertEqual(a[0, 2, 1].item(), 1)
            self.assertEqual(c[0, 4, 1].item(), 1)
            self.assertTrue(bool((a[2, 4:] == 0).all()))
            self.assertTrue(bool((c[2, 4:] == 0).all()))
        self.assertLess(p.logger.values["active_actor_fraction"], 1)
        self.assertGreater(p.logger.values["dead_actor_samples"], 0)
        self.assertEqual(p.logger.values["agent_update_skipped"], 0)

    def test_masked_auxiliary_latents_receive_no_direct_dead_sample_gradient(self):
        p, env = self.policy()
        batch = self.batch(p, env)
        batch["obs"][:, 3:, 1, 3] = 0
        p.mac.set_residual_actor_gate_for_t(3000000)
        p.mac._learning_clock = 3000000
        _, aux = replay_sequence(p.mac, batch, True)
        aux["z"].retain_grad()
        mask = batch["filled"][:, :-1].float().expand(-1, -1, p.args.n_agents)
        mask = mask * alive_at_action(batch, batch.max_seq_length - 1)
        losses = p.mac.agent.compute_auxiliary_losses(aux, batch["obs"], batch["actions_onehot"], mask)
        (losses["ahs_loss"] + losses["bbd_core_loss"] + losses["hyp_loss"] + losses["fwd_loss"]).backward()
        self.assertEqual(float(aux["z"].grad[mask == 0].abs().sum()), 0)
        self.assertGreater(float(aux["z"].grad[mask > 0].abs().sum()), 0)

    def test_all_alive_masking_preserves_original_four_epoch_updates(self):
        original, env = self.policy("legacy")
        masked, _ = self.policy("active_mask")
        batch = self.batch(original, env)
        for p in (original, masked):
            p.mac.set_residual_actor_gate_for_t(4000000)
            learn(p, batch, 4000000, 4)
        for key in ("mac",):
            self.assert_models_close(getattr(original, key).agent, getattr(masked, key).agent)
        self.assert_models_close(original.learner.critic, masked.learner.critic)
        self.assert_models_close(original.learner.target_critic, masked.learner.target_critic)
        self.assertEqual(masked.logger.values["active_actor_fraction"], 1)
        self.assertEqual(masked.logger.values["ppo_epochs"], 4)

    def test_death_mask_does_not_truncate_shared_critic_returns(self):
        original, env = self.policy("legacy")
        masked, _ = self.policy("active_mask")
        batch = self.batch(original, env)
        batch["obs"][:, 3:, 1, 3] = 0
        for p in (original, masked):
            p.mac.set_residual_actor_gate_for_t(4000000)
            learn(p, batch, 4000000, 4)
        self.assert_models_close(original.learner.critic, masked.learner.critic)
        self.assert_models_close(original.learner.target_critic, masked.learner.target_critic)

    def test_empty_active_batch_is_finite_and_does_not_update_parameters(self):
        p, env = self.policy()
        batch = self.batch(p, env)
        batch["obs"][..., 3] = 0
        before = {k:v.clone() for k,v in p.mac.agent.state_dict().items()}
        learn(p, batch, 3000000, 4)
        self.assertEqual(p.logger.values["empty_active_batch"], 1)
        self.assertEqual(p.learner.critic_training_steps, 0)
        for k,v in before.items():
            torch.testing.assert_close(v, p.mac.agent.state_dict()[k], atol=0, rtol=0)

    def test_one_million_schedule_and_teacher_start(self):
        p, env = self.policy()
        for clock, stage, gate, scale in ((999999,"warmup",0,0), (1000000,"joint",0,0),
                                        (2000000,"joint",.5,.5), (3000000,"joint",1,1),
                                        (5000000,"joint",1,.6)):
            self.assertEqual(p.learner._get_stage(clock), stage)
            self.assertAlmostEqual(p.mac.set_residual_actor_gate_for_t(clock), gate)
            self.assertAlmostEqual(p.learner._belief_scale(clock), scale)
        batch = self.batch(p, env)
        before = p.mac.agent.teacher_head.weight.detach().clone()
        p.mac.set_residual_actor_gate_for_t(999200)
        learn(p, batch, 999200, 4)
        torch.testing.assert_close(before, p.mac.agent.teacher_head.weight, atol=0, rtol=0)
        p.mac.set_residual_actor_gate_for_t(1000800)
        learn(p, batch, 1000800, 8)
        self.assertEqual(p.logger.values["dib_stage"], 2)
        self.assertGreater(p.logger.values["belief_loss_scale"], 0)
        self.assertFalse(torch.equal(before, p.mac.agent.teacher_head.weight))

    def test_alive_profile_native_and_accelerated_replay_updates_match(self):
        native, env = self.policy(native=True)
        fast, _ = self.policy()
        batch = self.batch(native, env)
        batch["obs"][:, 3:, 1, 3] = 0
        for clock in (999200, 1000800, 3000800):
            for p in (native, fast):
                p.mac.set_residual_actor_gate_for_t(clock)
                learn(p, batch, clock, 4)
            self.assert_models_close(native.mac.agent, fast.mac.agent)
            self.assert_models_close(native.learner.critic, fast.learner.critic)
            self.assertEqual(native.logger.values["active_actor_fraction"], fast.logger.values["active_actor_fraction"])

    def test_mhf_matches_proposal_times_latest_transition_likelihood(self):
        p, _ = self.policy()
        agent = p.mac.agent
        pi = torch.tensor([[.1,.2,.3,.4], [.4,.3,.2,.1]])
        z = torch.randn(2, 4, agent.z_dim) * .1
        old = torch.randn(2, agent.obs_dim) * .1
        obs = torch.randn_like(old) * .1
        action = torch.zeros(2, agent.n_actions)
        has_prev = torch.tensor([[1.],[0.]])
        weights, error = agent._completed_transition_filter(pi, z, old, obs, action, has_prev)
        expected = agent._normalise_weights(pi * torch.exp(-error.clamp_max(agent.error_clip) / agent.tau_d))
        expected[1] = pi[1]
        torch.testing.assert_close(weights, expected, atol=0, rtol=0)

    def test_mhf_history_weights_are_not_a_required_recursive_posterior(self):
        # main.tex, eq:dynamics-filter: w_t ∝ pi_t exp(-epsilon_t/tau),
        # not w_(t-1). The unused history field is redundant, not a missing
        # Bayesian recursion. Keep compatibility and do not invent mode alignment.
        p, env = self.policy()
        agent = p.mac.agent
        obs = torch.as_tensor(np.asarray(env.get_obs(), np.float32))[None]
        prev = torch.zeros(1, env.n_agents, agent.n_actions)
        with torch.no_grad():
            _, state = agent(obs, prev, agent.init_hidden(1))
            a = {k:v.clone() for k,v in state.items()}
            b = {k:v.clone() for k,v in state.items()}
            a["filter_weights"].fill_(.25)
            b["filter_weights"].zero_(); b["filter_weights"][...,0] = 1
            x, _ = agent(obs, prev, a)
            y, _ = agent(obs, prev, b)
            torch.testing.assert_close(x["weights"], y["weights"], atol=0, rtol=0)
            torch.testing.assert_close(x["logits_k"], y["logits_k"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
