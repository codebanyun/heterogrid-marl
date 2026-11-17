"""Checkpoint continuation contracts for the optional native EPyMARL backend."""
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

BACKEND = Path(os.environ.get("EPYMARL_ROOT", "/root/research/epymarl/src"))


@unittest.skipUnless(torch is not None and (BACKEND / "learners/ppo_learner.py").exists(),
                     "Optional torch/EPyMARL checkout not available")
class EPyMARLResumeTests(unittest.TestCase):
    def test_collector_populates_coma_joint_action_features(self):
        from algorithms.mappo.epymarl_transfer import build_policy, collect
        from heterogrid import EnvConfig, TaskSuite, TransferSMACAdapter
        from heterogrid.config import UnitSpec

        torch.set_num_threads(1)
        config = EnvConfig(map_size=(9, 9, 1), red=[UnitSpec("strike"), UnitSpec("hybrid")],
                           blue=[UnitSpec("hybrid")], opponent="static", max_steps=3)
        suite = TaskSuite({"duel": config}, {"train": ["duel"]})
        env = TransferSMACAdapter(suite)
        self.addCleanup(env.close)
        policy = build_policy(BACKEND, env.get_env_info(), torch.device("cpu"), algorithm="coma")
        rows, batch = collect(policy, [(env, "duel", 7)], 0, training=True)
        active = batch["filled"][:, :-1].bool().expand(-1, -1, policy.args.n_agents)
        onehot = batch["actions_onehot"][:, :-1]
        self.assertTrue(torch.all(onehot.sum(-1)[active] == 1))
        self.assertTrue(torch.equal(onehot.argmax(-1)[active], batch["actions"][:, :-1, :, 0][active]))
        critic = policy.learner.critic
        before = critic._build_inputs(batch).clone()
        original = batch["actions_onehot"][0, 0, 1].clone()
        batch["actions_onehot"][0, 0, 1] = original.roll(1)
        after = critic._build_inputs(batch)
        self.assertFalse(torch.equal(before[0, 0, 0], after[0, 0, 0]))
        self.assertTrue(torch.equal(before[0, 0, 1], after[0, 0, 1]))
        batch["actions_onehot"][0, 0, 1] = original
        policy.learner.train(batch, sum(row["length"] for row in rows), len(rows))
        self.assertTrue(all(torch.isfinite(p).all() for p in policy.mac.agent.parameters()))

    def test_checkpoint_restores_native_learner_and_rng_and_supports_legacy(self):
        from algorithms.mappo.epymarl_transfer import build_policy, checkpoint, collect, restore_checkpoint
        from heterogrid import EnvConfig, TaskSuite, TransferSMACAdapter
        from heterogrid.config import UnitSpec

        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        suite = TaskSuite({"duel": EnvConfig(map_size=(9, 9, 1), red=[UnitSpec("strike")],
                                           blue=[UnitSpec("hybrid")], opponent="static", max_steps=5)},
                          {"train": ["duel"]})
        envs = [TransferSMACAdapter(suite) for _ in range(2)]
        for env in envs:
            self.addCleanup(env.close)
        policy = build_policy(BACKEND, envs[0].get_env_info(), torch.device("cpu"))
        rows, batch = collect(policy, [(env, "duel", seed) for env, seed in zip(envs, (3, 4))], 0, training=True)
        steps = sum(row["length"] for row in rows)
        policy.learner.train(batch, steps, 2)
        rng = np.random.default_rng(42)
        rng.integers(0, 2**31, size=2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "latest.pt"
            checkpoint(path, policy, {}, steps, 1, counts={"duel": steps}, completed=2, rng=rng)
            saved = torch.load(path, weights_only=False)
            expected = (random.random(), np.random.rand(), rng.integers(0, 2**31), torch.rand(3))
            restored = build_policy(BACKEND, envs[0].get_env_info(), torch.device("cpu"))
            restored_rng = np.random.default_rng(0)
            counts, completed, exact = restore_checkpoint(saved, restored, output, ["duel"], 2, restored_rng)
            self.assertEqual((counts, completed, exact), ({"duel": steps}, 2, True))
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.rand(), expected[1])
            self.assertEqual(restored_rng.integers(0, 2**31), expected[2])
            torch.testing.assert_close(torch.rand(3), expected[3], rtol=0, atol=0)
            for name in ("critic", "target_critic"):
                for key, value in getattr(policy.learner, name).state_dict().items():
                    torch.testing.assert_close(getattr(restored.learner, name).state_dict()[key], value)
            self.assertEqual(restored.learner.critic_training_steps, 1)
            self.assertTrue(restored.learner.agent_optimiser.state_dict()["state"])
            for key, value in restored.mac.agent.state_dict().items():
                torch.testing.assert_close(restored.learner.old_mac.agent.state_dict()[key], value)

            legacy = {key: value for key, value in saved.items() if key != "training_state"}
            (output / "metrics.jsonl").write_text(json.dumps({"environment_steps": steps, "update": 1,
                                                              "task_environment_steps": {"duel": steps}}) + "\n")
            legacy_rng = np.random.default_rng(42)
            counts, completed, exact = restore_checkpoint(legacy, restored, output, ["duel"], 2, legacy_rng)
            self.assertEqual((counts, completed, exact), ({"duel": steps}, 2, False))
            expected_rng = np.random.default_rng(42)
            expected_rng.integers(0, 2**31)
            expected_rng.integers(0, 2**31)
            self.assertEqual(legacy_rng.bit_generator.state, expected_rng.bit_generator.state)
