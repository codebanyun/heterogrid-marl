"""Scientific contracts for the QMIX/QTRAN integration (no backend needed)."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from algorithms.common.sensitivity import balanced_roster, scenarios
from algorithms.common.value_decomposition.replay import EpisodeReplay


class SensitivityTests(unittest.TestCase):
    def test_matrix_preserves_task_and_reward_contracts(self):
        tasks = scenarios()
        self.assertEqual(len(tasks), 13)
        baseline = tasks["B0"].to_dict()
        allowed = {"M1": {"map_size"}, "M2": {"map_size"}, "M3": {"map_size"},
                   "N1": {"red", "blue"}, "N2": {"red", "blue"}, "N3": {"red", "blue"},
                   "C1": {"communication"}, "C2": {"communication"}, "C3": {"comm_radius"},
                   "C4": {"comm_radius"}, "C5": {"communication"},
                   "C6": {"communication", "dropout_probability"}}
        for name, cfg in tasks.items():
            self.assertEqual(cfg.opponent, "rule")
            self.assertEqual(cfg.max_steps, 200)
            self.assertEqual(cfg.target_sharing, "auto")
            self.assertEqual(len(cfg.red), len(cfg.blue))
            for roster in (cfg.red, cfg.blue):
                self.assertEqual([sum(u.role == role for u in roster) for role in
                                  ("scout", "strike", "hybrid")], [len(roster) // 3] * 3)
            if name != "B0":
                differences = {k for k, v in cfg.to_dict().items() if v != baseline[k]}
                self.assertEqual(differences, allowed[name])
        self.assertEqual(balanced_roster(6), tasks["B0"].red)
        self.assertEqual(tasks["N3"].map_size, (24, 24, 6))
        self.assertEqual(tasks["M3"].map_size, (48, 48, 12))


class ReplayTests(unittest.TestCase):
    def episode(self, length):
        return {"o": np.arange((length + 1) * 2 * 3, dtype=np.float32).reshape(length + 1, 2, 3),
                "s": np.arange((length + 1) * 4, dtype=np.float32).reshape(length + 1, 4),
                "avail": np.ones((length + 1, 2, 3), dtype=np.uint8),
                "u": np.ones((length, 2), dtype=np.uint8), "r": np.ones(length, dtype=np.float32)}

    def test_shift_terminal_and_padding_are_exact(self):
        replay = EpisodeReplay(3, 7)
        replay.add(self.episode(2)); replay.add(self.episode(5))
        batch, length = replay.sample(32, 3)
        self.assertEqual(length, 5)
        for i in range(32):
            t = int((1 - batch["padded"][i]).sum())
            source = self.episode(t)
            np.testing.assert_array_equal(batch["o"][i, :t], source["o"][:-1])
            np.testing.assert_array_equal(batch["o_next"][i, :t], source["o"][1:])
            np.testing.assert_array_equal(batch["s_next"][i, :t], source["s"][1:])
            self.assertEqual(batch["terminated"][i, t - 1, 0], 1)
            self.assertEqual(batch["terminated"][i, :t - 1].sum(), 0)
            self.assertEqual(batch["padded"][i, :t].sum(), 0)
            self.assertTrue(np.all(batch["avail_u_next"][i, t:, :, 0] == 1))

    def test_ring_capacity_and_float32_storage(self):
        replay = EpisodeReplay(2, 7)
        for length in (2, 3, 4):
            replay.add(self.episode(length))
        self.assertEqual(len(replay), 2)
        self.assertEqual(sorted(len(e["u"]) for e in replay.episodes), [3, 4])
        self.assertTrue(all(e["o"].dtype == np.float32 for e in replay.episodes))


class ReportingTests(unittest.TestCase):
    def test_missing_seed_and_unshared_checkpoints_never_become_zeroes(self):
        from algorithms.common.value_decomposition.report import summarize
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "experiment.json").write_text(json.dumps({"algorithm": "qmix", "seeds": [42, 43, 44],
                "steps": 5000000, "eval_episodes": 64, "heldout_episodes": 128}))
            for seed, steps in ((42, (0, 250000)), (43, (0,))):
                for step in steps:
                    path = root / "B0" / f"seed_{seed}" / "evaluations" / f"validation_{step:09d}"
                    path.mkdir(parents=True)
                    (path / "summary.json").write_text(json.dumps({"win_rate": 0.5, "episodes": 64}))
            with patch("algorithms.common.value_decomposition.report.plot_curves") as plotting:
                summarize(root)
                self.assertEqual(plotting.call_args[0][1]["B0"], [])
                path = root / "B0/seed_44/evaluations/validation_000000000"
                path.mkdir(parents=True)
                (path / "summary.json").write_text(json.dumps({"win_rate": 0.5, "episodes": 64}))
                summarize(root)
                points = plotting.call_args[0][1]["B0"]
                self.assertEqual(len(points), 1)
                self.assertEqual(points[0]["step"], 0)
                self.assertEqual(points[0]["mean"], 0.5)
                self.assertEqual(points[0]["sem"], 0.0)
                self.assertEqual(points[0]["n"], 3)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional torch not installed")
class RecurrentHistoryTests(unittest.TestCase):
    def test_target_history_starts_at_initial_observation(self):
        from types import SimpleNamespace
        from algorithms.common.value_decomposition.backend import recurrent_values

        class AdditiveRNN(torch.nn.Module):
            def forward(self, inputs, hidden):
                result = hidden + inputs[:, :1]
                return result.repeat(1, 2), result

        policy = SimpleNamespace(args=SimpleNamespace(rnn_hidden_dim=1), eval_rnn=AdditiveRNN(), target_rnn=AdditiveRNN())
        batch = {"o": torch.tensor([[[[1.]], [[2.]], [[3.]]]]),
                 "o_next": torch.tensor([[[[2.]], [[3.]], [[4.]]]]),
                 "u_onehot": torch.zeros(1, 3, 1, 2)}
        current, targets, _, _ = recurrent_values(policy, batch, 3)
        np.testing.assert_array_equal(current[0, :, 0, 0].numpy(), [1, 3, 6])
        np.testing.assert_array_equal(targets[0, :, 0, 0].numpy(), [3, 6, 10])
        self.assertFalse(targets.requires_grad)


if __name__ == "__main__":
    unittest.main()
