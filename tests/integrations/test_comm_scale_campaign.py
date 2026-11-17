import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from algorithms.common.comm_scale_campaign import matrix, equivalent, reuse_source, select_pending, memory_estimate
from algorithms.common.sensitivity import scenarios
from heterogrid import CommunicationConfig


class CommScaleTests(unittest.TestCase):
    def test_exact_matrix_and_roles(self):
        tasks = matrix(scenarios(True)["B0"])
        self.assertEqual(len(tasks), 15)
        for cond in ("R5", "P50", "C1S"):
            for scale, size, count in (("B0", (24, 24, 6), 6), ("M2", (36, 36, 9), 6),
                                       ("N2", (24, 24, 6), 9), ("D2", (27, 27, 7), 9)):
                cfg = tasks[cond + "_" + scale]
                self.assertEqual(cfg.map_size, size)
                self.assertEqual(len(cfg.red), count); self.assertEqual(len(cfg.blue), count)
                self.assertEqual(cfg.max_steps, 200); self.assertFalse(cfg.terminate_on_no_firepower)
                self.assertEqual(cfg.communication_for("blue").mode, "none" if cond == "C1S" else "distance")
                self.assertEqual(cfg.communication_for("red").dropout_probability, .5 if cond == "P50" else 0)
        for name, counts in (("R_scout", (3, 1, 2)), ("R_strike", (2, 3, 1)), ("R_hybrid", (1, 2, 3))):
            cfg = tasks["C1S_" + name]
            self.assertEqual(tuple(sum(u.role == role for u in cfg.red) for role in ("scout", "strike", "hybrid")), counts)
            self.assertEqual(cfg.red, cfg.blue)
            self.assertEqual(cfg.communication_for("blue").mode, "none")
        self.assertEqual(tasks["R5_D2"].reward_normalization, "reference")
        self.assertEqual(tasks["R5_N2"].reward_normalization, "none")

    def test_reuse_counts_and_no_false_v2_reuse(self):
        tasks = matrix(scenarios(True)["B0"])
        counts = Counter()
        for a in ("dib_v3", "qmix", "qtran", "maa2c", "coma"):
            for scene in tasks:
                for seed in (42, 43, 44):
                    if reuse_source(Path("/fake"), a, scene, seed): counts[a] += 1
        self.assertEqual(counts, {"dib_v3": 6, "qmix": 9, "qtran": 9, "maa2c": 9, "coma": 9})
        self.assertEqual(sum(counts.values()), 42)
        self.assertEqual(225 - sum(counts.values()), 183)

    def test_equivalence_does_not_ignore_dynamics_or_reward(self):
        tasks = matrix(scenarios(True)["B0"]); cfg = tasks["C1S_B0"]
        self.assertTrue(equivalent(cfg, scenarios(False)["C1"]))
        self.assertFalse(equivalent(cfg, replace(cfg, terminate_on_no_firepower=True)))
        self.assertFalse(equivalent(cfg, replace(cfg, blue_communication=CommunicationConfig("full"))))
        self.assertFalse(equivalent(cfg, replace(cfg, loss_reward=-11)))
        self.assertFalse(equivalent(cfg, replace(cfg, map_size=(25, 25, 6))))

    def test_scheduler_prioritizes_long_dib_but_leaves_baseline_slots(self):
        jobs = []
        for a in ("dib_v3", "qmix", "qtran"):
            for count in (6, 9):
                for i in range(10):
                    jobs.append(dict(algorithm=a, status="PENDING", phase="TRAINING", n_agents=count,
                                     scale="N2" if count == 9 else "M2", scenario=str(i), seed=42, training_reused=False))
        state = {"jobs": jobs, "dib_workers_while_baselines_pending": 12}
        chosen = select_pending(state, [], 20)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(sum(j["algorithm"] == "dib_v3" for j in chosen), 12)
        self.assertEqual(sum(j["algorithm"] == "dib_v3" and j["n_agents"] == 9 for j in chosen), 10)
        self.assertTrue(all(memory_estimate(j) > 0 for j in chosen))
        self.assertEqual(select_pending(state, [], 0), [])


if __name__ == "__main__":
    unittest.main()
