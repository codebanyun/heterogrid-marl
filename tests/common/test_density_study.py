import json
import math
import tempfile
import unittest
from pathlib import Path

from algorithms.common import communication_study, density_study
from algorithms.common.study_queue import jobs_for
from heterogrid import CommunicationConfig, EnvConfig
from heterogrid.core import GridCombat


class DensityStudyTests(unittest.TestCase):
    def setUp(self):
        self.base = EnvConfig.load(communication_study.mappo.DEFAULT_CONFIG)
        self.configs = density_study.conditions(self.base)

    def test_exact_d_scenarios_and_reward_normalization(self):
        self.assertEqual(tuple(self.configs), density_study.CONDITION_NAMES)
        for (name, config), count, size in zip(self.configs.items(), (6, 3, 9, 12),
                                               ((24, 24, 6), (19, 19, 5), (27, 27, 7), (30, 30, 8))):
            self.assertEqual(len(config.red), count)
            self.assertEqual(len(config.blue), count)
            self.assertEqual(config.map_size, size)
            self.assertEqual(config.max_steps, 200)
            self.assertTrue(config.diagnostics_enabled)
            self.assertEqual(config.reward_normalization, "reference")
            for team in ("red", "blue"):
                self.assertEqual(config.communication_for(team), CommunicationConfig("distance", 10, 0))
                for role in ("scout", "strike", "hybrid"):
                    self.assertEqual(sum(unit.role == role for unit in getattr(config, team)), count // 3)
            density = count / math.prod(size)
            self.assertLess(abs(density / (6 / 3456) - 1), .05)
            core = GridCombat(config)
            self.assertAlmostEqual(core.reward_scales["red"]["damage"], 6 / count)
            self.assertAlmostEqual(core.reward_scales["red"]["kill"], 6 / count)
            fixture = EnvConfig.load(Path("configs/scaling") / ("density_%dv%d.json" % (count, count)))
            self.assertEqual(config, fixture)

    def test_twenty_four_jobs_prioritize_larger_teams(self):
        manifest = dict(conditions=[a + "/" + c for c in self.configs for a in ("mappo", "smpe")],
                        seeds=[42, 43, 44], team_sizes={a + "/" + c: len(v.red) for c, v in self.configs.items()
                                                       for a in ("mappo", "smpe")})
        jobs = jobs_for(manifest, Path("/output"))
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({job["path"] for job in jobs}), 24)
        self.assertTrue(all("D3_12v12" in job["id"] for job in jobs[:6]))
        self.assertTrue(all("D1_3v3" in job["id"] for job in jobs[-6:]))

    def test_diagnostic_summary_preserves_nulls_and_counts(self):
        runs = []
        for seed, value in ((42, .2), (43, None), (44, .8)):
            result = dict(episodes=128, mean_team_episode_metrics={"red": {"diagnostics": {"attack_hit_rate": value}}},
                          team_episode_metrics_observed_episodes={"red": {"diagnostics": {"attack_hit_rate": 0 if value is None else 128}}})
            runs.append(dict(seed=seed, summary={"heldout": {"latest": {"greedy": result}}}))
        with tempfile.TemporaryDirectory() as directory:
            density_study.diagnostic_summary(Path(directory), runs)
            data = json.loads((Path(directory) / "diagnostics.json").read_text())
            row = data["metrics"][0]
            self.assertEqual(row["n_seeds"], 2)
            self.assertAlmostEqual(row["mean"], .5)
            self.assertAlmostEqual(row["sem"], .3)
            self.assertIsNone(row["seed_43"])


if __name__ == "__main__":
    unittest.main()
