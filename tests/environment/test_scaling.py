"""Reference rewards, density controls and diagnostic-only information boundaries."""
import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from heterogrid import CommunicationConfig, EnvConfig, RoleStats, SMACAdapter, UnitSpec, parallel_env
from heterogrid.core import GridCombat
from heterogrid.scenarios import balanced_roster, main, scaling_scenarios


def config(red=("strike",), blue=("hybrid",), **kwargs):
    return EnvConfig(map_size=(16, 8, 3), red=[UnitSpec(role) for role in red],
                     blue=[UnitSpec(role) for role in blue], opponent="dummy", **kwargs)


class NormalizationTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertEqual(EnvConfig().reward_normalization, "none")
        self.assertFalse(EnvConfig().diagnostics_enabled)
        for field, values in {"reward_normalization": ("all", True, None),
                              "reward_reference_team_size": (0, -1, 6.0, True),
                              "reward_reference_team_hp": (0, -1, True, "480", float("nan"), float("inf")),
                              "diagnostics_enabled": (1, "true", None)}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    EnvConfig(**{field: value})
        cfg = EnvConfig(reward_normalization="reference", diagnostics_enabled=True)
        self.assertEqual(EnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))), cfg)

    def test_balanced_ceiling_is_equal_across_sizes_and_baseline_unchanged(self):
        for count in (3, 6, 9, 12):
            roster = balanced_roster(count)
            core = GridCombat(EnvConfig(red=roster, blue=roster, reward_normalization="reference"))
            for team in ("red", "blue"):
                factors = core.reward_scales[team]
                self.assertAlmostEqual(factors["damage"], 6 / count)
                self.assertAlmostEqual(factors["kill"], 6 / count)
                self.assertAlmostEqual(.01 * 80 * count * factors["damage"] + count * factors["kill"], 10.8)
            if count == 6:
                self.assertEqual(core.reward_scales["red"], {"damage": 1., "kill": 1.})

    def test_asymmetric_initial_hp_and_counts_used_for_each_scoring_team(self):
        cfg = config(red=("strike", "strike"), blue=("hybrid",), reward_normalization="reference")
        core = GridCombat(cfg)
        self.assertEqual(core.reward_scales["red"], {"damage": 6., "kill": 6.})
        self.assertEqual(core.reward_scales["blue"], {"damage": 2.4, "kill": 3.})
        changed = replace(cfg, role_stats={**cfg.role_stats, "hybrid": RoleStats(160, 5, 4, 12)})
        self.assertEqual(GridCombat(changed).reward_scales["red"]["damage"], 3.)

    def test_damage_and_kill_scaled_but_terminal_time_and_information_are_not(self):
        core = GridCombat(config(red=("strike", "strike"), blue=("scout",),
            reward_normalization="reference", time_penalty=.005, diagnostics_enabled=True))
        core.reset(42, {"red_0": [0, 0, 0], "red_1": [0, 0, 0], "blue_0": [0, 0, 0]})
        core.units["blue_0"].hp = 30
        reward, info = core.step({"red_0": 7, "red_1": 7, "blue_0": 0})
        parts = info["reward_components"]["red"]
        self.assertAlmostEqual(parts["damage"], .01 * 30 * 8)
        self.assertEqual(parts["kill"], 6)
        self.assertEqual(parts["terminal"], 10)
        self.assertEqual(parts["time"], -.005)
        self.assertAlmostEqual(reward["red"], sum(parts.values()))
        self.assertEqual(core.episode_metrics["red"]["damage"], 30)
        self.assertEqual(core.episode_metrics["red"]["kills"], 1)
        # Information coefficients/cap are not modified by normalization.
        self.assertEqual(core.config.information_reward_cap, 1)

    def test_denominator_does_not_change_when_targets_die(self):
        core = GridCombat(config(blue=("scout", "scout"), reward_normalization="reference"))
        core.reset(42, {"red_0": [0, 0, 0], "blue_0": [0, 0, 0], "blue_1": [0, 0, 0]})
        for name in ("blue_0", "blue_1"):
            core.units[name].hp = 20
        _, first = core.step({"red_0": 7, "blue_0": 0, "blue_1": 0})
        _, second = core.step({"red_0": 8, "blue_1": 0})
        for row in (first, second):
            self.assertAlmostEqual(row["reward_components"]["red"]["damage"], .8)
            self.assertEqual(row["reward_components"]["red"]["kill"], 3)

    def test_sparse_stays_sparse(self):
        core = GridCombat(config(blue=("scout",), reward_normalization="reference", reward_mode="sparse"))
        core.reset(42, {"red_0": [0, 0, 0], "blue_0": [0, 0, 0]})
        core.units["blue_0"].hp = 20
        rewards, info = core.step({"red_0": 7, "blue_0": 0})
        self.assertEqual(rewards["red"], 10)
        self.assertEqual(info["reward_components"]["red"]["damage"], 0)
        self.assertEqual(info["reward_components"]["red"]["kill"], 0)

    def test_information_bonus_and_timeout_penalty_are_not_scaled(self):
        positions = {"red_0": [0, 0, 0], "red_1": [2, 0, 0], "blue_0": [8, 0, 0]}
        settings = dict(red=("scout", "strike"), blue=("strike",), communication="full",
                        target_sharing="auto", discovery_reward=.1, sharing_reward=.1,
                        information_reward_cap=.15, timeout_as_loss=True, max_steps=2, time_penalty=.005)
        cores = [GridCombat(config(**settings, reward_normalization=mode)) for mode in ("none", "reference")]
        transitions = []
        for core in cores:
            core.reset(42, positions)
            _, info = core.step({"red_0": 1, "red_1": 0, "blue_0": 0})
            self.assertAlmostEqual(info["reward_components"]["red"]["discovery"], .1)
            self.assertAlmostEqual(info["reward_components"]["red"]["sharing"], .05)
            transitions.append(core.step({"red_0": 0, "red_1": 0, "blue_0": 0}))
        self.assertEqual(transitions[0], transitions[1])
        self.assertEqual(transitions[1][1]["reward_components"]["red"]["terminal"], -10)


class DensityTests(unittest.TestCase):
    def test_shipped_configs_match_generator(self):
        directory = Path(__file__).resolve().parents[2] / "configs/scaling"
        baseline = EnvConfig.load(directory / "density_6v6.json")
        generated = scaling_scenarios(baseline, mode="constant_density")
        for name, cfg in generated.items():
            self.assertEqual(cfg, EnvConfig.load(directory / ("density_" + name + ".json")))

    def test_map_rounding_density_and_unchanged_other_settings(self):
        baseline = EnvConfig(target_sharing="auto", reward_normalization="reference", diagnostics_enabled=True,
                             blue_communication=CommunicationConfig("distance", 10, 0))
        fixed = scaling_scenarios(baseline)
        density = scaling_scenarios(baseline, mode="constant_density")
        sizes = [(19, 19, 5), (24, 24, 6), (27, 27, 7), (30, 30, 8)]
        for count, size in zip((3, 6, 9, 12), sizes):
            name = f"{count}v{count}"
            self.assertEqual(fixed[name].map_size, baseline.map_size)
            self.assertEqual(density[name].map_size, size)
            ratio = (count / math.prod(size)) / (6 / math.prod(baseline.map_size))
            self.assertLess(abs(ratio - 1), .05)
            for field, value in baseline.to_dict().items():
                if field not in ("red", "blue", "map_size"):
                    self.assertEqual(density[name].to_dict()[field], value)
            self.assertEqual(density[name].red, density[name].blue)
            self.assertEqual(len(density[name].red), count)
        self.assertEqual(density["6v6"].red, baseline.red)

    def test_invalid_counts_modes_and_baselines(self):
        for counts in ((), (3, 3), (0,), (4,), (True,), (6.,)):
            with self.assertRaises(ValueError):
                scaling_scenarios(EnvConfig(), counts)
        with self.assertRaises(ValueError):
            scaling_scenarios(EnvConfig(), mode="unknown")
        with self.assertRaises(ValueError):
            scaling_scenarios(config(red=("scout",), blue=("strike",)))

    def test_cli_writes_standalone_configs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.json"
            base.write_text(json.dumps(EnvConfig().to_dict()))
            output = Path(directory) / "generated"
            args = ["scenarios", "--base", str(base), "--output", str(output),
                    "--counts", "3", "12", "--reward-normalization", "reference", "--diagnostics"]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(len(list(output.glob("*.json"))), 2)
            self.assertEqual(EnvConfig.load(output / "12v12.json").map_size, (30, 30, 8))
            self.assertTrue(EnvConfig.load(output / "3v3.json").diagnostics_enabled)
            self.assertTrue((output / "README.md").exists())
            with patch("sys.argv", args), self.assertRaises(FileExistsError):
                main()


class DiagnosticsTests(unittest.TestCase):
    def test_initial_network_coverage_distance_and_no_resampling(self):
        core = GridCombat(config(red=("scout", "strike"), blue=("scout", "strike"),
                                 diagnostics_enabled=True, target_sharing="auto", communication="full"))
        core.reset(42, {"red_0": [3, 0, 0], "red_1": [0, 0, 0],
                        "blue_0": [6, 0, 0], "blue_1": [15, 0, 0]})
        metrics = core.team_episode_metrics()["red"]["diagnostics"]
        self.assertEqual(metrics["state_samples"], 1)
        self.assertEqual(metrics["initial_comm_degree"], 1)
        self.assertEqual(metrics["initial_scout_coverage_fraction"], .5)
        self.assertEqual(metrics["initial_strike_scout_link_fraction"], 1)
        self.assertEqual(metrics["initial_mean_nearest_enemy_distance"], 4.5)
        rng = copy.deepcopy(core.rng.bit_generator.state)
        for _ in range(3):
            core.observation("red_0"); core.team_episode_metrics(); core.snapshot()
            core.exchange_messages({"red_0": {"targets": ["blue_0"]}})
        self.assertEqual(core.rng.bit_generator.state, rng)
        self.assertEqual(core.team_episode_metrics()["red"]["diagnostics"]["state_samples"], 1)
        json.dumps(core.snapshot(), allow_nan=False)

    def test_simultaneous_overkill_survival_and_reset(self):
        core = GridCombat(config(red=("strike", "strike"), blue=("scout",), diagnostics_enabled=True))
        positions = {"red_0": [0, 0, 0], "red_1": [0, 0, 0], "blue_0": [0, 0, 0]}
        core.reset(42, positions); core.units["blue_0"].hp = 30
        core.step({"red_0": 7, "red_1": 7, "blue_0": 0})
        metrics = core.team_episode_metrics()
        red, blue = metrics["red"]["diagnostics"], metrics["blue"]["diagnostics"]
        for field, value in {"attack_attempts": 2, "hit_attacks": 2, "raw_hit_damage": 40,
                             "effective_damage": 30, "overkill_damage": 10, "overkill_fraction": .25,
                             "state_samples": 2, "final_alive_fraction": 1}.items():
            self.assertEqual(red[field], value)
        self.assertEqual(blue["final_alive_fraction"], 0)
        self.assertEqual(blue["mean_alive_fraction"], .5)
        self.assertIsNone(blue["attack_hit_rate"])
        self.assertIsNone(blue["mean_strike_scout_link_fraction"])
        core.reset(42, positions)
        self.assertEqual(core.team_episode_metrics()["red"]["diagnostics"]["attack_attempts"], 0)

    def test_masked_attack_and_stale_track_miss_are_distinct(self):
        core = GridCombat(config(red=("scout", "strike"), diagnostics_enabled=True))
        core.reset(42, {"red_0": [0, 0, 0], "red_1": [0, 0, 0], "blue_0": [3, 0, 0]})
        core.units["blue_0"].position[:] = [0, 4, 0]
        core.step({"red_0": 7, "red_1": 7, "blue_0": 0})
        row = core.team_episode_metrics()["red"]["diagnostics"]
        self.assertEqual(row["attack_attempts"], 2)
        self.assertEqual(row["invalid_attack_attempts"], 1)
        self.assertEqual(row["executed_attacks"], 1)
        self.assertEqual(row["missed_attacks"], 1)
        self.assertEqual(row["ineffective_attack_attempts"], 2)
        self.assertEqual(row["invalid_attack_rate"], .5)
        self.assertIsNone(row["overkill_fraction"])

    def test_enabling_diagnostics_preserves_actions_rng_physics_rewards_and_spaces(self):
        for mode in ("none", "full", "distance", "random"):
            baseline = EnvConfig(communication=mode, target_sharing="auto", max_steps=30)
            a, b = [SMACAdapter(config=replace(baseline, diagnostics_enabled=flag)) for flag in (False, True)]
            rng = np.random.default_rng(123)
            try:
                a.reset(42); b.reset(42)
                self.assertEqual(a.get_env_info(), b.get_env_info())
                while True:
                    np.testing.assert_equal(a.get_obs(), b.get_obs())
                    self.assertEqual(a.get_avail_actions(), b.get_avail_actions())
                    actions = [int(rng.choice(np.flatnonzero(mask))) for mask in a.get_avail_actions()]
                    ra, da, ia = a.step(actions); rb, db, ib = b.step(actions)
                    self.assertEqual((ra, da), (rb, db))
                    self.assertEqual(a.env.core.rng.bit_generator.state, b.env.core.rng.bit_generator.state)
                    self.assertNotIn("team_episode_metrics", ia)
                    self.assertEqual(set(ib["team_episode_metrics"]), {"red", "blue"})
                    if da:
                        break
            finally:
                a.close(); b.close()

    def test_parallel_info_replay_and_logging_without_leaking_to_blue_policy(self):
        class Policy:
            def reset(self, seed=None):
                pass
            def act(self, observations, infos):
                assert all("team_episode_metrics" not in info for info in infos.values())
                return {name: 0 for name in observations}
        cfg = replace(config(diagnostics_enabled=True, max_steps=1), opponent="pretrained")
        env = parallel_env(config=cfg, opponent_policy=Policy(), record_replay=True)
        try:
            with self.assertRaises(RuntimeError):
                env.episode_metrics()
            obs, _ = env.reset(42)
            self.assertEqual(set(env.episode_metrics()), {"red", "blue"})
            _, _, _, _, infos = env.step({name: 0 for name in env.agents})
            self.assertIn("team_episode_metrics", infos["red_0"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "replay.json"
                env.save_replay(path)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["frames"][-1]["episode_metrics"]["blue"]["diagnostics"]["state_samples"], 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
