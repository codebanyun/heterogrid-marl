"""Optional firepower adjudication is symmetric, terminal and legacy-safe."""
import unittest
from dataclasses import replace

import numpy as np

from heterogrid import EnvConfig, RoleStats, SMACAdapter, UnitSpec, parallel_env
from heterogrid.core import ATTACK_OFFSET, GridCombat


def config(**kwargs):
    base = EnvConfig(red=[UnitSpec("scout"), UnitSpec("strike")],
                     blue=[UnitSpec("scout"), UnitSpec("strike")],
                     opponent="dummy", terminate_on_no_firepower=True,
                     reward_mode="sparse", time_penalty=0.005, timeout_as_loss=True)
    return replace(base, **kwargs)


def core_for(cfg=None):
    core = GridCombat(cfg or config())
    core.reset(42, {"red_0": [0, 0, 0], "red_1": [1, 0, 0],
                    "blue_0": [0, 1, 0], "blue_1": [1, 1, 0]})
    return core


def wait(core):
    return core.step({n: 0 for n in core.all_names if core.units[n].alive})


class FirepowerTerminationTests(unittest.TestCase):
    def test_unilateral_firepower_loss_is_symmetric_and_paid_once(self):
        for loser, winner in (("red", "blue"), ("blue", "red")):
            with self.subTest(loser=loser):
                core = core_for()
                core.units[loser + "_1"].hp = 0
                rewards, info = wait(core)
                self.assertTrue(core.finished)
                self.assertEqual(core.reason, "no_firepower")
                self.assertEqual(core.winner, winner)
                self.assertAlmostEqual(rewards[loser], -10.005)
                self.assertAlmostEqual(rewards[winner], 9.995)
                self.assertEqual(info["kills"], {"red": 0, "blue": 0})
                self.assertEqual(info["died"], [])
                self.assertTrue(core.units[loser + "_0"].alive)
                with self.assertRaises(RuntimeError):
                    wait(core)

    def test_simultaneous_last_guns_die_with_scouts_surviving_is_draw(self):
        core = core_for(config(draw_reward=-0.25, max_steps=1))
        core.units["red_1"].hp = core.units["blue_1"].hp = 20
        rewards, info = core.step({"red_0": 0, "red_1": ATTACK_OFFSET + 1,
                                   "blue_0": 0, "blue_1": ATTACK_OFFSET + 1})
        self.assertEqual(core.reason, "mutual_no_firepower")
        self.assertIsNone(core.winner)
        self.assertEqual(info["kills"], {"red": 1, "blue": 1})
        self.assertEqual(info["damage"], {"red": 20, "blue": 20})
        for team in ("red", "blue"):
            self.assertAlmostEqual(rewards[team], -0.255)
            self.assertEqual(info["reward_components"][team]["terminal"], -0.25)
            self.assertTrue(core.units[team + "_0"].alive)

    def test_mutual_exhaustion_does_not_receive_timeout_loss(self):
        core = core_for(config(max_steps=1))
        core.units["red_1"].hp = core.units["blue_1"].hp = 0
        rewards, _ = wait(core)
        self.assertEqual(core.reason, "mutual_no_firepower")
        self.assertEqual(rewards, {"red": -0.005, "blue": -0.005})

    def test_elimination_has_priority_even_if_survivor_is_unarmed(self):
        core = core_for(config(max_steps=1))
        for name in ("red_1", "blue_0", "blue_1"):
            core.units[name].hp = 0
        rewards, _ = wait(core)
        self.assertEqual((core.reason, core.winner), ("elimination", "red"))
        self.assertAlmostEqual(rewards["red"], 9.995)

    def test_mutual_elimination_has_draw_reward_and_no_timeout_loss(self):
        core = core_for(config(draw_reward=0.5))
        core.units["red_0"].hp = core.units["blue_0"].hp = 0
        core.units["red_1"].hp = core.units["blue_1"].hp = 20
        rewards, _ = core.step({"red_1": ATTACK_OFFSET + 1, "blue_1": ATTACK_OFFSET + 1})
        self.assertEqual(core.reason, "mutual_elimination")
        self.assertEqual(rewards, {"red": 0.495, "blue": 0.495})

    def test_hidden_far_away_and_offline_armed_units_do_not_terminate(self):
        core = core_for(config(communication="none"))
        core.reset(42, {"red_0": [0, 0, 0], "red_1": [1, 0, 0],
                        "blue_0": [23, 23, 4], "blue_1": [23, 23, 5]})
        core.online = dict.fromkeys(core.all_names, False)
        self.assertFalse(core.action_mask("red_1")[ATTACK_OFFSET:].any())
        wait(core)
        self.assertFalse(core.finished)

    def test_capability_depends_on_damage_not_role_or_current_range(self):
        stats = {**EnvConfig().role_stats, "scout": RoleStats(60, 0, 0, 1),
                 "strike": RoleStats(100, 3, 6, 0)}
        core = core_for(config(role_stats=stats))
        wait(core)
        self.assertFalse(core.finished)  # Range zero can still hit a co-located enemy.
        core.units["red_0"].hp = 0
        wait(core)
        self.assertEqual((core.reason, core.winner), ("no_firepower", "blue"))

    def test_initial_unarmed_teams_end_on_first_transition_not_reset(self):
        cfg = config(red=[UnitSpec("scout")], blue=[UnitSpec("scout")])
        core = GridCombat(cfg)
        core.reset(42)
        self.assertFalse(core.finished)
        self.assertEqual(core.step_count, 0)
        wait(core)
        self.assertEqual(core.reason, "mutual_no_firepower")
        self.assertEqual(core.step_count, 1)

    def test_legacy_default_does_not_end_scout_only_episodes_early(self):
        self.assertFalse(EnvConfig().terminate_on_no_firepower)
        core = core_for(config(terminate_on_no_firepower=False, max_steps=3))
        core.units["red_1"].hp = 0
        for _ in range(2):
            wait(core)
            self.assertFalse(core.finished)
        rewards, _ = wait(core)
        self.assertEqual((core.reason, core.winner), ("timeout", None))
        self.assertAlmostEqual(rewards["red"], -10.005)

    def test_enabled_and_disabled_match_until_a_firepower_event(self):
        a, b = core_for(), core_for(config(terminate_on_no_firepower=False))
        for _ in range(5):
            self.assertEqual(wait(a), wait(b))
            self.assertEqual(a.snapshot(), b.snapshot())
            np.testing.assert_array_equal(a.state(), b.state())
            for name in a.all_names:
                for field, value in a.observation(name).items():
                    np.testing.assert_array_equal(value, b.observation(name)[field])

    def test_parallel_api_marks_firepower_end_terminated_not_truncated(self):
        env = parallel_env(config=config())
        try:
            env.reset(42)
            env.core.units["red_1"].hp = 0
            env.agents.remove("red_1")
            _, rewards, terms, truncs, _ = env.step({"red_0": 0})
            self.assertTrue(terms["red_0"])
            self.assertFalse(truncs["red_0"])
            self.assertAlmostEqual(rewards["red_0"], -10.005)
            self.assertEqual(env.agents, [])
        finally:
            env.close()

    def test_smac_marks_firepower_end_terminal_and_updates_win_stats(self):
        env = SMACAdapter(config=config())
        try:
            env.reset(42)
            env.env.core.units["blue_1"].hp = 0
            reward, done, info = env.step([0, 0])
            self.assertTrue(done)
            self.assertTrue(info["battle_won"])
            self.assertFalse(info["episode_limit"])
            self.assertEqual(info["termination_reason"], "no_firepower")
            self.assertAlmostEqual(reward, 9.995)
            self.assertEqual(env.get_stats()["battles_won"], 1)
            self.assertEqual(env.get_stats()["timeouts"], 0)
        finally:
            env.close()

    def test_config_round_trip_and_validation(self):
        cfg = config(draw_reward=-0.5)
        self.assertEqual(EnvConfig.from_dict(cfg.to_dict()), cfg)
        for value in (1, None, "true"):
            with self.assertRaises(ValueError):
                config(terminate_on_no_firepower=value)
        for value in (True, None, "0", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                config(draw_reward=value)


if __name__ == "__main__":
    unittest.main()
