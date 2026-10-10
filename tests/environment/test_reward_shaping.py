import unittest
from dataclasses import replace

from heterogrid import EnvConfig, SMACAdapter, UnitSpec, parallel_env
from heterogrid.core import GridCombat


def make_config(red=("scout",), blue=("strike",), **kwargs):
    return EnvConfig(map_size=(16, 8, 3), red=[UnitSpec(r) for r in red],
                     blue=[UnitSpec(r) for r in blue], opponent="dummy", **kwargs)


def wait(core):
    return core.step({n: 0 for n in core.all_names if core.units[n].alive})


class DiscoveryRewardTests(unittest.TestCase):
    def test_no_reset_bonus_and_team_discovery_is_only_paid_once(self):
        core = GridCombat(make_config(red=("scout", "scout"), discovery_reward=0.1, target_ttl=1))
        positions = {"red_0": [0, 0, 0], "red_1": [0, 0, 0], "blue_0": [8, 0, 0]}
        core.reset(1, positions)
        self.assertEqual(core.episode_returns["red"], 0)
        self.assertEqual(wait(core)[0]["red"], 0)
        rewards, info = core.step({"red_0": 1, "red_1": 1, "blue_0": 0})
        self.assertAlmostEqual(rewards["red"], 0.1)
        self.assertEqual(info["information_events"]["red"]["discovery"], 1)
        self.assertEqual(core.episode_metrics["red"]["first_contact_step"], 2)
        core.step({"red_0": 2, "red_1": 2, "blue_0": 0})
        wait(core)
        self.assertAlmostEqual(core.step({"red_0": 1, "red_1": 1, "blue_0": 0})[0]["red"], 0)
        core.reset(1, positions)
        self.assertAlmostEqual(core.step({"red_0": 1, "red_1": 1, "blue_0": 0})[0]["red"], 0.1)

    def test_initial_sightings_are_recorded_without_free_reward(self):
        core = GridCombat(make_config(discovery_reward=0.1))
        core.reset(1, {"red_0": [0, 0, 0], "blue_0": [5, 0, 0]})
        self.assertEqual(core.episode_metrics["red"]["first_contact_step"], 0)
        self.assertEqual(core.episode_metrics["red"]["discovered_targets"], 1)
        self.assertEqual(wait(core)[0]["red"], 0)

    def test_dead_targets_do_not_generate_discovery_reward(self):
        core = GridCombat(make_config(blue=("strike", "strike"), discovery_reward=0.1))
        core.reset(1, {"red_0": [0, 0, 0], "blue_0": [8, 0, 0], "blue_1": [15, 0, 0]})
        core.units["blue_0"].hp = 0
        rewards, _ = core.step({"red_0": 1, "blue_1": 0})
        self.assertEqual(rewards["red"], 0)

    def test_information_budget_clips_and_cannot_be_refilled(self):
        core = GridCombat(make_config(blue=("strike", "strike"), discovery_reward=0.1,
                                     information_reward_cap=0.15))
        core.reset(1, {"red_0": [0, 0, 0], "blue_0": [8, 0, 0], "blue_1": [9, 0, 0]})
        self.assertAlmostEqual(core.step({"red_0": 1, "blue_0": 0, "blue_1": 0})[0]["red"], 0.1)
        self.assertAlmostEqual(core.step({"red_0": 1, "blue_0": 0, "blue_1": 0})[0]["red"], 0.05)
        self.assertEqual(wait(core)[0]["red"], 0)
        self.assertAlmostEqual(core.episode_metrics["red"]["information_reward"], 0.15)


class SharingRewardTests(unittest.TestCase):
    def core(self, **kwargs):
        settings = dict(red=("scout", "strike"), communication="full", sharing_reward=0.1, target_ttl=1)
        settings.update(kwargs)
        core = GridCombat(make_config(**settings))
        core.reset(1, {"red_0": [3, 0, 0], "red_1": [0, 0, 0], "blue_0": [6, 0, 0]})
        return core

    def test_useful_report_is_deferred_and_deduplicated_after_expiry(self):
        core = self.core()
        report = {"red_0": {"targets": ["blue_0"]}}
        core.exchange_messages(report)
        core.exchange_messages(report)
        self.assertEqual(core.episode_returns["red"], 0)
        self.assertAlmostEqual(wait(core)[0]["red"], 0.1)
        wait(core)
        self.assertEqual(core.action_mask("red_1")[7], 0)
        core.exchange_messages(report)
        self.assertEqual(wait(core)[0]["red"], 0)
        self.assertEqual(core.episode_metrics["red"]["useful_reports"], 1)

    def test_blocked_links_do_not_receive_bonus(self):
        for settings in ({"communication": "none"}, {"communication": "distance", "comm_radius": 2},
                         {"communication": "random", "dropout_probability": 1}):
            core = self.core(**settings)
            core.exchange_messages({"red_0": {"targets": ["blue_0"]}})
            self.assertEqual(wait(core)[0]["red"], 0)
            self.assertEqual(core.episode_metrics["red"]["useful_reports"], 0)

    def test_known_targets_out_of_range_and_payload_only_are_not_useful_reports(self):
        for position in ([3, 0, 0], [0, 7, 0]):
            core = self.core()
            core.units["red_1"].position[:] = position
            core._refresh_perception()
            core.exchange_messages({"red_0": {"targets": ["blue_0"]}})
            self.assertEqual(wait(core)[0]["red"], 0)
        core = self.core()
        core.exchange_messages({"red_0": {"content": [0.5] * 8}})
        self.assertEqual(wait(core)[0]["red"], 0)

    def test_invalid_message_batch_does_not_create_reward(self):
        core = self.core()
        with self.assertRaises(ValueError):
            core.exchange_messages({"red_0": {"targets": ["blue_0"]},
                                    "red_1": {"targets": ["blue_0"]}})
        self.assertEqual(wait(core)[0]["red"], 0)
        self.assertEqual(core.episode_metrics["red"]["useful_reports"], 0)

    def test_initial_auto_reports_are_not_paid_again(self):
        core = self.core(target_sharing="auto")
        self.assertEqual(core.action_mask("red_1")[7], 1)
        self.assertEqual(wait(core)[0]["red"], 0)
        self.assertEqual(core.episode_metrics["red"]["useful_reports"], 1)

    def test_discovery_and_sharing_use_one_combined_budget(self):
        core = GridCombat(make_config(red=("scout", "strike"), discovery_reward=0.1,
            sharing_reward=0.1, information_reward_cap=0.15, communication="full", target_sharing="auto"))
        core.reset(1, {"red_0": [0, 0, 0], "red_1": [2, 0, 0], "blue_0": [8, 0, 0]})
        rewards, info = core.step({"red_0": 1, "red_1": 0, "blue_0": 0})
        self.assertAlmostEqual(rewards["red"], 0.15)
        self.assertAlmostEqual(info["reward_components"]["red"]["discovery"], 0.1)
        self.assertAlmostEqual(info["reward_components"]["red"]["sharing"], 0.05)

    def test_mixed_team_reports_have_independent_budgets_and_reset(self):
        core = GridCombat(make_config(red=("scout", "strike"), blue=("scout", "strike"),
            communication="full", sharing_reward=0.1, information_reward_cap=0.05))
        positions = {"red_0": [3, 0, 0], "red_1": [0, 0, 0],
                     "blue_0": [3, 0, 0], "blue_1": [6, 0, 0]}
        for _ in range(2):
            core.reset(1, positions)
            core.exchange_messages({"red_0": {"targets": ["blue_1"]},
                                    "blue_0": {"targets": ["red_1"]}})
            rewards, _ = wait(core)
            self.assertAlmostEqual(rewards["red"], 0.05)
            self.assertAlmostEqual(rewards["blue"], 0.05)


class RewardInterfaceTests(unittest.TestCase):
    def test_time_and_timeout_components_match_scalar_reward_and_truncation(self):
        env = SMACAdapter(config=make_config(red=("strike",), max_steps=1,
            time_penalty=0.005, timeout_as_loss=True))
        env.reset(1, {"positions": {"red_0": [0, 0, 0], "blue_0": [15, 0, 0]}})
        reward, done, info = env.step([0])
        self.assertTrue(done)
        self.assertTrue(info["episode_limit"])
        self.assertAlmostEqual(reward, -10.005)
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
        self.assertEqual(info["reward_components"]["time"], -0.005)
        self.assertEqual(info["reward_components"]["terminal"], -10)

    def test_sparse_mode_suppresses_information_but_keeps_explicit_time_cost(self):
        core = GridCombat(make_config(discovery_reward=0.1, time_penalty=0.005, reward_mode="sparse"))
        core.reset(1, {"red_0": [0, 0, 0], "blue_0": [8, 0, 0]})
        rewards, info = core.step({"red_0": 1, "blue_0": 0})
        self.assertAlmostEqual(rewards["red"], -0.005)
        self.assertEqual(info["reward_components"]["red"]["discovery"], 0)

    def test_scripted_broadcast_can_be_disabled_or_made_symmetric(self):
        positions = {"red_0": [0, 0, 0], "blue_0": [3, 0, 0], "blue_1": [6, 0, 0]}
        for sharing, scripted, hp in (("manual", False, 100), ("manual", True, 80), ("auto", False, 80)):
            cfg = replace(make_config(red=("strike",), blue=("scout", "strike"), communication="full"),
                          opponent="static", target_sharing=sharing, scripted_target_sharing=scripted)
            env = parallel_env(config=cfg)
            env.reset(1, {"positions": positions})
            env.step({"red_0": 0})
            self.assertEqual(env.core.units["red_0"].hp, hp)

    def test_shaping_configuration_validation_and_round_trip(self):
        for name in ("time_penalty", "discovery_reward", "sharing_reward", "information_reward_cap"):
            for value in (-0.1, float("nan"), float("inf"), True, "0.1"):
                with self.assertRaises(ValueError):
                    make_config(**{name: value})
        with self.assertRaises(ValueError):
            make_config(scripted_target_sharing="false")
        cfg = make_config(time_penalty=0.005, discovery_reward=0.05, sharing_reward=0.05,
                          scripted_target_sharing=False)
        self.assertEqual(EnvConfig.from_dict(cfg.to_dict()), cfg)


if __name__ == "__main__":
    unittest.main()
