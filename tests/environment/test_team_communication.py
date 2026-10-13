"""Independent red/blue communication without changing legacy experiments."""
import copy
import json
import unittest
from dataclasses import replace

import numpy as np

from heterogrid import CommunicationConfig, EnvConfig, SMACAdapter, UnitSpec, parallel_env
from heterogrid.core import ATTACK_OFFSET, GridCombat


class TeamCommunicationTests(unittest.TestCase):
    def config(self, **kwargs):
        return EnvConfig(map_size=(16, 8, 3),
                         red=[UnitSpec("scout"), UnitSpec("strike")],
                         blue=[UnitSpec("scout"), UnitSpec("strike")], **kwargs)

    def core(self, **kwargs):
        core = GridCombat(self.config(**kwargs))
        core.reset(42, {"red_0": [3, 2, 1], "red_1": [0, 2, 1],
                        "blue_0": [3, 2, 1], "blue_1": [6, 2, 1]})
        return core

    def test_serialization_and_override_precedence(self):
        cfg = self.config(communication="none", target_sharing="auto",
                          red_communication=CommunicationConfig("distance", 5, .25),
                          blue_communication=CommunicationConfig("full"))
        self.assertEqual(EnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))), cfg)
        self.assertEqual(cfg.communication_for("red"), CommunicationConfig("distance", 5, .25))
        self.assertEqual(cfg.communication_for("blue"), CommunicationConfig("full"))
        inherited = replace(cfg, red_communication=None)
        self.assertEqual(inherited.communication_for("red").mode, "none")
        self.assertEqual(inherited.communication_for("blue").mode, "full")
        with self.assertRaises(ValueError):
            cfg.communication_for("green")

    def test_invalid_team_settings_rejected(self):
        for values in ({"mode": "random"}, {"mode": "invalid"}, {"radius": -1},
                       {"radius": float("nan")}, {"radius": True}, {"radius": "10"},
                       {"dropout_probability": 1.1}, {"dropout_probability": -.1},
                       {"dropout_probability": float("inf")}, {"dropout_probability": False}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CommunicationConfig(**values)
        with self.assertRaises(ValueError):
            EnvConfig(blue_communication={"mode": "full"})
        with self.assertRaises(TypeError):
            EnvConfig.from_dict({"blue_communication": {"unknown": 1}})

    def test_legacy_modes_resolve_without_changing_their_meaning(self):
        for mode, expected, probability in (("none", "none", 0), ("full", "full", 0),
                                            ("distance", "distance", 0), ("random", "full", .4)):
            cfg = self.config(communication=mode, comm_radius=7, dropout_probability=.4)
            self.assertEqual(cfg.communication_for("red"), CommunicationConfig(expected, 7, probability))
            self.assertEqual(cfg.communication_for("red"), cfg.communication_for("blue"))

    def test_legacy_network_rng_consumption_is_exact(self):
        for mode in ("none", "full", "distance", "random"):
            core = self.core(communication=mode, dropout_probability=.4)
            core.units["red_0"].hp = 0
            reference = np.random.default_rng()
            reference.bit_generator.state = copy.deepcopy(core.rng.bit_generator.state)
            probability = .4 if mode == "random" else 0
            expected = {n: core.units[n].alive and mode != "none" and reference.random() >= probability
                        for n in core.all_names}
            core._sample_network()
            self.assertEqual(core.online, expected)
            self.assertEqual(core.rng.random(), reference.random())

    def test_auto_sharing_red_none_does_not_disable_blue(self):
        core = self.core(target_sharing="auto", scripted_target_sharing=False,
                         red_communication=CommunicationConfig("none"),
                         blue_communication=CommunicationConfig("full"))
        self.assertNotIn("blue_1", core.tracks["red_1"])
        self.assertIn("red_1", core.tracks["blue_1"])
        self.assertEqual(core.action_mask("red_1")[ATTACK_OFFSET + 1], 0)
        self.assertEqual(core.action_mask("blue_1")[ATTACK_OFFSET + 1], 1)
        self.assertEqual(core.episode_metrics["red"]["useful_reports"], 0)
        self.assertGreater(core.episode_metrics["blue"]["useful_reports"], 0)
        self.assertFalse(core.can_communicate("red_0", "blue_0"))
        self.assertFalse(core.can_communicate("blue_0", "blue_0"))

    def test_manual_messages_obey_each_teams_links(self):
        for disabled in ("red", "blue"):
            kwargs = {disabled + "_communication": CommunicationConfig("none"),
                      ("blue" if disabled == "red" else "red") + "_communication": CommunicationConfig("full")}
            core = self.core(**kwargs)
            core.exchange_messages({"red_0": {"targets": ["blue_1"], "content": [1.] * 8},
                                    "blue_0": {"targets": ["red_1"], "content": [1.] * 8}})
            for team, target in (("red", "blue_1"), ("blue", "red_1")):
                self.assertEqual(target in core.tracks[team + "_1"], team != disabled)
                self.assertEqual(bool(core.observation(team + "_1")["message_mask"].any()), team != disabled)

    def test_distance_topology_and_outages_compose(self):
        for radius, dropout, enabled in ((2, 0, False), (3, 0, True), (3, 1, False)):
            core = self.core(red_communication=CommunicationConfig("distance", radius, dropout),
                             blue_communication=CommunicationConfig("full"))
            self.assertEqual(core.can_communicate("red_0", "red_1"), enabled)
            self.assertTrue(core.can_communicate("blue_0", "blue_1"))
        core = self.core(red_communication=CommunicationConfig("full", 0, 0))
        self.assertTrue(core.can_communicate("red_0", "red_1"))

    def test_blue_random_stream_independent_of_red_mode_and_deaths(self):
        blue = CommunicationConfig("distance", 10, .4)
        configs = [self.config(red_communication=red, blue_communication=blue) for red in
                   (CommunicationConfig("none"), CommunicationConfig("full"),
                    CommunicationConfig("distance", 1, .75))]
        cores = [GridCombat(cfg) for cfg in configs]
        for core in cores:
            core.reset(200042)
        for tick in range(40):
            reference = [cores[0].online[n] for n in cores[0].names["blue"]]
            for core in cores[1:]:
                self.assertEqual([core.online[n] for n in core.names["blue"]], reference)
            if tick == 5:
                cores[1].units["red_0"].hp = 0
            for core in cores:
                core._sample_network()

    def test_seeded_outages_stable_between_observations_and_reset(self):
        core = self.core(red_communication=CommunicationConfig("distance", 10, .5),
                         blue_communication=CommunicationConfig("full", 10, .5))
        def sequence():
            core.reset(123)
            result = []
            for _ in range(10):
                before = dict(core.online)
                core.comm_mask("red"); core.observation("red_0"); core.exchange_messages({})
                self.assertEqual(core.online, before)
                result.append(before); core._sample_network()
            return result
        self.assertEqual(sequence(), sequence())

    def test_fixed_blue_rule_behaviour_with_same_red_actions(self):
        blue = CommunicationConfig("distance", 10, .4)
        adapters = [SMACAdapter(config=self.config(
            communication=mode, target_sharing="auto", blue_communication=blue))
            for mode in ("none", "full", "random")]
        try:
            for adapter in adapters:
                adapter.reset(seed=42)
            for _ in range(30):
                transitions = [adapter.step([0, 0]) for adapter in adapters]
                reference = adapters[0].env.core
                for adapter in adapters[1:]:
                    core = adapter.env.core
                    self.assertEqual([core.online[n] for n in core.names["blue"]],
                                     [reference.online[n] for n in reference.names["blue"]])
                    self.assertEqual(core.last_attacks, reference.last_attacks)
                    for name in core.all_names:
                        np.testing.assert_array_equal(core.units[name].position, reference.units[name].position)
                        self.assertEqual(core.units[name].hp, reference.units[name].hp)
                self.assertEqual([done for _, done, _ in transitions], [transitions[0][1]] * 3)
                if transitions[0][1]:
                    break
        finally:
            for adapter in adapters:
                adapter.close()

    def test_interfaces_keep_shapes_and_snapshot_has_effective_settings(self):
        cfg = self.config(target_sharing="auto", red_communication=CommunicationConfig("none"),
                          blue_communication=CommunicationConfig("distance", 10))
        adapter = SMACAdapter(config=cfg)
        try:
            obs, state = adapter.reset(seed=42)
            self.assertEqual(len(obs), 2)
            self.assertEqual(len(obs[0]), adapter.get_obs_size())
            self.assertEqual(len(state), adapter.get_state_size())
            reward, _, info = adapter.step([0, 0])
            self.assertIsInstance(reward, float)
            self.assertFalse(info["invalid_actions"])
            snapshot = adapter.env.snapshot()
            self.assertEqual(snapshot["team_communication"]["red"]["mode"], "none")
            self.assertEqual(snapshot["team_communication"]["blue"]["mode"], "distance")
        finally:
            adapter.close()
        env = parallel_env(config=cfg)
        try:
            observations, _ = env.reset(seed=42)
            for name, obs in observations.items():
                self.assertTrue(env.observation_space(name).contains(obs))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
