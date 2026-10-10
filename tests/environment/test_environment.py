import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, RoleStats, SMACAdapter, UnitSpec, env, parallel_env
from heterogrid.core import GridCombat, ATTACK_OFFSET
from heterogrid.io import SnapshotWriter
from heterogrid.policies import ScriptedPolicy


def configuration(red=("strike",), blue=("strike",), **kwargs):
    return EnvConfig(map_size=(12, 8, 4), red=[UnitSpec(r) for r in red], blue=[UnitSpec(r) for r in blue],
                     opponent="dummy", **kwargs)


class CombatTests(unittest.TestCase):
    def make_core(self, config=None, positions=None):
        c = GridCombat(config or configuration())
        c.reset(seed=7, positions=positions or {"red_0": [2, 2, 1], "blue_0": [4, 2, 1]})
        return c

    def test_simultaneous_elimination_and_overkill(self):
        c = self.make_core()
        c.units["red_0"].hp = c.units["blue_0"].hp = 5
        c._refresh_perception()
        rewards, info = c.step({"red_0": 7, "blue_0": 7})
        self.assertIsNone(c.winner)
        self.assertEqual(c.reason, "mutual_elimination")
        self.assertEqual(info["damage"], {"red": 5, "blue": 5})
        self.assertAlmostEqual(rewards["red"], 1.05)

    def test_elimination_reward_once(self):
        c = self.make_core()
        c.units["blue_0"].hp = 1
        c._refresh_perception()
        rewards, _ = c.step({"red_0": 7, "blue_0": 0})
        self.assertEqual(c.winner, "red")
        self.assertAlmostEqual(rewards["red"], 11.01)
        self.assertEqual(rewards["blue"], -10)
        with self.assertRaises(RuntimeError):
            c.step({"red_0": 7})

    def test_attack_and_move_use_same_tick_snapshot(self):
        c = self.make_core()
        c.step({"red_0": 7, "blue_0": 1})
        self.assertEqual(c.units["blue_0"].hp, 80)
        self.assertEqual(c.units["blue_0"].position.tolist(), [5, 2, 1])

    def test_boundary_and_masked_action(self):
        c = self.make_core(positions={"red_0": [0, 0, 0], "blue_0": [11, 7, 3]})
        self.assertEqual(c.action_mask("red_0")[2], 0)
        c.step({"red_0": 2, "blue_0": 0})
        self.assertEqual(c.units["red_0"].position.tolist(), [0, 0, 0])
        self.assertIn("red_0", c.last_metrics["invalid_actions"])

    def test_scout_cannot_attack(self):
        c = self.make_core(configuration(red=("scout",)))
        self.assertEqual(c.action_mask("red_0")[7], 0)

    def test_timeout_and_sparse_reward(self):
        c = self.make_core(configuration(max_steps=1, reward_mode="sparse"))
        rewards, _ = c.step({"red_0": 7, "blue_0": 0})
        self.assertEqual(c.reason, "timeout")
        self.assertEqual(rewards, {"red": 0, "blue": 0})

    def test_out_of_space_actions_rejected_without_mutation(self):
        for action in (-1, 99, 1.5, True):
            c = self.make_core()
            with self.assertRaises(ValueError):
                c.step({"red_0": action, "blue_0": 0})
            self.assertEqual(c.step_count, 0)
            self.assertEqual(c.units["blue_0"].hp, 100)

    def test_singleton_dimensions(self):
        config = replace(configuration(), map_size=(1, 1, 1))
        c = GridCombat(config)
        c.reset(0)
        self.assertTrue(np.all(np.isfinite(c.state())))
        np.testing.assert_array_equal(c.action_mask("red_0")[1:7], 0)

    def test_invalid_initial_positions_rejected_before_integer_cast(self):
        c = GridCombat(configuration())
        for p in ([2**32, 0, 0], [-1, 0, 0], [0.5, 0, 0], [float("nan"), 0, 0]):
            with self.assertRaises(ValueError):
                c.reset(0, {"red_0": p})


class CommunicationTests(unittest.TestCase):
    def make_core(self, **kwargs):
        c = GridCombat(configuration(red=("scout", "strike"), communication="full", **kwargs))
        c.reset(3, {"red_0": [3, 2, 1], "red_1": [0, 2, 1], "blue_0": [6, 2, 1]})
        return c

    def test_report_enables_attack_beyond_own_view(self):
        c = self.make_core()
        self.assertEqual(c.action_mask("red_1")[7], 0)
        c.exchange_messages({"red_0": {"targets": ["blue_0"]}})
        self.assertEqual(c.action_mask("red_1")[7], 1)
        c.step({"red_0": 0, "red_1": 7, "blue_0": 0})
        self.assertEqual(c.units["blue_0"].hp, 80)

    def test_stale_track_does_not_follow_hidden_target(self):
        c = self.make_core()
        c.exchange_messages({"red_0": {"targets": ["blue_0"]}})
        c.step({"red_0": 0, "red_1": 0, "blue_0": 1})
        self.assertEqual(c.action_mask("red_1")[7], 1)
        c.step({"red_0": 0, "red_1": 7, "blue_0": 0})
        self.assertEqual(c.units["blue_0"].hp, 100)
        self.assertFalse(c.last_attacks[0]["hit"])

    def test_remote_death_does_not_leak_in_action_mask(self):
        c = self.make_core()
        c.exchange_messages({"red_0": {"targets": ["blue_0"]}})
        c.units["blue_0"].hp = 0
        c._refresh_perception()
        self.assertEqual(c.action_mask("red_1")[7], 1)
        self.assertEqual(c.tracks["red_1"]["blue_0"].hp, 100)

    def test_track_expires(self):
        c = self.make_core(target_ttl=1)
        c.exchange_messages({"red_0": {"targets": ["blue_0"]}})
        for _ in range(2):
            c.step({"red_0": 0, "red_1": 0, "blue_0": 0})
        self.assertEqual(c.action_mask("red_1")[7], 0)

    def test_distance_transport_and_none(self):
        for mode, radius, expected in [("distance", 2, False), ("distance", 3, True), ("none", 10, False)]:
            cfg = replace(configuration(red=("scout", "strike")), communication=mode, comm_radius=radius)
            c = GridCombat(cfg)
            c.reset(3, {"red_0": [3, 2, 1], "red_1": [0, 2, 1], "blue_0": [6, 2, 1]})
            c.exchange_messages({"red_0": {"targets": ["blue_0"]}})
            self.assertEqual(bool(c.action_mask("red_1")[7]), expected)

    def test_random_outages_are_seeded_and_tick_stable(self):
        cfg = replace(configuration(red=("scout", "strike")), communication="random")
        a, b = GridCombat(cfg), GridCombat(cfg)
        a.reset(55)
        b.reset(55)
        self.assertEqual(a.online, b.online)
        before = a.comm_mask("red").copy()
        a.exchange_messages({})
        np.testing.assert_array_equal(a.comm_mask("red"), before)
        a.step({n: 0 for n in a.all_names})
        b.step({n: 0 for n in b.all_names})
        self.assertEqual(a.online, b.online)

    def test_dropout_one_blocks_all_nodes(self):
        cfg = replace(configuration(red=("scout", "strike")), communication="random", dropout_probability=1)
        c = GridCombat(cfg)
        c.reset(2)
        self.assertFalse(np.any(c.comm_mask("red")))

    def test_learned_payload_recipient_selection_and_age(self):
        c = self.make_core()
        content = [0.5] * c.config.message_dim
        c.exchange_messages({"red_0": {"content": content, "recipients": ["red_1"]}})
        obs = c.observation("red_1")
        np.testing.assert_allclose(obs["messages"][0], content)
        self.assertEqual(obs["message_mask"][0], 1)
        c.online["red_0"] = False
        c.exchange_messages({"red_0": {"content": [1] * c.config.message_dim}})
        np.testing.assert_allclose(c.observation("red_1")["messages"][0], content)

    def test_unseen_reports_rejected_atomically(self):
        c = self.make_core()
        with self.assertRaises(ValueError):
            c.exchange_messages({"red_0": {"content": [1] * c.config.message_dim},
                                 "red_1": {"targets": ["blue_0"]}})
        self.assertEqual(c.inboxes["red_1"], {})

    def test_bad_content_and_cross_team_recipients(self):
        c = self.make_core()
        for payload in ({"content": [float("nan")] * 8}, {"content": [2] * 8},
                        {"recipients": ["blue_0"]}, {"targets": ["missing"]}):
            with self.assertRaises(ValueError):
                c.exchange_messages({"red_0": payload})


class InterfaceTests(unittest.TestCase):
    def test_spaces_shapes_and_deterministic_seed(self):
        a, b = parallel_env(), parallel_env()
        obs_a, _ = a.reset(123)
        obs_b, _ = b.reset(123)
        self.assertTrue(a.state_space.contains(a.state()))
        for n in a.agents:
            self.assertTrue(a.observation_space(n).contains(obs_a[n]))
            np.testing.assert_array_equal(obs_a[n]["observation"], obs_b[n]["observation"])
            self.assertIs(a.action_space(n), a.action_space(n))
        np.testing.assert_array_equal(a.state(), b.state())

    def test_hidden_enemies_are_zero(self):
        e = parallel_env(config=configuration())
        obs, _ = e.reset(1, {"positions": {"red_0": [0, 0, 0], "blue_0": [11, 7, 3]}})
        np.testing.assert_array_equal(obs["red_0"]["observation"][12:], 0)

    def test_time_limit_is_truncation(self):
        e = parallel_env(config=configuration(max_steps=1))
        e.reset(1)
        _, _, term, trunc, info = e.step({"red_0": 0})
        self.assertFalse(term["red_0"])
        self.assertTrue(trunc["red_0"])
        self.assertEqual(info["red_0"]["termination_reason"], "timeout")
        self.assertEqual(e.agents, [])
        self.assertEqual(e.step({}), ({}, {}, {}, {}, {}))

    def test_deaths_remove_pettingzoo_agents(self):
        e = parallel_env(config=replace(configuration(red=("strike", "strike")), opponent="static"))
        e.reset(1, {"positions": {"red_0": [2, 2, 1], "red_1": [11, 7, 3], "blue_0": [3, 2, 1]}})
        e.core.units["red_0"].hp = 1
        e.core._refresh_perception()
        _, reward, term, trunc, _ = e.step({"red_0": 0, "red_1": 0})
        self.assertTrue(term["red_0"])
        self.assertFalse(term["red_1"])
        self.assertEqual(e.agents, ["red_1"])
        self.assertEqual(reward["red_0"], reward["red_1"])

    def test_smac_fixed_slots_and_scalar_reward(self):
        e = SMACAdapter(configuration(red=("strike", "strike")))
        obs, state = e.reset(1, {"positions": {"red_0": [2, 2, 1], "red_1": [2, 3, 1], "blue_0": [3, 2, 1]}})
        self.assertEqual(np.asarray(obs).shape, (2, e.get_obs_size()))
        self.assertEqual(state.shape, (e.get_state_size(),))
        reward, done, _ = e.step([7, 7])
        self.assertAlmostEqual(reward, 0.4)  # NOT per-agent summed 0.8
        self.assertFalse(done)
        e.env.core.units["red_0"].hp = 0
        e.env.core._refresh_perception()
        self.assertEqual(len(e.get_obs()), 2)
        self.assertEqual(e.get_avail_agent_actions(0), [1] + [0] * 7)
        np.testing.assert_array_equal(e.get_obs_agent(0), 0)

    def test_smac_timeout_info_and_seed(self):
        e = SMACAdapter(configuration(max_steps=1))
        e.seed(3)
        e.reset()
        _, done, info = e.step([0])
        self.assertTrue(done)
        self.assertTrue(info["episode_limit"])
        self.assertEqual(e.get_stats()["timeouts"], 1)

    def test_pretrained_callback_receives_local_obs_only(self):
        class Policy:
            def reset(self, seed=None):
                self.seed = seed
            def act(self, observations, infos):
                self.observations = observations
                return {n: 0 for n in observations}
        p = Policy()
        e = parallel_env(config=replace(configuration(), opponent="pretrained"), opponent_policy=p)
        e.reset(7)
        e.step({"red_0": 0})
        self.assertEqual(p.seed, 7)
        self.assertEqual(set(p.observations["blue_0"]), {"observation", "action_mask", "messages", "message_mask", "message_age"})

    def test_replay_and_atomic_snapshot(self):
        e = parallel_env(config=configuration(max_steps=2), record_replay=True)
        e.reset(3)
        e.step({"red_0": 0})
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(e.save_replay(Path(directory) / "episode.json"))
            self.assertEqual(len(json.loads(replay.read_text())["frames"]), 2)
            publisher = SnapshotWriter(Path(directory) / "latest.json", every=2)
            self.assertTrue(publisher.write(e))
            self.assertFalse(publisher.write(e))
            self.assertEqual(json.loads(publisher.path.read_text())["step"], 1)

    def test_configuration_json_roundtrip(self):
        a = EnvConfig()
        b = EnvConfig.from_dict(json.loads(json.dumps(a.to_dict())))
        self.assertEqual(a, b)
        for data in ({"map_size": [0, 2, 3]}, {"red": []}, {"communication": "wrong"},
                     {"message_dim": 0}, {"dropout_probability": 2}, {"unknown": 1}):
            with self.assertRaises((ValueError, TypeError)):
                EnvConfig.from_dict(data)

    def test_random_rollout_invariants(self):
        for mode in ("none", "full", "distance", "random"):
            e = parallel_env(config=EnvConfig(map_size=(10, 10, 4), max_steps=35, communication=mode))
            rng = np.random.default_rng(12)
            for seed in range(3):
                obs, _ = e.reset(seed)
                while e.agents:
                    actions = {n: int(rng.choice(np.flatnonzero(obs[n]["action_mask"]))) for n in e.agents}
                    obs, reward, term, trunc, info = e.step(actions)
                    for n, value in obs.items():
                        self.assertTrue(e.observation_space(n).contains(value), (mode, n, value))
                    self.assertTrue(e.state_space.contains(e.state()))
                    for u in e.core.units.values():
                        self.assertGreaterEqual(u.hp, 0)
                        self.assertLessEqual(u.hp, u.stats.hp)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional renderer")
    def test_offscreen_render(self):
        e = parallel_env(render_mode="rgb_array")
        e.reset(1)
        image = e.render()
        self.assertEqual(image.shape, (500, 700, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_pettingzoo_parallel_standard(self):
        from pettingzoo.test import parallel_api_test
        parallel_api_test(parallel_env(config=EnvConfig(max_steps=20)), num_cycles=100)

    def test_pettingzoo_seed_standard(self):
        from pettingzoo.test import parallel_seed_test
        parallel_seed_test(lambda: parallel_env(config=EnvConfig(max_steps=15)), num_cycles=30)

    def test_pettingzoo_aec_standard(self):
        from pettingzoo.test import api_test
        api_test(env(config=EnvConfig(max_steps=15)), num_cycles=30)

    def test_aec_early_death_keeps_other_agent_turns(self):
        cfg = replace(configuration(red=("strike", "strike"), max_steps=3), opponent="static")
        e = env(config=cfg)
        e.reset(1, {"positions": {"red_0": [2, 2, 1], "red_1": [11, 7, 3], "blue_0": [3, 2, 1]}})
        e.unwrapped.core.units["red_0"].hp = 1
        e.unwrapped.core._refresh_perception()
        living_turns = {"red_0": 0, "red_1": 0}
        for name in e.agent_iter():
            obs, reward, terminated, truncated, info = e.last()
            if terminated or truncated:
                e.step(None)
            else:
                living_turns[name] += 1
                e.step(0)
        self.assertEqual(living_turns, {"red_0": 1, "red_1": 3})


if __name__ == "__main__":
    unittest.main()
