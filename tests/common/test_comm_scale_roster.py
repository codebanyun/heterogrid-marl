import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from algorithms.common import comm_scale_roster as protocol, replication
from algorithms.common.training_seeds import EpisodeSeedSequence, episode_seeds
from heterogrid.core import GridCombat


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.configs = protocol.scale_configs(Path(__file__).resolve().parents[2])
        self.configs.update(protocol.supplied_roster_configs(self.configs["C1S_B0"]))

    def test_fifteen_distinct_configs_and_ninety_independent_runs(self):
        self.assertEqual(set(self.configs), set(protocol.CONDITIONS))
        self.assertEqual(len(protocol.CONDITIONS) * len(protocol.SEEDS) * 2, 90)
        self.assertNotIn("C1S_R_balanced", self.configs)
        for name, config in self.configs.items():
            protocol.validate_config(name, config)

    def test_bilateral_no_communication_overrides_top_level_legacy_settings(self):
        for name, value in self.configs.items():
            if name.startswith("C1S_"):
                config = protocol.validate_config(name, value)
                self.assertEqual(config.communication, "distance")
                self.assertEqual(config.communication_for("red").mode, "none")
                self.assertEqual(config.communication_for("blue").mode, "none")

    def test_dropout_is_full_topology_and_fixed_blue(self):
        for scale in protocol.SCALES:
            env = protocol.validate_config("P50_" + scale, self.configs["P50_" + scale])
            self.assertEqual(env.communication_for("red").mode, "full")
            self.assertEqual(env.communication_for("red").dropout_probability, .5)
            self.assertEqual(env.communication_for("blue").dropout_probability, 0)

    def test_n2_and_d2_reward_difference_is_not_removed(self):
        n2 = GridCombat(protocol.validate_config("R5_N2", self.configs["R5_N2"]))
        d2 = GridCombat(protocol.validate_config("R5_D2", self.configs["R5_D2"]))
        self.assertEqual(n2.reward_scales["red"]["damage"], 1)
        for component in ("damage", "kill"):
            self.assertAlmostEqual(d2.reward_scales["red"][component], 2 / 3)

    def test_supplied_platforms_and_role_order_are_preserved(self):
        expected = {
            "R_scout": ["scout/uav", "scout/usv", "scout/uav", "strike/robot", "hybrid/uav", "hybrid/robot_dog"],
            "R_strike": ["scout/uav", "scout/usv", "strike/robot", "strike/robot_dog", "strike/robot", "hybrid/uav"],
            "R_hybrid": ["scout/uav", "strike/robot", "strike/robot_dog", "hybrid/uav", "hybrid/robot_dog", "hybrid/uav"],
        }
        for name, units in expected.items():
            value = self.configs["C1S_" + name]
            for team in ("red", "blue"):
                self.assertEqual([u["role"] + "/" + u["platform"] for u in value[team]], units)

    def test_rejects_protocol_drift(self):
        cases = [("P50_B0", "red_communication", dict(mode="distance", radius=10, dropout_probability=.5)),
                 ("R5_B0", "blue_communication", dict(mode="none", radius=10, dropout_probability=0)),
                 ("R5_D2", "reward_normalization", "none"), ("C1S_R_scout", "terminate_on_no_firepower", True),
                 ("C1S_R_scout", "draw_reward", 1), ("R5_B0", "target_ttl", 3), ("R5_N2", "max_steps", 300)]
        for name, field, replacement in cases:
            value = copy.deepcopy(self.configs[name]); value[field] = replacement
            with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                protocol.validate_config(name, value)

    def test_seeds_are_64bit_contiguous_and_legacy_rng_stays_unchanged(self):
        sequence = EpisodeSeedSequence(44)
        self.assertEqual(episode_seeds(sequence, 2), [4_401_000_000, 4_401_000_001])
        self.assertEqual(episode_seeds(sequence, 1), [4_401_000_002])
        rng = np.random.default_rng(42); reference = np.random.default_rng(42)
        self.assertEqual(episode_seeds(rng, 3), [int(reference.integers(0, 2**31 - 1)) for _ in range(3)])

    def test_auc_and_threshold_use_scheduled_and_actual_steps_separately(self):
        metrics = {"test_battle_won_mean": {"steps": [0, 250000, 500000], "values": [0, .8, 1]},
                   "test_actual_environment_steps": {"values": [0, 251000, 500123]}}
        result = protocol.learning_statistics(metrics, 500000)
        self.assertAlmostEqual(result["validation_win_auc_normalized"], .65)
        self.assertEqual(result["first_80pct_actual_steps"], 251000)
        metrics["test_battle_won_mean"]["values"] = [0, .2, .3]
        result = protocol.learning_statistics(metrics, 500000)
        self.assertFalse(result["reached_80pct"])
        self.assertIsNone(result["first_80pct_actual_steps"])

    def test_training_command_uses_new_protocol_and_never_loads_a_checkpoint(self):
        manifest = dict(python="python", protocol=protocol.PROTOCOL,
                        algorithms={"smpe": dict(output="/smpe", backend_root="/backend")})
        job = dict(algorithm="smpe", condition="P50_D2", seed=44, device="cuda", path="/smpe/P50/D2/seed_44")
        command = protocol.training_command(manifest, job)
        for key, value in (("--steps", "5000000"), ("--eval-seed", "1910042"),
                           ("--heldout-seed", "2800042"), ("--environment-seed-mode", "episode-index")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("--checkpoint", command)


class ResourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.control = {"max_workers": 3, "cpu_workers": 2, "gpu_workers": 1,
            "gpu_algorithms": ["smpe"], "minimum_available_ram_gib": 6, "minimum_free_gpu_mib": 4000,
            "peak_ram_gib": {"smpe": {"cpu": {"6": 3, "9": 4}, "cuda": {"6": 2, "9": 3}}},
            "gpu_peak_mib": {"smpe": 4500}}
        self.job = dict(id="smpe/P50_N2/seed_42", algorithm="smpe", condition="P50_N2", seed=42)

    def test_gpu_preferred_when_memory_is_available(self):
        result = replication.select_job([self.job], [], self.control, 12 * replication.GIB, 16000, {})
        self.assertEqual(result["device"], "cuda")

    def test_cpu_used_if_gpu_has_no_headroom(self):
        result = replication.select_job([self.job], [], self.control, 12 * replication.GIB, 7000, {})
        self.assertEqual(result["device"], "cpu")

    def test_host_memory_floor_never_bypassed(self):
        self.assertIsNone(replication.select_job([self.job], [], self.control, 7 * replication.GIB, 16000, {}))

    def test_reserves_growth_of_already_running_jobs(self):
        active = dict(self.job, device="cpu", pid=101)
        with patch.object(replication, "process_rss", return_value=replication.GIB):
            self.assertIsNone(replication.select_job([self.job], [active], self.control, 10 * replication.GIB, 16000, {}))

    def test_global_worker_limit_never_bypassed(self):
        active = [dict(self.job, device="cpu", pid=i) for i in range(3)]
        self.assertIsNone(replication.select_job([self.job], active, self.control, 40 * replication.GIB, 32000, {}))


class QueueTests(unittest.TestCase):
    def test_shared_queue_publishes_per_algorithm_totals_and_finalizes_each_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            algorithms = {name: dict(output=str(root / name)) for name in ("smpe", "epymarl_mappo")}
            for record in algorithms.values():
                Path(record["output"]).mkdir()
            (root / "experiment.json").write_text(json.dumps(dict(algorithms=algorithms, repository=str(root))))
            (root / "resource_control.json").write_text(json.dumps(dict(max_workers=2, cpu_workers=2,
                gpu_workers=0, minimum_available_ram_gib=6, minimum_free_disk_gib=0)))
            def select(queued, active, *args):
                return dict(queued[0], device="cpu") if queued else None
            with patch.object(replication, "CONDITIONS", ("R5_B0",)), patch.object(replication, "SEEDS", (42,)), \
                    patch.object(replication, "select_job", side_effect=select), \
                    patch.object(replication, "training_command", return_value=["python"]), \
                    patch.object(replication.subprocess, "Popen", side_effect=[Mock(pid=101, poll=Mock(return_value=0)),
                                                                             Mock(pid=102, poll=Mock(return_value=0))]), \
                    patch.object(replication, "latest_progress", return_value={"environment_steps": 16}), \
                    patch.object(replication, "finalize_run", return_value={}) as finalize, \
                    patch.object(replication, "results"), patch.object(replication.time, "sleep"):
                replication.manage(root)
            self.assertEqual(finalize.call_count, 2)
            self.assertEqual(json.loads((root / "status.json").read_text())["total_environment_steps"], 32)
            for record in algorithms.values():
                status = json.loads((Path(record["output"]) / "status.json").read_text())
                self.assertEqual(status["state"], "COMPLETED")
                self.assertEqual(status["counts"]["COMPLETED"], 1)
                self.assertEqual(status["total_environment_steps"], 16)


if __name__ == "__main__":
    unittest.main()
