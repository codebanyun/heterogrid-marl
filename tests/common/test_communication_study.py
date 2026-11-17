import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.common import communication_study as study
from algorithms.common.study_queue import jobs_for
from heterogrid import CommunicationConfig, EnvConfig
from heterogrid.core import GridCombat


class FixedBlueStudyTests(unittest.TestCase):
    def setUp(self):
        self.base = EnvConfig.load(study.mappo.DEFAULT_CONFIG)
        self.configs = study.conditions(self.base)

    def test_seven_conditions_only_change_red_communication(self):
        self.assertEqual(tuple(self.configs), study.CONDITION_NAMES)
        base = self.configs["B0_baseline"]
        for name, config in self.configs.items():
            self.assertEqual(config.blue_communication, CommunicationConfig("distance", 10, 0))
            self.assertIsNotNone(config.red_communication)
            changed = {key for key in config.to_dict() if config.to_dict()[key] != base.to_dict()[key]}
            self.assertLessEqual(changed, {"red_communication"})
            for key in self.base.to_dict():
                if key not in ("communication", "comm_radius", "dropout_probability", "red_communication", "blue_communication"):
                    self.assertEqual(config.to_dict()[key], self.base.to_dict()[key])
            self.assertEqual(EnvConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)

    def test_outages_preserve_full_online_topology(self):
        for name, probability in (("C5_random25", .25), ("C6_random50", .5)):
            config = self.configs[name]
            self.assertEqual(config.red_communication, CommunicationConfig("full", 10, probability))
            self.assertEqual(config.blue_communication.dropout_probability, 0)

    def test_all_conditions_use_independent_rng_and_same_blue_under_fixed_actions(self):
        from heterogrid import SMACAdapter
        envs = [SMACAdapter(config=config) for config in self.configs.values()]
        try:
            for env in envs:
                env.reset(seed=200042)
                self.assertTrue(env.env.core._independent_communication)
            for _ in range(20):
                transitions = [env.step([0] * 6) for env in envs]
                reference = envs[0].env.core
                for env in envs:
                    core = env.env.core
                    self.assertEqual(core.last_attacks, reference.last_attacks)
                    for name in core.all_names:
                        self.assertEqual(core.units[name].position.tolist(), reference.units[name].position.tolist())
                        self.assertEqual(core.units[name].hp, reference.units[name].hp)
                    self.assertTrue(all(core.online[n] for n in core.names["blue"] if core.units[n].alive))
                if transitions[0][1]:
                    break
        finally:
            for env in envs:
                env.close()

    def test_new_baseline_does_not_use_legacy_network_rng(self):
        new = GridCombat(self.configs["B0_baseline"])
        old = GridCombat(self.base)
        new.reset(42)
        old.reset(42)
        self.assertTrue(new._independent_communication)
        self.assertFalse(old._independent_communication)
        self.assertNotEqual(new.rng.bit_generator.state, old.rng.bit_generator.state)

    def test_training_commands_and_shared_job_count(self):
        local = dict(python="python", mappo_root="/code/mappo", smpe_root="/code/smpe", device="cpu",
                     actual_steps_per_seed=5000192, target_steps_per_seed=5000192,
                     validation_seed=123, heldout_seed=100123, eval_episodes=64, heldout_episodes=128,
                     replay_episodes=2, eval_interval=250368, save_interval=2000384, log_interval=10240)
        manifest = dict(experiments={a: dict(local) for a in study.ALGORITHMS},
                        conditions=[a + "/" + c for c in study.CONDITION_NAMES for a in study.ALGORITHMS],
                        seeds=[42, 43, 44])
        output = Path("/output/coordinator")
        jobs = jobs_for(manifest, output)
        self.assertEqual(len(jobs), 42)
        self.assertEqual(len({job["path"] for job in jobs}), 42)
        for job in jobs[:2]:
            command = study.training_command(manifest, output, job)
            algorithm = job["condition"].split("/")[0]
            self.assertEqual(command[command.index("-m") + 1], "algorithms." + algorithm + ".train")
            self.assertEqual(command[command.index("--eval-seed") + 1], "123")
            self.assertEqual(command[command.index("--config") + 1], str(output / algorithm / "configs/B0_baseline.json"))

    def test_launcher_freezes_both_backends_and_uses_one_nohup_manager(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            (reference / "configs").mkdir(parents=True)
            (reference / "configs/B0_baseline.json").write_text(json.dumps(self.base.to_dict()))
            (reference / "experiment.json").write_text(json.dumps(dict(algorithm="mappo", validation_seed=100042, heldout_seed=200042)))
            (reference / "status.json").write_text('{"state":"COMPLETED"}')
            mappo, smpe = root / "raw_mappo", root / "raw_smpe"
            (mappo / "onpolicy").mkdir(parents=True)
            (mappo / "onpolicy/config.py").touch()
            for name in ("controllers", "learners", "model_learners", "modules", "components", "utils", "config/algs"):
                (smpe / name).mkdir(parents=True)
            for name in ("learners/actor_critic_learner.py", "model_learners/vae_controller.py", "config/default.yaml", "config/algs/smpe_lbf.yaml"):
                (smpe / name).touch()
            arguments = ["communication", "--reference", str(reference), "--mappo-root", str(mappo),
                         "--smpe-root", str(smpe), "--output-root", str(root / "output"), "--name", "check"]
            with patch("sys.argv", arguments), patch.object(study.subprocess, "Popen") as popen, \
                    patch.object(study.smpe, "hyperparameters", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                popen.return_value.pid = 123
                study.main()
            coordinator = root / "output/communication/check"
            manifest = json.loads((coordinator / "experiment.json").read_text())
            self.assertEqual(manifest["jobs"], 42)
            self.assertEqual(manifest["workers"], 12)
            self.assertEqual(popen.call_args[0][0][0], "nohup")
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(popen.call_args[1]["start_new_session"])
            self.assertTrue((Path(manifest["repository"]) / "heterogrid/core.py").exists())
            for algorithm in study.ALGORITHMS:
                path = root / "output" / algorithm / "check"
                self.assertEqual((coordinator / algorithm).resolve(), path)
                self.assertEqual(len(list((path / "configs").glob("*.json"))), 7)
                self.assertEqual(json.loads((path / "experiment.json").read_text())["jobs"], 21)
                self.assertEqual((path / "status.json").resolve(), coordinator / "status.json")
            self.assertEqual((coordinator / "mappo/configs/B0_baseline.json").read_text(),
                             (coordinator / "smpe/configs/B0_baseline.json").read_text())

    def test_comparison_reads_only_new_final_models_and_splits_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment.json").write_text(json.dumps({"seeds": [42, 43, 44]}))
            summaries = {}
            for algorithm in study.ALGORITHMS:
                for label in study.CONDITION_NAMES:
                    summaries[algorithm + "/" + label] = {"checked": True}
                    for seed in (42, 43, 44):
                        path = root / algorithm / label / ("seed_" + str(seed))
                        path.mkdir(parents=True)
                        (path / "summary.json").write_text(json.dumps({"heldout": {"latest": {"greedy": {"win_rate": .5}}}}))
            (root / "summary.json").write_text(json.dumps(summaries))
            study.write_comparison(root)
            self.assertEqual((root / "RESULTS.md").read_text().count("50.00 ±0.00"), 14)
            for algorithm in study.ALGORITHMS:
                self.assertEqual(len(json.loads((root / algorithm / "summary.json").read_text())), 7)


if __name__ == "__main__":
    unittest.main()
