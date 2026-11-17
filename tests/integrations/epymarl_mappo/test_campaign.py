import json
import tempfile
import unittest
from pathlib import Path

from algorithms.common.communication_study import conditions as communication_conditions
from algorithms.common.density_study import conditions as density_conditions
from algorithms.common.study_queue import jobs_for
from algorithms.epymarl_mappo import campaign
from heterogrid import EnvConfig


class NativeMAPPOCampaignTests(unittest.TestCase):
    def test_seventeen_configs_and_fifty_one_separate_output_jobs(self):
        base = EnvConfig.load(campaign.PROJECT_ROOT / "configs/scenarios/6v6_rule_distance.json")
        communication = communication_conditions(base)
        density = density_conditions(base)
        sensitivity = {name: base for name in ("M1_small", "M2_large", "M3_xlarge", "N1_3v3", "N2_9v9", "N3_12v12")}
        with tempfile.TemporaryDirectory() as temporary:
            references = []
            for i, variants in enumerate((communication, sensitivity, density)):
                root = Path(temporary) / str(i)
                (root / "configs").mkdir(parents=True)
                protocol = dict(conditions=list(variants), validation_seed=100042, heldout_seed=200042,
                                eval_episodes=64, heldout_episodes=128)
                (root / "experiment.json").write_text(json.dumps(protocol))
                (root / "status.json").write_text(json.dumps(dict(state="COMPLETED")))
                for name, config in variants.items():
                    (root / "configs" / (name + ".json")).write_text(json.dumps(config.to_dict()))
                references.append(root)
            configs, sources, protocol = campaign.reference_configs(references)
            self.assertEqual(len(configs), 17)
            manifest = dict(conditions=list(configs), seeds=[42, 43, 44], team_sizes={n: len(c["red"]) for n,c in configs.items()})
            jobs = jobs_for(manifest, Path("/output/epymarl_mappo/experiment"))
            self.assertEqual(len(jobs), 51)
            self.assertTrue(all("/epymarl_mappo/" in row["path"] for row in jobs))
            self.assertEqual(sources["C1_none"], str(references[0]))
            self.assertEqual(sources["N3_12v12"], str(references[1]))
            self.assertEqual(sources["D3_12v12"], str(references[2]))

    def test_command_uses_epymarl_not_other_mappo_or_smpe(self):
        manifest = dict(python="python", epymarl_root="/frozen/epymarl/src", target_steps_per_seed=5000192,
                        device="cpu", validation_seed=100042, heldout_seed=200042, eval_episodes=64,
                        heldout_episodes=128, eval_interval=250368, log_interval=10240, replay_episodes=1)
        command = campaign.training_command(manifest, Path("/output/epymarl_mappo"),
                                            dict(condition="D3_12v12", seed=42, path="/output/epymarl_mappo/D3/seed_42"))
        self.assertIn("algorithms.epymarl_mappo.train", command)
        self.assertNotIn("--mappo-root", command)
        self.assertNotIn("--smpe-root", command)
        self.assertEqual(command[command.index("--epymarl-root") + 1], manifest["epymarl_root"])


if __name__ == "__main__":
    unittest.main()
