import json
import unittest
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, parallel_env


ROOT = Path(__file__).resolve().parents[2]


class ScenarioConfigTests(unittest.TestCase):
    def test_baseline_matches_environment_defaults(self):
        config = EnvConfig.load(ROOT / "configs/scenarios/default_6v6.json")
        self.assertEqual(config, EnvConfig())

    def test_training_scenario_preserves_shaped_task(self):
        config = EnvConfig.load(ROOT / "configs/scenarios/6v6_rule_distance.json")
        previous = EnvConfig(target_sharing="auto", scripted_target_sharing=False,
                             timeout_as_loss=True, time_penalty=0.005,
                             discovery_reward=0.05, sharing_reward=0.05,
                             information_reward_cap=1.0)
        self.assertEqual(config, previous)

    def test_reward_ablation_changes_only_reward_fields(self):
        configs = [EnvConfig.load(ROOT / path).to_dict() for path in (
            "configs/ablations/6v6_shared.json", "configs/ablations/6v6_terminal.json",
            "configs/scenarios/6v6_rule_distance.json")]
        reward_fields = {"timeout_as_loss", "time_penalty", "discovery_reward",
                         "sharing_reward", "information_reward_cap"}
        tasks = [{k: v for k, v in c.items() if k not in reward_fields} for c in configs]
        self.assertEqual(tasks[0], tasks[1])
        self.assertEqual(tasks[1], tasks[2])

    def test_all_configs_are_distinct_and_support_legal_rollouts(self):
        paths = sorted((ROOT / "configs").rglob("*.json"))
        self.assertEqual(len(paths), 5)
        expanded = []
        for path in paths:
            with self.subTest(config=path.name):
                config = EnvConfig.load(path)
                expanded.append(json.dumps(config.to_dict(), sort_keys=True))
                environment = parallel_env(config=config)
                try:
                    observations, _ = environment.reset(seed=42)
                    random = np.random.default_rng(42)
                    for _ in range(10):
                        if not environment.agents:
                            break
                        actions = {name: int(random.choice(np.flatnonzero(observations[name]["action_mask"])))
                                   for name in environment.agents}
                        observations, _, _, _, infos = environment.step(actions)
                        self.assertTrue(all(not info.get("invalid_action", False) for info in infos.values()))
                finally:
                    environment.close()
        self.assertEqual(len(expanded), len(set(expanded)))


if __name__ == "__main__":
    unittest.main()
