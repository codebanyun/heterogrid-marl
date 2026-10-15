"""Single-sided interventions and bounded four-algorithm scheduling."""
import unittest

from algorithms.common.fixed_blue_campaign import combine_plans
from algorithms.common.sensitivity import communication_protocol, scenarios
from heterogrid import CommunicationConfig, EnvConfig


class FixedBlueTests(unittest.TestCase):
    def test_all_thirteen_treatments_keep_blue_and_other_rules_fixed(self):
        old, new = scenarios(), scenarios(fixed_blue=True)
        self.assertEqual(set(new), set(old))
        blue = CommunicationConfig("distance", 10.0, 0.0)
        baseline = new["B0"].to_dict()
        for name, config in new.items():
            self.assertEqual(config.blue_communication, blue)
            self.assertEqual(config.communication_for("red"), old[name].communication_for("red"))
            self.assertEqual(EnvConfig.from_dict(config.to_dict()), config)
            for field, value in old[name].to_dict().items():
                if field not in {"communication", "comm_radius", "dropout_probability",
                                 "red_communication", "blue_communication"}:
                    self.assertEqual(config.to_dict()[field], value, (name, field))
            if name.startswith("C"):
                changed = {key for key, value in config.to_dict().items() if value != baseline[key]}
                self.assertEqual(changed, {"red_communication"})
        self.assertEqual(new["C1"].red_communication.mode, "none")
        self.assertEqual(new["C3"].red_communication.radius, 5.0)
        self.assertEqual(new["C4"].red_communication.radius, 15.0)
        for name, probability in (("C5", 0.25), ("C6", 0.5)):
            self.assertEqual(new[name].red_communication.mode, "full")
            self.assertEqual(new[name].red_communication.dropout_probability, probability)

    def test_legacy_matrix_has_no_team_overrides(self):
        for config in scenarios().values():
            self.assertIsNone(config.red_communication)
            self.assertIsNone(config.blue_communication)
        self.assertEqual(communication_protocol(True)["scope"], "red_only")
        self.assertEqual(communication_protocol(False)["scope"], "both_teams")

    def test_merged_queue_keeps_commands_and_runtime_and_never_duplicates(self):
        plans = [{"runtime": "/value", "roots": {"qmix": "/qmix", "qtran": "/qtran"},
                  "jobs": [{"command": ["qmix"]}, {"command": ["qtran"]}], "requested_total_steps": 10},
                 {"runtime": "/actor", "roots": {"maa2c": "/maa2c", "coma": "/coma"},
                  "jobs": [{"command": ["maa2c"]}, {"command": ["coma"]}], "requested_total_steps": 10}]
        merged = combine_plans(plans, 3, "/queue")
        self.assertEqual(merged["workers"], 3)
        self.assertEqual(merged["requested_total_steps"], 20)
        self.assertEqual(len(merged["roots"]), 4)
        self.assertEqual([job["command"][0] for job in merged["jobs"]], ["qmix", "maa2c", "qtran", "coma"])
        self.assertEqual([job["runtime"] for job in merged["jobs"]], ["/value", "/actor", "/value", "/actor"])
        self.assertNotIn("runtime", plans[0]["jobs"][0])


if __name__ == "__main__":
    unittest.main()
