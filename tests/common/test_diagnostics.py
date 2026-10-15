import unittest

from algorithms.common.diagnostics import summarize_team_metrics


class DiagnosticReportingTests(unittest.TestCase):
    def test_missing_diagnostics_and_undefined_values_are_not_zeroes(self):
        self.assertEqual(summarize_team_metrics([{}]), {})
        rows = [{"team_episode_metrics": {"red": {"diagnostics": {"schema_version": 1, "attack_hit_rate": rate}},
                                           "blue": {"diagnostics": {"schema_version": 1, "attack_hit_rate": None}}}}
                for rate in (None, .5, 1)]
        result = summarize_team_metrics(rows)
        self.assertEqual(result["mean_team_episode_metrics"]["red"]["diagnostics"]["attack_hit_rate"], .75)
        self.assertEqual(result["team_episode_metrics_observed_episodes"]["red"]["diagnostics"]["attack_hit_rate"], 2)
        self.assertIsNone(result["mean_team_episode_metrics"]["blue"]["diagnostics"]["attack_hit_rate"])
        self.assertEqual(result["team_episode_metrics_observed_episodes"]["blue"]["diagnostics"]["attack_hit_rate"], 0)
        self.assertNotIn("schema_version", result["mean_team_episode_metrics"]["red"]["diagnostics"])
