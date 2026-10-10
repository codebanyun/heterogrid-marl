import importlib.util
import unittest

import numpy as np

from heterogrid import EnvConfig, SMACAdapter


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional MAPPO dependency torch is unavailable")
class MAPPOObservationTest(unittest.TestCase):
    def test_cached_observations_match_adapter(self):
        from algorithms.mappo.train import observe

        random = np.random.default_rng(42)
        for sharing in ("manual", "auto"):
            env = SMACAdapter(config=EnvConfig(target_sharing=sharing, max_steps=50))
            try:
                env.reset(seed=42)
                dead_slots = 0
                for step in range(150):
                    if step % 10 == 0:
                        env.env.share_visible_targets()
                    obs, state, available, active = observe(env)
                    np.testing.assert_array_equal(obs, env.get_obs())
                    np.testing.assert_array_equal(available, env.get_avail_actions())
                    np.testing.assert_array_equal(state[0], env.get_state())
                    dead_slots += int((active == 0).any())
                    actions = [int(random.choice(np.flatnonzero(row))) for row in available]
                    _, done, _ = env.step(actions)
                    if done:
                        env.reset()
                self.assertGreater(dead_slots, 0)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
