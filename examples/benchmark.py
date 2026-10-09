"""Local environment throughput only; no neural-network training or rendering."""
import argparse
import json
import time

import numpy as np

from heterogrid import EnvConfig, parallel_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    environment = parallel_env(config=EnvConfig(max_steps=200))
    rng = np.random.default_rng(42)
    observations, _ = environment.reset(seed=42)
    start = time.perf_counter()
    for _ in range(args.steps):
        if not environment.agents:
            observations, _ = environment.reset()
        actions = {n: int(rng.choice(np.flatnonzero(observations[n]["action_mask"]))) for n in environment.agents}
        observations, _, _, _, _ = environment.step(actions)
    elapsed = time.perf_counter() - start
    print(json.dumps({"environment_steps": args.steps, "seconds": round(elapsed, 3),
                      "steps_per_second": round(args.steps / elapsed, 1),
                      "rendering": False, "neural_network_inference": False}))
    environment.close()


if __name__ == "__main__":
    main()
