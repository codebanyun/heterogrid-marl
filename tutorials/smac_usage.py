"""Fixed-slot SMAC-style rollout without an external algorithm dependency."""
import numpy as np

from heterogrid import SMACAdapter


def main():
    environment = SMACAdapter()
    observations, state = environment.reset(seed=42)
    print(environment.get_env_info())
    rng = np.random.default_rng(42)
    done = False
    total_reward = 0.0
    while not done:
        actions = [int(rng.choice(np.flatnonzero(mask))) for mask in environment.get_avail_actions()]
        reward, done, info = environment.step(actions)
        observations, state = environment.get_obs(), environment.get_state()
        total_reward += reward
    print("Team reward:", total_reward, "outcome:", info)
    environment.close()


if __name__ == "__main__":
    main()
