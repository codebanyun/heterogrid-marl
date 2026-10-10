"""Algorithm-neutral PettingZoo usage; replace the action sampler with your actor."""
import numpy as np

from heterogrid import parallel_env


def main():
    environment = parallel_env()
    observations, infos = environment.reset(seed=42)
    rng = np.random.default_rng(42)
    total_team_reward = 0.0
    while environment.agents:
        actions = {name: int(rng.choice(np.flatnonzero(observations[name]["action_mask"])))
                   for name in environment.agents}
        observations, rewards, terminated, truncated, infos = environment.step(actions)
        total_team_reward += next(iter(rewards.values()))  # shared reward: take once, don't sum
    print("Team reward:", total_team_reward, "winner:", environment.core.winner)
    environment.close()


if __name__ == "__main__":
    main()
