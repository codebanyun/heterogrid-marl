"""Exchange messages before actions; reports cannot originate from unseen targets."""
import numpy as np

from heterogrid import EnvConfig, UnitSpec, parallel_env


def main():
    config = EnvConfig(map_size=(12, 8, 4), red=[UnitSpec("scout"), UnitSpec("strike")],
                       blue=[UnitSpec("strike")], opponent="dummy", communication="distance", comm_radius=4)
    environment = parallel_env(config=config)
    observations, infos = environment.reset(seed=1, options={"positions": {
        "red_0": [3, 2, 1], "red_1": [0, 2, 1], "blue_0": [6, 2, 1]}})
    print("Strike attack legal before reporting:", observations["red_1"]["action_mask"][7])
    observations, infos = environment.exchange_messages({"red_0": {
        "content": np.zeros(config.message_dim, dtype=np.float32),
        "targets": ["blue_0"], "recipients": ["red_1"]}})
    print("Strike attack legal after reporting:", observations["red_1"]["action_mask"][7])
    environment.step({"red_0": 0, "red_1": 7})
    print("Enemy remaining HP:", environment.snapshot()["units"][-1]["hp"])
    environment.close()


if __name__ == "__main__":
    main()
