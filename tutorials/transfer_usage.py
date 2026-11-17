"""Exercise the opt-in task interface without training or writing artifacts."""
from dataclasses import replace
import numpy as np

from heterogrid import EnvConfig, TaskSuite, TransferSMACAdapter
from heterogrid.scenarios import balanced_roster


def main():
    base=EnvConfig(max_steps=3,target_sharing="auto")
    tasks={str(n):replace(base,red=balanced_roster(n),blue=balanced_roster(n)) for n in (3,6)}
    suite=TaskSuite(tasks,{"train":["3","6"]})
    env=TransferSMACAdapter(suite)
    try:
        for task in ("3","6"):
            env.reset(seed=42,options={"task_id":task})
            done=False
            while not done:
                actions=[int(np.flatnonzero(mask)[0]) for mask in env.get_avail_actions()]
                _,done,info=env.step(actions)
            print({"task":info["task_id"],"actual_agents":info["actual_n_agents"],**env.get_env_info()})
    finally:env.close()


if __name__=="__main__":main()
