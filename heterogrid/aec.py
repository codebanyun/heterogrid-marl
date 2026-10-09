"""AEC conversion with self-contained, standard dead-agent bookkeeping.

The local PettingZoo installation writes to the read-only num_agents property in
its dead-step helper. Keep the conversion isolated here rather than modifying
the user's installed package. This implementation also works on upstream builds.
"""
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper


class GridAECWrapper(parallel_to_aec_wrapper):
    def _was_dead_step(self, action):
        if action is not None:
            raise ValueError("Dead agents must receive action None")
        agent = self.agent_selection
        if not (self.terminations[agent] or self.truncations[agent]):
            raise RuntimeError("Cannot remove a live agent")
        for mapping in (self.terminations, self.truncations, self.rewards, self._cumulative_rewards, self.infos):
            del mapping[agent]
        self.agents.remove(agent)
        dead = [n for n in self.agents if self.terminations[n] or self.truncations[n]]
        if dead:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = dead[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()


def as_aec(parallel):
    return OrderEnforcingWrapper(GridAECWrapper(parallel))
