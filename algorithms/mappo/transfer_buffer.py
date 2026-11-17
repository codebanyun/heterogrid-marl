"""Exclude absent/dead samples before PPO and value-normalizer updates.

The external trainer masks losses but can still update its value statistics from
padded returns. Filtering before minibatching also prevents empty-active batches.
Only feed-forward sampling is implemented; recurrent sequences require a separate
sequence-preserving implementation, not independent row filtering.
"""
import numpy as np


def active_buffer_class(base):
    class ActiveBuffer(base):
        def feed_forward_generator(self,advantages,num_mini_batch=None,mini_batch_size=None):
            valid=np.flatnonzero(self.active_masks[:-1].reshape(-1)>0)
            if not len(valid):
                raise ValueError("No active agent samples in rollout")
            if num_mini_batch is None:
                if not mini_batch_size or mini_batch_size<1:raise ValueError("Provide a positive batch size")
                num_mini_batch=max(1,len(valid)//mini_batch_size)
            if not 1<=num_mini_batch<=len(valid):raise ValueError("Too many mini-batches for active samples")
            order=np.random.permutation(valid)
            flatten=lambda a:a.reshape(-1,*a.shape[3:])
            arrays=(flatten(self.share_obs[:-1]),flatten(self.obs[:-1]),flatten(self.rnn_states[:-1]),
                    flatten(self.rnn_states_critic[:-1]),flatten(self.actions),flatten(self.value_preds[:-1]),
                    flatten(self.returns[:-1]),flatten(self.masks[:-1]),flatten(self.active_masks[:-1]),
                    flatten(self.action_log_probs),flatten(advantages),
                    flatten(self.available_actions[:-1]) if self.available_actions is not None else None)
            for indices in np.array_split(order,num_mini_batch):
                yield tuple(a[indices] if a is not None else None for a in arrays)
    return ActiveBuffer
