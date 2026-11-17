import unittest
import numpy as np
from algorithms.mappo.transfer_buffer import active_buffer_class

class ActiveBufferTests(unittest.TestCase):
    def test_inactive_returns_never_reach_value_normalizer_batches(self):
        class Base:pass
        b=active_buffer_class(Base)()
        # Two timesteps, one environment, three slots; only slot 0 is active.
        b.active_masks=np.zeros((3,1,3,1),np.float32);b.active_masks[:,:,0]=1
        for name in ("share_obs","obs","rnn_states","rnn_states_critic","value_preds","returns","masks","available_actions"):
            setattr(b,name,np.zeros((3,1,3,1),np.float32))
        b.returns[:,:,1:]=99999
        b.actions=np.zeros((2,1,3,1));b.action_log_probs=b.actions.copy()
        batches=list(b.feed_forward_generator(np.ones((2,1,3,1)),2))
        self.assertEqual(len(batches),2)
        self.assertEqual(sum(len(s[0]) for s in batches),2)
        for sample in batches:
            self.assertTrue(np.all(sample[8]==1))
            self.assertTrue(np.all(sample[6]==0))
