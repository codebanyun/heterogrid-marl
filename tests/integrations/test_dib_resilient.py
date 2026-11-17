import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from heterogrid import EnvConfig,SMACAdapter
from heterogrid.scenarios import balanced_roster
try:
    import torch
    from algorithms.dib_mappo.backend import build_policy,initial_memory,act,episode_batch
    from algorithms.dib_mappo.sequence import replay_sequence,learn
    from algorithms.dib_mappo.resilient import actor_support,episode_weights,geometric_features
    from algorithms.dib_mappo.resume import capture,restore
except ImportError:torch=None
ROOT=Path('/root/research/DIB-MAPPO/src')


@unittest.skipIf(torch is None or not ROOT.exists(),'Optional DIB backend absent')
class ResilientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def policy(self,native=False):
        r=balanced_roster(3);e=SMACAdapter(config=EnvConfig(red=r,blue=r,max_steps=8,opponent='dummy'))
        self.addCleanup(e.close);e.reset(42);torch.manual_seed(42)
        p,_=build_policy(SimpleNamespace(dib_root=ROOT,training_profile='resilient_v3',
            tactical_preset='belief_first',sequence_acceleration=not native),e)
        return p,e
    def batch(self,p,e):
        from algorithms.common.value_decomposition.train import observe,new_history
        o,s,a=observe(e);h=new_history(o,s,a);memory=initial_memory(p,1);rng=np.random.default_rng(42)
        while True:
            ac,memory=act(p,o[None],a[None],memory,rng);r,done,_=e.step(ac[0].tolist());o,s,a=observe(e)
            h['u'].append(ac[0]);h['r'].append(r);h['o'].append(o);h['s'].append(s);h['avail'].append(a)
            if done:return episode_batch(p,[h]*4)
    def test_capability_mask_is_pre_action_and_does_not_change_fill(self):
        obs=torch.zeros(2,5,3,20);obs[...,3]=1.;obs[:,:,1:,9]=1.
        obs[0,2:,1:,3]=0.;obs[1,4:,:,3]=0.
        filled=torch.ones(2,5,1);before=filled.clone()
        mask,alive=actor_support(obs,filled)
        self.assertEqual(float(mask[0,:2].sum()),6.)
        self.assertEqual(float(mask[0,2:].sum()),0.)
        self.assertEqual(float(alive[0,2:].sum()),3.)
        torch.testing.assert_close(filled,before,atol=0,rtol=0)
        w=episode_weights(mask)
        torch.testing.assert_close(w.sum((1,2)),torch.ones(2))
    def test_geometry_masks_missing_and_dead_entities_and_is_permutation_invariant(self):
        p,e=self.policy();agent=p.mac.agent;o=torch.as_tensor(np.asarray(e.get_obs(),np.float32))
        start=agent.entity_layout.target_start;o[:,start:start+33]=0
        base=geometric_features(agent,o);self.assertEqual(base.shape,(3,7,37))
        self.assertTrue(bool(torch.isfinite(base).all()))
        targets=o[:,start:start+33].reshape(3,3,11);targets[...,2:]=torch.randn(3,3,9)
        torch.testing.assert_close(geometric_features(agent,o),base,atol=0,rtol=0)
        targets[...,0]=1;targets[...,5]=.5;targets[...,2:5]=torch.randn(3,3,3)*.1
        one=geometric_features(agent,o);targets[:]=targets[:,[2,0,1]].clone()
        torch.testing.assert_close(geometric_features(agent,o),one,atol=1e-6,rtol=1e-6)
    def test_native_fused_probabilities_and_full_updates_agree(self):
        native,e=self.policy(True);fast,_=self.policy();batch=self.batch(native,e)
        batch['obs'][:,4:,1:,3]=0.
        for clock in [1000,750000]:
            for p in (native,fast):p.mac.set_residual_actor_gate_for_t(clock)
            native.mac.init_hidden(4)
            with torch.no_grad():
                a=torch.stack([native.mac.forward(batch,t) for t in range(batch.max_seq_length-1)],1)
                fast.mac._learning_clock=clock;b,_=replay_sequence(fast.mac,batch,True)
                torch.testing.assert_close(a,b,atol=2e-6,rtol=3e-5)
            for p in (native,fast):learn(p,batch,clock,4)
            for name,value in native.mac.agent.state_dict().items():
                torch.testing.assert_close(value,fast.mac.agent.state_dict()[name],atol=5e-5,rtol=5e-4)
        self.assertGreater(fast.logger.values['incapable_actor_samples'],0)
        self.assertAlmostEqual(fast.logger.values['episode_policy_weight_min'],1.,places=6)
    def test_execution_cannot_use_another_agents_current_observation(self):
        p,e=self.policy();p.mac.set_residual_actor_gate_for_t(1000000)
        o=torch.as_tensor(np.asarray(e.get_obs(),np.float32))[None];a=torch.zeros(1,3,p.n_actions)
        h=p.mac.agent.init_hidden(1)
        with torch.no_grad():
            one,_=p.mac.agent(o,a,h);o[:,1:]=torch.randn_like(o[:,1:]);two,_=p.mac.agent(o,a,h)
        torch.testing.assert_close(one['logits_k'][:,0],two['logits_k'][:,0],atol=0,rtol=0)
    def test_exact_resume_restores_new_parameters_and_next_update(self):
        p,e=self.policy();b=self.batch(p,e);p.mac.set_residual_actor_gate_for_t(750000)
        learn(p,b,750000,4);rng=np.random.default_rng(42);snapshot=capture(p,rng,{'steps':100},'v3')
        learn(p,b,750000,8);expected=copy.deepcopy(p.mac.agent.state_dict())
        fresh,_=self.policy();restore(fresh,rng,snapshot,'v3');learn(fresh,b,750000,8)
        for name,value in expected.items():torch.testing.assert_close(value,fresh.mac.agent.state_dict()[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
