import copy
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from heterogrid import EnvConfig,SMACAdapter
from heterogrid.scenarios import balanced_roster
try:
    import torch
    from algorithms.dib_mappo.backend import build_policy,initial_memory,episode_batch,act
    from algorithms.dib_mappo.sequence import learn,replay_sequence
    from algorithms.dib_mappo.tactical import gae,trajectory_teacher
    from algorithms.dib_mappo.resume import capture,restore
except ImportError:torch=None
ROOT=Path('/root/research/DIB-MAPPO/src')


@unittest.skipIf(torch is None or not ROOT.exists(),'Optional DIB backend absent')
class TacticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def policy(self,native=False):
        r=balanced_roster(3);e=SMACAdapter(config=EnvConfig(red=r,blue=r,max_steps=8,opponent='dummy'))
        self.addCleanup(e.close);e.reset(42);torch.manual_seed(42)
        p,_=build_policy(SimpleNamespace(dib_root=ROOT,training_profile='tactical_v2',
                                        sequence_acceleration=not native),e)
        return p,e
    def batch(self,p,e):
        from algorithms.common.value_decomposition.train import observe,new_history
        o,s,a=observe(e);h=new_history(o,s,a);mem=initial_memory(p,1);rng=np.random.default_rng(42)
        while True:
            ac,mem=act(p,o[None],a[None],mem,rng);r,done,_=e.step(ac[0].tolist());o,s,a=observe(e)
            h['u'].append(ac[0]);h['r'].append(r);h['o'].append(o);h['s'].append(s);h['avail'].append(a)
            if done:return episode_batch(p,[h]*4)
    def test_gae_terminal_padding_and_team_return(self):
        r=torch.tensor([[[1.],[2.],[3.],[99.]]]);v=torch.zeros(1,5,1)
        term=torch.tensor([[[0.],[0.],[1.],[0.]]]);mask=torch.tensor([[[1.],[1.],[1.],[0.]]])
        a,ret=gae(r,v,term,mask,1.,1.)
        torch.testing.assert_close(a,torch.tensor([[[6.],[5.],[3.],[0.]]]),atol=0,rtol=0)
        v[:,3:]=1000
        a,_=gae(r,v,term,mask,1.,1.)
        torch.testing.assert_close(a,torch.tensor([[[6.],[5.],[3.],[0.]]]),atol=0,rtol=0)
    def test_entity_attack_head_respects_permutation_and_missing_entities(self):
        p,e=self.policy();agent=p.mac.agent
        o=torch.as_tensor(np.asarray(e.get_obs(),np.float32));a=torch.zeros(3,agent.n_actions);ids=torch.eye(3)
        start=agent.entity_layout.target_start
        o[:,start:start+33]=torch.rand(3,33)
        changed=o.clone();target=changed[:,start:start+33].reshape(3,3,11)
        target[:]=target[:,[2,0,1]].clone()
        x=agent.encode_local(o,a,ids);xx=agent.encode_local(changed,a,ids)
        torch.testing.assert_close(x,xx,atol=1e-7,rtol=1e-6)
        logits=agent.action_logits(x,o);other=agent.action_logits(xx,changed)
        torch.testing.assert_close(logits[:,7:][:,[2,0,1]],other[:,7:],atol=1e-7,rtol=1e-6)
        self.assertTrue(bool(torch.isfinite(agent.encode_local(torch.zeros_like(o),a,ids)).all()))
        targets=o[:,start:start+33].reshape(3,3,11)
        features=agent.enemy_encoder(targets)
        dense=agent.attack_head(torch.cat((x[:,None].expand(-1,3,-1),features,o[:,:12,None].transpose(1,2).expand(-1,3,-1)),-1)).squeeze(-1)
        torch.testing.assert_close(dense,logits[:,7:],atol=1e-7,rtol=1e-6)

    def test_zero_gate_execution_fast_path_matches_native_actions(self):
        p,e=self.policy();p.mac.set_residual_actor_gate_for_t(100000)
        observations=torch.as_tensor(np.asarray(e.get_obs(),np.float32))[None]
        memory=initial_memory(p,1);av=np.asarray(e.get_avail_actions(),np.uint8)[None]
        with torch.no_grad():
            out,_=p.mac.agent(observations,torch.zeros(1,3,p.n_actions),p.mac.agent.init_hidden(1))
            probs=torch.softmax(out['logits_k'].masked_fill(torch.as_tensor(av).unsqueeze(2)==0,-1e10),-1)
            probs=(out['weights'].unsqueeze(-1)*probs).sum(2)
        actions,_=act(p,observations.numpy(),av,memory,np.random.default_rng(42),True)
        np.testing.assert_array_equal(actions,probs.argmax(-1).numpy())
    def test_native_streaming_and_sequence_probabilities_aux_and_updates(self):
        native,e=self.policy(True);fast,_=self.policy();b=self.batch(native,e)
        b['obs'][:,5:,1]=0;b['filled'][1,6:]=0;b['terminated'][1,5]=1
        for clock in [1000,100000,750000]:
            for p in [native,fast]:p.mac.set_residual_actor_gate_for_t(clock)
            native.mac.init_hidden(4)
            with torch.no_grad():
                steps=[native.mac.forward(b,t,return_aux=True) for t in range(b.max_seq_length-1)]
                pn=torch.stack([x[0] for x in steps],1);fast.mac._learning_clock=clock
                pf,aux=replay_sequence(fast.mac,b,True)
                torch.testing.assert_close(pn,pf,atol=2e-6,rtol=3e-5)
                if clock>50000:
                    for key in ['c','probe_mu','rho','z','weights','base_logits']:
                        torch.testing.assert_close(torch.stack([x[1][key] for x in steps],1),aux[key],atol=3e-6,rtol=5e-5)
            for p in [native,fast]:learn(p,b,clock,4)
            for first,second in [(native.mac.agent,fast.mac.agent),(native.learner.critic,fast.learner.critic)]:
                for k,v in first.state_dict().items():torch.testing.assert_close(v,second.state_dict()[k],atol=5e-5,rtol=5e-4)
        self.assertEqual(fast.logger.values['agent_update_skipped'],0)
        self.assertGreater(fast.logger.values['hyp_latent_spread'],.01)
    def test_actor_cannot_read_future_or_other_agents(self):
        p,e=self.policy();b=self.batch(p,e);p.mac.set_residual_actor_gate_for_t(1000000);p.mac._learning_clock=1000000
        with torch.no_grad():
            original,_=replay_sequence(p.mac,b,True)
            b['obs'][:,4:]=torch.randn_like(b['obs'][:,4:])
            changed,_=replay_sequence(p.mac,b,True)
            torch.testing.assert_close(original[:,:4],changed[:,:4],atol=0,rtol=0)
            obs=torch.as_tensor(np.asarray(e.get_obs(),np.float32))[None];action=torch.zeros(1,3,p.n_actions);hidden=p.mac.agent.init_hidden(1)
            one,_=p.mac.agent(obs,action,hidden);obs[:,1:]=torch.randn_like(obs[:,1:]);two,_=p.mac.agent(obs,action,hidden)
            torch.testing.assert_close(one['logits_k'][:,0],two['logits_k'][:,0],atol=0,rtol=0)
    def test_teacher_padding_is_not_future_evidence(self):
        p,e=self.policy();tokens=torch.randn(2,8,3,128);valid=torch.ones(2,8,3);valid[0,4:]=0
        with torch.no_grad():
            one=trajectory_teacher(p.mac.agent,tokens,valid)
            tokens[0,4:]=torch.randn_like(tokens[0,4:])*1000
            two=trajectory_teacher(p.mac.agent,tokens,valid)
        torch.testing.assert_close(one[0,:4],two[0,:4],atol=0,rtol=0)
    def test_resume_restores_optimizer_rng_and_next_identical_update(self):
        p,e=self.policy();b=self.batch(p,e);p.mac.set_residual_actor_gate_for_t(750000)
        learn(p,b,750000,4);rng=np.random.default_rng(987)
        snapshot=capture(p,rng,{'steps':1000,'episode_index':4},'identity')
        expected_rng=rng.random(7);expected_torch=torch.rand(7);expected_python=random.random()
        learn(p,b,750000,8);expected=copy.deepcopy(p.mac.agent.state_dict())
        fresh,_=self.policy();state=restore(fresh,rng,snapshot,'identity')
        self.assertEqual(state,{'steps':1000,'episode_index':4})
        np.testing.assert_array_equal(rng.random(7),expected_rng);torch.testing.assert_close(torch.rand(7),expected_torch,atol=0,rtol=0)
        self.assertEqual(random.random(),expected_python)
        learn(fresh,b,750000,8)
        for k,v in expected.items():torch.testing.assert_close(v,fresh.mac.agent.state_dict()[k],atol=0,rtol=0)
        with self.assertRaises(ValueError):restore(fresh,rng,snapshot,'different source')


if __name__=='__main__':unittest.main()
