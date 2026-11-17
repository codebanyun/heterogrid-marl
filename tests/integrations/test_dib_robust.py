"""Structured supervision, local-only filter, guarded actor and replay parity."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from heterogrid import EnvConfig, SMACAdapter
from heterogrid.scenarios import balanced_roster
try:
    import torch
    from algorithms.dib_mappo.robust import EntityLayout, compose_logits, clip_actor_gradients
    from algorithms.dib_mappo.backend import build_policy, episode_batch, initial_memory, act, settings
    from algorithms.dib_mappo.sequence import replay_sequence,learn
except ImportError:
    torch = None
ROOT = Path('/root/research/DIB-MAPPO/src')


@unittest.skipIf(torch is None or not (ROOT/'learners/dib_ppo_learner.py').exists(),'Optional DIB backend absent')
class RobustDIBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def policy(self,n=3,native=False):
        r=balanced_roster(n)
        env=SMACAdapter(config=EnvConfig(red=r,blue=r,max_steps=6,opponent='dummy'))
        env.reset(42);self.addCleanup(env.close);torch.manual_seed(42)
        p,_=build_policy(SimpleNamespace(dib_root=ROOT,training_profile='robust_comm_v1',sequence_acceleration=not native),env)
        return p,env

    def batch(self,p,e):
        from algorithms.common.value_decomposition.train import observe,new_history
        o,s,a=observe(e);h=new_history(o,s,a);mem=initial_memory(p,1);rng=np.random.default_rng(42)
        while True:
            ac,mem=act(p,o[None],a[None],mem,rng);r,done,_=e.step(ac[0].tolist());o,s,a=observe(e)
            h['u'].append(ac[0]);h['r'].append(r);h['o'].append(o);h['s'].append(s);h['avail'].append(a)
            if done:return episode_batch(p,[h]*4)

    def test_compact_shapes_and_labels_not_raw_zero_mask(self):
        p,e=self.policy(6);layout=p.mac.agent.entity_layout
        o=torch.as_tensor(np.asarray(e.get_obs(),np.float32))[None,None]
        y=p.mac.agent._build_other_agent_targets(o)
        self.assertEqual(y.shape,(1,1,6,390));self.assertEqual(layout.width,78)
        self.assertEqual(p.mac.agent.probe_head.out_features,780)
        # Message fields and fixed role/capability/time fields never become targets.
        z=o.clone();z[...,-60:]=.7;z[...,4:12]=.9
        torch.testing.assert_close(p.mac.agent._build_other_agent_targets(z),y,atol=0,rtol=0)
        compact=layout.pack(o)
        self.assertTrue(bool((layout.valid_features(compact)[...,1:5]==1).all()))
        # Coordinate zero is valid if the entity/record is observed.
        compact.zero_();compact[...,0]=1;compact[...,5]=1
        mask=layout.valid_features(compact)
        self.assertEqual(mask[0,0,0,6].item(),1.)

    def test_missing_attributes_have_zero_grad_but_negative_presence_learns(self):
        p,_=self.policy();layout=p.mac.agent.entity_layout
        target=torch.zeros(2,layout.width);target[:,0]=1
        prediction=torch.zeros_like(target,requires_grad=True)
        loss=layout.error(prediction,target).sum();loss.backward()
        start=5+5*(layout.n-1)
        self.assertGreater(prediction.grad[:,start].abs().sum().item(),0)
        self.assertEqual(prediction.grad[:,start+2:start+7].abs().sum().item(),0)
        # Dead owner: only its alive/dead label is supervised.
        target.zero_();prediction.grad=None;layout.error(prediction,target).sum().backward()
        self.assertGreater(prediction.grad[:,0].abs().sum().item(),0)
        self.assertEqual(prediction.grad[:,1:].abs().sum().item(),0)

    def test_valid_target_coordinates_receive_grad_even_when_zero(self):
        p,_=self.policy();layout=p.mac.agent.entity_layout
        target=torch.zeros(1,layout.width);target[:,0]=1;start=5+5*(layout.n-1);target[:,start]=1
        prediction=torch.full_like(target,.3,requires_grad=True)
        layout.error(prediction,target).sum().backward()
        self.assertGreater(prediction.grad[:,start+2:start+5].abs().sum().item(),0)

    def test_filter_uses_local_evidence_and_preserves_initial_prior(self):
        p,e=self.policy();agent=p.mac.agent
        obs=torch.as_tensor(np.asarray(e.get_obs(),np.float32));z=torch.zeros(e.n_agents,4,agent.z_dim)
        prior=torch.full((e.n_agents,4),.25);actions=torch.zeros(e.n_agents,agent.n_actions)
        w,error=agent._completed_transition_filter(prior,z,obs,obs,actions,torch.zeros(e.n_agents,1))
        torch.testing.assert_close(w,prior,atol=0,rtol=0)
        # Candidate-specific local record likelihood, with no other-agent data.
        target=agent.entity_layout.pack(obs)
        predictions=torch.zeros(e.n_agents,4,agent.entity_layout.width)
        predictions[:,0,0]=5.;predictions[:,1:,0]=-5.
        old=agent.predict_next_obs
        agent.predict_next_obs=lambda *args:predictions
        w,error=agent._completed_transition_filter(prior,z,obs,obs,actions,torch.ones(e.n_agents,1))
        agent.predict_next_obs=old
        self.assertTrue(bool((w[:,0]>w[:,1]).all()));self.assertTrue(bool(torch.isfinite(w).all()))
        # Equal evidence must not manufacture a confident winner.
        w,_=agent._completed_transition_filter(prior,z,obs,obs,actions,torch.ones(e.n_agents,1))
        torch.testing.assert_close(w,prior,atol=1e-6,rtol=1e-6)

    def test_residual_bound_zero_gate_and_learned_trust(self):
        p,_=self.policy();agent=p.mac.agent
        base=torch.randn(2,1,agent.n_actions);res=torch.full((2,4,agent.n_actions),1e5)
        h=torch.randn(2,agent.hidden_dim);c=torch.randn(2,agent.ahs_hidden_dim)
        agent.set_residual_actor_gate(0)
        logits,_,trust=compose_logits(agent,base,res,h,c)
        torch.testing.assert_close(logits,base.expand_as(res),atol=0,rtol=0)
        self.assertAlmostEqual(trust.mean().item(),.1192029,places=6)
        agent.set_residual_actor_gate(1)
        logits,_,trust=compose_logits(agent,base,res,h,c)
        self.assertLessEqual((logits-base).abs().max().item(),1.)
        logits.sum().backward();self.assertGreater(agent.belief_trust.bias.grad.abs().sum().item(),0)

    def test_probe_classification_uncertainty_does_not_use_unused_logstd(self):
        from algorithms.dib_mappo.robust import probe_entropy
        p,_=self.policy();a=p.mac.agent
        mu=torch.randn(2,a.y_dim);std=torch.zeros_like(mu)
        first=probe_entropy(a,mu,std)
        std[:,a.entity_layout.binary_features(mu).bool()]=1.
        torch.testing.assert_close(first,probe_entropy(a,mu,std),atol=0,rtol=0)

    def test_schedule_has_real_belief_pretraining_before_actor(self):
        p,e=self.policy();b=self.batch(p,e)
        for clock,gate,scale in [(999999,0,0),(1000000,0,0),(1250000,0,.5),
                                 (1500000,0,1),(2250000,.5,None),(3000000,1,None)]:
            self.assertAlmostEqual(p.mac.set_residual_actor_gate_for_t(clock),gate)
            if scale is not None:self.assertAlmostEqual(p.learner._belief_scale(clock),scale)
        teacher=p.mac.agent.teacher_head.weight.detach().clone()
        base=p.mac.agent.base_actor_head.weight.detach().clone()
        p.mac.set_residual_actor_gate_for_t(999000);learn(p,b,999000,4)
        torch.testing.assert_close(teacher,p.mac.agent.teacher_head.weight,atol=0,rtol=0)
        p.mac.set_residual_actor_gate_for_t(1250000);learn(p,b,1250000,8)
        self.assertFalse(torch.equal(teacher,p.mac.agent.teacher_head.weight))
        self.assertFalse(torch.equal(base,p.mac.agent.base_actor_head.weight))
        self.assertEqual(p.logger.values['belief_loss_scale'],.5)

    def test_separate_clipping_keeps_base_grad_and_handles_large_finite_values(self):
        p,_=self.policy();p.mac._learning_clock=0
        base=p.mac.agent.base_actor_head.bias;aux=p.mac.agent.student[0].bias
        base.grad=torch.ones_like(base)*.001;aux.grad=torch.ones_like(aux)*1e30
        before=base.grad.clone();norm=clip_actor_gradients(p.learner)
        self.assertTrue(bool(torch.isfinite(norm)))
        torch.testing.assert_close(base.grad,before,atol=0,rtol=0)
        self.assertLessEqual(float(aux.grad.double().norm()),5.00001)
        aux.grad.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):clip_actor_gradients(p.learner)

    def test_native_and_accelerated_probabilities_and_full_updates_match(self):
        native,e=self.policy(native=True);fast,_=self.policy();b=self.batch(native,e)
        b['obs'][:,3:,1,3]=0
        for clock in (999000,1250000,2250000,3000000):
            for p in (native,fast):p.mac.set_residual_actor_gate_for_t(clock)
            with torch.no_grad():
                fast.mac._learning_clock=clock
                pi,aux=replay_sequence(fast.mac,b,True)
                native.mac.init_hidden(b.batch_size)
                pn=torch.stack([native.mac.forward(b,t) for t in range(b.max_seq_length-1)],1)
                torch.testing.assert_close(pi,pn,atol=2e-6,rtol=3e-5)
            for p in (native,fast):learn(p,b,clock,4)
            for m,n in [(native.mac.agent,fast.mac.agent),(native.learner.critic,fast.learner.critic)]:
                for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],atol=5e-5,rtol=5e-4)
        self.assertGreater(fast.logger.values['belief_trust_mean'],0)
        self.assertEqual(fast.logger.values['agent_update_skipped'],0)

    def test_local_execution_has_no_future_or_other_observation_input(self):
        p,e=self.policy();a=p.mac.agent;a.set_residual_actor_gate(1)
        obs=torch.as_tensor(np.asarray(e.get_obs(),np.float32))[None]
        action=torch.zeros(1,e.n_agents,a.n_actions);hidden=a.init_hidden(1)
        with torch.no_grad():
            one,_=a(obs,action,hidden)
            # Change all *other* inputs in the batched actor. Agent0 cannot see them.
            altered=obs.clone();altered[:,1:]=torch.randn_like(altered[:,1:])
            two,_=a(altered,action,hidden)
        torch.testing.assert_close(one['logits_k'][:,0],two['logits_k'][:,0],atol=0,rtol=0)
        torch.testing.assert_close(one['weights'][:,0],two['weights'][:,0],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
