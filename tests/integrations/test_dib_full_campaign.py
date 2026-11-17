import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from algorithms.common.experiment_log import write_atomic
from algorithms.dib_mappo.full_campaign import task_matrix,assert_d0_equivalent,NEW_SCENARIOS,REPORT_SCENARIOS
from algorithms.dib_mappo.tactical_campaign import choose
from heterogrid import EnvConfig,SMACAdapter
from heterogrid.scenarios import balanced_roster
try:
    import torch
    from algorithms.dib_mappo.backend import build_policy,act,initial_memory,episode_batch
    from algorithms.dib_mappo.sequence import learn
    from algorithms.dib_mappo.resume import capture,restore
except ImportError:torch=None
ROOT=Path('/root/research/DIB-MAPPO/src')


class FullMatrixTests(unittest.TestCase):
    def test_scope_and_density_protocol(self):
        tasks=task_matrix()
        self.assertEqual(len(NEW_SCENARIOS),12);self.assertEqual(len(REPORT_SCENARIOS),15)
        self.assertFalse(set(NEW_SCENARIOS)&{'C1','C3','C5','C6','D0'})
        for name,count,size in [('D1',3,(19,19,5)),('D0',6,(24,24,6)),('D2',9,(27,27,7)),('D3',12,(30,30,8))]:
            config=tasks[name];self.assertEqual(len(config.red),count);self.assertEqual(config.map_size,size)
            self.assertEqual(config.reward_normalization,'reference')
            self.assertEqual(config.communication_for('blue').mode,'distance')
            self.assertEqual(config.communication_for('blue').radius,10.)
            self.assertEqual(config.max_steps,200)
        assert_d0_equivalent(tasks['B0'],tasks['D0'])
        with self.assertRaises(ValueError):assert_d0_equivalent(tasks['B0'],replace(tasks['D0'],damage_reward=.12))
    def test_fixed_preset_selection_handles_non_c_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            state={'presets':['belief_first'],'scenarios':['B0','D3'],'seeds':[42],
                'selection_targets':{},'selection_rule':'fixed previous validation selection','fixed_preset':True,'jobs':[]}
            for scene in state['scenarios']:
                path=Path(tmp)/scene;path.mkdir()
                write_atomic(path/'training_summary.json',{'validation':[{'win_rate':.5,'mean_return':1.}]*3})
                state['jobs'].append({'kind':'candidate','scenario':scene,'seed':42,'preset':'belief_first','output':str(path)})
            choice=choose(state);self.assertEqual(choice['winner'],'belief_first')
            self.assertTrue(choice['fixed_before_training']);self.assertFalse(choice['heldout_used'])


@unittest.skipIf(torch is None or not ROOT.exists(),'Optional DIB backend absent')
class CompactResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def policy(self):
        roster=balanced_roster(3);env=SMACAdapter(config=EnvConfig(red=roster,blue=roster,max_steps=4,opponent='dummy'))
        self.addCleanup(env.close);env.reset(42);torch.manual_seed(42)
        p,_=build_policy(SimpleNamespace(dib_root=ROOT,training_profile='tactical_v2',tactical_preset='belief_first'),env)
        return p,env
    def batch(self,p,env):
        from algorithms.common.value_decomposition.train import observe,new_history
        o,s,a=observe(env);h=new_history(o,s,a);memory=initial_memory(p,1);rng=np.random.default_rng(42)
        while True:
            chosen,memory=act(p,o[None],a[None],memory,rng);r,done,_=env.step(chosen[0].tolist());o,s,a=observe(env)
            h['u'].append(chosen[0]);h['r'].append(r);h['o'].append(o);h['s'].append(s);h['avail'].append(a)
            if done:return episode_batch(p,[h]*4)
    def test_compact_snapshot_is_lossless_smaller_and_next_update_identical(self):
        p,env=self.policy();batch=self.batch(p,env);p.mac.set_residual_actor_gate_for_t(750000)
        learn(p,batch,750000,4);rng=np.random.default_rng(43);state={'trained':False,'steps':100}
        full=capture(p,rng,state,'same');compact=capture(p,rng,state,'same',compact=True)
        self.assertIs(compact['actor'],compact['old_actor']);self.assertIs(compact['critic'],compact['target_critic'])
        first,second=io.BytesIO(),io.BytesIO();torch.save(full,first);torch.save(compact,second)
        self.assertLess(len(second.getvalue()),len(first.getvalue()))
        fresh,_=self.policy();restore(fresh,rng,torch.load(io.BytesIO(second.getvalue())),'same')
        learn(p,batch,750000,8);learn(fresh,batch,750000,8)
        for key,value in p.mac.agent.state_dict().items():torch.testing.assert_close(value,fresh.mac.agent.state_dict()[key],atol=0,rtol=0)
    def test_only_finished_budget_may_omit_optimizer_state(self):
        p,_=self.policy();rng=np.random.default_rng(42)
        snapshot=capture(p,rng,{'trained':True,'steps':5000000},'same',compact=True)
        self.assertTrue(snapshot['evaluation_only']);self.assertNotIn('actor_optimizer',snapshot)
        fresh,_=self.policy();self.assertTrue(restore(fresh,rng,snapshot,'same')['trained'])
        snapshot['state']['trained']=False
        with self.assertRaises(ValueError):restore(fresh,rng,snapshot,'same')


if __name__=='__main__':unittest.main()
