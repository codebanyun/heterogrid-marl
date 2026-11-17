import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from heterogrid import EnvConfig, SMACAdapter, TaskSuite, TransferCapacity, TransferSMACAdapter, UnitSpec
from heterogrid.scenarios import balanced_roster
from heterogrid.transfer_scenarios import make_transfer_suite


def suite():
    base=EnvConfig(max_steps=4,target_sharing="auto")
    tasks={str(n):replace(base,red=balanced_roster(n),blue=balanced_roster(n)) for n in (3,6,9,12)}
    return TaskSuite(tasks,{"train":["3","6"],"validation":["9"],"test":["12"]})


class TransferTests(unittest.TestCase):
    def test_suite_roundtrip_validation_and_no_overwrite(self):
        s=suite()
        self.assertEqual(s.to_dict(),TaskSuite.from_dict(json.loads(json.dumps(s.to_dict()))).to_dict())
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"suite.json";s.save(p)
            self.assertEqual(s.to_dict(),TaskSuite.load(p).to_dict())
            with self.assertRaises(FileExistsError):s.save(p)
        with self.assertRaises(ValueError):TaskSuite({"a":EnvConfig()},{"train":["a"],"test":["a"]})
        with self.assertRaises(ValueError):TaskSuite({"a":EnvConfig()},{"train":["absent"]})
        with self.assertRaises(ValueError):TaskSuite({"a":EnvConfig()},{"train":["a"]},TransferCapacity(3,3))
        with self.assertRaises(ValueError):TaskSuite({"a":EnvConfig(),"b":EnvConfig(message_dim=4)},{"train":["a","b"]})
        for field in ("max_red","max_blue","message_dim"):
            with self.assertRaises(ValueError):TransferCapacity(**{**dict(max_red=12,max_blue=12,message_dim=8),field:True})

    def test_stable_spaces_switch_only_at_boundary_and_no_test_sampling(self):
        env=TransferSMACAdapter(suite())
        try:
            expected=env.get_env_info()
            for task_id in ("3","6","3"):
                obs,state=env.reset(seed=42,options={"task_id":task_id})
                self.assertEqual(env.get_env_info(),expected)
                self.assertEqual(np.asarray(obs).shape,(12,expected["obs_shape"]))
                self.assertEqual(state.shape,(expected["state_shape"],))
                self.assertEqual(env.get_active_mask().sum(),int(task_id))
                with self.assertRaises(RuntimeError):env.reset(options={"task_id":"6" if task_id=="3" else "3"})
                done=False
                while not done:_,done,_=env.step([0]*12)
            with self.assertRaises(ValueError):env.reset(options={"task_id":"12"})
            with self.assertRaises(ValueError):env.reset(options={"map_size":(50,50,6)})
            for _ in range(10):
                env.reset()
                self.assertIn(env.task_id,("3","6"))
                for _ in range(4):env.step([0]*12)
        finally:env.close()

    def test_padding_is_not_an_extra_unit_and_target_actions_are_masked(self):
        env=TransferSMACAdapter(suite())
        try:
            env.reset(seed=42,options={"task_id":"3"})
            entities=env.get_entity_obs_agent(0)
            self.assertEqual(entities["ally_slots"].sum(),2)
            self.assertEqual(entities["target_slots"].sum(),3)
            self.assertEqual(entities["targets"][3:].sum(),0)
            self.assertEqual(env.get_entity_obs_agent(4)["self_slot"][0],0)
            self.assertTrue(np.array_equal(env.get_obs_agent(4),np.zeros(env.get_obs_size())))
            self.assertEqual(env.get_avail_agent_actions(4),[1]+[0]*18)
            self.assertEqual(env.get_avail_agent_actions(0)[10:],[0]*9)
            actions=[0]*12;actions[4]=1
            with self.assertRaises(ValueError):env.step(actions)
            actions=[0]*12;actions[0]=10
            with self.assertRaises(ValueError):env.step(actions)
            self.assertEqual(len(env.env.core.all_names),6)
        finally:env.close()

    def test_hidden_enemy_features_are_not_taken_from_global_state(self):
        base=EnvConfig(red=[UnitSpec("strike")],blue=[UnitSpec("hybrid")],map_size=(40,40,6),opponent="dummy")
        s=TaskSuite({"a":base},{"train":["a"]},TransferCapacity(3,3))
        env=TransferSMACAdapter(s)
        try:
            env.reset(seed=42,options={"positions":{"red_0":[0,0,0],"blue_0":[39,39,5]}})
            before=env.get_obs_agent(0).copy()
            self.assertEqual(env.get_entity_obs_agent(0)["targets"].sum(),0)
            self.assertEqual(env.get_entity_obs_agent(0)["target_slots"].tolist(),[1,0,0])
            env.env.core.units["blue_0"].position[:]=[30,30,4]
            env.env.core.units["blue_0"].hp=1
            np.testing.assert_array_equal(before,env.get_obs_agent(0))
            self.assertEqual(env.get_active_mask()[0,0],1)
        finally:env.close()

    def test_same_seed_and_actions_preserve_underlying_legacy_game(self):
        s=suite();cfg=s.config("6")
        old,new=SMACAdapter(config=cfg),TransferSMACAdapter(s)
        rng=np.random.default_rng(19)
        try:
            old.reset(seed=42);new.reset(seed=42,options={"task_id":"6"})
            for _ in range(4):
                np.testing.assert_array_equal(old.get_state(),new.env.state())
                np.testing.assert_array_equal(old.get_avail_actions(),np.asarray(new.get_avail_actions())[:6,:13])
                for i in range(6):
                    data=new.get_entity_obs_agent(i);raw=old.env.core.observation("red_"+str(i))
                    np.testing.assert_array_equal(data["self"],raw["observation"][:12])
                    np.testing.assert_array_equal(data["targets"][:6].ravel(),raw["observation"][52:])
                actions=[int(rng.choice(np.flatnonzero(mask))) for mask in old.get_avail_actions()]
                a,b=old.step(actions),new.step(actions+[0]*6)
                self.assertEqual(a[:2],b[:2])
                self.assertEqual(a[2]["reward_components"],b[2]["reward_components"])
                if a[1]:break
        finally:old.close();new.close()

    def test_reproducible_random_task_and_world_sequence(self):
        a,b=TransferSMACAdapter(suite()),TransferSMACAdapter(suite())
        try:
            for episode in range(8):
                a.reset(seed=42 if episode==0 else None);b.reset(seed=42 if episode==0 else None)
                self.assertEqual(a.task_id,b.task_id)
                np.testing.assert_array_equal(a.get_state(),b.get_state())
                for _ in range(4):a.step([0]*12);b.step([0]*12)
        finally:a.close();b.close()

    def test_generators_preserve_rules_and_only_expected_factors(self):
        base=EnvConfig()
        for family in ("scale_fixed","scale_density","map","composition"):
            s=make_transfer_suite(base,family)
            for task_id in s.task_ids:
                c=s.config(task_id)
                self.assertEqual(c.opponent,base.opponent)
                self.assertEqual(c.role_stats,base.role_stats)
                self.assertEqual(c.communication_for("blue"),base.communication_for("blue"))
                self.assertEqual(c.win_reward,base.win_reward)
                self.assertEqual(c.max_steps,base.max_steps)
            self.assertFalse(set(s.ids("train")) & set(s.ids("test")))


if __name__=="__main__":unittest.main()
