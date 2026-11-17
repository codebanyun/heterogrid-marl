import tempfile
import unittest
from pathlib import Path
from algorithms.common.experiment_log import write_atomic
from algorithms.dib_mappo.tactical_campaign import choose


class TacticalSelectionTests(unittest.TestCase):
    def state(self,root):
        result={'presets':['policy_first','balanced'],'scenarios':['C1','C3'],'seeds':[42,43],
                'selection_targets':{'C1':.05,'C3':.75},'selection_rule':'validation only','jobs':[]}
        for profile in result['presets']:
            for scene in result['scenarios']:
                for seed in result['seeds']:
                    output=root/profile/scene/str(seed);output.mkdir(parents=True)
                    win=.04 if scene=='C1' else .6
                    if profile=='balanced':win*=.5
                    write_atomic(output/'training_summary.json',{'validation':[{'win_rate':win,'mean_return':-1.}]*3})
                    result['jobs'].append({'kind':'candidate','preset':profile,'scenario':scene,'seed':seed,'output':str(output)})
        return result
    def test_selection_ignores_heldout_scores_and_chooses_one_global_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=self.state(root)
            for job in state['jobs']:
                write_atomic(Path(job['output'])/'summary.json',{'heldout':{'fake_win_rate':1 if job['preset']=='balanced' else 0}})
            selected=choose(state)
            self.assertEqual(selected['winner'],'policy_first');self.assertFalse(selected['heldout_used'])
    def test_incomplete_preset_is_not_selected_from_successful_seeds_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=self.state(Path(tmp));state['jobs'].pop(0)
            self.assertEqual(choose(state)['winner'],'balanced')


if __name__=='__main__':unittest.main()
