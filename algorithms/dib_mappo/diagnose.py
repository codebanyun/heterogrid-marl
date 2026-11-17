"""Read-only local-policy behaviour audit on development episodes, never a trainer."""
import argparse
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.common.value_decomposition.train import observe
from algorithms.dib_mappo.backend import build_policy,initial_memory,act
from algorithms.dib_mappo.train import restore_checkpoint
from heterogrid import EnvConfig,SMACAdapter


def audit(source,episodes,base_seed):
    source=Path(source);cfg=EnvConfig.load(source/'env.json')
    saved=json.loads((source/'config.json').read_text())
    env=SMACAdapter(config=cfg)
    cli=SimpleNamespace(dib_root=Path('/root/research/DIB-MAPPO/src'),
        training_profile=saved['training_profile'],tactical_preset=saved['tactical_preset'])
    policy,_=build_policy(cli,env);restore_checkpoint(source/'models/final.pt',policy)
    rng=np.random.default_rng(base_seed);counts=collections.defaultdict(collections.Counter);rows=[]
    try:
        for index in range(episodes):
            env.reset(base_seed+index);memory=initial_memory(policy,1);done=False
            first_combat_death=None;last_combat_death=None;snapshots={};total=0.
            while not done:
                obs,_,available=observe(env)
                actions,memory=act(policy,obs[None],available[None],memory,rng,True)
                core=env.env.core
                for i,name in enumerate(core.names['red']):
                    unit=core.units[name]
                    if not unit.alive:continue
                    role=unit.spec.role;c=counts[role];a=int(actions[0,i]);c['ticks']+=1
                    c['wait']+=a==0;c['move_xy']+=1<=a<=4;c['move_z']+=5<=a<=6;c['attack']+=a>=7
                    opportunity=bool(available[i,7:].any());c['attack_opportunities']+=opportunity
                    c['attacks_when_available']+=opportunity and a>=7
                    c['x_sum']+=int(unit.position[0]);c['y_sum']+=int(unit.position[1]);c['z_sum']+=int(unit.position[2])
                reward,done,_=env.step(actions[0].tolist());total+=reward
                red=[core.units[n] for n in core.names['red']];fighters=[u for u in red if u.stats.damage>0]
                living=sum(u.alive for u in fighters)
                if living<len(fighters) and first_combat_death is None:first_combat_death=core.step_count
                if not living and last_combat_death is None:last_combat_death=core.step_count
                if core.step_count in (20,40,80,120,200) or done:
                    positions=np.asarray([u.position for u in red if u.alive])
                    snapshots[str(core.step_count)]={'alive':{role:sum(u.alive and u.spec.role==role for u in red) for role in ('scout','strike','hybrid')},
                        'mean_position':positions.mean(0).tolist() if len(positions) else None,
                        'spread':float(np.linalg.norm(positions-positions.mean(0),axis=-1).mean()) if len(positions) else None}
            rows.append({'seed':base_seed+index,'win':core.winner=='red','reason':core.reason,
                'length':core.step_count,'return':total,'first_combat_death':first_combat_death,
                'last_combat_death':last_combat_death,'snapshots':snapshots,
                'red':dict(core.episode_metrics['red']),'blue':dict(core.episode_metrics['blue'])})
    finally:env.close()
    roles={}
    for role,c in counts.items():
        roles[role]={**dict(c),'attack_take_rate':c['attacks_when_available']/max(1,c['attack_opportunities']),
            'wait_fraction':c['wait']/c['ticks'],'z_move_fraction':c['move_z']/c['ticks'],
            'mean_position':[c[k]/c['ticks'] for k in ('x_sum','y_sum','z_sum')]}
    return {'source':str(source),'protocol':'development diagnostics only; no actor receives simulator diagnostics',
        'base_seed':base_seed,'episodes':episodes,'win_rate':sum(r['win'] for r in rows)/len(rows),
        'role_actions':roles,'results':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--base-seed',type=int,default=910042)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.episodes<1:p.error('Positive episodes required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.output.exists():raise FileExistsError('Do not overwrite an audit')
    result=audit(a.source,a.episodes,a.base_seed);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
