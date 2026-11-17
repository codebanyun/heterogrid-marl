"""Fresh fixed-budget training, boundary-exact recovery, deferred heldout tests."""
import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT
from algorithms.common.value_decomposition.train import observe,new_history,episode_row
from algorithms.dib_mappo.backend import build_policy,initial_memory,act,select_memory,assign_memory,episode_batch
from algorithms.dib_mappo.sequence import learn
from algorithms.dib_mappo.train import save_checkpoint,restore_checkpoint,evaluate
from algorithms.dib_mappo.resume import capture,restore,save,fingerprint
from algorithms.dib_mappo.tactical import PRESETS
from heterogrid import EnvConfig,SMACAdapter
from heterogrid.core import REWARD_COMPONENTS


def source_hash():
    digest=hashlib.sha256()
    for directory in ('algorithms','heterogrid'):
        for file in sorted((PROJECT_ROOT/directory).rglob('*.py')):
            digest.update(str(file.relative_to(PROJECT_ROOT)).encode());digest.update(file.read_bytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dib-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--tactical-preset',choices=tuple(PRESETS),default='balanced')
    parser.add_argument('--training-profile',choices=('tactical_v2','resilient_v3'),default='tactical_v2')
    parser.add_argument('--steps',type=int,default=5000000)
    parser.add_argument('--eval-interval',type=int,default=250000)
    parser.add_argument('--eval-episodes',type=int,default=64)
    parser.add_argument('--heldout-episodes',type=int,default=128)
    parser.add_argument('--eval-seed',type=int,default=700042)
    parser.add_argument('--heldout-seed',type=int,default=800042)
    parser.add_argument('--checkpoint-interval',type=int,default=50000)
    parser.add_argument('--storage-policy',choices=('full','compact'),default='full',
        help='Compact: one atomic exact training snapshot; eval-only state after budget; no duplicate episode log')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--stop-after',type=int,help='Functional recovery test only; not a shorter training budget')
    cli=parser.parse_args();cli.algorithm='dib_mappo'
    if min(cli.steps,cli.eval_interval,cli.eval_episodes,cli.heldout_episodes,cli.checkpoint_interval)<1:
        parser.error('Positive budgets required')
    if cli.evaluate_only and not cli.resume:parser.error('Heldout evaluation requires a saved trained run')
    if max(cli.eval_seed,cli.heldout_seed)<min(cli.eval_seed+cli.eval_episodes,cli.heldout_seed+cli.heldout_episodes):
        parser.error('Validation and heldout seeds overlap')
    cli.output=cli.output.resolve();cli.config=cli.config.resolve();cli.dib_root=cli.dib_root.resolve()
    random.seed(cli.seed);np.random.seed(cli.seed);torch.manual_seed(cli.seed)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=EnvConfig.load(cli.config)
    num_envs=8 if cli.training_profile=='resilient_v3' else 4
    envs=[SMACAdapter(config=cfg) for _ in range(num_envs)]
    policy,args=build_policy(cli,envs[0]);rng=np.random.default_rng(cli.seed)
    code=source_hash()
    identity=fingerprint(args,cfg,code+json.dumps({'seed':cli.seed,'budget':cli.steps,
        'eval_interval':cli.eval_interval,'eval_episodes':cli.eval_episodes,'heldout_episodes':cli.heldout_episodes,
        'eval_seed':cli.eval_seed,'heldout_seed':cli.heldout_seed,'storage_policy':cli.storage_policy},sort_keys=True))
    checkpoint=cli.output/'resume.pt'
    state={'steps':0,'updates':0,'episodes':0,'episode_index':0,'learned_steps':0,
        'train_seconds':0.,'eval_seconds':0.,'active_wall_seconds':0.,'validation':[],
        'metrics':[],'segments':[],'best_score':[-1.,-1e30],'best_step':0,
        'last_checkpoint':0,'next_validation':cli.eval_interval,'next_model_save':2000000,
        'next_metric':10000,'trained':False,'heldout':{},'phase':'INITIALIZING'}
    if cli.resume:
        if not checkpoint.exists():raise FileNotFoundError('No exact training checkpoint; weights alone are not resumable')
        state=restore(policy,rng,torch.load(checkpoint,map_location='cpu'),identity)
    else:
        cli.output.mkdir(parents=True,exist_ok=False)
        (cli.output/'models').mkdir();(cli.output/'training_parts').mkdir()
        write_atomic(cli.output/'env.json',cfg.to_dict())
        write_atomic(cli.output/'config.json',{'name':'dib_'+cli.training_profile,'algorithm':'dib_mappo',
            'variant':cli.training_profile,'training_profile':cli.training_profile,'tactical_preset':cli.tactical_preset,
            'cli':{k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()},
            'num_envs':num_envs,
            'storage_policy':cli.storage_policy,
            'seed':cli.seed,'environment':cfg.to_dict(),'algorithm_args':vars(args),
            'source_sha256':code,'resume_identity':identity,'budget':cli.steps,
            'validation_seed':cli.eval_seed,'heldout_seed':cli.heldout_seed,
            'validation_episodes':cli.eval_episodes,'heldout_episodes':cli.heldout_episodes,
            'fresh_training':True,'reward_shaping':False,'expert_policy':False,
            'execution_information':'own local obs/history/action/ID only; masks unchanged',
            'teacher_context':'training-only full valid episode, bidirectional; not original 8+4 window',
            'resume':'model/old-model/Adam/RNG/counters at complete-rollout boundaries; no weights-only restart',
            'budget_count':'team environment transitions, excludes all evaluation; final incomplete batch not learned',
            'versions':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version}})
    attempt=str(time.time_ns());wall=time.monotonic();previous_wall=state['active_wall_seconds']
    interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    metric_path=cli.output/'metrics.jsonl'
    if cli.resume and metric_path.exists():
        archive=cli.output/'recovery_archive';archive.mkdir(exist_ok=True)
        metric_path.rename(archive/('metrics_before_'+attempt+'.jsonl'))
    metric_file=metric_path.open('w',buffering=1)
    for row in state['metrics']:metric_file.write(json.dumps(row)+'\n')
    segment=None;segment_name=None;segment_rows=0
    def open_segment():
        nonlocal segment,segment_name,segment_rows
        segment_name='training_parts/part_%09d_%s.jsonl.gz'%(state['steps'],attempt)
        segment=gzip.open(cli.output/segment_name,'wt',compresslevel=3);segment_rows=0
    def seal_segment():
        nonlocal segment,segment_rows
        if segment is not None:
            segment.close();segment=None
            if segment_rows:state['segments'].append(segment_name)
    def progress(status='RUNNING'):
        value={'status':status,'pid':os.getpid(),'phase':state['phase'],'seed':cli.seed,
            'environment_steps':state['steps'],'target_steps':cli.steps,'updates':state['updates'],
            'training_episodes':state['episodes'],'learned_environment_steps':state['learned_steps'],
            'training_seconds':state['train_seconds'],'evaluation_seconds':state['eval_seconds'],
            'environment_steps_per_second':state['steps']/max(state['train_seconds'],1e-9),
            'active_wall_seconds':previous_wall+time.monotonic()-wall,
            'tactical_preset':cli.tactical_preset,'last_exact_checkpoint':state['last_checkpoint']}
        write_atomic(cli.output/'progress.json',value);write_atomic(cli.output/'run.json',value)
    def checkpoint_now(reopen=True):
        seal_segment();metric_file.flush();state['last_checkpoint']=state['steps']
        state['active_wall_seconds']=previous_wall+time.monotonic()-wall
        snap=capture(policy,rng,state,identity,compact=cli.storage_policy=='compact')
        if checkpoint.exists() and cli.storage_policy=='full':shutil.copy2(checkpoint,cli.output/'resume_previous.pt')
        save(checkpoint,snap)
        if reopen:open_segment()
    def validate(milestone):
        state['phase']='VALIDATION';progress();start=time.monotonic()
        path=cli.output/'evaluations'/('attempt_'+attempt)/('validation_%09d'%state['steps'])
        result=evaluate(policy,cfg,path,cli.eval_episodes,cli.eval_seed)
        state['eval_seconds']+=time.monotonic()-start
        state['validation'].append({'milestone':milestone,'step':state['steps'],
            'win_rate':result['win_rate'],'mean_return':result['mean_return'],'path':str(path.relative_to(cli.output))})
        score=[result['win_rate'],result['mean_return']]
        if tuple(score)>tuple(state['best_score']):
            state['best_score']=score;state['best_step']=state['steps']
            save_checkpoint(cli.output/'models/best.pt',policy,cli,cfg,state['steps'],state['updates'])
        state['phase']='TRAINING'
    try:
        if not cli.resume:
            if cli.storage_policy=='full':save_checkpoint(cli.output/'models/initial.pt',policy,cli,cfg,0,0)
            validate(0);checkpoint_now()
        elif not state['trained']:open_segment()
        while not state['trained'] and state['steps']<cli.steps:
            if interrupted:break
            started=time.monotonic();state['phase']='TRAINING'
            policy.batch_start_step=state['steps'];policy.mac.set_residual_actor_gate_for_t(state['steps'])
            memories=initial_memory(policy,num_envs);hist=[];obs=[];available=[];totals=[0.]*num_envs
            components=[dict.fromkeys(REWARD_COMPONENTS,0.) for _ in range(num_envs)];seeds=[]
            for i,e in enumerate(envs):
                seed=1000000+cli.seed*100000000+state['episode_index'];state['episode_index']+=1
                seeds.append(seed);e.reset(seed=seed);o,s,a=observe(e);hist.append(new_history(o,s,a));obs.append(o);available.append(a)
            active=list(range(num_envs))
            while active and state['steps']<cli.steps:
                slots=active[:cli.steps-state['steps']]
                actions,memory=act(policy,np.asarray(obs)[slots],np.asarray(available)[slots],select_memory(memories,slots),rng)
                assign_memory(memories,slots,memory);finished=[]
                for j,i in enumerate(slots):
                    reward,done,info=envs[i].step(actions[j].tolist());o,s,a=observe(envs[i])
                    if any(x.startswith('red_') for x in info['invalid_actions']):raise RuntimeError('Invalid learned action')
                    h=hist[i];h['u'].append(actions[j]);h['r'].append(reward);h['o'].append(o);h['s'].append(s);h['avail'].append(a)
                    obs[i]=o;available[i]=a;totals[i]+=reward
                    for key in components[i]:components[i][key]+=info['reward_components'][key]
                    state['steps']+=1
                    if done:
                        finished.append(i);state['episodes']+=1
                        row={**episode_row(envs[i],seeds[i],totals[i],components[i]),'environment_steps':state['steps']}
                        segment.write(json.dumps(row)+'\n');segment_rows+=1
                active=[i for i in active if i not in finished]
            if not active:
                batch=episode_batch(policy,hist);learn(policy,batch,policy.batch_start_step,state['episodes'])
                state['updates']+=1;state['learned_steps']+=sum(len(h['u']) for h in hist)
            state['train_seconds']+=time.monotonic()-started
            if state['steps']>=state['next_metric'] or state['steps']==cli.steps:
                row={'step':state['steps'],**policy.logger.values,'updates':state['updates'],
                     'training_seconds':state['train_seconds'],'evaluation_seconds':state['eval_seconds']}
                state['metrics'].append(row);metric_file.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                state['next_metric']=(state['steps']//10000+1)*10000
            if state['steps']>=state['next_model_save']:
                save_checkpoint(cli.output/'models'/('step_%09d.pt'%state['steps']),policy,cli,cfg,state['steps'],state['updates'])
                state['next_model_save']+=2000000
            if state['steps']>=state['next_validation'] or state['steps']==cli.steps:
                milestone=min(state['next_validation'],cli.steps);validate(milestone)
                state['next_validation']=(state['steps']//cli.eval_interval+1)*cli.eval_interval
            if cli.stop_after and state['steps']>=cli.stop_after:interrupted=True
            if state['steps']-state['last_checkpoint']>=cli.checkpoint_interval or interrupted:
                checkpoint_now(reopen=not interrupted)
            progress()
        if interrupted and not state['trained'] and state['steps']<cli.steps:
            checkpoint_now(False);progress('INTERRUPTED');return 130
        if not state['trained']:
            save_checkpoint(cli.output/'models/final.pt',policy,cli,cfg,state['steps'],state['updates'])
            state['trained']=True;state['phase']='TRAINED';checkpoint_now(False)
            write_atomic(cli.output/'training_summary.json',{k:v for k,v in state.items() if k not in ('metrics','heldout')})
        if not cli.evaluate_only:progress('TRAINED');return 0
        # Validation selection is performed by the parent before scheduling this.
        state['phase']='HELDOUT';progress()
        for model in ('final','best'):
            state['heldout'].setdefault(model,{})
            for mode in ('greedy','sampled'):
                if mode in state['heldout'][model]:continue
                restore_checkpoint(cli.output/'models'/(model+'.pt'),policy)
                started=time.monotonic()
                path=cli.output/'evaluations'/('attempt_'+attempt)/('heldout_'+model+'_'+mode)
                state['heldout'][model][mode]=evaluate(policy,cfg,path,cli.heldout_episodes,cli.heldout_seed,mode=='greedy')
                state['eval_seconds']+=time.monotonic()-started
                restore_checkpoint(cli.output/'models/final.pt',policy)
                checkpoint_now(False)
        xs=np.asarray([v['step'] for v in state['validation']]);ys=np.asarray([v['win_rate'] for v in state['validation']])
        reached=[v['step'] for v in state['validation'] if v['win_rate']>=.8]
        summary={'algorithm':'dib_mappo','variant':cli.training_profile,'tactical_preset':cli.tactical_preset,
            'seed':cli.seed,'environment_steps':state['steps'],'learned_environment_steps':state['learned_steps'],
            'updates':state['updates'],'training_episodes':state['episodes'],'validation':state['validation'],
            'best_validation_step':state['best_step'],'validation_auc_normalized':float(np.trapz(ys,xs)/cli.steps),
            'first_80_percent_step':min(reached) if reached else None,'heldout':state['heldout'],
            'training_seconds':state['train_seconds'],'evaluation_seconds':state['eval_seconds'],
            'environment_steps_per_second':state['steps']/state['train_seconds']}
        write_atomic(cli.output/'summary.json',summary)
        # Concatenated gzip members remain readable by gzip.open; no rows are
        # reconstructed from abandoned, uncommitted segments after a crash.
        if cli.storage_policy=='compact':
            write_atomic(cli.output/'training_episodes.manifest.json',{
                'format':'ordered gzip JSONL members','parts':state['segments'],
                'episodes':state['episodes'],'environment_steps':state['steps'],
                'read_instruction':'Read parts in order with gzip.open; no episodes are omitted or duplicated'})
        else:
            with (cli.output/'training_episodes.jsonl.gz').open('wb') as target:
                for part in state['segments']:
                    with (cli.output/part).open('rb') as source:shutil.copyfileobj(source,target)
        state['phase']='FINISHED';checkpoint_now(False);progress('COMPLETED');return 0
    except BaseException:
        state['phase']='FAILED';progress('FAILED');raise
    finally:
        seal_segment();metric_file.close()
        for e in envs:e.close()


if __name__=='__main__':sys.exit(main())
