"""Bounded integrated-preset search, validation-only choice, matched final tests."""
import argparse
import collections
import hashlib
import fcntl
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from algorithms.common.experiment_log import write_atomic,timestamp
from algorithms.common.sensitivity import PROJECT_ROOT,scenarios


def read(path):return json.loads(Path(path).read_text()) if Path(path).exists() else None


def prepare(a):
    from algorithms.dib_mappo.tactical import PRESETS

    root=a.output_root/'dib_mappo'/a.name;coordinator=a.output_root/'campaigns'/(a.name+'_dib_mappo')
    if root.exists() or coordinator.exists():raise FileExistsError('Fresh name required')
    if shutil.disk_usage(a.output_root.parent).free<4*1024**3:raise OSError('Need 4 GiB headroom')
    runtime=coordinator/'_runtime';backend=coordinator/'_backend';coordinator.mkdir(parents=True)
    for name in ['algorithms','heterogrid','configs']:
        shutil.copytree(PROJECT_ROOT/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['learners','controllers','modules','components','utils','config']:
        shutil.copytree(a.dib_root/name,backend/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    root.mkdir(parents=True);(root/'configs').mkdir()
    tasks=scenarios(True)
    for name in a.scenarios:write_atomic(root/'configs'/(name+'.json'),replace(tasks[name],diagnostics_enabled=True).to_dict())
    hashes={str(p.relative_to(coordinator)):hashlib.sha256(p.read_bytes()).hexdigest()
            for base in [runtime,backend] for p in sorted(base.rglob('*.py'))}
    jobs=[]
    for scene in a.scenarios:
        for seed in a.seeds:
            for preset in a.presets:
                out=root/preset/scene/('seed_'+str(seed))
                cmd=[sys.executable,'-u','-m','algorithms.dib_mappo.tactical_train','--dib-root',str(backend),
                     '--config',str(root/'configs'/(scene+'.json')),'--output',str(out),'--seed',str(seed),
                     '--tactical-preset',preset,'--steps',str(a.steps),'--eval-interval',str(a.eval_interval),
                     '--eval-episodes',str(a.eval_episodes),'--heldout-episodes',str(a.heldout_episodes),
                     '--training-profile',a.training_profile,'--eval-seed',str(a.eval_seed),
                     '--heldout-seed',str(a.heldout_seed)]
                jobs.append({'kind':'candidate','scenario':scene,'seed':seed,'preset':preset,
                             'output':str(out),'command':cmd,'status':'PENDING','pid':None})
    plan={'status':'PREPARED','stage':'TRAINING','manager_pid':None,'created':timestamp(),
        'runtime':str(runtime),'backend':str(backend),'root':str(root),'workers':a.workers,'jobs':jobs,
        'training_steps_per_run':a.steps,'requested_search_steps':a.steps*len(jobs),
        'presets':a.presets,'seeds':a.seeds,'scenarios':a.scenarios,'eval_episodes':a.eval_episodes,
        'heldout_episodes':a.heldout_episodes,'matched_baselines':a.matched_baselines,
        'baseline_root':str(a.baseline_root),'training_profile':a.training_profile,
        'reference_root':str(a.reference_root) if a.reference_root else None,
        'validation_seed':a.eval_seed,'heldout_seed':a.heldout_seed,'source_sha256':hashes,
        'selection_rule':'one global preset; mean capped(scene_win/target), then mean return; last 3 validation points; all seeds; no heldout access',
        'selection_targets':{'C1':.05,'C3':.75,'C5':.40,'C6':.05}}
    write_atomic(coordinator/'plan.json',plan);write_atomic(coordinator/'status.json',plan)
    write_atomic(root/'experiment.json',{**plan,'variant':a.training_profile,'not_upstream_equivalent':True,
        'hyperparameter_search_only':len(a.presets)>1,'search_parameter':{'tactical_aux_weight':{x:PRESETS[x] for x in a.presets}},
        'primary_model':'final.pt','best_model':'validation-only supplement',
        'teacher':'training-only full valid episode bidirectional context',
        'environment_changes':False,'action_changes':False,'reward_changes':False,'scripted_execution':False,
        'original_episodes_limit':200,'blue':'fixed rule, distance10, dropout0',
        'same_execution_information':'own observation/action/ID/history and legal mask only',
        'capacity_matching':False,'resume':'same-budget exact full boundary snapshots incl. Adam/RNG/counters',
        'sota_scope':'Only best among explicitly evaluated local baselines, not a universal literature SOTA claim'})
    (root/'README.md').write_text(
        f'# DIB {a.training_profile}：C组定向优化\n\n'
        f'场景{a.scenarios}，训练seed{a.seeds}，整合方案辅助强度{a.presets}，每次{a.steps:,}团队步。\n'
        f'搜索共{len(jobs)}次从头训练，显式总预算{a.steps*len(jobs):,}；不是单因素消融。\n'
        '环境、奖励、角色、蓝方策略、200步时限、通信与攻击掩码不变，无手写执行策略。\n'
        '新变体使用实体动作头、流式自适应历史、完整有效轨迹teacher、锚定候选和固定GAE优势。\n'
        f'只按验证{a.eval_seed}起选择一套跨场景统一preset，再对该配置的final/best做{a.heldout_seed}起独立测试。\n'
        '测试不参与选择；不能选各seed/场景测试最高的模型拼接。最终主报告final。\n'
        '默认每250k附近64局验证，独立每模型128局贪心+128局采样；参数搜索成本单列。\n'
        'selected符号链接在验证选择后生成；RESULTS.md在独立测试完成后更新。\n'
        '新检查点每50k附近完整rollout边界保存，含Adam、old actor、RNG与计数；支持同预算精确续训。\n'
        'canonical训练日志完成后为training_episodes.jsonl.gz；过程中封存于training_parts，废弃未提交记录保留不混入主数据。\n'
        '同一heldout集重新评估既有基准final模型，不改写旧模型或旧结果。\n'
        '使用冻结_runtime运行；中断后显式--resume-run，不安装系统自启动服务。\n'
        f'完整方法/配置和局限见仓库algorithms/dib_mappo/{"RESILIENT_V3.md" if a.training_profile=="resilient_v3" else "TACTICAL_V2.md"}。\n',encoding='utf-8')
    print(coordinator,flush=True);return coordinator


def choose(state):
    scores={}
    for preset in state['presets']:
        jobs=[x for x in state['jobs'] if x.get('preset')==preset and x['kind']=='candidate']
        expected={(scene,seed) for scene in state['scenarios'] for seed in state['seeds']}
        if len(jobs)!=len(expected) or {(j['scenario'],j['seed']) for j in jobs}!=expected:continue
        if not all(read(Path(j['output'])/'training_summary.json') for j in jobs):continue
        by_scene=collections.defaultdict(list);returns=[]
        for job in jobs:
            data=read(Path(job['output'])/'training_summary.json');last=data['validation'][-3:]
            by_scene[job['scenario']].append(statistics.mean(x['win_rate'] for x in last))
            returns.append(statistics.mean(x['mean_return'] for x in last))
        scene_means={name:statistics.mean(vals) for name,vals in by_scene.items()}
        score=statistics.mean(min(1.5,scene_means[x]/state['selection_targets'].get(x,1.)) for x in scene_means)
        scores[preset]={'score':score,'mean_return':statistics.mean(returns),'scene_validation':scene_means}
    if not scores:raise RuntimeError('No preset has a complete all-seed/all-scenario validation matrix')
    winner=max(scores,key=lambda p:(scores[p]['score'],scores[p]['mean_return']))
    return {'winner':winner,'scores':scores,'selected_at':timestamp(),'rule':state['selection_rule'],
            'fixed_before_training':bool(state.get('fixed_preset',False)),
            'heldout_used':False,'source_manifest':'experiment.json'}


def summary(state):
    root=Path(state['root']);selection=read(root/'selection.json')
    if not selection:return
    variant=state.get('training_profile','tactical_v2')
    algorithms=['dib_mappo_original','qmix','qtran','maa2c','coma']
    labels=['原DIB','QMIX','QTRAN','MAA2C','COMA']
    if state.get('reference_root'):algorithms.insert(0,'dib_tactical_v2');labels.insert(0,'DIB-v2')
    winner=selection['winner'];lines=['# DIB '+variant+' 独立测试','',
        ('配置在本轮训练前已冻结，不在新增场景重新选参：`' if state.get('fixed_preset') else
         '配置在访问独立测试前，由完整验证矩阵选定：`')+winner+'`。',
        f"所有数值为固定{state['training_steps_per_run']:,}步预算final模型的贪心胜率均值±SEM；训练seed为独立重复。",
        '容量/训练方法不匹配，不将均值领先解释为统计显著或全球SOTA。','',
        '| 场景 | 新DIB | '+' | '.join(labels)+' |','|---|'+ '---:|'*(len(labels)+1)]
    def number(paths):
        vals=[]
        for path,is_dib in paths:
            data=read(path)
            if data is None:return '未完成'
            vals.append(data['heldout']['final']['greedy']['win_rate'] if is_dib else data['win_rate'])
        if len(vals)<2:return '%.2f%% (n=%d)'%(100*statistics.mean(vals),len(vals))
        return '%.2f ± %.2f%%'%(100*statistics.mean(vals),100*statistics.stdev(vals)/math.sqrt(len(vals)))
    for scene in state.get('report_scenarios',state['scenarios']):
        values=[number([(root/winner/scene/('seed_'+str(seed))/'summary.json',True) for seed in state['seeds']])]
        for algo in algorithms:
            values.append(number([(root/'matched_baselines'/algo/scene/('seed_'+str(seed))/'summary.json',False) for seed in state['seeds']]))
        lines.append('| '+scene+' | '+' | '.join(values)+' |')
    lines+=['',f"搜索成本：{state['requested_search_steps']:,}团队训练步；评估步另计，不计入训练。",
            f"新旧模型统一测试{state['heldout_seed']}起的地图；不得把不同测试集的旧表直接当成配对结果。",'']
    if state.get('reuse_notes'):lines.extend(state['reuse_notes'])
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def supervise(coordinator,resuming=False):
    coordinator=Path(coordinator).resolve()
    lock=(coordinator/'manager.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another manager owns this campaign; duplicate launch refused')
    state=read(coordinator/'status.json')
    if not resuming and state['status']!='PREPARED':raise ValueError('Use --resume-run for existing campaigns')
    for relative,digest in state['source_sha256'].items():
        if hashlib.sha256((coordinator/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+relative)
    if resuming:
        for job in state['jobs']:
            pid=job.get('pid');file=Path('/proc/'+str(pid)+'/cmdline')
            if file.exists() and job['output'].encode() in file.read_bytes():raise RuntimeError('Worker already alive; no duplicate launch')
            progress=read(Path(job['output'])/'progress.json') or {}
            if job['kind']=='candidate' and read(Path(job['output'])/'training_summary.json'):job['status']='TRAINED'
            if job['kind']=='evaluation' and read(Path(job['output'])/'summary.json') and progress.get('status')=='COMPLETED':job['status']='COMPLETED'
            if job['kind']=='baseline' and read(Path(job['output'])/'summary.json') and read(Path(job['output'])/'provenance.json'):job['status']='COMPLETED'
            if job['status'] in ('RUNNING','INTERRUPTED'):
                if job['kind']!='baseline' and not (Path(job['output'])/'resume.pt').exists():
                    job['status']='FAILED';job['error']='No exact checkpoint; refuse weights-only continuation'
                else:job['status']='PENDING'
    state.update(manager_pid=os.getpid(),status='RUNNING',heartbeat=timestamp())
    live={};stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        for process,_,_ in live.values():
            if process.poll() is None:process.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    env.pop('PYTHONPATH',None)
    crashed=False
    try:
        while True:
            if shutil.disk_usage(coordinator).free<2*1024**3:state['stop_reason']='2 GiB disk reserve';stop()
            for key,(process,stream,job) in list(live.items()):
                result=process.poll()
                if result is None:
                    job['progress']=read(Path(job['output'])/'progress.json');continue
                stream.close();job['returncode']=result
                if result==0:job['status']='TRAINED' if job['kind']=='candidate' else 'COMPLETED'
                else:job['status']='INTERRUPTED' if stopping or result==130 else 'FAILED'
                job['finished']=timestamp();print(job['kind'],job.get('preset'),job['scenario'],job['seed'],job['status'],flush=True);del live[key]
            failures=sum(j['status']=='FAILED' for j in state['jobs'])
            if failures>=3 and not stopping:state['stop_reason']='three failures; inspect before continuing';stop()
            if stopping and not live:break
            pending=[j for j in state['jobs'] if j['status']=='PENDING']
            if not stopping:
                for job in pending[:max(0,state['workers']-len(live))]:
                    out=Path(job['output']);out.parent.mkdir(parents=True,exist_ok=True)
                    command=list(job['command'])
                    if (out/'resume.pt').exists() and '--resume' not in command:command.append('--resume')
                    stream=(out.parent/('seed_'+str(job['seed'])+'.log')).open('a',buffering=1)
                    process=subprocess.Popen(command,cwd=state['runtime'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    job.update(pid=process.pid,status='RUNNING',started=timestamp());live[process.pid]=(process,stream,job)
                    print('Started',job['kind'],job.get('preset'),job['scenario'],job['seed'],process.pid,flush=True)
            if not live and not any(j['status']=='PENDING' for j in state['jobs']):
                if stopping:break
                if state['stage']=='TRAINING':
                    root=Path(state['root']);selection=read(root/'selection.json') or choose(state)
                    write_atomic(root/'selection.json',selection)
                    selected=root/'selected'
                    if selected.is_symlink():
                        if os.readlink(selected)!=selection['winner']:raise RuntimeError('Existing selected preset differs')
                    else:selected.symlink_to(selection['winner'],target_is_directory=True)
                    originals=[j for j in state['jobs'] if j['kind']=='candidate' and j['preset']==selection['winner']]
                    for j in originals:
                        state['jobs'].append({**{k:v for k,v in j.items() if k not in ('progress','returncode','finished')},
                            'kind':'evaluation','command':j['command']+['--resume','--evaluate-only'],'status':'PENDING','pid':None})
                    if state['matched_baselines']:
                        # Read-only reuse of the pre-existing four baseline final models.
                        source_output=Path(state['baseline_root'])
                        algorithms=['qmix','qtran','maa2c','coma','dib_mappo_original']
                        if state.get('reference_root'):algorithms.append('dib_tactical_v2')
                        for algo in algorithms:
                            for scene in state['scenarios']:
                                for seed in state['seeds']:
                                    folder='dib_mappo' if algo=='dib_mappo_original' else algo
                                    experiment='20261018_fixed_blue_sensitivity_5m' if folder=='dib_mappo' else '20261013_fixed_blue_sensitivity_5m'
                                    source=source_output/folder/experiment/scene/('seed_'+str(seed))
                                    if algo=='dib_tactical_v2':source=Path(state['reference_root'])/scene/('seed_'+str(seed))
                                    out=root/'matched_baselines'/algo/scene/('seed_'+str(seed))
                                    if not (source/'models/final.pt').exists():
                                        state.setdefault('missing_baselines',[]).append({'algorithm':algo,'scenario':scene,'seed':seed,'reason':'No final checkpoint; not imputed as zero'})
                                        continue
                                    command=[sys.executable,'-u','-m','algorithms.dib_mappo.matched_eval','--source',str(source),
                                        '--config',str(root/'configs'/(scene+'.json')),'--output',str(out),
                                        '--episodes',str(state['heldout_episodes']),'--seed',str(state['heldout_seed'])]
                                    state['jobs'].append({'kind':'baseline','scenario':scene,'seed':seed,'algorithm':algo,
                                        'output':str(out),'command':command,'status':'PENDING','pid':None})
                    state['stage']='EVALUATION';state['selection']=selection
                else:break
            state['heartbeat']=timestamp();write_atomic(coordinator/'status.json',state)
            write_atomic(Path(state['root'])/'status.json',{'status':state['status'],'stage':state['stage'],
                'manager_pid':os.getpid(),'heartbeat':state['heartbeat'],
                'counts':dict(collections.Counter(j['status'] for j in state['jobs'])),'coordinator':str(coordinator)})
            if state['stage']=='EVALUATION':summary(state)
            time.sleep(5)
    except BaseException as error:
        crashed=True;state['error']=repr(error);stop();raise
    finally:
        state['status']='FAILED' if crashed else 'INTERRUPTED' if stopping else 'FAILED' if any(j['status']=='FAILED' for j in state['jobs']) else 'COMPLETED'
        state['heartbeat']=timestamp();write_atomic(coordinator/'status.json',state)
        write_atomic(Path(state['root'])/'status.json',{'status':state['status'],'stage':state['stage'],
            'manager_pid':os.getpid(),'heartbeat':state['heartbeat'],
            'counts':dict(collections.Counter(j['status'] for j in state['jobs'])),'coordinator':str(coordinator)})
        summary(state)
        lock.close()
    return 130 if stopping else int(state['status']=='FAILED')


def main():
    from algorithms.dib_mappo.tactical import PRESETS

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--resume-run',type=Path)
    p.add_argument('--name',default='20261103_tactical_v2_search_5m')
    p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'output')
    p.add_argument('--dib-root',type=Path,default=Path('/root/research/DIB-MAPPO/src'))
    p.add_argument('--workers',type=int,default=36)
    p.add_argument('--steps',type=int,default=5000000)
    p.add_argument('--eval-interval',type=int,default=250000)
    p.add_argument('--eval-episodes',type=int,default=64)
    p.add_argument('--heldout-episodes',type=int,default=128)
    p.add_argument('--training-profile',choices=('tactical_v2','resilient_v3'),default='tactical_v2')
    p.add_argument('--eval-seed',type=int,default=700042)
    p.add_argument('--heldout-seed',type=int,default=800042)
    p.add_argument('--scenarios',nargs='+',choices=('C1','C3','C5','C6'),default=['C1','C3','C5','C6'])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--presets',nargs='+',choices=tuple(PRESETS),default=list(PRESETS))
    p.add_argument('--matched-baselines',action='store_true')
    p.add_argument('--baseline-root',type=Path,default=PROJECT_ROOT/'output')
    p.add_argument('--reference-root',type=Path,help='Optional read-only Tactical-v2 preset directory to re-evaluate')
    a=p.parse_args()
    if a.run:return supervise(a.run)
    if a.resume_run:return supervise(a.resume_run,True)
    if min(a.workers,a.steps,a.eval_interval,a.eval_episodes,a.heldout_episodes)<1:p.error('Positive budgets required')
    if max(a.eval_seed,a.heldout_seed)<min(a.eval_seed+a.eval_episodes,a.heldout_seed+a.heldout_episodes):
        p.error('Validation and independent tests must not overlap')
    if a.reference_root:
        a.reference_root=a.reference_root.resolve()
        if not a.reference_root.is_dir():p.error('Reference directory does not exist')
    for field in ['seeds','scenarios','presets']:
        if len(set(getattr(a,field)))!=len(getattr(a,field)):p.error('Duplicate '+field)
    a.output_root=a.output_root.resolve();a.dib_root=a.dib_root.resolve();a.baseline_root=a.baseline_root.resolve();prepare(a);return 0


if __name__=='__main__':sys.exit(main())
