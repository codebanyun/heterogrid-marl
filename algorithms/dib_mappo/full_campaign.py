"""Extend the exact selected Tactical-v2 implementation to B/M/N/C2/C4/D.

Reuse C3/C5 results and D0=B0 without training duplicates. C1/C6 remain in the
separate Resilient-v3 study, never mixed into this version's result table.
"""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from algorithms.common.experiment_log import write_atomic,timestamp
from algorithms.common.sensitivity import PROJECT_ROOT,scenarios
from algorithms.dib_mappo.tactical_campaign import supervise
from heterogrid import EnvConfig
from heterogrid.core import GridCombat
from heterogrid.scenarios import scaling_scenarios

NEW_SCENARIOS=('B0','M1','M2','M3','N1','N2','N3','C2','C4','D1','D2','D3')
REPORT_SCENARIOS=('B0','M1','M2','M3','N1','N2','N3','C2','C3','C4','C5','D0','D1','D2','D3')
ALGORITHM_FILES=('backend.py','tactical.py','robust.py','sequence.py','auxiliary.py',
                 'alive.py','returns.py','train.py')
RUNNER_OVERLAYS=('full_campaign.py','tactical_campaign.py','tactical_train.py','resume.py','matched_eval.py')


def read(path):return json.loads(Path(path).read_text())


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_matrix():
    tasks={name:replace(cfg,diagnostics_enabled=True) for name,cfg in scenarios(True).items()}
    base=replace(tasks['B0'],reward_normalization='reference',reward_reference_team_size=6,
        reward_reference_team_hp=480.)
    scaled=scaling_scenarios(base,mode='constant_density')
    for name,count in [('D0',6),('D1',3),('D2',9),('D3',12)]:tasks[name]=scaled[f'{count}v{count}']
    assert_d0_equivalent(tasks['B0'],tasks['D0'])
    return tasks


def assert_d0_equivalent(baseline,d0):
    a,b=baseline.to_dict(),d0.to_dict()
    allowed={'reward_normalization','reward_reference_team_size','reward_reference_team_hp'}
    if {key for key in a if a[key]!=b[key]}-allowed:raise ValueError('D0 changes baseline task')
    if len(baseline.red)!=6 or len(baseline.blue)!=6:raise ValueError('D0 requires 6v6')
    for config in (baseline,d0):
        if any(v!=1. for team in GridCombat(config).reward_scales.values() for v in team.values()):
            raise ValueError('D0 effective reward coefficients differ')


def prepare(a):
    source=a.source.resolve();coordinator=a.output_root/'campaigns'/(a.name+'_dib_mappo')
    root=a.output_root/'dib_mappo'/a.name
    if coordinator.exists() or root.exists():raise FileExistsError('Fresh experiment name required')
    if shutil.disk_usage(a.output_root.parent).free<4*1024**3:raise OSError('Need 4 GiB before creating full campaign')
    old=read(source/'experiment.json');selection=read(source/'selection.json')
    if selection['winner']!='belief_first' or old['variant']!='tactical_v2':raise ValueError('Expected selected original Tactical-v2 belief_first')
    previous=Path(old['runtime']).parent;frozen=previous/'_runtime';backend_source=previous/'_backend'
    for path,sha in old['source_sha256'].items():
        if digest(previous/path)!=sha:raise ValueError('Previous frozen source changed: '+path)
    tasks=task_matrix();preset='belief_first'
    reused={}
    for scene in ('C3','C5'):
        path=source/preset/scene
        if tasks[scene].to_dict()!=EnvConfig.load(source/'configs'/(scene+'.json')).to_dict():raise ValueError('Reused scene environment mismatch')
        reused[scene]={'source':str(path),'training_repeated':False,'runs':[]}
        for seed in a.seeds:
            run=path/('seed_'+str(seed));s=read(run/'summary.json');test=s['heldout']['final']['greedy']
            if s['environment_steps']!=5000000:raise ValueError('Reused run did not finish 5M')
            if not a.preflight and (test['base_seed']!=800042 or test['episodes']!=128):raise ValueError('Reused independent-test protocol differs')
            reused[scene]['runs'].append({'seed':seed,'summary_sha256':digest(run/'summary.json'),
                'final_sha256':digest(run/'models/final.pt'),'original_budget':5000000})
    coordinator.mkdir(parents=True);root.mkdir(parents=True);(root/'configs').mkdir()
    runtime=coordinator/'_runtime';backend=coordinator/'_backend'
    # Preserve every computational module from the exact successful experiment;
    # replace only orchestration/checkpoint serialization, never the algorithm.
    shutil.copytree(frozen,runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(backend_source,backend,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in RUNNER_OVERLAYS:shutil.copy2(PROJECT_ROOT/'algorithms/dib_mappo'/name,runtime/'algorithms/dib_mappo'/name)
    same={name:digest(runtime/'algorithms/dib_mappo'/name) for name in ALGORITHM_FILES}
    for name,sha in same.items():
        if sha!=digest(frozen/'algorithms/dib_mappo'/name):raise ValueError('Algorithm must remain identical')
    for name in REPORT_SCENARIOS:write_atomic(root/'configs'/(name+'.json'),tasks[name].to_dict())
    (root/preset).mkdir()
    for scene in ('C3','C5'):(root/preset/scene).symlink_to(source/preset/scene,target_is_directory=True)
    (root/preset/'D0').symlink_to('B0',target_is_directory=True)
    reused['D0']={'source':'belief_first/B0','training_repeated':False,'independent_repeat':False,
        'reason':'6v6 reference normalization has unit damage/kill scales; all dynamics/observations/diagnostics identical',
        'declared_config':'configs/D0.json','data_config':'configs/B0.json'}
    write_atomic(root/'reused_results.json',reused)
    # Reuse old paired baselines only when the same 800042 test protocol is used.
    for algo in ['qmix','qtran','maa2c','coma','dib_mappo_original']:
        dest=root/'matched_baselines'/algo;dest.mkdir(parents=True)
        for scene in ('C3','C5'):
            origin=source/'matched_baselines'/algo/scene
            if origin.is_dir():(dest/scene).symlink_to(origin,target_is_directory=True)
        (dest/'D0').symlink_to('B0',target_is_directory=True)
    jobs=[]
    for scene in NEW_SCENARIOS:
        for seed in a.seeds:
            out=root/preset/scene/('seed_'+str(seed))
            cmd=[sys.executable,'-u','-m','algorithms.dib_mappo.tactical_train','--dib-root',str(backend),
                '--config',str(root/'configs'/(scene+'.json')),'--output',str(out),'--seed',str(seed),
                '--training-profile','tactical_v2','--tactical-preset',preset,'--storage-policy','compact',
                '--steps',str(a.steps),'--eval-interval',str(a.eval_interval),'--eval-episodes',str(a.eval_episodes),
                '--heldout-episodes',str(a.heldout_episodes),'--eval-seed','700042','--heldout-seed','800042']
            jobs.append({'kind':'candidate','scenario':scene,'seed':seed,'preset':preset,
                'output':str(out),'command':cmd,'status':'PENDING','pid':None})
    hashes={str(p.relative_to(coordinator)):digest(p) for base in (runtime,backend) for p in sorted(base.rglob('*.py'))}
    notes=['C3/C5直接复用20261103已选定belief_first的完整5M结果，不重新训练、不重新挑seed。',
        'D0链接B0：参考归一化系数均为1，动力学/奖励/诊断等价，不算额外独立训练重复。',
        'C1/C6本轮排除；Resilient-v3在独立目录进行，不混入DIB-v2结果。',
        '此为已公开C3/C5测试结果之后的固定配置扩展，不能当成覆盖所有场景的全新盲测方法选择。']
    if a.preflight:notes.insert(0,'仅流水线预检；新短任务与复用5M任务预算不同，不作性能比较。')
    state={'status':'PREPARED','stage':'TRAINING','manager_pid':None,'created':timestamp(),
        'runtime':str(runtime),'backend':str(backend),'root':str(root),'workers':a.workers,'jobs':jobs,
        'training_profile':'tactical_v2','training_steps_per_run':a.steps,'requested_search_steps':a.steps*len(jobs),
        'presets':[preset],'fixed_preset':True,'seeds':a.seeds,'scenarios':list(NEW_SCENARIOS),
        'report_scenarios':list(REPORT_SCENARIOS),'eval_episodes':a.eval_episodes,'heldout_episodes':a.heldout_episodes,
        'validation_seed':700042,'heldout_seed':800042,'matched_baselines':a.matched_baselines,
        'baseline_root':str(a.baseline_root.resolve()),'reference_root':None,'source_sha256':hashes,
        'selection_rule':'belief_first .30 fixed from previous validation selection before extension; no new hyperparameter search',
        'selection_targets':{},'reuse_notes':notes,'reused_results':reused,
        'excluded_scenarios':{'C1':'separate v3 study','C6':'separate v3 study'},
        'reproduction_source':str(source),'previous_selection_sha256':digest(source/'selection.json'),
        'unchanged_algorithm_sha256':same,'runner_overlays':list(RUNNER_OVERLAYS),'preflight_only':a.preflight,
        'storage_policy':'compact','disk_free_at_prepare':shutil.disk_usage(root).free}
    write_atomic(coordinator/'plan.json',state);write_atomic(coordinator/'status.json',state)
    write_atomic(root/'experiment.json',{**state,'variant':'tactical_v2','environment_changes':False,
        'algorithm_changes':False,'new_training_runs':len(jobs),'reused_previous_training_runs':2*len(a.seeds),
        'd0_reused_baseline_runs':len(a.seeds),'hyperparameter_search':False,
        'primary_model':'fixed-budget final','best_model':'validation-only supplement'})
    write_atomic(root/'status.json',{'status':'PREPARED','coordinator':str(coordinator),'new_training_runs':len(jobs)})
    (root/'README.md').write_text(
        '# DIB Tactical-v2：固定配置全组补充实验\n\n'
        f'固定belief_first=.30；{len(jobs)}次新训练，每次{a.steps:,}团队步，共{a.steps*len(jobs):,}步。\n\n'
        f'新训练：{", ".join(NEW_SCENARIOS)}；训练seed：{a.seeds}。\n\n'+
        '\n\n'.join(notes)+'\n\n'
        '算法计算源码及外部backend逐字节复用20261103冻结版本，只覆盖运行/存储管理文件。\n'
        'hidden128、K4、z64、lr5e-4、gamma.99、GAE.95、4环境/PPO4轮、原辅助/残差调度均保持。\n'
        '固定蓝方rule/distance10/无失联；200步、原观测/动作/射击限制不变；D组沿用reference奖励协议。\n'
        '每250k附近64局验证700042起；最终final/best各128局贪心+128局采样800042起。\n'
        '子目录belief_first/<场景>/seed_<seed>；C3/C5为旧数据链接，D0为B0链接。\n'
        'compact存储：训练期间保留一个原子完整Adam/RNG恢复点，等值网络共享存储但不降精度；\n'
        '5M结束后恢复点仅用于评估，不再保留Adam；不另存initial或重复拼接逐局日志。\n'
        '逐局训练数据全部保留于training_parts/，最终training_episodes.manifest.json列出有序有效段。\n'
        '保留2M/4M/final/best模型；固定预算内仍可精确续训，不能扩预算或权重式冒充续训。\n'
        '原子完整快照不保留previous备份，存在单文件损坏后的恢复风险；磁盘2GiB保护仍启用。\n'
        f'后台管理目录：{coordinator}；中断后从其_runtime显式--resume-run。\n'
        '结果表训练/测试完成后自动生成RESULTS.md；不宣称统计显著或全球SOTA。\n',encoding='utf-8')
    print(coordinator,flush=True);return coordinator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--resume-run',type=Path)
    p.add_argument('--name',default='20261107_tactical_v2_full_groups_5m')
    p.add_argument('--source',type=Path,default=PROJECT_ROOT/'output/dib_mappo/20261103_tactical_v2_search_5m')
    p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'output')
    p.add_argument('--baseline-root',type=Path,default=PROJECT_ROOT/'output')
    p.add_argument('--workers',type=int,default=36);p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--steps',type=int,default=5000000);p.add_argument('--eval-interval',type=int,default=250000)
    p.add_argument('--eval-episodes',type=int,default=64);p.add_argument('--heldout-episodes',type=int,default=128)
    p.add_argument('--matched-baselines',action='store_true');p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if a.run:return supervise(a.run)
    if a.resume_run:return supervise(a.resume_run,True)
    if min(a.workers,a.steps,a.eval_interval,a.eval_episodes,a.heldout_episodes)<1:p.error('Positive budgets required')
    if len(set(a.seeds))!=len(a.seeds):p.error('Duplicate training seeds')
    if not a.preflight and (a.seeds!=[42,43,44] or a.steps!=5000000 or a.eval_interval!=250000 or a.eval_episodes!=64 or a.heldout_episodes!=128):
        p.error('Formal extension uses the original fixed 3-seed/5M/64-validation/128-test protocol')
    a.output_root=a.output_root.resolve();prepare(a);return 0


if __name__=='__main__':raise SystemExit(main())
