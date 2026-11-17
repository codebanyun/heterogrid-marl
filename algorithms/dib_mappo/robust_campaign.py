"""Fresh integrated robust_comm_v1 training on C1/C5/C6; no ablation matrix."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from algorithms.common.experiment_log import write_atomic
from algorithms.common.sensitivity import PROJECT_ROOT, scenarios, communication_protocol, LABELS
from algorithms.common.value_decomposition.campaign import now, supervise
from algorithms.dib_mappo.backend import settings


def prepare(args):
    root = args.output_root/'dib_mappo'/args.name
    coordinator = args.output_root/'campaigns'/(args.name+'_dib_mappo')
    if root.exists() or coordinator.exists():
        raise FileExistsError('Fresh name required; never overwrite existing results')
    if shutil.disk_usage(args.output_root.parent).free < 4*1024**3:
        raise OSError('Need at least 4 GiB free; running campaigns retain a 2 GiB reserve')
    runtime = coordinator/'_runtime';backend=coordinator/'_backend'
    coordinator.mkdir(parents=True)
    for name in ('heterogrid','algorithms','configs'):
        shutil.copytree(PROJECT_ROOT/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('learners','controllers','modules','components','utils','config'):
        shutil.copytree(args.dib_root/name,backend/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    root.mkdir(parents=True);(root/'configs').mkdir()
    tasks=scenarios(fixed_blue=True)
    for name in args.scenarios:
        write_atomic(root/'configs'/(name+'.json'),replace(tasks[name],diagnostics_enabled=True).to_dict())
    hashes={str(p.relative_to(coordinator)):hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (runtime,backend) for p in sorted(directory.rglob('*.py'))}
    meta={'algorithm':'dib_mappo','variant':'robust_comm_v1','training_profile':'robust_comm_v1',
          'created':now(),'fresh_training':True,'seeds':args.seeds,'steps':args.steps,
          'planned_runs':len(args.scenarios)*len(args.seeds),'eval_interval':args.eval_interval,
          'eval_episodes':args.eval_episodes,'heldout_episodes':args.heldout_episodes,
          'eval_seed':500042,'heldout_seed':600042,'primary_model':'final.pt',
          'selection':'validation only; test seeds never select configurations',
          'groups':{'communication':args.scenarios},'labels':{x:LABELS[x] for x in args.scenarios},
          'communication_protocol':communication_protocol(True),'num_envs':4,
          'hyperparameters':settings(backend,'robust_comm_v1'),'coordinator':str(coordinator),
          'runtime_snapshot':str(runtime),'backend_snapshot':str(backend),'source_sha256':hashes,
          'not_upstream_equivalent':True,
          'changes':['structured compact entity supervision','conditional attribute validity; negative labels retained',
                     'local-only grouped MHF evidence in log space','task-learned bounded residual trust',
                     '1M auxiliary-off; 0.5M belief pretraining; actor fully ramped at 3M',
                     'uncapped scheduled auxiliary objectives; separate FP64-norm gradient clipping',
                     'entropy floor .003; no AHS length penalty'],
          'unchanged':['environment','reward','action space and target-record firing restriction',
                       'blue rule and distance10 communication','hidden128 z64 K4 lr3e-4 PPO4 4env'],
          'capacity_note':'Compact target heads and a new 257-parameter trust head; not parameter matched',
          'execution_information':'own observation/action/history only; no other-agent or future labels',
          'checkpoint_resume':'network-only checkpoints; cannot exactly resume training'}
    write_atomic(root/'experiment.json',meta)
    jobs=[]
    for name in args.scenarios:
        for seed in args.seeds:
            out=root/name/('seed_'+str(seed))
            command=[sys.executable,'-m','algorithms.dib_mappo.train','--dib-root',str(backend),
                     '--training-profile','robust_comm_v1','--config',str(root/'configs'/(name+'.json')),
                     '--output',str(out),'--seed',str(seed),'--steps',str(args.steps),
                     '--eval-interval',str(args.eval_interval),'--eval-episodes',str(args.eval_episodes),
                     '--heldout-episodes',str(args.heldout_episodes),'--eval-seed','500042',
                     '--heldout-seed','600042','--replay-episodes','0']
            jobs.append({'algorithm':'dib_mappo','scenario':name,'seed':seed,'output':str(out),
                         'runtime':str(runtime),'command':command,'status':'PENDING','pid':None})
    plan={'created':now(),'status':'PREPARED','manager_pid':None,'runtime':str(runtime),
          'workers':args.workers,'jobs':jobs,'roots':{'dib_mappo':str(root)},
          'requested_total_steps':args.steps*len(jobs),'report_options_list':[],
          'python':sys.executable,'source_sha256':hashes,'host_cpu_count':os.cpu_count()}
    write_atomic(coordinator/'plan.json',plan);write_atomic(coordinator/'status.json',plan)
    write_atomic(root/'status.json',{'status':'PREPARED','coordinator':str(coordinator),'counts':{'PENDING':len(jobs)}})
    (root/'README.md').write_text(
        '# DIB robust_comm_v1：无通信／受干扰定向训练\n\n'
        f'场景{args.scenarios}，训练seed={args.seeds}，各{args.steps:,}团队步，{len(jobs)}次从头训练。\n'
        '这是一套整合方法变体，不是原DIB的等价优化，也不是逐项消融。\n'
        '环境、角色、动作、攻击条件、奖励和蓝方rule不变；蓝方固定distance10无失联。\n'
        '仅训练监督标签采用紧凑实体格式，执行输入仍是完整原始局部观测；不补充隐藏状态或虚拟目标报告。\n'
        '分类保留无记录／不可见／阵亡负标签；只有有定义的属性参与回归；消息恒空字段和固定角色等不重建。\n'
        '滤波按自身新观测中的有效实体字段评分，不递推旧候选权重，不强制置信度变尖。\n'
        '前1M辅助关闭；1–1.5M辅助预训练且策略门控关闭；1.5–3M渐入有界、任务学习信任门控的残差。\n'
        '训练actor/辅助采用动作前存活掩码；团队critic不截断；基础策略和信念分支分别裁剪有限梯度。\n'
        '每250k验证64局（正式默认）；独立测试每模型128贪心+128采样；主报告final。\n'
        '验证500042起，测试600042起；不能用独立测试结果选配置。\n'
        '保留旧实验不变。输出含原始压缩训练记录、模型、验证和测试，配置与源码SHA已冻结。\n'
        '详细实现说明见仓库algorithms/dib_mappo/ROBUST_COMM.md。\n',encoding='utf-8')
    print(coordinator,flush=True)
    return coordinator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--dib-root',type=Path,default=Path('/root/research/DIB-MAPPO/src'))
    parser.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'output')
    parser.add_argument('--name',default='20261023_robust_comm_v1_5m')
    parser.add_argument('--scenarios',nargs='+',choices=('B0','C1','C3','C5','C6'),default=['C1','C5','C6'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--workers',type=int,default=9)
    parser.add_argument('--steps',type=int,default=5000000)
    parser.add_argument('--eval-interval',type=int,default=250000)
    parser.add_argument('--eval-episodes',type=int,default=64)
    parser.add_argument('--heldout-episodes',type=int,default=128)
    a=parser.parse_args()
    if a.run:return supervise(a.run)
    if min(a.workers,a.steps,a.eval_interval,a.eval_episodes,a.heldout_episodes)<1:
        parser.error('Budgets and workers must be positive')
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.scenarios))!=len(a.scenarios):
        parser.error('Duplicate seeds/scenarios')
    a.dib_root=a.dib_root.resolve();a.output_root=a.output_root.resolve()
    if (a.dib_root/'src').is_dir():a.dib_root=a.dib_root/'src'
    prepare(a)
    return 0


if __name__=='__main__':sys.exit(main())
