"""Opt-in padded MAPPO baseline: online sources, held-out evaluation, adaptation.

Uses the external MAPPO implementation. No proposed research architecture is
silently substituted for the baseline; legacy train.py semantics are preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from heterogrid import TaskSuite, TransferSMACAdapter
from algorithms.common.evaluation import observe
from algorithms.common.experiment_log import ExperimentRecorder
from algorithms.mappo.settings import add_backend_argument, validate_backend, PROJECT_ROOT
from algorithms.mappo.transfer_buffer import active_buffer_class


def save_checkpoint(path,policy,trainer,args,suite,interface,steps,phase,*,training_state=None):
    path=Path(path);temporary=path.with_suffix(".tmp")
    torch.save({"schema":"heterogrid-mappo-transfer-v1","actor":policy.actor.state_dict(),
                "critic":policy.critic.state_dict(),"value_normalizer":trainer.value_normalizer.state_dict(),
                "actor_optimizer":policy.actor_optimizer.state_dict(),
                "critic_optimizer":policy.critic_optimizer.state_dict(),"mappo_args":vars(args),
                "suite":suite.to_dict(),"interface":interface,"environment_steps":steps,"phase":phase,
                "training_state":training_state},temporary)
    temporary.replace(path)


def read_checkpoint(path,interface,device="cpu"):
    saved=torch.load(path,map_location=device,weights_only=False)
    if saved.get("schema")!="heterogrid-mappo-transfer-v1":
        raise ValueError("Expected a transfer checkpoint; legacy flat-observation checkpoints use a different interface")
    def signature(spec):
        if not isinstance(spec,dict):return None
        return {**spec,"env_info":{k:v for k,v in spec.get("env_info",{}).items() if k!="episode_limit"}}
    if signature(saved.get("interface"))!=signature(interface):
        raise ValueError("Checkpoint capacity/feature layout differs from the target suite; use the same declared interface")
    return saved


def restore_weights(saved,policy,trainer):
    policy.actor.load_state_dict(saved["actor"],strict=True)
    policy.critic.load_state_dict(saved["critic"],strict=True)
    trainer.value_normalizer.load_state_dict(saved["value_normalizer"])
    # Intentionally fresh optimizers: this is transfer initialization, not resume.


@torch.no_grad()
def evaluate_tasks(policy,args,suite,task_ids,episodes,seed):
    cpu_rng=torch.random.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng=np.random.get_state()
    actor_training=policy.actor.training
    env=TransferSMACAdapter(suite,task_ids=task_ids)
    results={}
    try:
        policy.actor.eval()
        for task_id in task_ids:
            rows=[]
            for episode in range(episodes):
                episode_seed=seed+episode
                torch.manual_seed(episode_seed)
                env.reset(seed=episode_seed,options={"task_id":task_id})
                hidden=np.zeros((env.n_agents,args.recurrent_N,args.hidden_size),np.float32)
                masks=np.ones((env.n_agents,1),np.float32)
                total=0.;done=False
                while not done:
                    obs,_,available,active=observe(env)
                    action,h=policy.act(obs,hidden,masks,available,deterministic=True)
                    selected=action.cpu().numpy().reshape(env.n_agents).astype(np.int64)
                    if np.any(available[np.arange(env.n_agents),selected]!=1):
                        raise RuntimeError("Evaluation selected a masked action")
                    reward,done,info=env.step(selected.tolist());total+=reward
                    hidden=h.cpu().numpy();masks=env.get_active_mask()
                    hidden[masks[:,0]==0]=0
                rows.append({"seed":episode_seed,"return":total,"won":info["battle_won"],
                             "reason":info["termination_reason"],"length":env.env.core.step_count,
                             **info["episode_metrics"]})
            contacts=[r["first_contact_step"] for r in rows if r["first_contact_step"] is not None]
            results[task_id]={"episodes":episodes,"win_rate":float(np.mean([r["won"] for r in rows])),
                              "mean_return":float(np.mean([r["return"] for r in rows])),
                              "timeout_rate":float(np.mean([r["reason"]=="timeout" for r in rows])),
                              "contact_rate":len(contacts)/episodes,
                              "mean_first_contact_step":float(np.mean(contacts)) if contacts else None,
                              "results":rows}
    finally:
        env.close();policy.actor.train(actor_training)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    add_backend_argument(parser)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--algorithm",choices=("mappo",),default="mappo")
    parser.add_argument("--steps",type=int,default=65536)
    parser.add_argument("--num-envs",type=int,default=4)
    parser.add_argument("--rollout-length",type=int,default=128)
    parser.add_argument("--ppo-epochs",type=int,default=5)
    parser.add_argument("--mini-batches",type=int,default=4)
    parser.add_argument("--hidden-size",type=int,default=64)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--threads",type=int,default=1,help="PyTorch CPU thread count")
    parser.add_argument("--eval-episodes",type=int,default=16)
    parser.add_argument("--eval-split",choices=("validation","test"),default="validation")
    parser.add_argument("--eval-seed",type=int)
    parser.add_argument("--eval-interval",type=int,default=0)
    parser.add_argument("--skip-initial-evaluation",action="store_true")
    parser.add_argument("--checkpoint-minutes",type=float,
                        help="Save only models/latest.pt on this interval and at completion")
    parser.add_argument("--checkpoint",type=Path,help="Zero-gradient evaluation; target suite is retained")
    parser.add_argument("--resume",action="store_true",help="Continue OUTPUT/models/latest.pt with saved optimizers")
    parser.add_argument("--init-checkpoint",type=Path,help="Transfer weights/value statistics, using fresh optimizers")
    parser.add_argument("--adapt-task",help="Explicitly train on this target task; requires --init-checkpoint")
    cli=parser.parse_args(argv)
    if min(cli.steps,cli.num_envs,cli.rollout_length,cli.ppo_epochs,cli.mini_batches,cli.hidden_size,cli.eval_episodes,cli.threads)<1:
        parser.error("Sizes must be positive")
    if cli.checkpoint_minutes is not None and (not np.isfinite(cli.checkpoint_minutes) or cli.checkpoint_minutes<=0):
        parser.error("--checkpoint-minutes must be positive and finite")
    batch_steps=cli.num_envs*cli.rollout_length
    if cli.steps%batch_steps or cli.eval_interval<0 or cli.eval_interval%batch_steps:
        parser.error("Steps/evaluation interval must be multiples of rollout batch size")
    if cli.mini_batches>batch_steps:parser.error("Too many mini-batches")
    if cli.checkpoint and (cli.init_checkpoint or cli.adapt_task):parser.error("Evaluation and training initialization are mutually exclusive")
    if cli.resume and (cli.checkpoint or cli.init_checkpoint or cli.adapt_task):parser.error("Resume requires source training without transfer initialization")
    if cli.adapt_task and not cli.init_checkpoint:parser.error("--adapt-task requires --init-checkpoint")
    suite=TaskSuite.load(cli.suite)
    if cli.adapt_task and cli.adapt_task not in suite.task_ids:parser.error("Unknown adaptation task")
    train_ids=(cli.adapt_task,) if cli.adapt_task else suite.ids("train")
    eval_ids=(cli.adapt_task,) if cli.adapt_task else suite.ids(cli.eval_split)
    phase="evaluation" if cli.checkpoint else ("target_adaptation" if cli.adapt_task else "online_source_training")
    cli.mappo_root=validate_backend(parser,cli.mappo_root)
    sys.path.insert(0,str(cli.mappo_root.resolve()))
    from onpolicy.config import get_config
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
    from onpolicy.utils.shared_buffer import SharedReplayBuffer

    random.seed(cli.seed);np.random.seed(cli.seed);torch.manual_seed(cli.seed);torch.set_num_threads(cli.threads)
    device=torch.device(cli.device)
    probe=TransferSMACAdapter(suite,task_ids=train_ids)
    interface=probe.interface_spec();info=probe.get_env_info();probe.close()
    checkpoint_path=cli.output/"models/latest.pt" if cli.resume else (cli.checkpoint or cli.init_checkpoint)
    saved=read_checkpoint(checkpoint_path,interface,device) if checkpoint_path else None
    args=argparse.Namespace(**saved["mappo_args"]) if saved else get_config().parse_args([])
    if not saved:
        args.algorithm_name=cli.algorithm;args.hidden_size=cli.hidden_size
        args.use_recurrent_policy=cli.algorithm=="rmappo";args.use_naive_recurrent_policy=False
    args.env_name="HeteroGridTransfer";args.use_wandb=False;args.use_centralized_V=True
    args.use_policy_active_masks=True;args.use_value_active_masks=True;args.use_proper_time_limits=False
    args.n_rollout_threads=cli.num_envs;args.episode_length=cli.rollout_length;args.data_chunk_length=16
    args.ppo_epoch=cli.ppo_epochs;args.num_mini_batch=cli.mini_batches;args.seed=cli.seed;args.num_env_steps=cli.steps
    if args.use_recurrent_policy:parser.error("This reference transfer runner supports feed-forward MAPPO; use a sequence-aware masked buffer for recurrent training")
    obs_space=gym.spaces.Box(-np.inf,np.inf,(info["obs_shape"],),dtype=np.float32)
    state_space=gym.spaces.Box(-np.inf,np.inf,(info["state_shape"],),dtype=np.float32)
    action_space=gym.spaces.Discrete(info["n_actions"])
    policy=R_MAPPOPolicy(args,obs_space,state_space,action_space,device=device)
    trainer=R_MAPPO(args,policy,device=device)
    if saved:restore_weights(saved,policy,trainer)
    start_steps=0
    continuation=saved.get("training_state") if cli.resume else None
    if cli.resume:
        if saved["phase"]!="online_source_training" or saved["suite"]!=suite.to_dict():
            raise ValueError("Resume requires the unchanged source task suite")
        for key,value in vars(args).items():
            if key not in ("num_env_steps",) and saved["mappo_args"].get(key)!=value:
                raise ValueError("Cannot resume with changed MAPPO argument "+key)
        start_steps=saved["environment_steps"]
        if start_steps>cli.steps:raise ValueError("Checkpoint exceeds requested cumulative budget")
        policy.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        policy.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        from algorithms.mappo.epymarl_transfer import resume_logs
        archive=resume_logs(cli.output,start_steps,continuation["completed"] if continuation else 0)
        for path in cli.output.glob("evaluation_*.json"):
            if json.loads(path.read_text(encoding="utf-8"))["environment_steps"]>start_steps:
                path.replace(cli.output/archive/path.name)
    else:
        cli.output.mkdir(parents=True,exist_ok=False)
        (cli.output/"models").mkdir()
        suite.save(cli.output/"suite.json")
    metadata={"name":args.algorithm_name+"_transfer","algorithm":args.algorithm_name,"phase":phase,"interface":interface,"suite":suite.to_dict(),
              "train_task_ids":list(train_ids),"evaluation_task_ids":list(eval_ids),
              "source_checkpoint":str(checkpoint_path) if saved else None,
              "optimizer_restore":cli.resume,"mappo_args":vars(args),"environment_step_unit":"team transition, not agent count",
              "cli":{k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()}}
    sources=[Path(__file__),cli.suite]+list((PROJECT_ROOT/"heterogrid").glob("*.py"))
    sources+=[Path(__file__).with_name("transfer_buffer.py"),PROJECT_ROOT/"algorithms/common/evaluation.py",
              PROJECT_ROOT/"algorithms/common/experiment_log.py",cli.mappo_root/"onpolicy/config.py"]
    sources+=list((cli.mappo_root/"onpolicy/algorithms/r_mappo").rglob("*.py"))
    sources+=list((cli.mappo_root/"onpolicy/utils").glob("*.py"))
    recorder=ExperimentRecorder(cli.output,None,metadata,sources=sources,resume_step=start_steps if cli.resume else None)
    if cli.resume:
        recorder.run["resumptions"][-1].update(rng_restored=continuation is not None,
                                              environment_state_restored=continuation is not None)
        recorder.flush()
        print(json.dumps({"event":"resumed","environment_steps":start_steps,"exact_state_available":continuation is not None}),flush=True)
    eval_seed=cli.eval_seed if cli.eval_seed is not None else (200042 if cli.eval_split=="test" or cli.adapt_task else 100042)
    def evaluate_at(step):
        results=evaluate_tasks(policy,args,suite,eval_ids,cli.eval_episodes,eval_seed)
        path=cli.output/f"evaluation_{step:09d}.json"
        path.write_text(json.dumps({"environment_steps":step,"phase":phase,"tasks":results},indent=2,allow_nan=False)+"\n",encoding="utf-8")
        score=float(np.mean([v["win_rate"] for v in results.values()]))
        recorder.log(step,{"transfer_eval_win_rate":score})
        print(json.dumps({"evaluation_steps":step,"phase":phase,"tasks":{k:{x:v[x] for x in ("win_rate","mean_return")} for k,v in results.items()}}),flush=True)
        return results
    if cli.checkpoint:
        result=evaluate_at(saved["environment_steps"])
        recorder.complete({"evaluation_only":True,"tasks":result})
        return 0

    envs=continuation["envs"] if continuation else [TransferSMACAdapter(suite,task_ids=train_ids) for _ in range(cli.num_envs)]
    buffer=active_buffer_class(SharedReplayBuffer)(args,info["n_agents"],obs_space,state_space,action_space)
    try:
        for i,env in enumerate(envs):
            if not continuation:env.reset(seed=cli.seed+i+start_steps)
            ob,st,av,active=observe(env)
            buffer.obs[0,i]=ob;buffer.share_obs[0,i]=st
            buffer.available_actions[0,i]=av;buffer.active_masks[0,i]=active
        fields=("obs","share_obs","available_actions","active_masks","masks","bad_masks","rnn_states","rnn_states_critic")
        if continuation:
            for name in fields:getattr(buffer,name)[0]=continuation["buffer"][name]
            random.setstate(continuation["python_rng"])
            np.random.set_state(continuation["numpy_rng"])
            torch.set_rng_state(continuation["torch_rng"].cpu())
            if cli.device=="cuda":torch.cuda.set_rng_state_all([v.cpu() for v in continuation["cuda_rng"]])
        periodic=cli.checkpoint_minutes is not None
        if not periodic:
            save_checkpoint(cli.output/"models/initial.pt",policy,trainer,args,suite,interface,0,phase)
        if not cli.skip_initial_evaluation:
            evaluate_at(0)
        episode_returns=continuation["episode_returns"] if continuation else np.zeros(cli.num_envs)
        completed=continuation["completed"] if continuation else 0
        task_steps=continuation["task_steps"] if continuation else {k:0 for k in train_ids}
        if cli.resume and not continuation:
            records=[json.loads(line) for line in (cli.output/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            task_steps=next(row["task_environment_steps"] for row in reversed(records) if row["environment_steps"]==start_steps)
        def training_state():
            return {"envs":envs,"buffer":{name:getattr(buffer,name)[0].copy() for name in fields},
                    "episode_returns":episode_returns,"completed":completed,"task_steps":task_steps,
                    "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all() if cli.device=="cuda" else None}
        last_checkpoint=perf_counter()
        log_buffer=1 << 20 if periodic else -1
        mode="a" if cli.resume else "w"
        with (cli.output/"training_episodes.jsonl").open(mode,encoding="utf-8",buffering=log_buffer) as episodes, (cli.output/"metrics.jsonl").open(mode,encoding="utf-8",buffering=log_buffer) as metrics:
            for update in range(start_steps//batch_steps,cli.steps//batch_steps):
                update_start=perf_counter()
                trainer.prep_rollout()
                for step in range(cli.rollout_length):
                    with torch.no_grad():
                        output=policy.get_actions(np.concatenate(buffer.share_obs[step]),np.concatenate(buffer.obs[step]),
                              np.concatenate(buffer.rnn_states[step]),np.concatenate(buffer.rnn_states_critic[step]),
                              np.concatenate(buffer.masks[step]),np.concatenate(buffer.available_actions[step]))
                    value,action,logp,rnn,crnn=[np.asarray(np.split(x.detach().cpu().numpy(),cli.num_envs)) for x in output]
                    rewards=np.zeros((cli.num_envs,info["n_agents"],1),np.float32)
                    masks=np.ones_like(rewards);obs_rows=[];state_rows=[];avail_rows=[];active_rows=[]
                    for i,env in enumerate(envs):
                        actions=action[i,:,0].astype(np.int64)
                        if np.any(buffer.available_actions[step,i,np.arange(env.n_agents),actions]!=1):
                            raise RuntimeError("Policy selected a masked action")
                        task=env.task_id;task_steps[task]+=1
                        reward,done,detail=env.step(actions.tolist());episode_returns[i]+=reward;rewards[i,:,0]=reward
                        if done:
                            completed+=1
                            episodes.write(json.dumps({"task_id":task,"return":float(episode_returns[i]),
                                "winner":detail["winner"],"length":env.env.core.step_count,"reason":detail["termination_reason"]})+"\n")
                            episode_returns[i]=0;env.reset();masks[i]=0;rnn[i]=0;crnn[i]=0
                        ob,st,av,active=observe(env)
                        if not done:masks[i]=active
                        rnn[i,active[:,0]==0]=0;crnn[i,active[:,0]==0]=0
                        obs_rows.append(ob);state_rows.append(st);avail_rows.append(av);active_rows.append(active)
                    buffer.insert(np.asarray(state_rows),np.asarray(obs_rows),rnn,crnn,action,logp,value,rewards,masks,
                                  active_masks=np.asarray(active_rows),available_actions=np.asarray(avail_rows))
                with torch.no_grad():
                    next_values=policy.get_values(np.concatenate(buffer.share_obs[-1]),np.concatenate(buffer.rnn_states_critic[-1]),np.concatenate(buffer.masks[-1]))
                buffer.compute_returns(np.asarray(np.split(next_values.cpu().numpy(),cli.num_envs)),trainer.value_normalizer)
                trainer.prep_training();losses={k:float(v) for k,v in trainer.train(buffer).items()}
                if not all(np.isfinite(v) for v in losses.values()):raise RuntimeError("Non-finite training loss")
                steps=(update+1)*batch_steps
                row={"environment_steps":steps,"phase":phase,"task_environment_steps":dict(task_steps),
                     "update_seconds":perf_counter()-update_start,**losses}
                buffer.after_update()
                save_due=periodic and (steps==cli.steps or perf_counter()-last_checkpoint>=cli.checkpoint_minutes*60)
                metrics.write(json.dumps(row,allow_nan=False)+"\n")
                flush=not periodic or save_due or update==start_steps//batch_steps
                recorder.log(steps,losses,flush=flush)
                if flush:
                    metrics.flush()
                    episodes.flush()
                    print(json.dumps(row),flush=True)
                if save_due:
                    save_checkpoint(cli.output/"models/latest.pt",policy,trainer,args,suite,interface,steps,phase,training_state=training_state())
                    last_checkpoint=perf_counter()
                if cli.eval_interval and steps%cli.eval_interval==0 and steps<cli.steps:
                    evaluate_at(steps)
            if not periodic:
                save_checkpoint(cli.output/"models/latest.pt",policy,trainer,args,suite,interface,cli.steps,phase,training_state=training_state())
            final=evaluate_at(cli.steps)
        recorder.complete({"phase":phase,"environment_steps":cli.steps,"task_environment_steps":task_steps,
                           "tasks":final,"checkpoint":"models/latest.pt"})
    finally:
        for env in envs:env.close()
    return 0


if __name__=="__main__":raise SystemExit(main())
