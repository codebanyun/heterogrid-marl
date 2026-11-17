"""Exact same-budget continuation at complete-rollout/update boundaries."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def fingerprint(args,config,source_hash):
    value={'args':vars(args),'environment':config.to_dict(),'source_sha256':source_hash}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def capture(policy,rng,state,identity,compact=False):
    l=policy.learner
    result={'format':'dib-tactical-boundary-v1','identity':identity,'state':copy.deepcopy(state),
        'actor':copy.deepcopy(policy.mac.agent.state_dict()),'old_actor':copy.deepcopy(l.old_mac.agent.state_dict()),
        'critic':copy.deepcopy(l.critic.state_dict()),'target_critic':copy.deepcopy(l.target_critic.state_dict()),
        'actor_optimizer':copy.deepcopy(l.agent_optimiser.state_dict()),
        'critic_optimizer':copy.deepcopy(l.critic_optimiser.state_dict()),
        'actor_gate':policy.mac.agent.residual_actor_gate,'old_gate':l.old_mac.agent.residual_actor_gate,
        'batch_clock':policy.batch_start_step,'critic_training_steps':l.critic_training_steps,
        'last_target_update_step':l.last_target_update_step,'log_stats_t':l.log_stats_t,
        'logger':copy.deepcopy(policy.logger.values),'numpy_generator':copy.deepcopy(rng.bit_generator.state),
        'numpy_global':np.random.get_state(),'python_random':random.getstate(),'torch_rng':torch.get_rng_state()}
    if compact:
        # Share identical immutable snapshot tensors in the serialized payload,
        # never parameters of the live networks. No dtype/precision conversion.
        for first,second in [('actor','old_actor'),('critic','target_critic')]:
            a,b=result[first],result[second]
            if a.keys()==b.keys() and all(torch.equal(a[key],b[key]) for key in a):result[second]=a
        result['storage_mode']='compact'
        if state.get('trained',False):
            # At a fixed exhausted budget only evaluation can resume. Retain
            # every network/RNG/counter; no training is allowed past this point.
            result.pop('actor_optimizer');result.pop('critic_optimizer')
            result['evaluation_only']=True
    return result


def restore(policy,rng,snapshot,identity):
    if snapshot.get('format')!='dib-tactical-boundary-v1' or snapshot['identity']!=identity:
        raise ValueError('Exact resume refused: source, hyperparameters or environment differ')
    l=policy.learner
    for model,key in [(policy.mac.agent,'actor'),(l.old_mac.agent,'old_actor'),
                      (l.critic,'critic'),(l.target_critic,'target_critic')]:model.load_state_dict(snapshot[key])
    if snapshot.get('evaluation_only',False):
        if not snapshot['state'].get('trained',False):raise ValueError('Missing Adam state before budget completion')
    else:
        l.agent_optimiser.load_state_dict(snapshot['actor_optimizer']);l.critic_optimiser.load_state_dict(snapshot['critic_optimizer'])
    policy.mac.set_residual_actor_gate(snapshot['actor_gate']);l.old_mac.set_residual_actor_gate(snapshot['old_gate'])
    policy.batch_start_step=snapshot['batch_clock']
    for key in ['critic_training_steps','last_target_update_step','log_stats_t']:setattr(l,key,snapshot[key])
    policy.logger.values=copy.deepcopy(snapshot['logger']);rng.bit_generator.state=copy.deepcopy(snapshot['numpy_generator'])
    np.random.set_state(snapshot['numpy_global']);random.setstate(snapshot['python_random']);torch.set_rng_state(snapshot['torch_rng'])
    return copy.deepcopy(snapshot['state'])


def save(path,snapshot):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(snapshot,temporary);temporary.replace(path)
    with path.open('rb') as stream:os.fsync(stream.fileno())
