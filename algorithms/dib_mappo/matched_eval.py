"""Read-only fixed-final baseline inference on the new common heldout seeds."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from algorithms.common.experiment_log import write_atomic
from heterogrid import EnvConfig,SMACAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=800042)
    p.add_argument('--episodes',type=int,default=128)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    existing=a.output/'summary.json';provenance=a.output/'provenance.json'
    if existing.exists() and provenance.exists():
        data=json.loads(existing.read_text());saved=json.loads(provenance.read_text())
        if data['base_seed']!=a.seed or data['episodes']!=a.episodes or saved['source']!=str(a.source):
            raise ValueError('Existing comparison result has a different identity')
        return 0
    if a.output.exists():
        archive=a.output.parent/'recovery_archive';archive.mkdir(exist_ok=True)
        a.output.rename(archive/(a.output.name+'_'+str(time.time_ns())))
    metadata=json.loads((a.source/'config.json').read_text())
    if 'cli' in metadata:cli=SimpleNamespace(**metadata['cli'])
    else:
        # Tactical-v2 archived metadata stores reconstructible args but no CLI.
        checkpoint_meta=torch.load(a.source/'models/final.pt',map_location='cpu')
        cli=SimpleNamespace(**checkpoint_meta['cli'])
    cli.output=a.output;cli.config=Path(a.config)
    config=EnvConfig.load(a.config);original=EnvConfig.load(a.source/'env.json')
    first=config.to_dict();second=original.to_dict()
    first.pop('diagnostics_enabled',None);second.pop('diagnostics_enabled',None)
    if first!=second:raise ValueError('Baseline environment differs; paired comparison refused')
    environment=SMACAdapter(config=config)
    if cli.algorithm=='dib_mappo':
        from algorithms.dib_mappo.backend import build_policy
        from algorithms.dib_mappo.train import restore_checkpoint,evaluate
        cli.dib_root=Path(cli.dib_root)
        cli.training_profile=metadata.get('training_profile','legacy')
        cli.tactical_preset=metadata.get('tactical_preset',getattr(cli,'tactical_preset','balanced'))
    elif cli.algorithm in ('qmix','qtran'):
        from algorithms.common.value_decomposition.backend import build_policy
        from algorithms.common.value_decomposition.train import restore_checkpoint,evaluate
        cli.marl_root=Path(cli.marl_root)
    else:
        from algorithms.common.actor_critic.backend import build_policy
        from algorithms.common.actor_critic.train import restore_checkpoint,evaluate
        cli.epymarl_root=Path(cli.epymarl_root)
    policy,_=build_policy(cli,environment)
    checkpoint=a.source/'models/final.pt';restore_checkpoint(checkpoint,policy);environment.close()
    # Both original evaluate APIs default to a deterministic greedy policy.
    result=evaluate(policy,config,a.output,a.episodes,a.seed)
    write_atomic(a.output/'provenance.json',{'source':str(a.source),'checkpoint':str(checkpoint),
        'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'model':'fixed-budget final',
        'episodes':a.episodes,'base_seed':a.seed,'algorithm':cli.algorithm,'no_training':True})
    return 0


if __name__=='__main__':raise SystemExit(main())
