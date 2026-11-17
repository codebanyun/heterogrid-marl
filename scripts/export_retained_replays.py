"""Export demonstrations, not new evaluation statistics, in a frozen runtime.

Invoke with python -c/runpy from the run's ORIGINAL _runtime directory. This
script deliberately imports its environment and backend from that directory.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def export(run):
    import numpy as np
    import torch
    from heterogrid import EnvConfig, SMACAdapter
    from algorithms.common.value_decomposition.train import observe

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    meta = json.loads((run / 'config.json').read_text())
    first_model = run / 'models/final.pt'
    saved = torch.load(first_model, map_location='cpu')
    cli_data = {**saved.get('cli', {}), **meta.get('cli', {})}
    algorithm = meta['algorithm']
    cli_data['algorithm'] = algorithm
    for key in ('config', 'output', 'marl_root', 'epymarl_root', 'dib_root'):
        if cli_data.get(key) is not None:
            cli_data[key] = Path(cli_data[key])
    cli_data['output'] = run
    cli_data['config'] = run / 'env.json'
    cli_data['training_profile'] = meta.get('training_profile', cli_data.get('training_profile', 'legacy'))
    cli_data['tactical_preset'] = meta.get('tactical_preset', cli_data.get('tactical_preset', 'balanced'))
    cli = SimpleNamespace(**cli_data)
    config = EnvConfig.load(run / 'env.json')
    family = ('algorithms.common.value_decomposition' if algorithm in ('qmix', 'qtran')
              else 'algorithms.common.actor_critic' if algorithm in ('maa2c', 'coma')
              else 'algorithms.dib_mappo')
    backend = importlib.import_module(family + '.backend')
    trainer = importlib.import_module(family + '.train')
    random.seed(int(meta['seed']))
    np.random.seed(int(meta['seed']))
    torch.manual_seed(int(meta['seed']))
    env = SMACAdapter(config=config)
    policy, args = backend.build_policy(cli, env)
    summary = json.loads((run / 'summary.json').read_text()) if (run / 'summary.json').exists() else {}
    base_seed = int(meta.get('heldout_seed', cli_data.get('heldout_seed', 200042)))
    target = run / 'replays/retained'
    if target.exists():
        raise FileExistsError('Refusing to overwrite retained replays: ' + str(target))
    target.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.retained-', dir=str(target.parent)))
    blue_actions = {}
    original_act = env.env.opponent_policy.act

    def capture_blue(observations, infos=None):
        selected = original_act(observations, infos)
        blue_actions.clear()
        blue_actions.update({k: int(v) for k, v in selected.items()})
        return selected

    env.env.opponent_policy.act = capture_blue
    results = {}
    try:
        for kind, filename in (('latest', 'final.pt'), ('best', 'best.pt')):
            checkpoint = run / 'models' / filename
            checkpoint_sha = sha(checkpoint)
            trainer.restore_checkpoint(checkpoint, policy)
            seed = int(summary.get('heldout', {}).get('final' if kind == 'latest' else 'best', {})
                       .get('greedy', {}).get('base_seed', base_seed))
            rng = np.random.default_rng(seed)
            env.env.record_replay = True
            env.reset(seed=seed)
            if algorithm in ('qmix', 'qtran'):
                hidden = np.zeros((1, env.n_agents, args.rnn_hidden_dim), np.float32)
                previous = np.zeros((1, env.n_agents, policy.n_actions), np.float32)
            elif algorithm in ('maa2c', 'coma'):
                hidden = np.zeros((1, env.n_agents, args.hidden_dim), np.float32)
            else:
                hidden = backend.initial_memory(policy, 1)
            dest = temporary / kind
            dest.mkdir()
            done, steps = False, 0
            with gzip.open(dest / 'actions.jsonl.gz', 'wt', encoding='utf8', compresslevel=6) as out:
                while not done:
                    obs, _, avail = observe(env)
                    living = list(env.env.agents)
                    if algorithm in ('qmix', 'qtran'):
                        selected, hidden, previous = backend.act(policy, obs[None], avail[None], hidden, previous, rng, 0.0)
                    else:
                        selected, hidden = backend.act(policy, obs[None], avail[None], hidden, rng, True)
                    red_slots = [int(a) for a in selected[0]]
                    red = {n: red_slots[i] for i, n in enumerate(env.env.possible_agents) if n in living}
                    reward, done, info = env.step(red_slots)
                    if any(n.startswith('red_') for n in info['invalid_actions']):
                        raise RuntimeError('Illegal red action')
                    steps += 1
                    out.write(json.dumps({'step': steps, 'red': red, 'red_fixed_slots': red_slots,
                                          'blue': blue_actions, 'reward': float(reward), 'done': bool(done),
                                          'invalid_actions': info['invalid_actions']}, separators=(',', ':')) + '\n')
            env.save_replay(dest / 'replay.json')
            replay = json.loads((dest / 'replay.json').read_text())
            with gzip.open(dest / 'actions.jsonl.gz', 'rt') as source:
                actions = [json.loads(line) for line in source]
            assert len(replay['frames']) == steps + 1 == len(actions) + 1
            assert replay['frames'][-1]['finished'] and actions[-1]['done']
            assert sha(checkpoint) == checkpoint_sha
            provenance = {'purpose': 'regenerated demonstration; NOT a new statistical evaluation or model selection',
                          'model': str(checkpoint), 'model_sha256': checkpoint_sha, 'kind': kind,
                          'environment_seed': seed, 'greedy': True, 'steps': steps,
                          'winner': env.env.core.winner, 'runtime': str(Path.cwd()),
                          'environment_module': importlib.import_module('heterogrid').__file__,
                          'backend_module': backend.__file__, 'algorithm': algorithm,
                          'action_semantics': 'actual submitted actions, living red/blue agents; dead red fixed slots also included',
                          'replay_sha256': sha(dest / 'replay.json'),
                          'actions_sha256': sha(dest / 'actions.jsonl.gz')}
            (dest / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            results[kind] = provenance
        temporary.rename(target)
        print(json.dumps({'run': str(run), 'replays': results}), flush=True)
    finally:
        env.close()
        if temporary.exists():
            # Leave partial exports for diagnosis; retention never prunes originals on failure.
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    export(parser.parse_args().run.resolve())
