"""Audited retention of completed HeteroGrid runs; never follows directory links.

First create a frozen inventory with --phase plan; inspect plan.json. Subsequent
phases require --apply. Run storage before replays; all original summaries,
metrics, configurations, reports, source snapshots, and active runs are retained.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

ALGORITHMS = ('qmix', 'qtran', 'maa2c', 'coma', 'dib_mappo')
LOCK = threading.Lock()
AUDIT = None


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def fingerprint(stream):
    h, size, rows = hashlib.sha256(), 0, 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
        size += len(block)
        rows += block.count(b'\n')
    return {'sha256': h.hexdigest(), 'bytes': size, 'lines': rows}


def identity(path, compressed=False):
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as f:
        return fingerprint(f)


def audit(event, **values):
    row = {'time': time.time(), 'event': event, **values}
    with LOCK:
        with AUDIT.open('a') as out:
            out.write(json.dumps(row, sort_keys=True) + '\n')


def write_json(path, value):
    temporary = path.with_name(path.name + '.retention-tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def walk(path):
    for root, dirs, files in os.walk(str(path), followlinks=False):
        p = Path(root)
        dirs[:] = [d for d in dirs if d not in ('_sources', 'sources', '__pycache__', '_runtime', '_backend')
                   and not (p / d).is_symlink()]
        yield p, files


def active_roots(output):
    roots, live_pids = set(), []
    for status in (output / 'campaigns').glob('*/status.json'):
        data = read_json(status, {})
        jobs = data.get('jobs', [])
        if isinstance(jobs, dict):
            jobs = list(jobs.values())
        for job in jobs:
            if not isinstance(job, dict) or not job.get('pid') or not job.get('output'):
                continue
            pid, run = int(job['pid']), Path(job['output'])
            cmdline = Path('/proc') / str(pid) / 'cmdline'
            try:
                command = cmdline.read_bytes()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if str(run).encode() in command:
                rel = run.relative_to(output)
                roots.add(str(output / rel.parts[0] / rel.parts[1]))
                live_pids.append(pid)
    return sorted(roots), sorted(set(live_pids))


def complete(run):
    meta = read_json(run / 'config.json', {})
    budget = int(meta.get('budget', meta.get('cli', {}).get('steps', 5000000)))
    summary, training = read_json(run / 'summary.json', {}), read_json(run / 'training_summary.json', {})
    return (summary.get('environment_steps') == budget or
            (training.get('trained') is True and training.get('steps') == budget))


def frozen_runtime(run):
    sources = read_json(run / 'run.json', {}).get('experiment', {}).get('sources', [])
    for source in sources:
        path = source.get('path', '')
        if '/_runtime/' in path:
            runtime = Path(path.split('/_runtime/')[0]) / '_runtime'
            if (runtime / 'heterogrid').is_dir():
                return str(runtime)
    for ancestor in run.parents:
        experiment = read_json(ancestor / 'experiment.json', {})
        for key in ('runtime', 'runtime_snapshot'):
            if experiment.get(key):
                runtime = Path(experiment[key])
                if (runtime / 'heterogrid').is_dir():
                    return str(runtime)
    raise ValueError('Original frozen runtime not found: ' + str(run))


def protected_hashes(run):
    result = {}
    important = {'summary.json', 'metrics.json', 'metrics.jsonl', 'info.json', 'run.json',
                 'config.json', 'env.json', 'training_summary.json'}
    for directory, files in walk(run):
        for name in files:
            path = directory / name
            if name in important and not path.is_symlink():
                result[str(path.relative_to(run))] = identity(path)['sha256']
    return result


def make_plan(output, destination):
    protected, pids = active_roots(output)
    eligible, skipped = [], []
    for algorithm in ALGORITHMS:
        for run, files in walk(output / algorithm):
            if not run.name.startswith('seed_') or 'config.json' not in files or not (run / 'models').is_dir():
                continue
            if any(run == Path(p) or Path(p) in run.parents for p in protected) or not complete(run):
                skipped.append(str(run))
                continue
            if not all((run / 'models' / name).is_file() for name in ('final.pt', 'best.pt')):
                skipped.append(str(run))
                continue
            eligible.append({'run': str(run), 'algorithm': algorithm, 'runtime': frozen_runtime(run),
                             'models': {name: identity(run / 'models' / name)['sha256']
                                        for name in ('final.pt', 'best.pt')},
                             'protected_hashes': protected_hashes(run)})
            if len(eligible) % 50 == 0:
                print(json.dumps({'inventory_completed': len(eligible)}), flush=True)
    plan = {'created': time.time(), 'output': str(output), 'active_roots': protected, 'active_pids': pids,
            'runs': eligible, 'skipped': skipped, 'disk_before': shutil.disk_usage(output)._asdict(),
            'policy': 'completed only; latest/best with byte-identical final alias; all statistics unchanged; lossless episode logs; 1 demonstration/model'}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'plan.json', plan)
    print(json.dumps({'eligible': len(eligible), 'algorithms': dict(Counter(r['algorithm'] for r in eligible)),
                      'protected_or_incomplete': len(skipped), 'active_pids': len(pids), 'manifest': str(destination / 'plan.json')}), flush=True)
    return plan


def guard(run, plan):
    if run.is_symlink() or not complete(run):
        raise ValueError('Not a canonical completed run: ' + str(run))
    for part in [run, *run.parents]:
        if part.is_symlink():
            raise ValueError('Symlink directory rejected: ' + str(run))
    protected, _ = active_roots(Path(plan['output']))
    for p in plan['active_roots'] + protected:
        if run == Path(p) or Path(p) in run.parents:
            raise ValueError('Active experiment is protected: ' + str(run))
    # Also reject a currently executing worker even if its coordinator is stale.
    for cmd in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            data = cmd.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if str(run).encode() in data and any(b'algorithms.' in arg for arg in data):
            raise ValueError('Live worker references completed run: ' + str(run))


def remove(path, reason):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Refusing non-regular deletion: ' + str(path))
    original = identity(path)
    audit('delete_intent', path=str(path), reason=reason, **original)
    path.unlink()
    audit('deleted', path=str(path), reason=reason, **original)


def compress(path):
    if path.is_symlink():
        raise ValueError('Refusing linked log: ' + str(path))
    before = path.stat()
    original = identity(path)
    target = path.with_name(path.name + '.gz')
    if not target.exists():
        temporary = path.with_name(path.name + '.retention-tmp.gz')
        with path.open('rb') as source, temporary.open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=6, mtime=0) as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)
        if identity(temporary, compressed=True) != original:
            raise ValueError('Compression verification failed: ' + str(path))
        temporary.replace(target)
    if identity(target, compressed=True) != original:
        raise ValueError('Existing gzip is different, kept both: ' + str(path))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Log changed while compressing: ' + str(path))
    audit('lossless_gzip_verified', original=str(path), retained=str(target), **original)
    remove(path, 'byte-identical decoded gzip retained')
    return {'path': str(target), **original, 'gzip_sha256': identity(target)['sha256']}


def combine_segments(run):
    state = read_json(run / 'training_summary.json', {})
    parts = state.get('segments', [])
    if not parts:
        return None
    target = run / 'training_episodes.jsonl.gz'
    paths = [run / p for p in parts]
    if not all(p.is_file() and not p.is_symlink() and run in p.resolve().parents for p in paths):
        retained = read_json(run / 'retention.json', {})
        if target.is_file() and retained.get('training_log', {}).get('sha256') == identity(target, True)['sha256']:
            return retained['training_log']
        raise ValueError('Missing or unsafe committed training segment: ' + str(run))
    expected, expected_bytes = hashlib.sha256(), 0
    for path in paths:
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                expected.update(block)
                expected_bytes += len(block)
    if not target.exists():
        temporary = target.with_name(target.name + '.retention-tmp')
        with temporary.open('wb') as dest:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, dest, 1024 * 1024)
        temporary.replace(target)
    if identity(target)['sha256'] != expected.hexdigest():
        raise ValueError('Combined training log differs from committed segments: ' + str(run))
    decoded = identity(target, True)
    if decoded['lines'] != state['episodes']:
        raise ValueError('Episode count mismatch: ' + str(run))
    result = {'path': str(target), **decoded, 'gzip_sha256': expected.hexdigest(),
              'replaces_segments': parts, 'episode_count_verified': True}
    # Record the canonical replacement before removing any historical segment.
    previous = read_json(run / 'retention.json', {})
    write_json(run / 'retention.json', {**previous, 'training_log': result})
    audit('committed_segments_verified', run=str(run), **result)
    for path in paths:
        remove(path, 'byte-identical ordered gzip members retained in training_episodes.jsonl.gz')
    return result


def storage(item, plan):
    run = Path(item['run'])
    guard(run, plan)
    result = read_json(run / 'retention.json', {})
    # Logs first: this frees capacity before exporting any replays.
    for name in ('training_episodes.jsonl',):
        path = run / name
        if path.exists():
            result['training_log'] = compress(path)
    log = combine_segments(run)
    if log:
        result['training_log'] = log
    existing_gzip = run / 'training_episodes.jsonl.gz'
    if not result.get('training_log') and existing_gzip.is_file():
        result['training_log'] = {'path': str(existing_gzip), **identity(existing_gzip, True),
                                  'gzip_sha256': identity(existing_gzip)['sha256'],
                                  'already_compressed': True}
    models = run / 'models'
    if models.is_symlink():
        raise ValueError('Symlink model directory rejected: ' + str(models))
    for name, expected in item['models'].items():
        if identity(models / name)['sha256'] != expected:
            raise ValueError('Retained model changed before cleanup: ' + str(models / name))
    latest, final = models / 'latest.pt', models / 'final.pt'
    if not latest.exists():
        if final.is_symlink():
            latest.symlink_to('final.pt')  # D0 points to the original B0 model.
        else:
            final.rename(latest)
            final.symlink_to('latest.pt')
        audit('latest_alias', final=str(final), latest=str(latest), model_sha256=item['models']['final.pt'])
    required_targets = {(models / n).resolve() for n in ('final.pt', 'latest.pt', 'best.pt')}
    for path in sorted(models.iterdir()):
        if path.is_symlink() or path.resolve() in required_targets:
            continue
        if path.is_file() and (path.name in ('initial.pt', 'interrupted.pt') or re.fullmatch(r'step_\d+\.pt', path.name)):
            remove(path, 'completed run: only final/latest and best checkpoints retained')
    for name in ('resume.pt', 'resume_previous.pt'):
        path = run / name
        if path.is_file() and not path.is_symlink():
            remove(path, 'completed training: full optimizer/resume snapshot no longer required')
    # Archived metric prefixes are redundant only if byte-for-byte a prefix.
    canonical = run / 'metrics.jsonl'
    archive = run / 'recovery_archive'
    if canonical.is_file() and archive.is_dir() and not archive.is_symlink():
        for path in sorted(archive.glob('metrics_before_*.jsonl')):
            if path.is_file() and not path.is_symlink():
                with path.open('rb') as old, canonical.open('rb') as new:
                    same = True
                    for block in iter(lambda: old.read(1024 * 1024), b''):
                        if block != new.read(len(block)):
                            same = False
                            break
                if same:
                    remove(path, 'byte-identical prefix preserved in canonical metrics.jsonl')
    result.update({'policy': plan['policy'], 'model_sha256': item['models'],
                   'models': {'latest': 'models/latest.pt', 'best': 'models/best.pt', 'compatibility': 'models/final.pt'},
                   'exact_training_resume_retained': False, 'maintenance_manifest': str(AUDIT.parent),
                   'storage_completed': True, 'statistics_unchanged': True})
    write_json(run / 'retention.json', result)
    verify_item(item, replay=False)
    audit('storage_completed', run=str(run))
    return str(run)


def verify_replays(run, item):
    base = run / 'replays/retained'
    for kind, model in (('latest', 'final.pt'), ('best', 'best.pt')):
        directory = base / kind
        provenance = read_json(directory / 'provenance.json', {})
        if provenance.get('model_sha256') != item['models'][model]:
            raise ValueError('Replay/model provenance mismatch: ' + str(directory))
        if identity(directory / 'replay.json')['sha256'] != provenance['replay_sha256']:
            raise ValueError('Replay hash mismatch: ' + str(directory))
        if identity(directory / 'actions.jsonl.gz')['sha256'] != provenance['actions_sha256']:
            raise ValueError('Action hash mismatch: ' + str(directory))
        data = read_json(directory / 'replay.json')
        with gzip.open(directory / 'actions.jsonl.gz', 'rt') as f:
            actions = [json.loads(line) for line in f]
        if len(actions) != provenance['steps'] or len(data['frames']) != len(actions) + 1:
            raise ValueError('Incomplete replay/actions: ' + str(directory))
        if not actions[-1]['done'] or not data['frames'][-1]['finished']:
            raise ValueError('Truncated replay/actions: ' + str(directory))


def replays(item, plan):
    run = Path(item['run'])
    guard(run, plan)
    target = run / 'replays/retained'
    if not target.exists():
        script = Path(__file__).with_name('export_retained_replays.py').resolve()
        command = [sys.executable, '-c', 'import runpy; runpy.run_path(' + repr(str(script)) + ", run_name='__main__')",
                   '--run', str(run)]
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        log = AUDIT.parent / 'replay_export_logs' / (str(run.relative_to(Path(plan['output']))).replace('/', '__') + '.log')
        log.parent.mkdir(exist_ok=True)
        with log.open('w') as f:
            subprocess.run(command, cwd=item['runtime'], env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
    verify_replays(run, item)
    for directory, files in walk(run):
        for name in files:
            if re.fullmatch(r'episode_\d+\.json(?:\.gz)?', name):
                path = directory / name
                if not path.is_symlink():
                    remove(path, 'canonical complete replay and actual actions retained per final/best model')
    result = read_json(run / 'retention.json', {})
    result['replays'] = {'latest': 'replays/retained/latest', 'best': 'replays/retained/best',
                         'purpose': 'regenerated demonstrations, original evaluation statistics unchanged',
                         'complete_replays': 2, 'actual_action_logs': 2}
    write_json(run / 'retention.json', result)
    verify_item(item, replay=True)
    audit('replays_completed', run=str(run))
    return str(run)


def verify_item(item, replay=False):
    run = Path(item['run'])
    for name, expected in item['models'].items():
        if identity(run / 'models' / name)['sha256'] != expected:
            raise ValueError('Retained checkpoint SHA changed: ' + str(run / 'models' / name))
    for name, expected in item['protected_hashes'].items():
        if identity(run / name)['sha256'] != expected:
            raise ValueError('Statistics/configuration SHA changed: ' + str(run / name))
    retained = read_json(run / 'retention.json', {})
    if retained.get('training_log'):
        log = retained['training_log']
        if identity(Path(log['path']), True)['sha256'] != log['sha256']:
            raise ValueError('Retained training log SHA mismatch: ' + str(run))
    if replay:
        verify_replays(run, item)
    return str(run)


def main():
    global AUDIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=('plan', 'storage', 'replays', 'verify'), default='plan')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', help='Substring filter for smoke tests')
    args = parser.parse_args()
    output, destination = args.output.resolve(), args.manifest.resolve()
    if args.phase == 'plan':
        make_plan(output, destination)
        return
    plan = read_json(destination / 'plan.json')
    if not plan or plan['output'] != str(output):
        parser.error('A matching frozen plan is required')
    if not args.apply:
        parser.error('Non-plan phases require explicit --apply')
    AUDIT = destination / 'actions.jsonl'
    items = [r for r in plan['runs'] if not args.only or args.only in r['run']]
    if args.limit:
        items = items[:args.limit]
    errors, completed, started = [], 0, time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        function = ((lambda item: storage(item, plan)) if args.phase == 'storage'
                    else (lambda item: replays(item, plan)) if args.phase == 'replays'
                    else (lambda item: verify_item(item, replay=True)))
        pending = {pool.submit(function, item): item for item in items}
        for task in concurrent.futures.as_completed(pending):
            item = pending[task]
            try:
                task.result()
                completed += 1
            except Exception as error:
                errors.append({'run': item['run'], 'error': repr(error)})
                audit('error', phase=args.phase, run=item['run'], error=repr(error), traceback=traceback.format_exc())
                print(json.dumps(errors[-1]), flush=True)
            if (completed + len(errors)) % 10 == 0 or completed + len(errors) == len(items):
                print(json.dumps({'phase': args.phase, 'completed': completed, 'errors': len(errors),
                                  'total': len(items), 'elapsed_seconds': round(time.time() - started, 1),
                                  'free_gib': round(shutil.disk_usage(output).free / 2**30, 3)}), flush=True)
    result = {'phase': args.phase, 'completed': completed, 'errors': errors, 'elapsed_seconds': time.time() - started,
              'disk_after': shutil.disk_usage(output)._asdict(), 'active_after': active_roots(output)}
    suffix = ('_' + re.sub(r'[^a-zA-Z0-9_-]', '_', args.only)) if args.only else ''
    write_json(destination / (args.phase + suffix + '_result.json'), result)
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
