"""Retention safety tests use temporary synthetic data, never experiment outputs."""
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'completed_retention', Path(__file__).parents[1] / 'scripts/retain_completed_experiments.py')
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


class CompletedRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='retention-safety-')
        self.output = Path(self.temporary.name)
        self.run = self.output / 'qmix/study/B0/seed_42'
        (self.run / 'models').mkdir(parents=True)
        (self.output / 'campaigns').mkdir()
        retention.AUDIT = self.output / 'audit.jsonl'
        for name in ('final.pt', 'best.pt', 'initial.pt', 'step_002000000.pt'):
            (self.run / 'models' / name).write_bytes(name.encode())
        (self.run / 'config.json').write_text(json.dumps({'cli': {'steps': 5}}))
        (self.run / 'summary.json').write_text(json.dumps({'environment_steps': 5, 'win_rate': 0.9}))
        (self.run / 'metrics.jsonl').write_bytes(b'{"win":0.9}\n')
        self.plan = {'output': str(self.output), 'active_roots': [], 'policy': 'test'}
        self.item = {'run': str(self.run), 'models': {name: retention.identity(self.run / 'models' / name)['sha256']
                                                  for name in ('final.pt', 'best.pt')},
                     'protected_hashes': retention.protected_hashes(self.run)}

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_storage_preserves_data_and_model_aliases(self):
        data = b'{"episode":1,"return":2.5}\n' * 13
        (self.run / 'training_episodes.jsonl').write_bytes(data)
        (self.run / 'resume.pt').write_bytes(b'resume')
        (self.run / 'models/user_note.txt').write_text('keep unknown files')
        retention.storage(self.item, self.plan)
        self.assertFalse((self.run / 'models/initial.pt').exists())
        self.assertFalse((self.run / 'models/step_002000000.pt').exists())
        self.assertFalse((self.run / 'resume.pt').exists())
        self.assertTrue((self.run / 'models/final.pt').is_symlink())
        self.assertEqual((self.run / 'models/latest.pt').read_bytes(), b'final.pt')
        self.assertTrue((self.run / 'models/user_note.txt').exists())
        with gzip.open(self.run / 'training_episodes.jsonl.gz', 'rb') as f:
            self.assertEqual(f.read(), data)
        retention.verify_item(self.item)
        retention.storage(self.item, self.plan)  # Idempotent.

    def test_incomplete_run_is_never_modified(self):
        (self.run / 'summary.json').write_text('{"environment_steps": 4}')
        with self.assertRaises(ValueError):
            retention.storage(self.item, self.plan)
        self.assertTrue((self.run / 'models/initial.pt').exists())

    def test_active_root_is_never_modified(self):
        self.plan['active_roots'] = [str(self.run.parent.parent)]
        with self.assertRaises(ValueError):
            retention.storage(self.item, self.plan)
        self.assertTrue((self.run / 'models/initial.pt').exists())

    def test_differing_gzip_never_overwrites_or_deletes_raw(self):
        path = self.run / 'training_episodes.jsonl'
        path.write_bytes(b'original\n')
        with gzip.open(str(path) + '.gz', 'wb') as f:
            f.write(b'different\n')
        with self.assertRaises(ValueError):
            retention.compress(path)
        self.assertEqual(path.read_bytes(), b'original\n')

    def test_committed_gzip_members_retain_exact_rows(self):
        parts = []
        for i in range(2):
            path = self.run / ('part%d.jsonl.gz' % i)
            with gzip.open(path, 'wb') as f:
                f.write(('{"episode":%d}\n' % i).encode())
            parts.append(path.name)
        (self.run / 'training_summary.json').write_text(json.dumps({'segments': parts, 'episodes': 2}))
        result = retention.combine_segments(self.run)
        self.assertEqual(result['lines'], 2)
        self.assertTrue(all(not (self.run / p).exists() for p in parts))
        self.assertEqual(retention.combine_segments(self.run)['sha256'], result['sha256'])

    def test_symlinked_run_is_rejected(self):
        alias = self.run.parent / 'seed_43'
        alias.symlink_to(self.run, target_is_directory=True)
        with self.assertRaises(ValueError):
            retention.guard(alias, self.plan)

    def test_reused_models_keep_original_targets(self):
        source = self.output / 'original-model.pt'
        source.write_bytes(b'original bytes')
        for name in ('final.pt', 'best.pt'):
            path = self.run / 'models' / name
            path.unlink()
            path.symlink_to(source)
            self.item['models'][name] = retention.identity(path)['sha256']
        retention.storage(self.item, self.plan)
        self.assertEqual(source.read_bytes(), b'original bytes')
        self.assertTrue((self.run / 'models/latest.pt').is_symlink())
        self.assertEqual((self.run / 'models/latest.pt').resolve(), source)

    def test_existing_compressed_episode_data_is_indexed(self):
        with gzip.open(self.run / 'training_episodes.jsonl.gz', 'wb') as f:
            f.write(b'{"episode":1}\n')
        retention.storage(self.item, self.plan)
        data = retention.read_json(self.run / 'retention.json')
        self.assertEqual(data['training_log']['lines'], 1)


if __name__ == '__main__':
    unittest.main()
