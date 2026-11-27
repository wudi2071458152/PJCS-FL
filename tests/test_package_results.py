import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('package_results', ROOT / 'tools/package_results.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PackageResultTests(unittest.TestCase):
    def fixture(self):
        base = ROOT / 'archive/test_runs'
        base.mkdir(parents=True, exist_ok=True)
        root = base / ('package_results_' + uuid.uuid4().hex)
        root.mkdir()
        def cleanup():
            if not root.resolve().is_relative_to(base.resolve()):
                raise ValueError('Cleanup escaped test workspace')
            shutil.rmtree(root)
        self.addCleanup(cleanup)
        jobs = [dict(dataset='cifar10', alpha=a, seed=2025, method='fedavg', phase='first_seed')
                for a in ('0p05', '0p07')]
        for job in jobs:
            folder = root / job['dataset'] / job['alpha'] / str(job['seed']) / job['method']
            folder.mkdir(parents=True)
            protocol = dict(suite='pjcs_portable_lr001', dataset=job['dataset'], alpha=job['alpha'],
                            seed=job['seed'], route=job['method'], rounds=1500)
            data = dict(protocol=protocol, rounds=[dict(round=i) for i in range(1, 1501)],
                        summary=dict(rounds=1500), status=dict(state='complete'))
            marker = dict(rounds=1500)
            for name, value in data.items():
                payload = json.dumps(value).encode('utf-8')
                (folder / (name + '.json')).write_bytes(payload)
                if name != 'status': marker[name + '_sha256'] = hashlib.sha256(payload).hexdigest()
            (folder / 'COMPLETED.json').write_text(json.dumps(marker), encoding='utf-8')
            (folder / 'resume.pt').write_bytes(b'checkpoint must not be packaged')
        return root, jobs

    def test_verified_json_whitelist_checksum_and_matching_reuse(self):
        root, jobs = self.fixture()
        output = root / 'final_results.tar.gz'
        result = worker.package(root, jobs, output)
        self.assertFalse(result['reused'])
        self.assertEqual(result['runs'], 2)
        with tarfile.open(output) as archive:
            names = archive.getnames()
            self.assertEqual(len(names), 12)
            self.assertTrue(all(name.endswith('.json') for name in names))
            self.assertNotIn('resume.pt', '\n'.join(names))
        expected = hashlib.sha256(output.read_bytes()).hexdigest() + '  final_results.tar.gz\n'
        self.assertEqual(output.with_name(output.name + '.sha256').read_text(), expected)
        self.assertTrue(worker.package(root, jobs, output)['reused'])

    def test_missing_identity_and_hash_damage_fail_without_publishing(self):
        root, jobs = self.fixture()
        output = root / 'final_results.tar.gz'
        folder = root / 'cifar10/0p05/2025/fedavg'
        protocol_path = folder / 'protocol.json'
        original = protocol_path.read_bytes()
        changed = json.loads(original); changed['seed'] = 2027
        protocol_path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'identity'):
            worker.package(root, jobs, output)
        self.assertFalse(output.exists())
        protocol_path.write_bytes(original)
        (folder / 'summary.json').write_text('{"rounds":1500,"changed":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            worker.package(root, jobs, output)
        self.assertFalse(output.exists())
        (folder / 'COMPLETED.json').unlink()
        with self.assertRaises(FileNotFoundError):
            worker.package(root, jobs, output)
        self.assertFalse(output.exists())

    def test_round_coverage_and_no_clobber_publication(self):
        root, jobs = self.fixture()
        output = root / 'final_results.tar.gz'
        folder = root / 'cifar10/0p05/2025/fedavg'
        rounds = folder / 'rounds.json'
        original = rounds.read_bytes()
        rounds.write_text('[{"round":1}]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'round coverage'):
            worker.package(root, jobs, output)
        rounds.write_bytes(original)
        real_link = worker.os.link
        def competing(source, destination):
            if Path(destination) == output:
                output.write_bytes(b'foreign archive')
            return real_link(source, destination)
        with patch.object(worker.os, 'link', side_effect=competing), self.assertRaises(FileExistsError):
            worker.package(root, jobs, output)
        self.assertEqual(output.read_bytes(), b'foreign archive')


if __name__ == '__main__':
    unittest.main()
