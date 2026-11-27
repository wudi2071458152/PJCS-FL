import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
from unittest.mock import patch
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('accept_initializations', ROOT / 'tools/accept_initializations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InitializationAcceptanceTests(unittest.TestCase):
    def workspace(self):
        base = ROOT / 'archive/test_runs'
        base.mkdir(parents=True, exist_ok=True)
        folder = base / ('accept_initializations_' + uuid.uuid4().hex)
        folder.mkdir()
        self.addCleanup(lambda: shutil.rmtree(folder) if folder.resolve().is_relative_to(base.resolve()) else None)
        return folder

    def fixture(self, folder, mutation=None):
        models, payload, datasets = [], {}, {'cifar10': {}, 'fmnist': {}}
        for d in datasets:
            for seed in (2027, 2028, 2029, 2030):
                state = {'weight': torch.tensor([float(seed)])}
                fingerprint = module.state_fingerprint(state)
                datasets[d][str(seed)] = fingerprint
                memory = io.BytesIO(); torch.save(state, memory); data = memory.getvalue()
                name = f'metadata/{d}/initializations/seed_{seed}.pt'
                payload[name] = data
                models.append(dict(dataset=d, seed=seed, asset_path=name, sha256=hashlib.sha256(data).hexdigest(),
                                   state_sha256=fingerprint, bytes=len(data),
                                   tensors={'weight': dict(shape=[1], dtype='torch.float32')}))
        manifest = folder / 'expected.json'
        manifest.write_text(json.dumps(dict(version=1, datasets=datasets)), encoding='utf-8')
        index = dict(version=1, kind='paper_initializations', expected_manifest_sha256=module.digest(manifest),
                     models=models, training_started=False)
        if mutation:
            mutation(payload, index, manifest)
        payload['initialization_export_manifest.json'] = json.dumps(index).encode()
        archive = folder / 'initializations.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            for name, data in payload.items():
                member = tarfile.TarInfo(name); member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
        checksum = folder / 'initializations.tar.gz.sha256'
        checksum.write_text(module.digest(archive) + '  ' + archive.name + '\n', encoding='ascii')
        return archive, checksum, manifest, folder / 'artifacts'

    def test_accepts_eight_states_and_idempotent_existing_matches(self):
        args = self.fixture(self.workspace())
        self.assertEqual(module.accept(*args)['copied_files'], 8)
        self.assertEqual(module.accept(*args)['copied_files'], 0)
        self.assertEqual(len(list(args[3].rglob('*.pt'))), 8)

    def test_archive_checksum_fails_before_deserialization(self):
        args = self.fixture(self.workspace())
        args[1].write_text('0' * 64 + '  ' + args[0].name + '\n', encoding='ascii')
        with patch.object(torch, 'load', side_effect=AssertionError('must not deserialize')):
            with self.assertRaisesRegex(ValueError, 'Archive SHA256'):
                module.accept(*args)
        self.assertFalse(args[3].exists())

    def test_missing_and_unsafe_members_rejected(self):
        for mutate in (lambda p, i, m: p.pop(next(iter(p))),
                       lambda p, i, m: p.update({'../escape': b'bad'})):
            args = self.fixture(self.workspace(), mutate)
            with self.assertRaises(ValueError): module.accept(*args)
            self.assertFalse(args[3].exists())

    def test_file_hash_rejected_before_deserialization(self):
        def mutate(payload, index, manifest):
            name = next(iter(payload)); payload[name] += b'changed'
        args = self.fixture(self.workspace(), mutate)
        with patch.object(torch, 'load', side_effect=AssertionError('must not deserialize')):
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                module.accept(*args)

    def test_canonical_mismatch_and_foreign_target_preserved(self):
        def mutate(payload, index, manifest):
            row = index['models'][0]
            row['state_sha256'] = '0' * 64
            value = json.loads(manifest.read_text()); value['datasets'][row['dataset']][str(row['seed'])] = '0' * 64
            manifest.write_text(json.dumps(value), encoding='utf-8')
            index['expected_manifest_sha256'] = module.digest(manifest)
        args = self.fixture(self.workspace(), mutate)
        with self.assertRaisesRegex(ValueError, 'Canonical'):
            module.accept(*args)
        self.assertFalse(args[3].exists())
        args = self.fixture(self.workspace())
        path = args[3] / 'metadata/cifar10/initializations/seed_2027.pt'
        path.parent.mkdir(parents=True); path.write_bytes(b'foreign')
        with self.assertRaises(FileExistsError): module.accept(*args)
        self.assertEqual(path.read_bytes(), b'foreign')
        self.assertEqual(len(list(args[3].rglob('*.pt'))), 1)


if __name__ == '__main__':
    unittest.main()
