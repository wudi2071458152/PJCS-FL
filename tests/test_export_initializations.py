import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import unittest
from unittest.mock import patch
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_initializations', ROOT / 'tools/export_initializations.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@contextlib.contextmanager
def workspace():
    base = ROOT / 'archive/test_runs'
    base.mkdir(parents=True, exist_ok=True)
    root = base / ('export_initializations_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(base.resolve()):
            raise ValueError('Test cleanup escaped workspace')
        shutil.rmtree(root)


def fake_factory(dataset, seed):
    torch.manual_seed(seed + (1 if dataset == 'fmnist' else 0))
    state = torch.nn.Linear(2, 2).state_dict()
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().numpy()
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes())
    return state, digest.hexdigest()


def manifest():
    return dict(version=1, datasets={dataset: {str(seed): fake_factory(dataset, seed)[1]
                 for seed in tool.SEEDS} for dataset in tool.DATASETS})


class InitializationExportTests(unittest.TestCase):
    def test_preflight_creates_no_output(self):
        with workspace() as root:
            path = root / 'expected.json'
            path.write_text(json.dumps(manifest()), encoding='utf-8')
            output = root / 'states.tar.gz'
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                tool.main(['--manifest', str(path), '--output', str(output)], fake_factory)
            self.assertEqual(json.loads(captured.getvalue())['state'], 'preflight_ok')
            self.assertFalse(output.exists())
            self.assertEqual({p.name for p in root.iterdir()}, {'expected.json'})

    def test_mismatched_environment_cannot_create_archive(self):
        with workspace() as root:
            value = manifest()
            value['datasets']['cifar10']['2027'] = '0' * 64
            path = root / 'expected.json'
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Initialization mismatch'):
                tool.main(['--manifest', str(path), '--output', str(root / 'states.tar.gz'), '--export'], fake_factory)
            self.assertEqual({p.name for p in root.iterdir()}, {'expected.json'})

    def test_valid_export_has_eight_states_and_verified_checksum(self):
        with workspace() as root:
            output = root / 'states.tar.gz'
            prepared = tool.prepare(manifest(), fake_factory)
            digest = tool.export(prepared, output, 'a' * 64)
            self.assertEqual(tool.sha256(output), digest)
            self.assertEqual(output.with_name(output.name + '.sha256').read_text(encoding='ascii'),
                             digest + '  states.tar.gz\n')
            with tarfile.open(output, 'r:gz') as archive:
                index = json.load(archive.extractfile('initialization_export_manifest.json'))
                self.assertEqual(len(archive.getmembers()), 9)
                self.assertEqual(len(index['models']), 8)
                self.assertFalse(index['training_started'])
                for row in index['models']:
                    content = archive.extractfile(row['asset_path']).read()
                    self.assertEqual(hashlib.sha256(content).hexdigest(), row['sha256'])
                    self.assertIn('/initializations/seed_', row['asset_path'])
                    self.assertEqual(row['state_sha256'], manifest()['datasets'][row['dataset']][str(row['seed'])])
            self.assertEqual({p.name for p in root.iterdir()}, {'states.tar.gz', 'states.tar.gz.sha256'})

    def test_existing_output_is_never_overwritten(self):
        with workspace() as root:
            output = root / 'states.tar.gz'
            output.write_bytes(b'foreign')
            with self.assertRaises(FileExistsError):
                tool.export(tool.prepare(manifest(), fake_factory), output, 'a' * 64)
            self.assertEqual(output.read_bytes(), b'foreign')

    def test_racing_output_is_never_overwritten(self):
        with workspace() as root:
            output = root / 'states.tar.gz'
            original = os.link
            def race(source, target):
                if Path(target) == output:
                    output.write_bytes(b'foreign')
                return original(source, target)
            with patch.object(tool.os, 'link', race), self.assertRaises(FileExistsError):
                tool.export(tool.prepare(manifest(), fake_factory), output, 'a' * 64)
            self.assertEqual(output.read_bytes(), b'foreign')
            self.assertEqual({p.name for p in root.iterdir()}, {'states.tar.gz'})


if __name__ == '__main__':
    unittest.main()
