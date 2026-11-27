import hashlib
import importlib.util
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_release', ROOT / 'tools/verify_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@contextmanager
def temporary_workspace():
    base = ROOT / 'archive/test_runs'
    base.mkdir(parents=True, exist_ok=True)
    root = base / ('release_metadata_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(base.resolve()):
            raise ValueError('Test cleanup escaped workspace')
        shutil.rmtree(root)


class ReleaseMetadataTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'assets').mkdir()
        (root / 'legacy').mkdir()
        (root / 'legacy/kernel.py').write_bytes(b'# frozen\r\n')
        source = {'files': [{'path': 'legacy/kernel.py', 'sha256': hashlib.sha256(b'# frozen\r\n').hexdigest()}]}
        (root / 'assets/source_manifest.json').write_text(json.dumps(source), encoding='utf-8')
        assets = [dict(dataset=d, alpha=a, client=i, sha256='a' * 64,
                       asset_path=f'proxies/{d}/{a}/{i}.pt')
                  for d in ('cifar10', 'fmnist') for a in ('0p05', '0p07', '0p1', '0p2', '0p3')
                  for i in range(32)]
        (root / 'assets/proxy_manifest.json').write_text(json.dumps({'assets': assets}), encoding='utf-8')

    def test_frozen_bytes_and_inventory(self):
        with temporary_workspace() as root:
            self.fixture(root)
            self.assertEqual(module.verify(root)['proxy_inventory'], 320)
            (root / 'legacy/kernel.py').write_bytes(b'# frozen\n')
            with self.assertRaisesRegex(ValueError, 'Source fingerprint changed'):
                module.verify(root)

    def test_incomplete_inventory_and_escaping_paths_rejected(self):
        with temporary_workspace() as root:
            self.fixture(root)
            path = root / 'assets/proxy_manifest.json'
            value = json.loads(path.read_text(encoding='utf-8'))
            value['assets'].pop()
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'cover ten groups'):
                module.verify(root)
            for name in ('../outside', '/absolute', 'C:/outside', 'a\\b'):
                with self.assertRaises(ValueError):
                    module.safe_path(root, name)


if __name__ == '__main__':
    unittest.main()
