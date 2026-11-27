import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('get_assets', ROOT / 'tools/get_assets.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class AssetInstallTests(unittest.TestCase):
    def fixture(self, members=None):
        parent = ROOT / 'archive/test_runs'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / ('asset_install_test_' + uuid.uuid4().hex)
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root) if root.resolve().is_relative_to(parent.resolve()) else None)
        members = members or {'proxies/x/sample.pt': b'tensor fixture'}
        archive = root / 'data.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
        item = dict(name=archive.name, bytes=archive.stat().st_size, sha256=tool.digest(archive),
                    members=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                             for name, data in members.items()])
        return root, archive, item

    def test_install_repeat_and_preserve_different_file(self):
        root, archive, item = self.fixture()
        target = root / 'artifacts'
        self.assertEqual(tool.install(item, archive, target)['installed_files'], 1)
        self.assertEqual(tool.install(item, archive, target)['installed_files'], 1)
        (target / 'proxies/x/sample.pt').write_bytes(b'foreign')
        with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
            tool.install(item, archive, target)
        self.assertEqual((target / 'proxies/x/sample.pt').read_bytes(), b'foreign')

    def test_bad_member_checksum_or_path_cannot_publish(self):
        root, archive, item = self.fixture()
        item['members'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'member fingerprint'):
            tool.install(item, archive, root / 'artifacts')
        self.assertFalse((root / 'artifacts').exists())
        root, archive, item = self.fixture({'../escape': b'bad'})
        with self.assertRaisesRegex(ValueError, 'Unsafe asset path'):
            tool.install(item, archive, root / 'artifacts')
        self.assertFalse((root / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
