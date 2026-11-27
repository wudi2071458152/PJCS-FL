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
spec = importlib.util.spec_from_file_location('package_data', ROOT / 'tools/package_data.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class PackageDataTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'archive/test_runs'
        base.mkdir(parents=True, exist_ok=True)
        self.root = base / ('package_data_' + uuid.uuid4().hex)
        self.root.mkdir()
        def cleanup():
            self.assertTrue(self.root.resolve().is_relative_to(base.resolve()))
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        self.asset = self.root / 'proxies/cifar10/a/client_00/iter_32000.pt'
        self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(b'proxy fixture')
        self.license = self.root / 'DATA_LICENSE.txt'
        self.license.write_text('test license only\n', encoding='utf-8')
        self.row = dict(dataset='cifar10', asset_path=self.asset.relative_to(self.root).as_posix(),
                        sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest())
        self.manifest = dict(version=1, assets=[self.row], support_files=[])
        self.output = self.root / 'release/cifar10.tar.gz'

    def package(self):
        return tool.package_dataset(self.manifest, self.root, 'cifar10', self.output, self.license)

    def test_whitelist_hashes_license_and_no_overwrite(self):
        result = self.package()
        self.assertEqual(len(result['members']), 3)
        self.assertEqual(result['sha256'], tool.sha256(self.output))
        with tarfile.open(self.output) as archive:
            index = json.load(archive.extractfile(result['manifest_path']))
            self.assertEqual(len(index['files']), 2)
            self.assertTrue(all(member.isfile() for member in archive.getmembers()))
        with self.assertRaises(FileExistsError): self.package()

    def test_bad_hash_is_rejected_without_publishing(self):
        self.asset.write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError, 'changed while packaging'): self.package()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name(self.output.name + '.partial').exists())

    def test_unsafe_and_duplicate_paths_rejected(self):
        for name in ('../outside', '/absolute', 'C:/outside', 'a\\b'):
            with self.assertRaises(ValueError): tool.source_path(self.root, name)
        self.manifest['support_files'].append(dict(self.row))
        with self.assertRaisesRegex(ValueError, 'Duplicate'): self.package()

    def test_concurrent_final_file_is_preserved(self):
        link = tool.os.link
        def competitor(source, destination):
            if Path(destination) == self.output:
                self.output.write_bytes(b'other owner')
            return link(source, destination)
        with patch.object(tool.os, 'link', competitor), self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(self.output.read_bytes(), b'other owner')


if __name__ == '__main__':
    unittest.main()
