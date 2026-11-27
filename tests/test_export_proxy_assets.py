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

SPEC = importlib.util.spec_from_file_location('export_proxy_assets',
    Path(__file__).resolve().parents[1] / 'tools/export_proxy_assets.py')
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class ExportTests(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).resolve().parents[1] / 'archive/test_runs'
        runs.mkdir(parents=True, exist_ok=True)
        self.root = runs.resolve() / ('proxy_export_' + uuid.uuid4().hex)
        self.root.mkdir()
        def cleanup():
            self.assertTrue(self.root.resolve().is_relative_to(runs.resolve()))
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        self.source = self.root / 'old/synthesis/client_00/snapshots/iter_32000.pt'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'synthetic fixture, not a torch model')
        self.source.parent.parent.joinpath('config.json').write_text('{}', encoding='utf-8')
        self.source.parent.parent.joinpath('COMPLETED_32000.json').write_text('{}', encoding='utf-8')
        self.row = dict(dataset='cifar10', alpha='0p05', client=0,
            sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            server_relative_path=self.source.relative_to(self.root).as_posix(),
            asset_path='proxies/cifar10/0p05/client_00/snapshots/iter_32000.pt', required_export=True)
        self.manifest = dict(version=1, assets=[self.row])
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.output = self.root / 'export/proxies.tar.gz'

    def test_preflight_is_read_only(self):
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            tool.main(['--project-root', str(self.root), '--manifest', str(self.manifest_path),
                       '--output', str(self.output)])
        result = json.loads(capture.getvalue())
        self.assertEqual((result['state'], result['snapshots'], result['metadata_files']),
                         ('preflight_ok', 1, 2))
        self.assertFalse(self.output.parent.exists())

    def test_valid_archive_and_checksum(self):
        self.source.parent.parent.joinpath('optimizer.pt').write_bytes(b'excluded')
        files, assets = tool.collect(self.manifest, self.root)
        sha = tool.export(files, assets, self.output)
        self.assertEqual(sha, hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(self.output.with_name(self.output.name+'.sha256').read_text(),
                         sha+'  '+self.output.name+'\n')
        with tarfile.open(self.output, 'r:gz') as archive:
            self.assertEqual(set(archive.getnames()), set(files) | {'export_manifest.json'})
            for name, (_, expected) in files.items():
                self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), expected)
            self.assertEqual(json.load(archive.extractfile('export_manifest.json'))['assets'], assets)
        with self.assertRaises(FileExistsError):
            tool.export(files, assets, self.output)

    def test_missing_or_mismatched_snapshot(self):
        self.row['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            tool.collect(self.manifest, self.root)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            tool.collect(self.manifest, self.root)
        self.assertFalse(self.output.exists())

    def test_unsafe_and_duplicate_paths(self):
        for field in ('server_relative_path', 'asset_path'):
            original = self.row[field]
            for unsafe in ('../escape', '/absolute', 'C:/absolute', 'a\\b', 'a/../b'):
                self.row[field] = unsafe
                with self.assertRaises(ValueError):
                    tool.collect(self.manifest, self.root)
            self.row[field] = original
        self.manifest['assets'].append(dict(self.row))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tool.collect(self.manifest, self.root)

    def test_reused_asset_not_read(self):
        self.manifest['assets'].append(dict(self.row, client=1, required_export=False,
            server_relative_path='not_present.pt', asset_path='proxies/reused.pt'))
        files, assets = tool.collect(self.manifest, self.root)
        self.assertEqual(len(assets), 1)
        self.assertEqual(len(files), 3)

    def test_support_integrity_and_metadata_deduplication(self):
        partition = self.root / 'inputs/split.pt'
        partition.parent.mkdir()
        partition.write_bytes(b'exact partition fixture')
        config = self.source.parent.parent / 'config.json'
        self.manifest['support_files'] = [dict(server_relative_path=path.relative_to(self.root).as_posix(),
            asset_path=member, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path, member in ((partition, 'inputs/split.pt'),
                (config, 'proxies/cifar10/0p05/client_00/config.json'))]
        files, assets = tool.collect(self.manifest, self.root)
        self.assertEqual(len(files), 4)
        tool.export(files, assets, self.output)
        with tarfile.open(self.output, 'r:gz') as archive:
            self.assertEqual(archive.extractfile('inputs/split.pt').read(), partition.read_bytes())
        self.manifest['support_files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Support SHA256 mismatch'):
            tool.collect(self.manifest, self.root)

    def test_changed_source_is_not_published(self):
        files, assets = tool.collect(self.manifest, self.root)
        self.source.write_bytes(b'changed after preflight')
        with self.assertRaisesRegex(ValueError, 'Archive verification failed'):
            tool.export(files, assets, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name(self.output.name+'.partial').exists())

    def test_competing_partial_is_not_removed(self):
        files, assets = tool.collect(self.manifest, self.root)
        partial = self.output.with_name(self.output.name+'.partial')
        original_open = Path.open
        def competing_open(path, mode='r', *args, **kwargs):
            if path == partial and mode == 'xb':
                partial.write_bytes(b'other exporter')
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', competing_open), self.assertRaises(FileExistsError):
            tool.export(files, assets, self.output)
        self.assertEqual(partial.read_bytes(), b'other exporter')

    def test_competing_final_is_not_overwritten(self):
        files, assets = tool.collect(self.manifest, self.root)
        checksum_partial = self.output.with_name(self.output.name+'.sha256.partial')
        original_open = Path.open
        def competing_open(path, mode='r', *args, **kwargs):
            if path == checksum_partial and mode == 'x':
                self.output.write_bytes(b'other final archive')
            return original_open(path, mode, *args, **kwargs)
        def posix_rename(source, destination):
            os.replace(source, destination)
            return Path(destination)
        with patch.object(Path, 'open', competing_open), patch.object(Path, 'rename', posix_rename):
            with self.assertRaises(FileExistsError):
                tool.export(files, assets, self.output)
        self.assertEqual(self.output.read_bytes(), b'other final archive')
        self.assertFalse(checksum_partial.exists())

    def test_competing_checksum_is_not_overwritten(self):
        files, assets = tool.collect(self.manifest, self.root)
        checksum = self.output.with_name(self.output.name+'.sha256')
        checksum_partial = checksum.with_name(checksum.name+'.partial')
        original_open = Path.open
        def competing_open(path, mode='r', *args, **kwargs):
            if path == checksum_partial and mode == 'x':
                checksum.write_text('other checksum', encoding='ascii')
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', competing_open), self.assertRaises(FileExistsError):
            tool.export(files, assets, self.output)
        self.assertEqual(checksum.read_text(encoding='ascii'), 'other checksum')
        self.assertTrue(tarfile.is_tarfile(self.output))
        self.assertFalse(checksum_partial.exists())


if __name__ == '__main__':
    unittest.main()
