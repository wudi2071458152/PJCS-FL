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
spec = importlib.util.spec_from_file_location('accept_proxy_assets', ROOT/'tools/accept_proxy_assets.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.base = ROOT/'archive/test_runs'
        self.base.mkdir(parents=True, exist_ok=True)
        self.root = self.base/('asset_accept_'+uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.base.resolve()))
        shutil.rmtree(self.root)

    def fixture(self, missing=False, unsafe=False):
        self.archive = self.root/'data.tar.gz'
        snapshot, support = b'proxy', b'partition'
        exported = dict(dataset='cifar10',alpha='0p05',client=0,
            asset_path='proxies/cifar10/0p05/client_00/iter_32000.pt',
            sha256=hashlib.sha256(snapshot).hexdigest(),required_export=True)
        reused = dict(exported, client=1, asset_path='proxies/cifar10/0p05/client_01/iter_32000.pt',
            required_export=False,local_relative_path='old/iter_32000.pt')
        helper = dict(asset_path='metadata/cifar10/0p05/splits/a.pt', sha256=hashlib.sha256(support).hexdigest())
        self.manifest = dict(version=1,assets=[exported,reused], support_files=[helper])
        self.local = self.root/'local'
        (self.local/'old').mkdir(parents=True)
        (self.local/'old/iter_32000.pt').write_bytes(snapshot)
        contents={exported['asset_path']:snapshot}
        if not missing: contents[helper['asset_path']]=support
        if unsafe: contents['../outside']=b'bad'
        index=dict(version=1,assets=[{k:v for k,v in exported.items() if k!='required_export'}],
                   files=[dict(path=p,sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for p,v in contents.items()])
        contents['export_manifest.json']=(json.dumps(index)+'\n').encode()
        with tarfile.open(self.archive,'w:gz') as stream:
            for name,data in contents.items():
                member=tarfile.TarInfo(name); member.size=len(data)
                stream.addfile(member,io.BytesIO(data))
        self.checksum=self.root/'data.tar.gz.sha256'
        self.checksum.write_text(worker.digest(self.archive)+'  data.tar.gz\n',encoding='ascii')

    def run_accept(self):
        return worker.accept(self.archive,self.checksum,self.manifest,self.root/'artifacts',self.local)

    def test_valid_copy_and_source_unchanged(self):
        self.fixture()
        result=self.run_accept()
        self.assertEqual(result['proxy_snapshots_verified'],2)
        self.assertEqual(result['support_files_verified'],1)
        destination=self.root/'artifacts'/self.manifest['assets'][1]['asset_path']
        destination.write_bytes(b'independent')
        self.assertEqual((self.local/'old/iter_32000.pt').read_bytes(),b'proxy')
        with self.assertRaises(FileExistsError): self.run_accept()

    def test_bad_archive_hash_rejected_before_staging(self):
        self.fixture()
        self.checksum.write_text('0'*64+'  data.tar.gz\n',encoding='ascii')
        with self.assertRaisesRegex(ValueError,'Archive SHA256'): self.run_accept()
        self.assertFalse((self.root/'artifacts').exists())

    def test_missing_support_rejected(self):
        self.fixture(missing=True)
        with self.assertRaisesRegex(ValueError,'Required input missing'): self.run_accept()
        self.assertFalse((self.root/'artifacts').exists())

    def test_unsafe_member_rejected(self):
        self.fixture(unsafe=True)
        with self.assertRaises(ValueError): self.run_accept()
        self.assertFalse((self.root/'outside').exists())

    def test_copy_failure_cleans_only_own_destination(self):
        destination=self.root/'copy.pt'
        with self.assertRaises(FileNotFoundError):
            worker.copy_checked(self.root/'missing.pt',destination,'0'*64)
        self.assertFalse(destination.exists())
        destination.write_bytes(b'other')
        with self.assertRaises(FileExistsError):
            worker.copy_checked(self.root/'missing.pt',destination,'0'*64)
        self.assertEqual(destination.read_bytes(),b'other')


if __name__=='__main__': unittest.main()
