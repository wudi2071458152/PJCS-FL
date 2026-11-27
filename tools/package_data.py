"""Package accepted proxy inputs as checksummed public assets; never train."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('cifar10', 'fmnist')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_path(root, name):
    relative = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or relative.is_absolute()
            or '..' in relative.parts or relative.as_posix() != name or name == '.'):
        raise ValueError('Unsafe artifact path: ' + str(name))
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or escaping artifact: ' + name)
    return path


def collect(manifest, artifacts, dataset):
    if manifest.get('version') != 1 or dataset not in DATASETS:
        raise ValueError('Expected version-1 proxy manifest and supported dataset')
    rows = [row for key in ('assets', 'support_files') for row in manifest.get(key, [])
            if row['dataset'] == dataset]
    files = {}
    for row in rows:
        name = row['asset_path']
        if name in files or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Duplicate artifact or invalid SHA256: ' + name)
        path = source_path(Path(artifacts), name)
        files[name] = (path, row['sha256'], path.stat().st_size)
    if not files:
        raise ValueError('No artifacts for ' + dataset)
    return files


class CheckedReader:
    def __init__(self, stream):
        self.stream, self.value = stream, hashlib.sha256()

    def read(self, length):
        data = self.stream.read(length)
        self.value.update(data)
        return data


def member_info(name, size):
    info = tarfile.TarInfo(name)
    info.size, info.mode = size, 0o644
    return info


def package_dataset(manifest, artifacts, dataset, output, license_file):
    output = Path(output).absolute()
    partial = output.with_name(output.name + '.partial')
    checksum = output.with_name(output.name + '.sha256')
    check_partial = checksum.with_name(checksum.name + '.partial')
    for path in (output, partial, checksum, check_partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Refusing to overwrite: ' + str(path))
    files = collect(manifest, artifacts, dataset)
    license_path = Path(license_file).resolve()
    license_name = 'metadata/' + dataset + '/DATA_LICENSE.txt'
    index_name = 'metadata/' + dataset + '/release_manifest.json'
    if license_name in files or index_name in files:
        raise ValueError('Reserved dataset metadata path')
    files[license_name] = (license_path, sha256(license_path), license_path.stat().st_size)
    members = [dict(path=name, sha256=sha, bytes=size) for name, (_, sha, size) in sorted(files.items())]
    index = dict(version=1, dataset=dataset, synthesis_iterations=32000, files=members)
    payload = (json.dumps(index, indent=2) + '\n').encode('utf-8')
    members = members + [dict(path=index_name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))]
    output.parent.mkdir(parents=True, exist_ok=True)
    own_partial = own_check = False
    try:
        stream = partial.open('xb')
        own_partial = True
        with stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, (path, wanted, size) in sorted(files.items()):
                    with path.open('rb') as incoming:
                        reader = CheckedReader(incoming)
                        archive.addfile(member_info(name, size), reader)
                        if reader.value.hexdigest() != wanted or incoming.read(1):
                            raise ValueError('Artifact bytes changed while packaging: ' + name)
                archive.addfile(member_info(index_name, len(payload)), io.BytesIO(payload))
        expected = {row['path']: row for row in members}
        with tarfile.open(partial, 'r:gz') as archive:
            entries = archive.getmembers()
            if (len(entries) != len(expected) or {row.name for row in entries} != set(expected)
                    or not all(row.isfile() and row.size == expected[row.name]['bytes'] for row in entries)):
                raise ValueError('Public archive whitelist or sizes differ')
            for entry in entries:
                with archive.extractfile(entry) as incoming:
                    value = hashlib.sha256()
                    for block in iter(lambda: incoming.read(1024 * 1024), b''):
                        value.update(block)
                if value.hexdigest() != expected[entry.name]['sha256']:
                    raise ValueError('Public archive bytes differ: ' + entry.name)
        digest = sha256(partial)
        if partial.stat().st_size >= 2 * 1024 ** 3:
            raise ValueError('Public asset exceeds 2 GiB')
        outgoing = check_partial.open('x', encoding='ascii', newline='\n')
        own_check = True
        with outgoing:
            outgoing.write(digest + '  ' + output.name + '\n')
        os.link(partial, output)
        os.link(check_partial, checksum)
        return dict(name=output.name, dataset=dataset, sha256=digest,
                    bytes=output.stat().st_size, manifest_path=index_name, members=members)
    finally:
        if own_partial:
            partial.unlink(missing_ok=True)
        if own_check:
            check_partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'assets/proxy_manifest.json')
    parser.add_argument('--artifacts-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--license-file', type=Path, default=ROOT / 'DATA_LICENSE.txt')
    parser.add_argument('--download-manifest', type=Path, default=ROOT / 'assets/download_manifest.json')
    parser.add_argument('--package', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    counts = {dataset: len(collect(manifest, args.artifacts_root, dataset)) for dataset in DATASETS}
    if not args.package:
        print(json.dumps(dict(state='preflight_ok', files=counts, training_started=False)))
        return
    if args.download_manifest.exists():
        raise FileExistsError('Refusing to overwrite download manifest')
    files = [package_dataset(manifest, args.artifacts_root, dataset,
             args.output_dir / ('pjcs_' + dataset + '_32k_v1.tar.gz'), args.license_file) for dataset in DATASETS]
    base = 'https://github.com/wudi2071458152/PJCS-FL/releases/download/v1.0.0/'
    for row in files:
        row['url'] = base + row['name']
    args.download_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.download_manifest.open('x', encoding='utf-8') as stream:
        json.dump(dict(version=1, release_tag='v1.0.0', published=False, files=files), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(state='packaged', files=[row['name'] for row in files], training_started=False)))


if __name__ == '__main__':
    main()
