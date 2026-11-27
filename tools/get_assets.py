"""List, download or safely install the checksummed public data release; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def safe(root, name):
    path = PurePosixPath(name)
    if (not name or name == '.' or path.is_absolute() or '..' in path.parts
            or ':' in name or '\\' in name or path.as_posix() != name):
        raise ValueError('Unsafe asset path: ' + name)
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes root: ' + name)
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def download(item, folder):
    target = safe(folder, item['name'])
    if target.exists():
        if digest(target) != item['sha256']:
            raise ValueError('Existing download has a different checksum: ' + target.name)
        return target
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (item['name'] + '.partial.' + uuid.uuid4().hex)
    try:
        with urllib.request.urlopen(item['url'], timeout=60) as incoming, temporary.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
            raise ValueError('Downloaded archive fingerprint differs: ' + item['name'])
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def install(item, archive, destination):
    if archive.stat().st_size != item['bytes'] or digest(archive) != item['sha256']:
        raise ValueError('Archive fingerprint differs: ' + item['name'])
    expected = {row['path']: row for row in item['members']}
    if len(expected) != len(item['members']):
        raise ValueError('Duplicate release manifest member')
    for name in expected:
        safe(destination, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / ('asset_install_' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        with tarfile.open(archive, 'r:gz') as stream:
            members = stream.getmembers()
            if (len(members) != len(expected) or {m.name for m in members} != set(expected)
                    or any(not m.isfile() for m in members)):
                raise ValueError('Archive member whitelist differs')
            for member in members:
                row = expected[member.name]
                if member.size != row['bytes']:
                    raise ValueError('Archive member size differs: ' + member.name)
                target = safe(stage, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if digest(target) != row['sha256']:
                    raise ValueError('Archive member fingerprint differs: ' + member.name)
        # Validate every existing destination before publishing any new file.
        for name, row in expected.items():
            target = safe(destination, name)
            if target.exists() and (not target.is_file() or digest(target) != row['sha256']):
                raise ValueError('Refusing to replace a different installed file: ' + name)
        for name in expected:
            target = safe(destination, name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(safe(stage, name), target)
    finally:
        if stage.resolve().parent != destination.parent.resolve():
            raise ValueError('Staging cleanup escaped destination parent')
        shutil.rmtree(stage)
    return dict(archive=item['name'], installed_files=len(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'assets/download_manifest.json')
    parser.add_argument('--dataset', choices=('all', 'cifar10', 'fmnist'), default='all')
    parser.add_argument('--archive-dir', type=Path, default=ROOT / 'downloads')
    parser.add_argument('--artifacts-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    files = [item for item in manifest['files'] if args.dataset == 'all'
             or item['dataset'] in (args.dataset, 'initializations')]
    if args.download and not manifest.get('published', False):
        parser.error('The configured public release is not published yet; install from --archive-dir instead')
    results = []
    for item in files:
        archive = download(item, args.archive_dir) if args.download else safe(args.archive_dir, item['name'])
        if args.install:
            results.append(install(item, archive, args.artifacts_root))
    print(json.dumps(dict(archives=[item['name'] for item in files], results=results,
                          downloaded=args.download, installed=args.install, training_started=False), indent=2))


if __name__ == '__main__':
    main()
