"""Accept a checksummed proxy export and copy the already audited local subset."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def target(root, name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or ':' in name or path.as_posix() != name or name == '.'):
        raise ValueError('Unsafe asset path: ' + name)
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Asset escapes root: ' + name)
    return result


def copy_checked(source, destination, expected):
    """Create an independent copy, never a hardlink to an original experiment file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = hashlib.sha256()
    created = False
    try:
        outgoing = destination.open('xb')
        created = True
        with outgoing, Path(source).open('rb') as incoming:
            for block in iter(lambda: incoming.read(1024 * 1024), b''):
                outgoing.write(block)
                value.update(block)
    except Exception:
        # A failed exclusive open must never remove somebody else's file.
        if created:
            destination.unlink(missing_ok=True)
        raise
    if value.hexdigest() != expected:
        destination.unlink()
        raise ValueError('Copied bytes differ: ' + str(destination))


def accept(archive, checksum, manifest, artifacts, local_project):
    archive, artifacts, local_project = map(Path, (archive, artifacts, local_project))
    line = Path(checksum).read_text(encoding='ascii').strip()
    match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
    if not match or match[2] != archive.name:
        raise ValueError('Checksum must identify this archive basename')
    archive_sha = digest(archive)
    if archive_sha != match[1]:
        raise ValueError('Archive SHA256 mismatch')
    if manifest.get('version') != 1:
        raise ValueError('Unsupported proxy manifest')
    exported = [a for a in manifest['assets'] if a['required_export']]
    reused = [a for a in manifest['assets'] if not a['required_export']]
    support = manifest.get('support_files', [])
    required = {a['asset_path']: a['sha256'] for a in exported + support}
    allowed = set(required)
    for row in exported:
        parent = PurePosixPath(row['asset_path']).parent
        allowed.update((parent / name).as_posix() for name in ('config.json', 'COMPLETED_32000.json'))
    if len(required) != len(exported) + len(support):
        raise ValueError('Duplicate expected asset paths')
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    stage = artifacts.parent / ('.accept_proxy_' + uuid.uuid4().hex)
    stage.mkdir()
    observed, index = {}, None
    try:
        # Streaming mode avoids one full gzip seek/decompression for every member.
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                if not member.isfile() or member.name in observed:
                    raise ValueError('Nonregular or duplicate archive member: ' + member.name)
                path = target(stage, member.name)
                if member.name == 'export_manifest.json':
                    if index is not None:
                        raise ValueError('Duplicate export index')
                    index = json.load(source.extractfile(member))
                    continue
                if member.name not in allowed:
                    raise ValueError('Unlisted archive member: ' + member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                sha = hashlib.sha256()
                size = 0
                with source.extractfile(member) as incoming, path.open('xb') as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b''):
                        sha.update(block)
                        size += len(block)
                        outgoing.write(block)
                if size != member.size:
                    raise ValueError('Truncated archive member')
                observed[member.name] = dict(sha256=sha.hexdigest(), bytes=size)
        if index is None or index.get('version') != 1:
            raise ValueError('Missing version-1 export index')
        indexed = {a['path']: dict(sha256=a['sha256'], bytes=a['bytes']) for a in index['files']}
        if len(indexed) != len(index['files']) or indexed != observed:
            raise ValueError('Export index differs from archive contents')
        expected_assets = [{k: a[k] for k in ('dataset', 'alpha', 'client', 'asset_path', 'sha256')}
                           for a in exported]
        if index.get('assets') != expected_assets:
            raise ValueError('Export snapshot identities differ from expected manifest')
        for name, expected in required.items():
            if name not in observed or observed[name]['sha256'] != expected:
                raise ValueError('Required input missing or changed: ' + name)
        # Copy the old, audited 96-client subset into the same independent layout.
        for row in reused:
            name = row['asset_path']
            if name in observed:
                raise ValueError('Duplicate reused asset path')
            source = target(local_project, row['local_relative_path'])
            path = target(stage, name)
            copy_checked(source, path, row['sha256'])
            observed[name] = dict(sha256=row['sha256'], bytes=path.stat().st_size)
        # Refuse all differing existing targets before publishing any new files.
        for name, value in observed.items():
            path = target(artifacts, name)
            if path.exists() and (not path.is_file() or digest(path) != value['sha256']):
                raise FileExistsError('Existing asset differs: ' + name)
        for name, value in observed.items():
            destination = target(artifacts, name)
            if not destination.exists():
                copy_checked(target(stage, name), destination, value['sha256'])
                if digest(destination) != value['sha256']:
                    raise ValueError('Published asset fingerprint differs: ' + name)
        report = dict(version=1, state='accepted', archive_name=archive.name, archive_sha256=archive_sha,
            archive_files=len(indexed), exported_snapshots=len(exported), reused_snapshots=len(reused),
            proxy_snapshots_verified=len(exported)+len(reused), support_files_verified=len(support),
            accepted_files=len(observed), accepted_uncompressed_bytes=sum(x['bytes'] for x in observed.values()),
            asset_paths=sorted(observed), formal_training_started=False)
        return report
    finally:
        if not stage.resolve().is_relative_to(artifacts.parent.resolve()):
            raise ValueError('Cleanup escaped the asset workspace')
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'checksum', 'manifest', 'artifacts-root', 'local-project-root', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    result = accept(args.archive, args.checksum, manifest, args.artifacts_root, args.local_project_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'asset_paths'}, indent=2))


if __name__ == '__main__':
    main()
