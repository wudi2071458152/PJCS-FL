"""Verify and export only missing proxy snapshots; default is read-only preflight."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def safe_name(value):
    if not isinstance(value, str):
        raise ValueError('Relative paths must be strings')
    name = PurePosixPath(value)
    if (not value or value == '.' or '\\' in value or ':' in value
            or name.is_absolute() or '..' in name.parts or name.as_posix() != value):
        raise ValueError('Unsafe relative path: ' + str(value))
    return name


def collect(manifest, project_root):
    if manifest.get('version') != 1 or not isinstance(manifest.get('assets'), list):
        raise ValueError('Expected version-1 proxy asset manifest')
    root = Path(project_root).resolve()
    files, assets, names = {}, [], set()
    for row in manifest['assets']:
        relative, archive = map(safe_name, (row['server_relative_path'], row['asset_path']))
        if archive.as_posix() in names or archive.as_posix() == 'export_manifest.json':
            raise ValueError('Duplicate or reserved asset path: ' + str(archive))
        names.add(archive.as_posix())
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid expected SHA256')
        if type(row['required_export']) is not bool:
            raise ValueError('required_export must be a boolean')
        if not row['required_export']:
            continue
        source = (root / Path(relative)).resolve()
        if not source.is_relative_to(root):
            raise ValueError('Source escapes project root: ' + str(relative))
        with source.open('rb') as stream:
            actual = digest(stream)
        if actual != row['sha256']:
            raise ValueError('Snapshot SHA256 mismatch: ' + str(relative))
        files[archive.as_posix()] = (source, actual)
        assets.append({key: row[key] for key in ('dataset', 'alpha', 'client', 'asset_path', 'sha256')})
        source_client = source.parent.parent if source.parent.name == 'snapshots' else source.parent
        archive_client = archive.parent.parent if archive.parent.name == 'snapshots' else archive.parent
        for filename in ('config.json', 'COMPLETED_32000.json'):
            metadata = source_client / filename
            if metadata.exists():
                metadata = metadata.resolve()
                member = safe_name((archive_client / filename).as_posix()).as_posix()
                if not metadata.is_relative_to(root) or member in files or member in names:
                    raise ValueError('Unsafe or duplicate metadata path: ' + member)
                with metadata.open('rb') as stream:
                    files[member] = (metadata, digest(stream))
    for row in manifest.get('support_files', []):
        relative, archive = map(safe_name, (row['server_relative_path'], row['asset_path']))
        member = archive.as_posix()
        if member in names or member == 'export_manifest.json':
            raise ValueError('Duplicate or reserved support path: ' + member)
        names.add(member)
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid expected support SHA256')
        source = (root / Path(relative)).resolve()
        if not source.is_relative_to(root):
            raise ValueError('Support source escapes project root: ' + str(relative))
        with source.open('rb') as stream:
            actual = digest(stream)
        if actual != row['sha256']:
            raise ValueError('Support SHA256 mismatch: ' + str(relative))
        if member in files and files[member] != (source, actual):
            raise ValueError('Conflicting metadata/support path: ' + member)
        files[member] = (source, actual)
    if not assets:
        raise ValueError('No assets marked required_export')
    return files, assets


def export(files, assets, output):
    output = Path(output).absolute()
    checksum = output.with_name(output.name + '.sha256')
    partial = output.with_name(output.name + '.partial')
    checksum_partial = checksum.with_name(checksum.name + '.partial')
    for path in (output, checksum, partial, checksum_partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Refusing to overwrite: ' + str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    index = dict(version=1, assets=assets, files=[dict(path=name, sha256=sha,
        bytes=source.stat().st_size) for name, (source, sha) in sorted(files.items())])
    payload = (json.dumps(index, indent=2) + '\n').encode('utf-8')
    partial_created = checksum_created = False
    try:
        stream = partial.open('xb')
        partial_created = True
        with stream:
            with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
                for name, (source, _) in sorted(files.items()):
                    info = archive.gettarinfo(str(source), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with source.open('rb') as content:
                        archive.addfile(info, content)
                info = tarfile.TarInfo('export_manifest.json')
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        with tarfile.open(partial, 'r:gz') as archive:
            members = archive.getmembers()
            if (len(members) != len(files) + 1 or not all(m.isfile() for m in members)
                    or {m.name for m in members} != set(files) | {'export_manifest.json'}):
                raise ValueError('Archive member whitelist mismatch')
            for name, (_, sha) in files.items():
                with archive.extractfile(name) as stream:
                    if digest(stream) != sha:
                        raise ValueError('Archive verification failed: ' + name)
            if archive.extractfile('export_manifest.json').read() != payload:
                raise ValueError('Archive index mismatch')
        with partial.open('rb') as stream:
            archive_sha = digest(stream)
        stream = checksum_partial.open('x', encoding='ascii', newline='\n')
        checksum_created = True
        with stream:
            stream.write(archive_sha + '  ' + output.name + '\n')
        os.link(partial, output)
        os.link(checksum_partial, checksum)
        partial.unlink()
        checksum_partial.unlink()
    except Exception:
        if partial_created:
            partial.unlink(missing_ok=True)
        if checksum_created:
            checksum_partial.unlink(missing_ok=True)
        raise
    return archive_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export', action='store_true', help='write archive after input verification')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    files, assets = collect(manifest, args.project_root)
    result = dict(state='preflight_ok', snapshots=len(assets), metadata_files=len(files)-len(assets),
        declared_support_files=len(manifest.get('support_files', [])),
        uncompressed_bytes=sum(source.stat().st_size for source, _ in files.values()),
        training_started=False, output=str(args.output.absolute()))
    if args.export:
        result.update(state='export_complete', sha256=export(files, assets, args.output))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
