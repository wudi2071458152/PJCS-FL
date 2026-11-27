"""Verify complete paper runs and package only compact JSON records, never tensors."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


ROUNDS = 1500


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return digest(stream)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def collect(root, jobs):
    files, index, identities = {}, [], set()
    if not jobs:
        raise ValueError('Cannot package an empty result matrix')
    for job in jobs:
        identity = tuple(job[k] for k in ('dataset', 'alpha', 'seed', 'method'))
        if identity in identities:
            raise ValueError('Duplicate expected run')
        identities.add(identity)
        if (type(job['seed']) is not int or any(not isinstance(v, str) or not v or v in ('.', '..')
                or any(c in v for c in '/\\:') for v in (job['dataset'], job['alpha'], job['method']))):
            raise ValueError('Unsafe result identity')
        relative = '/'.join(str(v) for v in identity)
        folder = (root / relative).resolve()
        if not folder.is_relative_to(root.resolve()):
            raise ValueError('Result path escapes output root')
        payload = {}
        for name in ('protocol', 'rounds', 'summary', 'COMPLETED'):
            path = folder / (name + '.json')
            payload[name] = json.loads(path.read_text(encoding='utf-8'))
            files[relative + '/' + path.name] = (path, file_sha(path), path.stat().st_size)
        protocol, marker, rows = (payload[k] for k in ('protocol', 'COMPLETED', 'rounds'))
        if (protocol.get('suite') != 'pjcs_portable_lr001' or protocol.get('rounds') != ROUNDS
                or marker.get('rounds') != ROUNDS or payload['summary'].get('rounds') != ROUNDS
                or tuple(protocol.get(k) for k in ('dataset', 'alpha', 'seed', 'route')) != identity):
            raise ValueError('Result identity or target differs: ' + relative)
        if len(rows) != ROUNDS or [row.get('round') for row in rows] != list(range(1, ROUNDS + 1)):
            raise ValueError('Incomplete round coverage: ' + relative)
        for name in ('protocol', 'rounds', 'summary'):
            if marker.get(name + '_sha256') != files[relative + '/' + name + '.json'][1]:
                raise ValueError('Result fingerprint changed: ' + relative + '/' + name)
        status = folder / 'status.json'
        if status.exists():
            if json.loads(status.read_text(encoding='utf-8')).get('state') != 'complete':
                raise ValueError('Result status is not complete: ' + relative)
            files[relative + '/status.json'] = (status, file_sha(status), status.stat().st_size)
        index.append(dict(dataset=job['dataset'], alpha=job['alpha'], seed=job['seed'],
                          route=job['method'], rounds=ROUNDS, path=relative))
    data = encode(dict(version=1, runs=len(jobs), results=index))
    files['result_index.json'] = (data, hashlib.sha256(data).hexdigest(), len(data))
    data = encode(dict(version=1, files=[dict(path=name, sha256=sha, bytes=size)
                                        for name, (_, sha, size) in sorted(files.items())]))
    files['packaging_manifest.json'] = (data, hashlib.sha256(data).hexdigest(), len(data))
    return files


def verify_archive(path, files):
    seen = set()
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in files or member.name in seen:
                raise ValueError('Unexpected archive member: ' + member.name)
            seen.add(member.name)
            if member.size != files[member.name][2]:
                raise ValueError('Archive size differs: ' + member.name)
            with archive.extractfile(member) as stream:
                if digest(stream) != files[member.name][1]:
                    raise ValueError('Archive fingerprint differs: ' + member.name)
    if seen != set(files):
        raise ValueError('Archive member coverage differs')


def package(root, jobs, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    files = collect(root, jobs)
    checksum = output.with_name(output.name + '.sha256')
    if output.exists():
        verify_archive(output, files)
        sha = file_sha(output)
        if checksum.read_text(encoding='ascii') != sha + '  ' + output.name + '\n':
            raise ValueError('Existing archive checksum differs')
        return dict(archive=output.name, sha256=sha, runs=len(jobs), reused=True)
    partial, check_partial = (p.with_name(p.name + '.partial') for p in (output, checksum))
    for path in (output, checksum, partial, check_partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Refusing to overwrite: ' + str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    own_partial = own_check = False
    try:
        stream = partial.open('xb'); own_partial = True
        with stream, tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
            for name, (source, _, size) in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = size, 0o644
                with io.BytesIO(source) if isinstance(source, bytes) else source.open('rb') as content:
                    archive.addfile(member, content)
        verify_archive(partial, files)
        sha = file_sha(partial)
        stream = check_partial.open('x', encoding='ascii', newline='\n'); own_check = True
        with stream:
            stream.write(sha + '  ' + output.name + '\n')
        os.link(partial, output)
        os.link(check_partial, checksum)
        return dict(archive=output.name, sha256=sha, runs=len(jobs), reused=False)
    finally:
        if own_partial: partial.unlink(missing_ok=True)
        if own_check: check_partial.unlink(missing_ok=True)
