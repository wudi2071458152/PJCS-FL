"""Verify the prepared source snapshot and the public proxy inventory; never train."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def safe_path(root, name):
    value = PurePosixPath(name)
    if not name or value.is_absolute() or '..' in value.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe manifest path: ' + name)
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes root: ' + name)
    return path


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(root=ROOT, assets_root=None):
    source = json.loads((root / 'assets/source_manifest.json').read_text(encoding='utf-8'))
    # The source manifest contains exact bytes, not a promise of a full numerical rerun.
    hashes = {item['path']: item['sha256'] for item in source['files']}
    if len(hashes) != len(source['files']):
        raise ValueError('Duplicate source manifest path')
    for name, expected in hashes.items():
        path = safe_path(root, name)
        if sha256(path) != expected:
            raise ValueError('Source fingerprint changed: ' + name)
    inventory = json.loads((root / 'assets/proxy_manifest.json').read_text(encoding='utf-8'))
    assets = inventory['assets']
    keys = [(a['dataset'], a['alpha'], a['client']) for a in assets]
    groups = Counter((d, a) for d, a, _ in keys)
    expected_groups = {(d, a) for d in ('cifar10', 'fmnist')
                       for a in ('0p05', '0p07', '0p1', '0p2', '0p3')}
    if (len(keys) != 320 or len(set(keys)) != 320 or set(groups) != expected_groups
            or any(n != 32 for n in groups.values()) or any(c not in range(32) for _, _, c in keys)):
        raise ValueError('Proxy inventory must cover ten groups of 32 clients exactly')
    paths = set()
    for item in assets + inventory.get('support_files', []):
        name = item['asset_path']
        if name in paths:
            raise ValueError('Duplicate asset path: ' + name)
        paths.add(name)
        safe_path(root, name)
        digest = item['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid asset fingerprint: ' + name)
        if assets_root is not None and sha256(safe_path(assets_root, name)) != digest:
            raise ValueError('Asset fingerprint changed: ' + name)
    return dict(source_files_verified=len(hashes), proxy_inventory=320,
                proxy_tensors_verified=320 if assets_root is not None else 0,
                formal_training_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(assets_root=args.assets_root), indent=2))
