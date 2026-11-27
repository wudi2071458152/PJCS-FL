"""Accept the eight checksummed server initial states without training."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from accept_proxy_assets import copy_checked, digest, target


def state_fingerprint(state):
    sys.path.insert(0, str(ROOT / 'legacy/synthesis_label2_study'))
    from run_dirichlet_multiseed import state_fingerprint as retained
    return retained(state)


def accept(archive, checksum, manifest_path, artifacts):
    import torch
    archive, manifest_path, artifacts = map(Path, (archive, manifest_path, artifacts))
    line = Path(checksum).read_text(encoding='ascii').strip()
    match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
    if not match or match[2] != archive.name:
        raise ValueError('Checksum must identify the archive basename')
    archive_sha = digest(archive)
    if archive_sha != match[1]:
        raise ValueError('Archive SHA256 mismatch')
    expected = json.loads(manifest_path.read_text(encoding='utf-8'))
    if expected.get('version') != 1:
        raise ValueError('Expected version-1 initialization manifest')
    identities = {(d, s) for d in ('cifar10', 'fmnist') for s in (2027, 2028, 2029, 2030)}
    paths = {f'metadata/{d}/initializations/seed_{s}.pt' for d, s in identities}
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    stage = artifacts.parent / ('.accept_initializations_' + uuid.uuid4().hex)
    stage.mkdir()
    observed, index, names = {}, None, set()
    try:
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                if not member.isfile() or member.name in names:
                    raise ValueError('Nonregular or duplicate archive member')
                names.add(member.name)
                path = target(stage, member.name)
                if member.name == 'initialization_export_manifest.json':
                    index = json.load(source.extractfile(member))
                    continue
                if member.name not in paths:
                    raise ValueError('Unexpected initialization member: ' + member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                value, size = hashlib.sha256(), 0
                with source.extractfile(member) as incoming, path.open('xb') as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b''):
                        value.update(block); size += len(block); outgoing.write(block)
                if size != member.size:
                    raise ValueError('Truncated initialization member')
                observed[member.name] = dict(sha256=value.hexdigest(), bytes=size)
        if names != paths | {'initialization_export_manifest.json'} or index is None:
            raise ValueError('Initialization archive must contain exactly eight states and its index')
        if (index.get('version') != 1 or index.get('kind') != 'paper_initializations'
                or index.get('expected_manifest_sha256') != digest(manifest_path)
                or index.get('training_started') is not False):
            raise ValueError('Initialization index identity or expected manifest differs')
        records = index.get('models', [])
        if len(records) != 8 or {(r['dataset'], r['seed']) for r in records} != identities:
            raise ValueError('Initialization identities differ from the paper matrix')
        for row in records:
            dataset, seed = row['dataset'], row['seed']
            name = f'metadata/{dataset}/initializations/seed_{seed}.pt'
            if (row['asset_path'] != name or observed[name] != dict(sha256=row['sha256'], bytes=row['bytes'])
                    or row['state_sha256'] != expected['datasets'][dataset][str(seed)]):
                raise ValueError('Initialization member fingerprint or identity differs')
        # Only deserialize after the archive/file hashes and formal manifest are accepted.
        for row in records:
            state = torch.load(target(stage, row['asset_path']), map_location='cpu', weights_only=True)
            if not isinstance(state, dict) or not state or any(
                    not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in state.values()):
                raise ValueError('Initial state must contain finite named tensors')
            tensors = {n: dict(shape=list(t.shape), dtype=str(t.dtype)) for n, t in state.items()}
            if tensors != row['tensors'] or state_fingerprint(state) != row['state_sha256']:
                raise ValueError('Canonical initial state differs from formal tensor fingerprint')
        for name, value in observed.items():
            destination = target(artifacts, name)
            if destination.exists() and (not destination.is_file() or digest(destination) != value['sha256']):
                raise FileExistsError('Existing initialization differs: ' + name)
        copied = 0
        for name, value in observed.items():
            destination = target(artifacts, name)
            if not destination.exists():
                copy_checked(target(stage, name), destination, value['sha256'])
                copied += 1
        return dict(version=1, state='accepted', archive_name=archive.name, archive_sha256=archive_sha,
                    expected_manifest_sha256=digest(manifest_path), initializations_verified=8,
                    copied_files=copied, models=records, formal_training_started=False)
    finally:
        if not stage.resolve().is_relative_to(artifacts.parent.resolve()):
            raise ValueError('Cleanup escaped the artifact workspace')
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--checksum', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'assets/initialization_manifest.json')
    parser.add_argument('--artifacts-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--report', type=Path, default=ROOT / 'assets/accepted_initializations_report.json')
    args = parser.parse_args()
    result = accept(args.archive, args.checksum, args.manifest, args.artifacts_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}, indent=2))


if __name__ == '__main__':
    main()
