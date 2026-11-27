"""Export the eight missing paper initial states; no SGD, data load or training."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import uuid

DATASETS = ('cifar10', 'fmnist')
SEEDS = (2027, 2028, 2029, 2030)
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def model_factory(dataset, seed):
    # Retain the original initializers and original tensor-content fingerprint.
    sys.path.insert(0, str(ROOT / 'legacy/synthesis_label2_study'))
    import dirichlet_screen_common as common
    import fmnist_common as fashion
    import run_dirichlet_multiseed as multi
    net = (common.model if dataset == 'cifar10' else fashion.model)(seed)
    fingerprint = multi.state_sha256(net)
    state = {name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()}
    return state, fingerprint


def prepare(manifest, factory=model_factory):
    if manifest.get('version') != 1:
        raise ValueError('Expected version-1 initialization manifest')
    prepared = []
    for dataset in DATASETS:
        for seed in SEEDS:
            expected = manifest['datasets'][dataset][str(seed)]
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in '0123456789abcdef' for c in expected)):
                raise ValueError('Invalid expected initialization fingerprint')
            state, actual = factory(dataset, seed)
            if actual != expected:
                raise ValueError(f'Initialization mismatch: {dataset} seed {seed}; '
                                 'use the recorded compatible server environment; do not change expected hashes')
            prepared.append(dict(dataset=dataset, seed=seed, state=state, state_sha256=actual))
    return prepared


def export(prepared, output, manifest_sha256):
    import torch
    output = Path(output).absolute()
    checksum = output.with_name(output.name + '.sha256')
    partial = output.with_name(output.name + '.partial')
    check_partial = checksum.with_name(checksum.name + '.partial')
    for path in (output, checksum, partial, check_partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Refusing to overwrite: ' + str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('initializations_' + uuid.uuid4().hex)
    stage.mkdir()
    own_partial = own_check_partial = False
    try:
        records = []
        for item in prepared:
            relative = f"metadata/{item['dataset']}/initializations/seed_{item['seed']}.pt"
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(item['state'], path)
            records.append(dict(dataset=item['dataset'], seed=item['seed'], asset_path=relative,
                sha256=sha256(path), state_sha256=item['state_sha256'], bytes=path.stat().st_size,
                tensors={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                         for name, value in item['state'].items()}))
        index = dict(version=1, kind='paper_initializations',
                     expected_manifest_sha256=manifest_sha256, models=records, training_started=False)
        payload = (json.dumps(index, indent=2) + '\n').encode('utf-8')
        stream = partial.open('xb')
        own_partial = True
        with stream, tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
            for row in records:
                info = archive.gettarinfo(str(stage / row['asset_path']), arcname=row['asset_path'])
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with (stage / row['asset_path']).open('rb') as content:
                    archive.addfile(info, content)
            info = tarfile.TarInfo('initialization_export_manifest.json')
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with tarfile.open(partial, 'r:gz') as archive:
            members = archive.getmembers()
            if (len(members) != 9 or not all(member.isfile() for member in members)
                    or {member.name for member in members}
                    != {row['asset_path'] for row in records} | {'initialization_export_manifest.json'}):
                raise ValueError('Initialization archive whitelist mismatch')
            for row in records:
                with archive.extractfile(row['asset_path']) as content:
                    if hashlib.sha256(content.read()).hexdigest() != row['sha256']:
                        raise ValueError('Initialization archive member mismatch')
            if archive.extractfile('initialization_export_manifest.json').read() != payload:
                raise ValueError('Initialization archive manifest mismatch')
        digest = sha256(partial)
        stream = check_partial.open('x', encoding='ascii', newline='\n')
        own_check_partial = True
        with stream:
            stream.write(digest + '  ' + output.name + '\n')
        # Same-directory hard links provide atomic no-clobber publication on Linux too.
        os.link(partial, output)
        os.link(check_partial, checksum)
        return digest
    finally:
        if own_partial:
            partial.unlink(missing_ok=True)
        if own_check_partial:
            check_partial.unlink(missing_ok=True)
        if stage.resolve().parent != output.parent.resolve():
            raise ValueError('Initialization staging cleanup escaped output directory')
        shutil.rmtree(stage)


def main(argv=None, factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'assets/initialization_manifest.json')
    parser.add_argument('--output', type=Path, default=Path('pjcs_initializations.tar.gz'))
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args(argv)
    prepared = prepare(json.loads(args.manifest.read_text(encoding='utf-8')), factory or model_factory)
    result = dict(state='preflight_ok', initializations=len(prepared), training_started=False)
    if args.export:
        result.update(state='export_complete', output=str(args.output.absolute()),
                      sha256=export(prepared, args.output, sha256(args.manifest)))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
