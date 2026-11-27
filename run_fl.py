"""Portable input binding for the retained LR=0.01 FL kernels; default is preflight."""
import argparse
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'legacy'))
sys.path.insert(0, str(ROOT / 'legacy/synthesis_label2_study'))
import numpy as np
import torch
import fmnist_common as fm
import run_fmnist_fl as worker
import schedule_dirichlet_screen as queue

c, legacy, multi = worker.c, worker.legacy, worker.multi
CONFIG = json.loads((ROOT / 'configs/paper_lr001.json').read_text(encoding='utf-8'))
PROXY = ('score_real', 'cluster_score_real_k3', 'representative_real_eta1p0')


def safe_path(root, relative):
    relative = str(relative)
    if not relative or '\\' in relative or ':' in relative or '..' in Path(relative).parts or Path(relative).is_absolute():
        raise ValueError('Expected a safe relative input path: ' + relative)
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Input path escapes root: ' + relative)
    return path


def isolated_output(output, *inputs):
    output = Path(output).resolve()
    for source in map(lambda p: Path(p).resolve(), inputs):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output overlaps immutable input directory: ' + str(source))


def partition_indices(value, size):
    rows = [np.asarray(row, dtype=np.int64) for row in value['client_indices']]
    if len(rows) != 32 or any(not len(row) for row in rows) or not np.array_equal(np.sort(np.concatenate(rows)), np.arange(size)):
        raise ValueError('Partition must cover the real training set exactly with 32 nonempty clients')
    return rows


def initial_model(dataset, alpha, seed, artifacts, initial_path, bind):
    net = fm.model(seed) if dataset == 'fmnist' else c.model(seed)
    manifest = ROOT / 'assets/initialization_manifest.json'
    expected = json.loads(manifest.read_text(encoding='utf-8'))['datasets'][dataset][str(seed)]
    expected = expected['state_sha256'] if isinstance(expected, dict) else expected
    bind(manifest, 'release/initialization_manifest.json')
    if seed == 2025:
        net.load_state_dict(c.load(initial_path))
    else:
        saved = safe_path(artifacts, f'metadata/{dataset}/initializations/seed_{seed}.pt')
        if saved.is_file():
            # Exported initializations are plain tensor state_dicts. No pickle objects are needed.
            bind(saved, 'initialization/seed_model.pt')
            net.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True))
    actual = multi.state_sha256(net)
    if actual != expected:
        raise RuntimeError(f'Initial model differs from the recorded {dataset} seed {seed} state. '
                           'Use the recorded environment or the checksummed server initialization export; '
                           'do not replace the expected fingerprint with a locally generated one.')
    return net, actual


def prepare(a):
    if (a.dataset not in CONFIG['dataset_names'] or a.alpha not in CONFIG['alphas']
            or a.seed not in CONFIG['fl_seeds'] or a.route not in CONFIG['routes'] or not 1 <= a.rounds <= 1500):
        raise ValueError('Unsupported dataset, alpha, seed, route or round target')
    artifacts, data = Path(a.artifacts_root).resolve(), Path(a.data_root).resolve()
    isolated_output(a.output, artifacts, data, ROOT / 'legacy')
    paths, hashes = {}, {}
    def bind(path, role, expected=None):
        path = Path(path).resolve()
        actual = c.digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Input SHA256 mismatch: ' + str(path))
        paths[role], hashes[role] = str(path), actual
        return path
    for name in ('run_fl.py', 'configs/paper_lr001.json', 'assets/proxy_manifest.json', 'assets/source_manifest.json'):
        bind(ROOT / name, 'release/' + name)
    for item in c.read(ROOT / 'assets/source_manifest.json')['files']:
        bind(safe_path(ROOT, item['path']), 'code/' + item['path'], item['sha256'])
    inventory = c.read(ROOT / 'assets/proxy_manifest.json')
    group = [r for r in inventory['support_files'] if (r['dataset'], r['alpha']) == (a.dataset, a.alpha)]
    support = {r['role']: bind(safe_path(artifacts, r['asset_path']), r['role'], r['sha256']) for r in group}
    manifest = c.read(support['inputs/manifest.json'])
    for name, wanted in manifest['data'].items():
        bind(safe_path(data, name), 'data/' + name, wanted)
    dataset = fm.dataset if a.dataset == 'fmnist' else c.dataset
    x, y = dataset(data)
    expected_size = 60000 if a.dataset == 'fmnist' else 50000
    if len(y) != expected_size:
        raise ValueError('Unexpected benchmark training-set size')
    split = support['inputs/splits/alpha' + a.alpha + '.pt']
    indices = partition_indices(c.load(split), len(y))
    counts = [torch.bincount(y[row], minlength=10).tolist() for row in indices]
    recorded = next(s['class_counts'] for s in c.read(support['inputs/counts.json'])['scenarios']
                    if str(s['alpha']).replace('.', 'p') == a.alpha)
    if recorded != counts:
        raise ValueError('Exact partition class counts differ from the recorded inputs')
    real = [(x[row], y[row]) for row in indices]
    weights = np.asarray([len(row) for row in indices], dtype=np.float64) / len(y)
    syn = None
    if a.route in PROXY:
        syn = []
        rows = sorted((r for r in inventory['assets'] if (r['dataset'], r['alpha']) == (a.dataset, a.alpha)), key=lambda r: r['client'])
        if [r['client'] for r in rows] != list(range(32)):
            raise ValueError('Proxy inventory must contain each client exactly once')
        shape = (1, 28, 28) if a.dataset == 'fmnist' else (3, 32, 32)
        for item in rows:
            path = bind(safe_path(artifacts, item['asset_path']), f"proxy/client_{item['client']:02d}/snapshot.pt", item['sha256'])
            value = c.load(path)
            images, labels = value['images'], value['labels']
            if (value.get('iteration') != 32000 or tuple(images.shape[1:]) != shape or len(images) != len(labels)
                    or not len(labels) or not torch.isfinite(images).all() or labels.min() < 0 or labels.max() >= 10):
                raise ValueError('Invalid final proxy snapshot: ' + str(path))
            class_weights = c.class_weights(counts[item['client']], torch.bincount(labels, minlength=10).tolist())
            syn.append((images, labels, class_weights))
    initial_path = next(path for role, path in support.items() if role in ('source/initial_model.pt', 'source/reference/round_0000.pt'))
    net, state_sha = initial_model(a.dataset, a.alpha, a.seed, artifacts, initial_path, bind)
    protocol = dict(suite='pjcs_portable_lr001', dataset=a.dataset, alpha=a.alpha, route=a.route,
        seed=a.seed, rounds=a.rounds, clients=32, selected=3, local_steps=3, batch=256,
        real_lr=.01, proxy_lr=.01 if syn is not None else None, real_ce='ordinary',
        proxy_ce=CONFIG['proxy_ce'] if syn is not None else None, augmentation=False,
        iteration=32000 if syn is not None else 0, weights=weights.tolist(),
        model_parameters=sum(p.numel() for p in net.parameters()),
        initial_model_state_sha256=legacy.cv.tensor_fingerprint(net.state_dict()),
        fl_initialization=dict(seed=a.seed, state_sha256=state_sha, shared_across_methods=True),
        files_paths=paths, files_sha256=hashes, communication=legacy.costs.policy(net, syn),
        software=legacy.costs.runtime(), round_seeds=dict(base=f'{a.seed}+round*1000', real='+client',
            uniform='+17', score='+400000', proxy='+10000000+client', joint_branch='+600000', joint_subset='+700000'))
    protocol.update(worker.route_definition(a.route))
    return net, real, syn, weights, dataset(data, train=False), protocol


def verify_output(folder, expected_rounds=1500):
    folder = Path(folder)
    protocol, marker = c.read(folder / 'protocol.json'), c.read(folder / 'COMPLETED.json')
    if protocol['suite'] != 'pjcs_portable_lr001' or protocol['rounds'] != expected_rounds or marker['rounds'] != expected_rounds:
        raise ValueError('Result identity or target differs')
    for key in ('protocol', 'rounds', 'summary'):
        if c.digest(folder / (key + '.json')) != marker[key + '_sha256']:
            raise ValueError('Result fingerprint changed: ' + key)
    for role, wanted in protocol['files_sha256'].items():
        if c.digest(protocol['files_paths'][role]) != wanted:
            raise ValueError('Bound input changed: ' + role)
    rows = c.read(folder / 'rounds.json')
    if len(rows) != expected_rounds:
        raise ValueError('Incomplete round coverage')
    worker.verify_records(protocol, rows)
    return protocol


def train(a, prepared):
    folder = Path(a.output)
    if folder.exists() and not (folder / 'protocol.json').exists() and any(p.name != 'scheduler.lock' for p in folder.iterdir()):
        raise ValueError('Refusing to reuse a nonempty output without a bound protocol')
    # Rebind only input-independent dispatch and final verification. The retained loop
    # owns SGD, RNG, apply/evaluation, costs, checkpoint integrity, resume and summaries.
    kernel = legacy.clone(legacy.train, initial_state=worker.initial_state, one_round=worker.one_round,
        verify_records=worker.verify_records, validate_state=worker.validate_state, verify=verify_output)
    return kernel(a, prepared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=CONFIG['dataset_names'], required=True)
    parser.add_argument('--alpha', choices=CONFIG['alphas'], required=True)
    parser.add_argument('--route', choices=CONFIG['routes'], required=True)
    parser.add_argument('--seed', type=int, choices=CONFIG['fl_seeds'], default=2025)
    parser.add_argument('--artifacts-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=1500, help='paper target is 1500; smaller values are diagnostics')
    parser.add_argument('--run', action='store_true')
    a = parser.parse_args(argv)
    a.deadline = None
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Formal FL requires CUDA; preflight and file tests can run on CPU')
    def stop(*_):
        a.deadline = 0.
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with queue.lock(a.output):
        raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__':
    main()
