"""Fashion-MNIST inputs and a scoped adapter around the frozen numerical kernels."""
import gzip
import hashlib
import http.client
import json
import os
import ssl
import struct
import urllib.error
import urllib.request
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import dirichlet_screen_common as c
from utils import split_dataset_indices

ROOT = c.ROOT
SUITE = 'fmnist_20261016'
INPUTS = ROOT / 'inputs' / SUITE
ALPHAS = ('0p05', '0p07', '0p1')
CONFIG = 'weighted_weak_x10'
SEEDS = (2025, 2026)
FL_SEEDS = (2025, 2027, 2028, 2029, 2030)
STAGES = (4000, 8000)
RAW = Path('FashionMNIST/raw')
MEAN, STD = .2861, .3530
# The four checksums are published in the dataset authors' README.
RESOURCES = {
    'train-images-idx3-ubyte.gz': ('8d4fb7e6c68d591d4c3dfef9ec88bf0d', 60000, 3),
    'train-labels-idx1-ubyte.gz': ('25c81989df183df01b3e8a0aad5dffbe', 60000, 1),
    't10k-images-idx3-ubyte.gz': ('bef4ecab320f06d8554ea6380940ec79', 10000, 3),
    't10k-labels-idx1-ubyte.gz': ('bb300cfdad3c16e7a12a480ee83cd310', 10000, 1),
}
BASE_URL = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
DOWNLOAD_URLS = (BASE_URL, 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
_legacy_protocol = c.protocol


def _md5(path):
    h = hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _idx(path, count, dimensions):
    with gzip.open(path, 'rb') as stream: raw = stream.read()
    header_size = 4 * (dimensions + 1)
    if len(raw) < header_size: raise RuntimeError('Truncated IDX header: ' + str(path))
    header = struct.unpack('>' + 'I' * (dimensions + 1), raw[:header_size])
    shape = (count, 28, 28) if dimensions == 3 else (count,)
    if header != (0x800 + dimensions, *shape) or len(raw) != header_size + int(np.prod(shape)):
        raise RuntimeError('Fashion-MNIST IDX shape/magic/size mismatch: ' + str(path))
    result = np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape).copy()
    if dimensions == 1 and (np.any(result > 9) or len(np.unique(result)) != 10):
        raise RuntimeError('Invalid Fashion-MNIST labels: ' + str(path))
    if dimensions == 1 and not np.array_equal(np.bincount(result, minlength=10), np.full(10, count // 10)):
        raise RuntimeError('Fashion-MNIST class frequency mismatch: ' + str(path))
    return result


class DownloadFailure(RuntimeError):
    pass


def _fetch(url, target):
    try:
        source = urllib.request.urlopen(url, timeout=120)
    except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException) as error:
        raise DownloadFailure(str(error)) from error
    with source, Path(target).open('wb') as stream:
        while True:
            try:
                block = source.read(64 * 1024)
            except (OSError, http.client.HTTPException) as error:
                raise DownloadFailure(str(error)) from error
            if not block: break
            stream.write(block)


def _download(path, name, md5, count, dimensions):
    temporary = path.with_name(path.name + f'.{os.getpid()}.part'); failures = []
    sources = DOWNLOAD_URLS * 2
    for attempt, base in enumerate(sources, 1):
        url = base + name
        print(f'Downloading {name}: attempt {attempt}/{len(sources)} from {base}', flush=True)
        try:
            _fetch(url, temporary)
            if _md5(temporary) != md5:
                raise DownloadFailure('Downloaded Fashion-MNIST MD5 mismatch: ' + name)
            try:
                _idx(temporary, count, dimensions)
            except (RuntimeError, EOFError, gzip.BadGzipFile, zlib.error) as error:
                raise DownloadFailure(str(error)) from error
            os.replace(temporary, path)
            return
        except DownloadFailure as error:
            failures.append(url + ': ' + str(error))
            print('Download attempt failed: ' + str(error), flush=True)
        finally:
            if temporary.exists(): temporary.unlink()
    raise RuntimeError('Fashion-MNIST download failed after finite retries; verified files retained:\n' + '\n'.join(failures))


def ensure_data(data_root, download=False):
    """Verify existing bytes; only an explicit download flag creates missing files."""
    root = Path(data_root)
    for name, (md5, count, dimensions) in RESOURCES.items():
        path = root / RAW / name
        if not path.exists():
            if not download: raise FileNotFoundError('Missing Fashion-MNIST file; run prepare --download: ' + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            _download(path, name, md5, count, dimensions)
        if not path.is_file() or _md5(path) != md5:
            raise RuntimeError('Existing Fashion-MNIST file differs from official MD5; retained unchanged: ' + str(path))
        _idx(path, count, dimensions)
    return {str(RAW / name).replace('\\', '/'): c.digest(root / RAW / name) for name in RESOURCES}


def dataset(data_root, train=True):
    prefix = 'train' if train else 't10k'
    images = _idx(Path(data_root) / RAW / (prefix + '-images-idx3-ubyte.gz'), 60000 if train else 10000, 3)
    labels = _idx(Path(data_root) / RAW / (prefix + '-labels-idx1-ubyte.gz'), len(images), 1)
    x = torch.from_numpy(images).unsqueeze(1).float().div_(255.).sub_(MEAN).div_(STD)
    return x, torch.from_numpy(labels.astype(np.int64))


def model(seed=2025):
    c.seed_all(seed)
    return c.ConvNet(1, 10, 128, 3, 'relu', 'instancenorm', 'avgpooling', (28, 28)).to(c.DEVICE)


def budget_table(counts):
    counts = np.asarray(counts, dtype=int); sizes = counts.sum(1)
    capacities = c.bounded_allocate(sizes, 12000, (counts > 0).sum(1), sizes)
    synthetic = [c.bounded_allocate(row, int(n), (row > 0).astype(int), row).tolist()
                 for row, n in zip(counts, capacities)]
    return dict(real_counts=counts.tolist(), real_sizes=sizes.tolist(),
                synthetic_sizes=capacities.tolist(), synthetic_counts=synthetic)


def validate_config(path=None):
    value = c.read(Path(path) if path is not None else INPUTS / 'config.json')
    expected = dict(suite=SUITE, dataset='FashionMNIST', alphas=list(ALPHAS), partition_seed=35,
        clients=32, selected=3, budget_ratio=.2, budget_total=12000, reference_seeds=list(SEEDS),
        fl_seeds=list(FL_SEEDS), synthesis_iterations=8000, snapshot_iterations=[0, 4000, 8000],
        fl_rounds=1500, local_steps=3, batch=256, real_lr=.01, proxy_lr=.01)
    for name, item in expected.items():
        if value.get(name) != item: raise RuntimeError('Frozen Fashion-MNIST setting changed: ' + name)
    nested = dict(model=dict(architecture='ConvNet', channel=1, im_size=[28, 28], num_classes=10,
        width=128, depth=3, activation='relu', normalization='instancenorm', pooling='avgpooling',
        grayscale_first_layer_padding=3, mean=[MEAN], std=[STD]),
        synthesis=dict(config=CONFIG, loss='original', weak_dsa='crop_flip', optimizer_lr=1.,
            optimizer_momentum=.5, reference_rounds=1500, train_pool=[list(g) for g in c.TRAIN_POOL],
            calibration_rounds=list(c.CAL_ROUNDS), calibration_repeats=2),
        diagnostics=dict(iterations=[8000], clients=32, reference_seeds=list(SEEDS),
            model_rounds=list(c.EVAL_ROUNDS), repeats=30, classifier_accuracy=False))
    for group, settings in nested.items():
        for name, item in settings.items():
            if value.get(group, {}).get(name) != item:
                raise RuntimeError('Frozen Fashion-MNIST setting changed: ' + group + '.' + name)
    return value


@contextmanager
def context(inputs_root=INPUTS, bind_protocol=True):
    changes = dict(INPUTS=Path(inputs_root), ALPHAS=ALPHAS, SEEDS=SEEDS, CONFIGS=(CONFIG,),
        SELECTED={alpha: list(range(32)) for alpha in ALPHAS}, dataset=dataset, model=model,
        budget_table=budget_table, validate_snapshot=validate_snapshot)
    if bind_protocol: changes['protocol'] = lambda: protocol(inputs_root)
    old = {name: getattr(c, name) for name in changes}
    for name, value in changes.items(): setattr(c, name, value)
    try: yield
    finally:
        for name, value in old.items(): setattr(c, name, value)


def protocol(inputs_root=INPUTS):
    inputs_root = Path(inputs_root); config = validate_config(inputs_root / 'config.json')
    manifest = c.read(inputs_root / 'manifest.json')
    if c.digest(inputs_root / 'counts.json') != manifest['counts_sha256']:
        raise RuntimeError('Fashion-MNIST counts fingerprint changed')
    if set(manifest['splits']) != {'splits/alpha' + alpha + '.pt' for alpha in ALPHAS}:
        raise RuntimeError('Fashion-MNIST split coverage changed')
    for rel, sha in manifest['splits'].items():
        if c.digest(inputs_root / rel) != sha: raise RuntimeError('Fashion-MNIST split fingerprint changed: ' + rel)
    with context(inputs_root, bind_protocol=False): value = _legacy_protocol()
    value.update(suite=SUITE, dataset='FashionMNIST', channel=1, image_size=[28, 28],
        normalization=dict(mean=[MEAN], std=[STD]), first_conv_padding=3,
        iterations=list(STAGES), budget_total=12000,
        partition=dict(seed=35, clients=32, min_per_client=1),
        classifier_evaluation_enabled=False, diagnostic_clients=list(range(32)),
        diagnostic_contexts_per_alpha=384, diagnostic_repeats=30,
        synthesis_order='all clients through 4000 to 8000, retaining images and SGD momentum',
        config_sha256=c.digest(inputs_root / 'config.json'))
    for name in ('fmnist_common.py', 'run_fmnist_synthesis.py', 'dirichlet_dsa4000_worker.py'):
        rel = 'synthesis_label2_study/' + name; value['code_sha256'][rel] = c.digest(ROOT / rel)
    return json.loads(json.dumps(value))


def prepare_inputs(data_root, inputs_root=INPUTS, download=False):
    inputs_root = Path(inputs_root); validate_config(inputs_root / 'config.json')
    data_hashes = ensure_data(data_root, download)
    manifest_path = inputs_root / 'manifest.json'
    if manifest_path.exists(): return preflight(data_root, inputs_root)
    if (inputs_root / 'counts.json').exists() or (inputs_root / 'splits').exists():
        raise RuntimeError('Unbound partial Fashion-MNIST inputs exist; do not overwrite')
    labels = _idx(Path(data_root) / RAW / 'train-labels-idx1-ubyte.gz', 60000, 1)
    source = SimpleNamespace(targets=labels); scenarios = []; split_hashes = {}
    for alpha in ALPHAS:
        indices = split_dataset_indices(source, num_clients=32, mode='dirichlet',
            alpha=float(alpha.replace('p', '.')), seed=35, min_per_client=1)
        counts = [np.bincount(labels[row], minlength=10).tolist() for row in indices]
        rel = 'splits/alpha' + alpha + '.pt'
        c.save(inputs_root / rel, dict(client_indices=indices, num_clients=32,
            alpha=float(alpha.replace('p', '.')), seed=35, min_per_client=1))
        split_hashes[rel] = c.digest(inputs_root / rel)
        scenarios.append(dict(alpha=float(alpha.replace('p', '.')), class_counts=counts))
    c.atomic_json(inputs_root / 'counts.json', dict(dataset='FashionMNIST', scenarios=scenarios))
    c.atomic_json(manifest_path, dict(dataset='FashionMNIST', data=data_hashes, splits=split_hashes,
        counts_sha256=c.digest(inputs_root / 'counts.json'), source_url=BASE_URL,
        source_md5={name: values[0] for name, values in RESOURCES.items()},
        partition=dict(seed=35, clients=32, min_per_client=1)))
    return preflight(data_root, inputs_root)


def preflight(data_root, inputs_root=INPUTS):
    inputs_root = Path(inputs_root); expected = protocol(inputs_root)
    manifest = c.read(inputs_root / 'manifest.json')
    if manifest.get('dataset') != 'FashionMNIST' or ensure_data(data_root) != manifest['data']:
        raise RuntimeError('Fashion-MNIST data fingerprint changed')
    labels = _idx(Path(data_root) / RAW / 'train-labels-idx1-ubyte.gz', 60000, 1)
    scenarios = c.read(inputs_root / 'counts.json')['scenarios']
    if [str(row['alpha']).replace('.', 'p') for row in scenarios] != list(ALPHAS):
        raise RuntimeError('Fashion-MNIST alpha coverage changed')
    with context(inputs_root):
        for alpha, scenario in zip(ALPHAS, scenarios):
            rel = 'splits/alpha' + alpha + '.pt'; path = inputs_root / rel
            if c.digest(path) != manifest['splits'].get(rel): raise RuntimeError('Changed original split: ' + rel)
            saved = c.load(path); indices = c.split(alpha)
            settings = dict(num_clients=32, alpha=float(alpha.replace('p', '.')), seed=35, min_per_client=1)
            if any(saved.get(k) != v for k, v in settings.items()): raise RuntimeError('Partition definition changed')
            if len(indices) != 32 or any(len(row) < 1 for row in indices) or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(60000)):
                raise RuntimeError('Invalid Fashion-MNIST split coverage')
            wanted = split_dataset_indices(SimpleNamespace(targets=labels), mode='dirichlet', **settings)
            if any(not np.array_equal(row, fixed) for row, fixed in zip(indices, wanted)):
                raise RuntimeError('Frozen seed35 partition indices changed')
            counts = [np.bincount(labels[row], minlength=10).tolist() for row in indices]
            if counts != scenario['class_counts']: raise RuntimeError('Fashion-MNIST class counts changed')
            table = expected['budgets'][alpha]; real = np.array(counts); synthetic = np.array(table['synthetic_counts'])
            if real.sum() != 60000 or synthetic.sum() != 12000 or np.any(synthetic > real) or np.any((real > 0) & (synthetic < 1)):
                raise RuntimeError('Invalid frozen Fashion-MNIST budget')
    return dict(state='preflight_ok', dataset='FashionMNIST', alphas=list(ALPHAS),
        data_root=str(Path(data_root).resolve()), input_root=str(inputs_root.resolve()),
        reference_jobs=6, calibration_jobs=3, synthesis_jobs=96, vector_jobs=3,
        fl_jobs=270, image_shape=[1, 28, 28], formal_training_started=False)


def validate_snapshot(root, alpha, config, client, iteration):
    path = c.snapshot(root, alpha, config, client, iteration); item = c.load(path)
    table = c.read(Path(root) / 'protocol.json')['budgets'][alpha]
    counts = table['synthetic_counts'][client]
    if alpha not in ALPHAS or config != CONFIG or iteration not in (0, *STAGES):
        raise RuntimeError('Unexpected Fashion-MNIST snapshot identity')
    coeff = Path(root) / 'calibration' / alpha / 'coefficients.json'
    expected = dict(alpha=alpha, client=client, config=config,
        lambda_dm=c.read(coeff)['weighted_weak']['r'] * 10,
        calibration_sha256=c.digest(coeff), real_counts=table['real_counts'][client], synthetic_counts=counts)
    settings = c.read(c.route(root, alpha, config, client) / 'config.json')
    if any(settings.get(k) != v for k, v in expected.items()): raise RuntimeError('Snapshot configuration changed')
    if item['images'].shape != (sum(counts), 1, 28, 28) or item['images'].dtype != torch.float32 or item['labels'].dtype != torch.long:
        raise RuntimeError('Invalid grayscale snapshot shape/dtype')
    if item['labels'].shape != (sum(counts),) or torch.any((item['labels'] < 0) | (item['labels'] > 9)):
        raise RuntimeError('Invalid Fashion-MNIST snapshot labels')
    if not torch.equal(torch.bincount(item['labels'], minlength=10), torch.tensor(counts)):
        raise RuntimeError('Fashion-MNIST synthetic class capacities changed')
    if item.get('iteration', 0) != iteration or not torch.isfinite(item['images']).all():
        raise RuntimeError('Invalid Fashion-MNIST snapshot progress/values')
    done = c.route(root, alpha, config, client) / f'COMPLETED_{iteration}.json'
    if done.exists() and c.read(done) != dict(iteration=iteration, snapshot_sha256=c.digest(path)):
        raise RuntimeError('Fashion-MNIST snapshot checksum changed')
    return path
