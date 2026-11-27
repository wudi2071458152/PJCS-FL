"""Fixed protocol and numerical primitives for the independent Dirichlet screen."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from networks import ConvNet
from utils import DiffAugment, ParamDiffAug

INPUTS = ROOT / 'synthesis_label2_study/dirichlet_screen_inputs'
ALPHAS = ('0p1', '0p2', '0p3')
SEEDS = (2025, 2026)
TRAIN_POOL = ([0, 10, 30, 75], [150, 250, 400, 600], [750, 1000, 1200])
EVAL_ROUNDS = (20, 100, 300, 500, 900, 1500)
CAL_ROUNDS = (30, 400, 1000)
MODEL_ROUNDS = sorted({0, 1500, *EVAL_ROUNDS, *CAL_ROUNDS, *(r for group in TRAIN_POOL for r in group)})
CONFIGS = tuple(f'{ce}_{dsa}_x{scale}' for ce in ('ordinary', 'weighted')
                for dsa in ('weak', 'strong') for scale in (1, 10))
SELECTED = {'0p1': [24, 31, 25, 28, 15, 13, 2, 1],
            '0p2': [26, 25, 5, 0, 30, 31, 14, 8],
            '0p3': [1, 5, 25, 21, 0, 12, 7, 8]}
STRATEGIES = {'weak': 'crop_flip', 'strong': 'color_crop_cutout_flip_scale_rotate'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)


def file_path(path):
    path = Path(path)
    if os.name == 'nt':
        absolute = str(path.absolute())
        if len(absolute) >= 240 and not absolute.startswith('\\\\?\\'):
            prefix = '\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute
            path = Path(prefix)
    return path


def digest(path):
    h = hashlib.sha256()
    with file_path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(file_path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = file_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def save(path, value):
    path = file_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    # A Python file handle also supports Windows paths beyond the C++ writer's path limit.
    with temp.open('wb') as file: torch.save(value, file)
    os.replace(temp, path)


def load(path, device='cpu'):
    with file_path(path).open('rb') as file:
        return torch.load(file, map_location=device, weights_only=False)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def expired(deadline):
    return deadline is not None and time.time() >= deadline


def bounded_allocate(weights, total, lower, upper):
    """Clipped proportional quotas followed by stable largest-remainder rounding."""
    w = np.asarray(weights, dtype=float); lo = np.asarray(lower, dtype=int); hi = np.asarray(upper, dtype=int)
    if np.any(lo < 0) or np.any(hi < lo) or not lo.sum() <= total <= hi.sum():
        raise ValueError('Infeasible allocation')
    if np.any((w <= 0) & (hi > 0)): raise ValueError('Positive capacity needs positive weight')
    left, right = 0., 1.
    while np.clip(right * w, lo, hi).sum() < total: right *= 2
    for _ in range(80):
        mid = (left + right) / 2
        if np.clip(mid * w, lo, hi).sum() < total: left = mid
        else: right = mid
    quota = np.clip(right * w, lo, hi)
    result = np.floor(quota + 1e-10).astype(int)
    order = sorted(range(len(w)), key=lambda i: (-(quota[i] - result[i]), i))
    for i in order:
        if result.sum() == total: break
        if result[i] < hi[i]: result[i] += 1
    if result.sum() != total: raise AssertionError('Allocation rounding failed')
    return result


def budget_table(counts):
    counts = np.asarray(counts, dtype=int)
    sizes = counts.sum(1); present = (counts > 0).sum(1)
    budgets = bounded_allocate(sizes, 10000, present, sizes)
    classes = [bounded_allocate(row, int(m), (row > 0).astype(int), row).tolist()
               for row, m in zip(counts, budgets)]
    return dict(real_counts=counts.tolist(), real_sizes=sizes.tolist(),
                synthetic_sizes=budgets.tolist(), synthetic_counts=classes)


def protocol():
    metadata = read(INPUTS / 'counts.json')
    tables = {str(s['alpha']).replace('.', 'p'): budget_table(s['class_counts']) for s in metadata['scenarios']}
    code = [Path(__file__), ROOT / 'networks.py', ROOT / 'utils.py']
    code += [ROOT / 'synthesis_label2_study' / name for name in
             ('run_dirichlet_reference.py', 'run_dirichlet_screen.py', 'schedule_dirichlet_screen.py')]
    return dict(version=1, alphas=ALPHAS, selected=SELECTED, configs=CONFIGS, budget_ratio=.2,
                budgets=tables, model_seeds=SEEDS, reference_rounds=1500, reference_selected=3,
                reference_aggregation='selected sample weighted FedAvg', local_steps=3, batch=256,
                shuffle=True, drop_last=False, wrap='reshuffle iterator', local_lr=.01,
                local_momentum=0., local_weight_decay=0., synthetic_optimizer=dict(lr=1., momentum=.5),
                train_pool=TRAIN_POOL, eval_rounds=EVAL_ROUNDS, calibration_rounds=CAL_ROUNDS,
                calibration_repeats=2, calibration_clients=32, calibration='median image gradient norm ratio',
                update_loss='squared delta error / (squared real delta norm + 1e-12)',
                weighted_ce='mean(M*n_c/(n*m_c) * per-image CE), no batch renormalization',
                dm='mean class normalized feature-mean error; min(256,n_c) real without replacement',
                dsa=STRATEGIES, dm_augmentation_mode='M', seed=2025, iterations=[500, 1000],
                diagnostic_repeats=30, diagnostic_seed=2000002025,
                diagnostic_noise='norm(mean(P-R))^2 - trace(sample_cov(P-R))/repeats; keep negatives',
                classifier=dict(seed=2025, epochs=1000, batch=256, ce='ordinary', lr=.01,
                                momentum=.9, weight_decay=.0005, lr_after_epoch500=.001,
                                dsa='strong', augmentation_mode='S'),
                inputs=read(INPUTS / 'manifest.json'),
                code_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in code})


def bind(path, value):
    """Reject accidental continuation under changed inputs or algorithms."""
    value = json.loads(json.dumps(value))
    if Path(path).exists():
        if read(path) != value: raise RuntimeError(f'Protocol/fingerprint mismatch: {path}')
        return
    atomic_json(path, value)


def preflight(data_root):
    manifest = read(INPUTS / 'manifest.json'); failures = []
    for rel, expected in manifest['data'].items():
        p = Path(data_root) / rel
        if not p.is_file(): failures.append(f'MISSING data: {p}')
        elif digest(p) != expected: failures.append(f'HASH MISMATCH data: {p}')
    for rel, expected in manifest['splits'].items():
        p = INPUTS / rel
        if not p.is_file() or digest(p) != expected: failures.append(f'MISSING/MISMATCH original split: {p}')
    if failures: raise RuntimeError('\n'.join(failures))
    # Independently verify every index and label count, not just stored metadata.
    import pickle
    targets = []
    for i in range(1, 6):
        with (Path(data_root) / f'cifar-10-batches-py/data_batch_{i}').open('rb') as f:
            targets.extend(pickle.load(f, encoding='latin1')['labels'])
    targets = np.asarray(targets)
    metadata = read(INPUTS / 'counts.json')
    for s in metadata['scenarios']:
        alpha = str(s['alpha']).replace('.', 'p'); indices = split(alpha)
        flat = np.concatenate(indices)
        if len(indices) != 32 or not np.array_equal(np.sort(flat), np.arange(50000)):
            raise RuntimeError(f'Invalid partition {alpha}')
        counts = [np.bincount(targets[row], minlength=10).tolist() for row in indices]
        if counts != s['class_counts']: raise RuntimeError(f'Label-count mismatch {alpha}')
        if not np.all(np.asarray(counts)[SELECTED[alpha]].sum(0) > 0): raise RuntimeError('Selected classes incomplete')
    if torch.cuda.is_available():
        with torch.no_grad():
            probe = model().eval()(torch.zeros(2, 3, 32, 32, device=DEVICE))
        if not torch.isfinite(probe).all(): raise RuntimeError('CUDA model forward produced nonfinite values')
    return dict(state='preflight_ok', data_root=str(Path(data_root).resolve()), cuda=torch.cuda.is_available(),
                visible_gpu_count=torch.cuda.device_count(),
                selected=SELECTED, synthesis_jobs=192, calibration='all 96 clients', formal_training_started=False)


def split(alpha):
    return [np.asarray(row, dtype=np.int64) for row in load(INPUTS / 'splits' / f'alpha{alpha}.pt')['client_indices']]


def dataset(data_root, train=True):
    # download=False: server must have the verified original CIFAR files.
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(str(data_root), train=train, download=False)
    x = torch.from_numpy(ds.data.copy()).permute(0, 3, 1, 2).float().div_(255.)
    x.sub_(torch.tensor([.4914, .4822, .4465])[None, :, None, None])
    x.div_(torch.tensor([.2023, .1994, .2010])[None, :, None, None])
    return x, torch.tensor(ds.targets, dtype=torch.long)


def model(seed=2025):
    seed_all(seed)
    return ConvNet(3, 10, 128, 3, 'relu', 'instancenorm', 'avgpooling', (32, 32)).to(DEVICE)


def model_path(root, alpha, seed, round_number):
    return Path(root) / 'reference' / alpha / str(seed) / 'models' / f'round_{round_number:04d}.pt'


def reference(root, alpha, seed, round_number):
    net = model(seed); net.load_state_dict(load(model_path(root, alpha, seed, round_number), DEVICE))
    return net


def batches(n, seed, steps=3, batch_size=256):
    loader = DataLoader(torch.arange(n), batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=0, generator=torch.Generator().manual_seed(int(seed)))
    iterator = iter(loader); result = []
    for _ in range(steps):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        result.append(batch)
    return result


def ce_loss(logits, y, weights=None):
    losses = F.cross_entropy(logits, y, reduction='none')
    return (losses if weights is None else losses * weights[y]).mean()


def update(net, x, y, seed, weights=None, differentiable=False):
    params = OrderedDict((name, p.detach().clone().requires_grad_(True)) for name, p in net.named_parameters())
    initial = list(params.values()); buffers = OrderedDict((name, b.detach().clone()) for name, b in net.named_buffers())
    for batch in batches(len(y), seed):
        logits = torch.func.functional_call(net, (params, buffers), (x[batch].to(DEVICE),))
        loss = ce_loss(logits, y[batch].to(DEVICE), weights)
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=differentiable)
        params = OrderedDict((name, p - .01 * g) for (name, p), g in zip(params.items(), gradients))
    delta = torch.cat([(p - old).reshape(-1) for p, old in zip(params.values(), initial)])
    return delta if differentiable else delta.detach()


@torch.no_grad()
def add_delta(net, delta):
    offset = 0
    for p in net.parameters():
        p.add_(delta[offset:offset + p.numel()].view_as(p)); offset += p.numel()


def initial_images(x, y, indices, counts, alpha, client):
    rng = np.random.default_rng(2025 + client * 1009 + ALPHAS.index(alpha) * 100000)
    chosen = []
    for label, n in enumerate(counts):
        members = indices[y[indices].numpy() == label]
        if n: chosen.extend(rng.choice(members, n, replace=False).tolist())
    return x[chosen].clone().to(DEVICE), y[chosen].clone().to(DEVICE)


def class_weights(real_counts, synthetic_counts):
    real = torch.tensor(real_counts, dtype=torch.float, device=DEVICE)
    syn = torch.tensor(synthetic_counts, dtype=torch.float, device=DEVICE)
    return syn.sum() * real / (real.sum() * syn.clamp_min(1))


def dm_terms(net, images, labels, real_x, real_y, seed, mode):
    rng = np.random.default_rng(seed)
    param = ParamDiffAug(); param.aug_mode = 'M'
    classes = sorted(set(labels.tolist()))
    for offset, label in enumerate(classes):
        members = torch.where(real_y == label)[0].numpy()
        ids = rng.choice(members, min(256, len(members)), replace=False)
        aug_seed = seed + offset * 10
        with torch.no_grad():
            real = DiffAugment(real_x[ids].to(DEVICE), STRATEGIES[mode], seed=aug_seed, param=param)
            mean_real = net.embed(real).mean(0)
        synthetic = DiffAugment(images[labels == label], STRATEGIES[mode], seed=aug_seed, param=param)
        mean_syn = net.embed(synthetic).mean(0)
        yield (mean_syn - mean_real).square().sum() / (mean_real.square().sum() + 1e-12) / len(classes)


def route(root, alpha, config, client):
    return Path(root) / 'synthesis' / alpha / config / f'client_{client:02d}'


def snapshot(root, alpha, config, client, iteration):
    return route(root, alpha, config, client) / 'snapshots' / f'iter_{iteration:04d}.pt'


def validate_snapshot(root, alpha, config, client, iteration):
    """Validate actual exported content, including the exact per-class budget."""
    path = snapshot(root, alpha, config, client, iteration)
    item = load(path)
    counts = read(Path(root)/'protocol.json')['budgets'][alpha]['synthetic_counts'][client]
    settings = read(route(root, alpha, config, client)/'config.json')
    ce, mode, multiplier = config.split('_')
    coefficient_file = Path(root)/'calibration'/alpha/'coefficients.json'
    coefficient = read(coefficient_file)[f'{ce}_{mode}']['r'] * int(multiplier[1:])
    table = read(Path(root)/'protocol.json')['budgets'][alpha]
    expected_settings = dict(alpha=alpha, client=client, config=config, lambda_dm=coefficient,
                             calibration_sha256=digest(coefficient_file), real_counts=table['real_counts'][client],
                             synthetic_counts=counts)
    if any(settings.get(key) != value for key,value in expected_settings.items()):
        raise RuntimeError(f'Snapshot configuration mismatch: {path}')
    expected = torch.tensor(counts)
    if item['images'].shape != (sum(counts), 3, 32, 32) or item['labels'].dtype != torch.long:
        raise RuntimeError(f'Invalid snapshot shape/dtype: {path}')
    if item['labels'].shape != (sum(counts),) or torch.any((item['labels'] < 0) | (item['labels'] > 9)):
        raise RuntimeError(f'Invalid snapshot labels: {path}')
    if not torch.equal(torch.bincount(item['labels'], minlength=10), expected):
        raise RuntimeError(f'Snapshot class budget mismatch: {path}')
    if item.get('iteration', 0) != iteration or not torch.isfinite(item['images']).all():
        raise RuntimeError(f'Invalid snapshot iteration/values: {path}')
    marker = route(root, alpha, config, client)/f'COMPLETED_{iteration}.json'
    if marker.exists() and read(marker) != dict(iteration=iteration, snapshot_sha256=digest(path)):
        raise RuntimeError(f'Snapshot checksum mismatch: {path}')
    return path


def geometry(proxy, real):
    p = proxy.double(); r = real.double(); pn = float(p.norm()); rn = float(r.norm())
    cosine = float(torch.dot(p, r) / (p.norm() * r.norm()).clamp_min(1e-30))
    return dict(relative_error=float((p-r).norm()) / max(rn, 1e-15), proxy_norm=pn, real_norm=rn,
                norm_ratio=pn/max(rn, 1e-15), cosine=cosine,
                angle_degrees=float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))


class Moments:
    def __init__(self): self.n = 0; self.mean = None; self.m2 = 0.

    def add(self, value):
        value = value.detach().cpu().double(); self.n += 1
        if self.mean is None: self.mean = torch.zeros_like(value)
        d = value - self.mean; self.mean += d / self.n
        self.m2 += float(torch.dot(d, value - self.mean))

    def variance(self): return self.m2 / (self.n - 1) if self.n > 1 else 0.

    def corrected_squared_mean(self):
        return float(self.mean.square().sum()) - self.variance() / self.n
