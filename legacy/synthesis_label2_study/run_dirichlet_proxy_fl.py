"""Independent 32-client FL routes; preparation is the default, training needs --run."""
import argparse
import ast
import hashlib
import json
import platform
import signal
import time
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c
from run_dirichlet_screen import evaluate

METHODS = ('fedavg', 'uniform_real', 'score_real', 'residual_uniform', 'residual_score')
CONFIG = 'weighted_weak_x10'
SEED = 2025
EPS = 1e-12
EXPECTED = Path(__file__).with_name('dirichlet_continue32000_source_protocol.json')


def fixed_aocs_sample(u, m, rng):
    """固定大小的 dependent rounding，保留返回的边际入选概率 p。"""
    u = np.asarray(u, dtype=np.float64)
    if not 0 <= m <= len(u) or not np.isfinite(u).all() or np.any(u < 0):
        raise ValueError("Invalid AOCS scores or sample size")
    # 极小正数保证不足 m 个非零分数时仍可分配完整预算。
    weights = np.maximum(u / max(float(u.max()), EPS), EPS)
    p = np.zeros(len(u))
    active = np.ones(len(u), dtype=bool)
    budget = float(m)
    while active.any():
        proposal = budget * weights[active] / weights[active].sum()
        saturated = proposal >= 1.0
        ids = np.flatnonzero(active)
        if not saturated.any():
            p[ids] = proposal
            break
        p[ids[saturated]] = 1.0
        active[ids[saturated]] = False
        budget -= int(saturated.sum())
    q = p.copy()
    while True:
        fractional = np.flatnonzero((q > 1e-12) & (q < 1 - 1e-12))
        if len(fractional) < 2:
            break
        i, j = rng.choice(fractional, size=2, replace=False)
        a, b = min(1 - q[i], q[j]), min(q[i], 1 - q[j])
        if rng.random() < b / (a + b):
            q[i] += a
            q[j] -= a
        else:
            q[i] -= b
            q[j] += b
    selected = np.flatnonzero(q > 0.5).tolist()
    assert len(selected) == m
    return selected, p


def one_round(net, real, syn, weights, method, seed, round_number, update_fn=None):
    """Compute P for all clients when needed, R only for the three selected clients."""
    if method not in METHODS: raise ValueError('Unknown FL method')
    update_fn = update_fn or c.update
    weights = np.asarray(weights, dtype=float); n = len(real)
    if n < 3 or weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid global sample weights')
    base = seed + round_number * 1000; net.train()
    proxies = None; scores = None
    if method not in ('fedavg', 'uniform_real'):
        if syn is None or len(syn) != n: raise ValueError('This method requires all client proxies')
        proxies = [update_fn(net, x, y, base + 10000000 + i, w) for i, (x, y, w) in enumerate(syn)]
        scores = np.asarray([float(p.norm()) / (3 * .01) for p in proxies])
    if method in ('score_real', 'residual_score'):
        selected, pi = fixed_aocs_sample(weights * scores, 3, np.random.default_rng(base + 400000))
    else:
        selected = sorted(np.random.default_rng(base + 17).choice(n, 3, replace=False).tolist())
        pi = np.full(n, 3 / n)
    if len(selected) != 3 or len(set(selected)) != 3 or any(not 0 <= i < n for i in selected) or pi.shape != (n,) or not np.isfinite(pi).all() or np.any(pi <= 0) or np.any(pi > 1) or not np.isclose(pi.sum(), 3.):
        raise RuntimeError('Invalid selection or inclusion probabilities')
    observed = {i: update_fn(net, *real[i], base + i) for i in selected}
    factors = weights[selected] / (weights[selected].sum() if method == 'fedavg' else pi[selected])
    true_delta = sum(float(w) * observed[i] for i, w in zip(selected, factors))
    delta = true_delta; proxy_delta = correction = None
    if method.startswith('residual_'):
        proxy_delta = sum(float(w) * p for w, p in zip(weights, proxies))
        correction = sum(float(w) * (observed[i] - proxies[i]) for i, w in zip(selected, factors))
        delta = proxy_delta + correction
    if not torch.isfinite(delta).all(): raise RuntimeError('Nonfinite federated update')
    diagnostics = [dict(client=i,
        global_weight=float(weights[i]), inclusion_probability=float(pi[i]), aggregation_weight=float(w),
        **(c.geometry(proxies[i], observed[i]) if proxies is not None else dict(real_norm=float(observed[i].norm()))))
        for i, w in zip(selected, factors)]
    record = dict(round=round_number, selected=selected, pi=pi.tolist(),
        scores=None if scores is None else scores.tolist(),
        importance=None if scores is None else (weights * scores).tolist(),
        proxy_norms=None if proxies is None else [float(p.norm()) for p in proxies],
        selected_real=diagnostics, selected_sample_fraction=float(weights[selected].sum()),
        true_weight_sum=float((weights[selected] / pi[selected]).sum()),
        aggregation_weight_sum=float(factors.sum()), update_norm=float(delta.norm()),
        selected_real_aggregate_norm=float(true_delta.norm()),
        proxy_aggregate_norm=None if proxy_delta is None else float(proxy_delta.norm()),
        correction_norm=None if correction is None else float(correction.norm()))
    return delta, record


def prepare(a):
    """Validate frozen source, inputs, exact proxy budgets and effective loss settings."""
    source = Path(a.source_root); source_protocol = c.read(source / 'protocol.json')
    output = Path(a.output).resolve()
    for path in (source, getattr(a, 'proxy_root', None)):
        if path is not None and (output.is_relative_to(Path(path).resolve()) or Path(path).resolve().is_relative_to(output)):
            raise RuntimeError('FL output must be disjoint from immutable source/proxy directories')
    expected = c.read(EXPECTED)
    if source_protocol != expected: raise RuntimeError('Source differs from the immutable completed 24000 protocol')
    if a.method not in METHODS or a.alpha not in c.ALPHAS: raise ValueError('Invalid FL route')
    baseline = a.method in ('fedavg', 'uniform_real')
    if not baseline and a.iteration not in (8000, 24000, 32000): raise ValueError('Invalid proxy iteration')
    files = {}; cache = {}
    def fingerprint(path, name):
        path = Path(path); key = str(path.resolve())
        if key not in cache: cache[key] = c.digest(path)
        files[name] = cache[key]; return cache[key]
    fingerprint(EXPECTED, 'code/' + EXPECTED.name)
    fingerprint(source / 'protocol.json', 'source/protocol.json')
    manifest = c.read(c.INPUTS / 'manifest.json')
    if manifest != expected['inputs']: raise RuntimeError('Original input manifest changed')
    for name in ('manifest.json', 'counts.json'): fingerprint(c.INPUTS / name, 'inputs/' + name)
    for rel, sha in manifest['splits'].items():
        if fingerprint(c.INPUTS / rel, 'inputs/' + rel) != sha: raise RuntimeError('Original split changed: ' + rel)
    for rel, sha in manifest['data'].items():
        if fingerprint(Path(a.data_root) / rel, 'data/' + rel) != sha: raise RuntimeError('CIFAR data changed: ' + rel)
    for rel, sha in expected['code_sha256'].items():
        if fingerprint(c.ROOT / rel, 'code/' + rel) != sha: raise RuntimeError('Original numerical code changed: ' + rel)
    for path in (Path(__file__), Path(c.__file__), Path(evaluate.__code__.co_filename), c.ROOT / 'networks.py', c.ROOT / 'utils.py'):
        fingerprint(path, 'code/' + path.relative_to(c.ROOT).as_posix())
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8-sig'))
    sampler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fixed_aocs_sample')
    sampler_sha = hashlib.sha256(ast.dump(sampler, include_attributes=False).encode()).hexdigest()
    indices = c.split(a.alpha); x, y = c.dataset(a.data_root); table = expected['budgets'][a.alpha]
    if len(indices) != 32 or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(50000)):
        raise RuntimeError('Invalid 32-client partition')
    counts = [torch.bincount(y[row], minlength=10).tolist() for row in indices]
    if counts != table['real_counts'] or sum(table['synthetic_sizes']) != 10000:
        raise RuntimeError('Real counts or 20% proxy budget changed')
    weights = np.asarray([len(row) for row in indices], dtype=float) / 50000
    model_file = c.model_path(source, a.alpha, SEED, 0)
    model_sha = fingerprint(model_file, 'source/reference/round_0000.pt')
    marker_file = model_file.parent.parent / 'COMPLETED.json'
    fingerprint(marker_file, 'source/reference/COMPLETED.json')
    if c.read(marker_file)['model_sha256']['0'] != model_sha: raise RuntimeError('Initial reference model changed')
    net = c.reference(source, a.alpha, SEED, 0); syn = None; loss = None
    if not baseline:
        proxy = Path(a.proxy_root); proxy_protocol = c.read(proxy / 'protocol.json')
        numerical = ('budgets', 'local_steps', 'batch', 'local_lr', 'weighted_ce', 'synthetic_optimizer',
                     'seed', 'update_loss', 'dm', 'dsa', 'dm_augmentation_mode')
        if any(proxy_protocol.get(k) != expected[k] for k in numerical): raise RuntimeError('Proxy numerical protocol differs from source')
        fingerprint(proxy / 'protocol.json', 'proxy/protocol.json')
        loss = proxy_protocol.get('loss', 'original'); gamma = None
        if a.iteration != 32000 and proxy.resolve() != source.resolve(): raise RuntimeError('8k/24k proxies must use the immutable source')
        if a.iteration == 32000:
            if loss not in ('original', 'low_dm_direction') or proxy_protocol.get('baseline_iteration') != 24000 or proxy_protocol.get('source_protocol_sha256') != c.digest(EXPECTED):
                raise RuntimeError('Unexpected 32000 continuation protocol')
            if loss == 'original' and any(proxy_protocol['loss_settings'].get(k) != v for k, v in
                dict(dm_factor=1., direction_layers=[], gamma={}).items()): raise RuntimeError('Original loss settings changed')
            if loss == 'low_dm_direction':
                import dirichlet_continue32000_worker as worker
                gamma = worker.gamma_manifest(proxy)
                settings = proxy_protocol['loss_settings']
                if settings['gamma'] != gamma or settings['dm_factor'] != .1 or settings['direction_layers'] != ['features.4.weight', 'features.8.weight']:
                    raise RuntimeError('New loss changed its fixed gamma or DM/direction definition')
                for path in (Path(worker.__file__), Path(worker.p.__file__), Path(worker.p.d.__file__)):
                    fingerprint(path, 'code/' + path.relative_to(c.ROOT).as_posix())
                for alpha in c.ALPHAS:
                    for path in (proxy / 'calibration' / alpha).glob('*.json'):
                        fingerprint(path, 'proxy/' + path.relative_to(proxy).as_posix())
        syn = []
        for i in range(32):
            file = c.validate_snapshot(proxy, a.alpha, CONFIG, i, a.iteration)
            fingerprint(file, f'proxy/client_{i:02d}/snapshot.pt')
            folder = c.route(proxy, a.alpha, CONFIG, i)
            fingerprint(folder / 'config.json', f'proxy/client_{i:02d}/config.json')
            src = c.route(source, a.alpha, CONFIG, i)
            fingerprint(src / 'config.json', f'source/client_{i:02d}/config.json')
            if c.read(folder / 'config.json') != c.read(src / 'config.json'): raise RuntimeError('Inherited proxy route configuration changed')
            coefficient = proxy / 'calibration' / a.alpha / 'coefficients.json'
            if fingerprint(coefficient, 'proxy/calibration/coefficients.json') != fingerprint(source / 'calibration' / a.alpha / 'coefficients.json', 'source/calibration/coefficients.json'):
                raise RuntimeError('Proxy DM calibration differs from source')
            if loss == 'low_dm_direction':
                settings_file = folder / 'loss_settings.json'
                actual = c.read(settings_file); source_lambda = c.read(src / 'config.json')['lambda_dm']
                wanted = dict(loss=loss, source_iteration=24000, source_lambda_dm=source_lambda,
                    effective_lambda_dm=.1 * source_lambda, gamma=gamma[a.alpha]['gamma'], gamma_fingerprints=gamma[a.alpha],
                    source_config_sha256=fingerprint(src / 'config.json', f'source/client_{i:02d}/config.json'),
                    source_resume_sha256=fingerprint(src / 'resume.pt', f'source/client_{i:02d}/resume.pt'),
                    source_snapshot_sha256=fingerprint(c.snapshot(source, a.alpha, CONFIG, i, 24000), f'source/client_{i:02d}/iter_24000.pt'),
                    optimizer=dict(lr=1., momentum=.5, initial_momentum='inherit24000'))
                if actual != wanted: raise RuntimeError('New loss settings/source mismatch: ' + str(settings_file))
                fingerprint(settings_file, f'proxy/client_{i:02d}/loss_settings.json')
            item = c.load(file)
            syn.append((item['images'], item['labels'], c.class_weights(table['real_counts'][i], table['synthetic_counts'][i])))
    protocol = dict(version=1, alpha=a.alpha, method=a.method, iteration=0 if baseline else a.iteration,
        proxy_loss=loss, seed=SEED, rounds=a.rounds, clients=32, selected=3, local_steps=3, batch=256,
        real_lr=.01, proxy_lr=.01, real_ce='ordinary', proxy_ce=expected['weighted_ce'], augmentation=False,
        score='norm(P)/(3*0.01)', importance='w_i*score_i', sampling='uniform without replacement' if a.method in ('fedavg', 'uniform_real', 'residual_uniform') else 'fixed_aocs_sample, dependent rounding; exact marginals',
        aggregation={'fedavg': 'sum_selected(n_i/sum_selected(n_j)*R_i)',
            'uniform_real': 'sum_selected(w_i/pi_i*R_i)', 'score_real': 'sum_selected(w_i/pi_i*R_i)',
            'residual_uniform': 'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))',
            'residual_score': 'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))'}[a.method],
        residual_alpha=1., renormalize=a.method == 'fedavg', weights=weights.tolist(),
        round_seeds=dict(base='2025+round*1000', uniform='+17', score='+400000', real='+client', proxy='+10000000+client'),
        files_sha256=files, sampler_ast_sha256=sampler_sha, sampler_copied_from='main_train.py:fixed_aocs_sample/EPS=1e-12',
        software=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__))
    return net, [(x[row], y[row]) for row in indices], syn, weights, c.dataset(a.data_root, train=False), protocol


def summarize(rows):
    if not rows: return dict(rounds=0)
    last = rows[-1]
    result = dict(rounds=len(rows), final_test_accuracy=last['test_accuracy'], final_test_loss=last['test_loss'],
        best_test_accuracy=max(r['test_accuracy'] for r in rows),
        mean_last100_test_accuracy=float(np.mean([r['test_accuracy'] for r in rows[-100:]])),
        elapsed_seconds=last['elapsed_seconds'])
    for name, start, stop in (('early500', 1, 500), ('mid500', 501, 1000), ('late500', 1001, 1500),
                              ('round601_900', 601, 900), ('last300', max(1, len(rows)-299), len(rows))):
        values = [r['test_accuracy'] for r in rows if start <= r['round'] <= stop]
        if values: result['mean_' + name + '_test_accuracy'] = float(np.mean(values))
    return result


def train(a, prepared):
    net, real, syn, weights, test, protocol = prepared
    folder = Path(a.output); folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder / 'protocol.json', protocol); protocol_sha = c.digest(folder / 'protocol.json')
    if (folder / 'COMPLETED.json').exists(): verify(folder, a.rounds); return True
    checkpoint = folder / 'resume.pt'; start = 0; rows = []; elapsed = 0.
    if checkpoint.exists():
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != protocol_sha: raise RuntimeError('Checkpoint protocol/fingerprint mismatch')
        start = old['round']; rows = old['records']; elapsed = old['elapsed_seconds']
        if not 0 <= start <= a.rounds or [r['round'] for r in rows] != list(range(1, start + 1)):
            raise RuntimeError('Checkpoint rounds are inconsistent')
        net.load_state_dict(old['model']); c.restore_rng(old['rng'])
    else: c.seed_all(SEED)
    tick = time.monotonic()
    def persist(state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows,
            elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=protocol_sha))
        c.atomic_json(folder / 'rounds.json', rows)
        c.atomic_json(folder / 'summary.json', summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder / 'status.json', dict(state=state, round=len(rows), target=a.rounds, elapsed_seconds=duration,
            eta_seconds=float(np.mean(recent)) * (a.rounds-len(rows)) if recent else None))
    for rnd in range(start + 1, a.rounds + 1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        round_tick = time.monotonic()
        delta, row = one_round(net, real, syn, weights, a.method, SEED, rnd)
        c.add_delta(net, delta); row.update(evaluate(net, *test))
        row.update(round_seconds=time.monotonic() - round_tick, elapsed_seconds=elapsed + time.monotonic() - tick)
        rows.append(row)
        if rnd % 10 == 0 or rnd == a.rounds or c.expired(a.deadline): persist('training')
    persist('complete')
    c.atomic_json(folder / 'COMPLETED.json', dict(rounds=a.rounds, protocol_sha256=protocol_sha,
        rounds_sha256=c.digest(folder / 'rounds.json'), summary_sha256=c.digest(folder / 'summary.json')))
    verify(folder, a.rounds); return True


def verify(output, expected_rounds=1500):
    folder = Path(output); marker = c.read(folder / 'COMPLETED.json'); rows = c.read(folder / 'rounds.json')
    protocol = c.read(folder / 'protocol.json')
    if expected_rounds < 1 or protocol.get('rounds', expected_rounds) != expected_rounds:
        raise RuntimeError('FL protocol has an unexpected round target')
    if marker.get('rounds') != expected_rounds or len(rows) != expected_rounds or [r['round'] for r in rows] != list(range(1, expected_rounds + 1)):
        raise RuntimeError('FL result has missing or unexpected rounds')
    for name in ('protocol', 'rounds', 'summary'):
        if marker.get(name + '_sha256') != c.digest(folder / (name + '.json')):
            raise RuntimeError('FL result checksum mismatch: ' + name)
    for row in rows:
        pi = np.asarray(row['pi'])
        selected = row['selected']; clients = protocol.get('clients', len(pi))
        if len(selected) != 3 or len(set(selected)) != 3 or len(pi) != clients or any(not isinstance(i, int) or not 0 <= i < clients for i in selected) or not np.isfinite(pi).all() or np.any(pi <= 0) or np.any(pi > 1) or not np.isclose(pi.sum(), 3.):
            raise RuntimeError('Invalid selection or inclusion probabilities')
        if any(not np.isfinite(row[k]) for k in ('test_accuracy', 'test_loss', 'update_norm')):
            raise RuntimeError('Nonfinite FL result')
    if c.read(folder / 'summary.json') != summarize(rows): raise RuntimeError('FL summary differs from records')
    return marker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--proxy-root', type=Path)
    p.add_argument('--iteration', type=int, default=0); p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--alpha', choices=c.ALPHAS, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--rounds', type=int, default=1500)
    p.add_argument('--run', action='store_true'); p.add_argument('--deadline', type=float)
    a = p.parse_args()
    if a.rounds < 1: p.error('--rounds must be positive')
    if a.method not in ('fedavg', 'uniform_real') and a.proxy_root is None: p.error('Proxy methods need --proxy-root')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]), ensure_ascii=False)); return
    def stop(_signal, _frame): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as scheduler
    with scheduler.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
