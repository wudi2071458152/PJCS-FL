"""Fixed-image, independent FL seeds; delegate all existing numerical kernels."""
import argparse
import copy
import hashlib
import json
import signal
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import run_dirichlet_proxy_fl as base
import run_dirichlet_proxy_fl_fingerprintfix as fix
import run_dirichlet_cluster_history as cluster
import run_dirichlet_cluster_k456 as k456
import run_dirichlet_ema_selection as weight
import run_dirichlet_joint_selection as joint
import joint_direction_sampling_certfix as precision

c = base.c
precision.install()
NEW_SEEDS = (2027, 2028, 2029, 2030)
ROUTES = ('fedavg', 'uniform_real', 'score_real', 'cluster_score_real_k2',
          'cluster_score_real_k3', 'cluster_score_real_k4',
          'representative_real_eta0p5', 'representative_real_eta1p0',
          'residual_uniform', 'residual_score', 'weight', 'history_uniform_zero')
MODULES = (base, fix, cluster, k456, weight, joint, precision, precision.previous)
EXTENSIONS = ('clustering', 'history', 'beta', 'initialization', 'ema_update', 'unseen_prior',
              'diagnostic', 'target', 'eta', 'aggregation_kind', 'joint_objective',
              'joint_constraints', 'joint_update', 'zero_direction', 'solver',
              'fl_initialization', 'joint_numerical_policy', 'synthetic_updates_per_round', 'route')


@contextmanager
def seed_context(seed):
    if seed not in (2025, *NEW_SEEDS): raise ValueError('Unsupported FL seed')
    with patch.object(base, 'SEED', seed): yield


def state_fingerprint(state):
    """Hash tensor contents, avoiding torch.save container timestamp differences."""
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha256(model):
    return state_fingerprint(model.state_dict())


@lru_cache(maxsize=5)
def initialization_sha256(seed):
    return state_sha256(c.model(seed))


def route_protocol(template, route, seed):
    if route not in ROUTES or seed not in (2025, *NEW_SEEDS): raise ValueError('Invalid route/seed')
    original = copy.deepcopy(template)
    for key in EXTENSIONS: original.pop(key, None)
    frozen = dict(clients=32, selected=3, local_steps=3, batch=256, real_lr=.01,
                  proxy_lr=.01, real_ce='ordinary', augmentation=False,
                  proxy_ce=c.read(base.EXPECTED)['weighted_ce'])
    if any(original.get(k) != v for k, v in frozen.items()):
        raise RuntimeError('Frozen multi-seed training settings changed')
    if not 1 <= original.get('rounds', 0) <= 1500: raise ValueError('Invalid round target')
    no_proxy = route in ('fedavg', 'uniform_real', 'history_uniform_zero')
    method = route if route in base.METHODS else 'uniform_real' if no_proxy else 'residual_score'
    residual = method.startswith('residual_')
    original.update(version=1, seed=2025, method=method, iteration=0 if no_proxy else 32000,
        proxy_loss=None if no_proxy else 'original', residual_alpha=1., renormalize=method == 'fedavg',
        score='norm(P)/(3*0.01)', importance='w_i*score_i',
        sampling='fixed_aocs_sample, dependent rounding; exact marginals' if method in ('score_real', 'residual_score')
                 else 'uniform without replacement',
        aggregation='sum_selected(n_i/sum_selected(n_j)*R_i)' if method == 'fedavg' else
                    'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))' if residual else
                    'sum_selected(w_i/pi_i*R_i)')
    original['round_seeds'] = dict(base='2025+round*1000', uniform='+17', score='+400000',
                                  real='+client', proxy='+10000000+client')
    if route.startswith('cluster_'):
        k = int(route[-1])
        builder = k456.route_protocol if k == 4 else cluster.route_protocol
        protocol = builder(original, 'cluster_score_real', k)
    elif route.startswith('representative_'):
        protocol = joint.route_protocol(original, 'representative', 'real', .5 if route.endswith('0p5') else 1.)
    elif route == 'weight': protocol = weight.route_protocol(original, 'weight', None)
    elif route == 'history_uniform_zero': protocol = cluster.route_protocol(original, route, 0)
    else: protocol = original
    protocol.update(version=4, route=route, seed=seed, synthetic_updates_per_round=0 if no_proxy else 32,
        fl_initialization=dict(seed=seed, initializer='dirichlet_screen_common.model(seed)',
            state_sha256=initialization_sha256(seed), source_reference_seed=2025,
            source_reference_role='frozen synthesis lineage; new FL model generated independently',
            source2025_initializer_parity='exact state_dict equality checked during prepare'))
    protocol['round_seeds']['base'] = f'{seed}+round*1000'
    if 'history_initial_proxy' in protocol['round_seeds']:
        protocol['round_seeds']['history_initial_proxy'] = f'{seed}+10000000+client at theta0'
    if route.startswith('representative_'):
        protocol['joint_numerical_policy'] = dict(bound_tolerance=precision.previous.BOUND_TOL,
            retry_dual_tolerance=precision.DUAL_TOL, certificate_tolerance=precision.legacy.CERTIFICATE_TOL,
            policy='record bounded roundoff cleanup; retry identical LP only after certificate gap')
    for module in MODULES:
        file = Path(module.__file__).resolve()
        protocol['files_sha256']['code/' + file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    file = Path(__file__).resolve()
    protocol['files_sha256']['code/' + file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    return protocol


def prepare(a):
    if a.iteration != 32000 or a.proxy_root is None: raise ValueError('Fixed old-loss 32000 proxies required')
    mapped = copy.copy(a); mapped.method = 'residual_score'
    with seed_context(2025): prepared = fix.prepare(mapped)
    if prepared[-1]['proxy_loss'] != 'original': raise RuntimeError('Only original-loss images are supported')
    source_model = prepared[0]; regenerated = c.model(2025)
    first, second = source_model.state_dict(), regenerated.state_dict()
    if first.keys() != second.keys() or any(not torch.equal(first[k], second[k]) for k in first):
        raise RuntimeError('Source seed2025 initial model differs from the agreed initializer')
    net = c.model(a.seed)
    protocol = route_protocol(prepared[-1], a.route, a.seed)
    if state_sha256(net) != protocol['fl_initialization']['state_sha256']:
        raise RuntimeError('Generated initial model fingerprint differs')
    return net, *prepared[1:-1], protocol


def one_round(net, real, syn, weights, route, seed, rnd, history=None, last_seen=None, update_fn=None):
    with seed_context(seed):
        if route in base.METHODS:
            return base.one_round(net, real, syn, weights, route, seed, rnd, update_fn)
        if route == 'weight':
            return weight.one_round(net, real, syn, weights, route, None, rnd, history, update_fn)
        if route == 'history_uniform_zero':
            return cluster.one_round(net, real, syn, weights, route, 0, rnd, history, last_seen, update_fn)
        if route.startswith('cluster_') and route in ROUTES:
            k = int(route[-1]); kernel = k456.one_round if k == 4 else cluster.one_round
            return kernel(net, real, syn, weights, 'cluster_score_real', k, rnd, update_fn=update_fn)
        if route.startswith('representative_') and route in ROUTES:
            return joint.one_round(net, real, syn, weights, 'representative', 'real',
                                   .5 if route.endswith('0p5') else 1., rnd, update_fn)
    raise ValueError('Invalid route')


def replay(protocol, rows):
    route = protocol['route']; weights = np.asarray(protocol['weights'], dtype=float)
    seen = np.zeros(len(weights), dtype=int)
    with seed_context(protocol['seed']):
        rebuilt = weight.replay(protocol, rows) if route == 'weight' else None
        if route.startswith('representative_'): joint.replay(protocol, rows)
        for row in rows:
            seed = protocol['seed'] + row['round'] * 1000
            if route.startswith('representative_'): selected, pi = row['selected'], np.asarray(row['pi'])
            elif route == 'weight':
                selected, pi, _ = weight.selection(weights, route, row['round'], None)
            elif route.startswith('cluster_'):
                d = row['clustering']; k = protocol['clustering']['k']; labels = np.asarray(d['labels'])
                if labels.shape != (32,) or set(labels) != set(range(k)) or np.bincount(labels, minlength=k).tolist() != d['sizes']:
                    raise RuntimeError('Invalid cluster membership')
                quotas = cluster.mixed_quotas(d['sizes'], d['spread'], .1)
                if not np.allclose(quotas, d['expected_quotas'], rtol=0, atol=1e-12): raise RuntimeError('Cluster quota mismatch')
                selected, pi, counts = cluster.stratified_sample(labels, quotas, np.asarray(row['importance']), True,
                    np.random.default_rng(seed + 400000), base.fixed_aocs_sample)
                if counts.tolist() != d['actual_quotas']: raise RuntimeError('Cluster rounded quota mismatch')
            elif route in ('score_real', 'residual_score'):
                scores = np.asarray(row['proxy_norms']) / .03
                if not np.allclose(row['scores'], scores, rtol=1e-12, atol=1e-15) or not np.allclose(row['importance'], weights * scores, rtol=1e-12, atol=1e-15):
                    raise RuntimeError('Proxy score mismatch')
                selected, pi = base.fixed_aocs_sample(weights * scores, 3, np.random.default_rng(seed + 400000))
            else:
                selected = sorted(np.random.default_rng(seed + 17).choice(32, 3, replace=False).tolist())
                pi = np.full(32, 3 / 32)
            if selected != row['selected'] or not np.allclose(pi, row['pi'], rtol=0, atol=1e-12):
                raise RuntimeError('Selection or inclusion probability mismatch')
            if route.startswith('cluster_') or route == 'residual_uniform':
                scores = np.asarray(row['proxy_norms']) / .03
                if not np.allclose(row['scores'], scores, rtol=1e-12, atol=1e-15) or not np.allclose(row['importance'], weights * scores, rtol=1e-12, atol=1e-15):
                    raise RuntimeError('Proxy score mismatch')
            factors = weights[selected] / (weights[selected].sum() if route == 'fedavg' else pi[selected])
            if [r['client'] for r in row['selected_real']] != selected: raise RuntimeError('Missing real observations')
            for item, factor in zip(row['selected_real'], factors):
                i = item['client']
                if not np.allclose([item['global_weight'], item['inclusion_probability'], item['aggregation_weight']],
                                   [weights[i], pi[i], factor], rtol=1e-12, atol=1e-15):
                    raise RuntimeError('Aggregation weight mismatch')
            if not np.isclose(row['aggregation_weight_sum'], factors.sum(), rtol=1e-12, atol=1e-15):
                raise RuntimeError('Aggregate weight sum mismatch')
            if not np.isclose(row['true_weight_sum'], (weights[selected] / pi[selected]).sum(), rtol=1e-12, atol=1e-15):
                raise RuntimeError('HT weight sum mismatch')
            if route == 'history_uniform_zero':
                if row['history_last_seen_before'] != seen.tolist(): raise RuntimeError('History updated before aggregation')
                seen[selected] = row['round']
    return rebuilt, seen if route == 'history_uniform_zero' else None


def verify_protocol(folder):
    protocol = c.read(Path(folder) / 'protocol.json')
    if protocol != route_protocol(protocol, protocol['route'], protocol['seed']):
        raise RuntimeError('Multi-seed protocol or numerical source changed')
    for name, sha in protocol['files_sha256'].items():
        if not name.startswith('code/'): continue
        path = c.ROOT / name[5:]
        if not path.is_file(): path = Path(__file__).parent / name[5:]
        if not path.is_file() or c.digest(path) != sha: raise RuntimeError('Numerical code fingerprint changed: ' + name)
    return protocol


def verify(folder, rounds=1500):
    folder = Path(folder); marker = base.verify(folder, rounds)
    protocol = verify_protocol(folder)
    replay(protocol, c.read(folder / 'rounds.json'))
    return marker


def train(a, prepared):
    net, real, syn, weights, test, protocol = prepared
    if (a.route, a.seed, a.rounds) != (protocol['route'], protocol['seed'], protocol['rounds']):
        raise RuntimeError('Arguments differ from prepared route')
    folder = Path(a.output); folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder / 'protocol.json', protocol); sha = c.digest(folder / 'protocol.json')
    if (folder / 'COMPLETED.json').exists(): verify(folder, a.rounds); return True
    checkpoint = folder / 'resume.pt'; rows = []; elapsed = 0.; history = last_seen = None
    if checkpoint.exists():
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != sha: raise RuntimeError('Checkpoint protocol/fingerprint mismatch')
        rows, elapsed = old['records'], old['elapsed_seconds']
        if old['round'] != len(rows) or len(rows) > a.rounds or [r['round'] for r in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError('Checkpoint round coverage mismatch')
        rebuilt, seen = replay(protocol, rows); history, last_seen = old['history'], old['last_seen']
        if a.route == 'weight':
            for key, value in rebuilt.items():
                if not np.array_equal(value, history[key], equal_nan=True): raise RuntimeError('Checkpoint weight history mismatch')
        elif a.route == 'history_uniform_zero':
            width = sum(p.numel() for p in net.parameters())
            if history is None or len(history) != 32 or any(h.shape != (width,) or not torch.isfinite(h).all() for h in history):
                raise RuntimeError('Invalid checkpoint history cache')
            if not np.array_equal(last_seen, seen): raise RuntimeError('Checkpoint history observation rounds mismatch')
            if old.get('history_state_sha256') != state_fingerprint({str(i): h for i, h in enumerate(history)}):
                raise RuntimeError('Checkpoint history vector fingerprint mismatch')
            history = [h.to(c.DEVICE) for h in history]
        elif history is not None or last_seen is not None: raise RuntimeError('Unexpected checkpoint history')
        net.load_state_dict(old['model']); c.restore_rng(old['rng'])
    else:
        if state_sha256(net) != protocol['fl_initialization']['state_sha256']: raise RuntimeError('Initial model differs from protocol')
        c.seed_all(a.seed)
        if a.route == 'weight': history = weight.initial_history(32)
        elif a.route == 'history_uniform_zero':
            history = cluster.initial_history(net, real, syn, a.route); last_seen = np.zeros(32, dtype=int)
    tick = time.monotonic()
    def persist(state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows,
            history=history, last_seen=last_seen,
            history_state_sha256=state_fingerprint({str(i): h for i, h in enumerate(history)}) if a.route == 'history_uniform_zero' else None,
            elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=sha))
        c.atomic_json(folder / 'rounds.json', rows); c.atomic_json(folder / 'summary.json', base.summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder / 'status.json', dict(state=state, round=len(rows), target=a.rounds,
            elapsed_seconds=duration, eta_seconds=float(np.mean(recent)) * (a.rounds - len(rows)) if recent else None))
    for rnd in range(len(rows) + 1, a.rounds + 1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        start = time.monotonic()
        delta, row = one_round(net, real, syn, weights, a.route, a.seed, rnd, history, last_seen)
        c.add_delta(net, delta); row.update(fix.evaluate(net, *test))
        row.update(round_seconds=time.monotonic() - start, elapsed_seconds=elapsed + time.monotonic() - tick)
        rows.append(row)
        if rnd % 10 == 0 or rnd == a.rounds or c.expired(a.deadline): persist('training')
    persist('complete')
    c.atomic_json(folder / 'COMPLETED.json', dict(rounds=a.rounds, protocol_sha256=sha,
        rounds_sha256=c.digest(folder / 'rounds.json'), summary_sha256=c.digest(folder / 'summary.json')))
    verify(folder, a.rounds); return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'proxy-root', 'data-root', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--alpha', choices=c.ALPHAS, required=True)
    parser.add_argument('--route', choices=ROUTES, required=True); parser.add_argument('--seed', type=int, choices=NEW_SEEDS, required=True)
    parser.add_argument('--rounds', type=int, default=1500); parser.add_argument('--iteration', type=int, choices=(32000,), default=32000)
    parser.add_argument('--deadline', type=float); parser.add_argument('--run', action='store_true')
    a = parser.parse_args()
    if a.rounds != 1500: parser.error('Formal multi-seed routes require 1500 rounds')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]), ensure_ascii=False)); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
