"""Independent 39-route extension; current proxy clusters or latest-real history."""
import argparse
import copy
import json
import signal
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import run_dirichlet_proxy_fl as base
import run_dirichlet_proxy_fl_fingerprintfix as fix
import run_dirichlet_proxy_fl_gamma24 as gammafl
import dirichlet_continue32000_worker as continuation
import dirichlet_gamma24 as gamma
from cluster_sampling import spherical_clusters, stratified_sample
from cluster_factorial_common import full_direction_features, mixed_quotas

c = base.c
METHODS = ('cluster_score_real', 'cluster_residual_score', 'history_uniform_zero',
           'history_score_zero', 'history_score_proxy')
CLUSTERING = dict(representation='exact full proxy Gram embedding; spherical clustering',
                  quota_rho=.1, quota_mix=.1, min_one=False, within='weighted proxy score AOCS')
CODE = ('synthesis_label2_study/run_dirichlet_cluster_history.py',
        'synthesis_label2_study/schedule_dirichlet_cluster_history.py',
        'synthesis_label2_study/cluster_factorial_common.py',
        'cluster_sampling.py', 'cluster_sampling_guarded.py')


def route_protocol(original, method, clusters):
    if method not in METHODS or clusters not in ((2, 3) if method.startswith('cluster_') else (0,)):
        raise ValueError('Invalid method/cluster count')
    protocol = copy.deepcopy(original)
    protocol.update(version=2, method=method, renormalize=False, residual_alpha=1.,
        clustering=dict(CLUSTERING, k=clusters) if clusters else None,
        history=dict(initialization='proxy_at_theta0' if method.endswith('_proxy') else 'zero',
            update='replace selected cache with current R AFTER aggregation; retain unselected', ema=False)
            if method.startswith('history_') else None,
        sampling=('dependent quota rounding then within-cluster AOCS; unconditional exact marginals'
                  if clusters else 'uniform without replacement' if method == 'history_uniform_zero'
                  else 'fixed_aocs_sample, dependent rounding; exact marginals'),
        aggregation=('sum_selected(w_i/pi_i*R_i)' if method == 'cluster_score_real' else
            'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))' if clusters else
            'sum_all(w_i*H_i)+sum_selected(w_i/pi_i*(R_i-H_i)); H is pre-round cache'))
    protocol['round_seeds'].update(cluster='+400000', history_initial_proxy='2025+10000000+client at theta0')
    for name in CODE:
        protocol['files_sha256']['code/' + name] = c.digest(c.ROOT / name)
    return protocol


def prepare(a):
    mapped = copy.copy(a)
    mapped.method = 'uniform_real' if a.method == 'history_uniform_zero' else 'residual_score'
    if a.gamma24:
        with patch.object(continuation, 'gamma_manifest', gamma.gamma_manifest), patch.object(gammafl, 'PREPARE', fix.prepare):
            prepared = gammafl.prepare(mapped)
    else:
        prepared = fix.prepare(mapped)
        if mapped.method != 'uniform_real' and prepared[-1]['proxy_loss'] != 'original':
            raise RuntimeError('Old-image entry requires original-loss 32000 images')
    if mapped.method != 'uniform_real' and a.iteration != 32000:
        raise RuntimeError('This extension only uses the agreed 32000 images')
    return (*prepared[:-1], route_protocol(prepared[-1], a.method, a.clusters))


def initial_history(net, real, syn, method, update_fn=None):
    if not method.startswith('history_'): return None
    if method == 'history_score_proxy':
        net.train(); update_fn = update_fn or c.update
        return [update_fn(net, x, y, base.SEED + 10000000 + i, w).detach().clone()
                for i, (x, y, w) in enumerate(syn)]
    zero = torch.cat([torch.zeros_like(p).reshape(-1) for p in net.parameters()])
    return [zero.clone() for _ in real]


def one_round(net, real, syn, weights, method, clusters, rnd, history=None, last_seen=None, update_fn=None):
    if method not in METHODS: raise ValueError('Unknown extension method')
    update_fn = update_fn or c.update
    weights = np.asarray(weights, dtype=float); n = len(real); seed = base.SEED + rnd * 1000
    if n != 32 or weights.shape != (n,) or np.any(weights <= 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid 32-client global weights')
    net.train(); proxies = scores = None; detail = None
    if method != 'history_uniform_zero':
        if syn is None or len(syn) != n: raise ValueError('All current proxies are required for scoring')
        proxies = [update_fn(net, x, y, seed + 10000000 + i, w) for i, (x, y, w) in enumerate(syn)]
        scores = np.asarray([float(p.norm()) / .03 for p in proxies])
    rng = np.random.default_rng(seed + 400000)
    if clusters:
        if clusters not in (2, 3) or not method.startswith('cluster_'): raise ValueError('Invalid clustering route')
        directions = torch.stack(proxies).cpu().numpy().astype(np.float64)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
        features = full_direction_features(directions)
        labels, sizes, spread = spherical_clusters(features, clusters, seed + 400000)
        quotas = mixed_quotas(sizes, spread, CLUSTERING['quota_mix'])
        selected, pi, counts = stratified_sample(labels, quotas, weights * scores, True, rng, base.fixed_aocs_sample)
        detail = dict(labels=labels.tolist(), sizes=sizes.tolist(), spread=spread.tolist(),
                      expected_quotas=quotas.tolist(), actual_quotas=counts.tolist())
    elif method == 'history_uniform_zero':
        selected = sorted(np.random.default_rng(seed + 17).choice(n, 3, replace=False).tolist())
        pi = np.full(n, 3 / n)
    else:
        selected, pi = base.fixed_aocs_sample(weights * scores, 3, rng)
    if len(set(selected)) != 3 or pi.shape != (n,) or not np.isfinite(pi).all() or np.any(pi <= 0) or np.any(pi > 1) or not np.isclose(pi.sum(), 3.):
        raise RuntimeError('Invalid true inclusion probabilities')
    observed = {i: update_fn(net, *real[i], seed + i) for i in selected}
    factors = weights[selected] / pi[selected]
    truth = sum(float(w) * observed[i] for i, w in zip(selected, factors))
    cache = history if method.startswith('history_') else proxies if method == 'cluster_residual_score' else None
    if method.startswith('history_') and (history is None or len(history) != n or last_seen is None):
        raise RuntimeError('Missing pre-round history cache')
    baseline = sum(float(w) * h for w, h in zip(weights, cache)) if cache is not None else None
    correction = sum(float(w) * (observed[i] - cache[i]) for i, w in zip(selected, factors)) if cache is not None else None
    delta = truth if cache is None else baseline + correction
    if not torch.isfinite(delta).all(): raise RuntimeError('Nonfinite aggregate')
    row = dict(round=rnd, selected=selected, pi=pi.tolist(), scores=None if scores is None else scores.tolist(),
        importance=None if scores is None else (weights * scores).tolist(), clustering=detail,
        proxy_norms=None if proxies is None else [float(p.norm()) for p in proxies],
        history_norms=None if history is None else [float(h.norm()) for h in history],
        history_last_seen_before=None if last_seen is None else last_seen.tolist(),
        selected_real=[dict(client=i, global_weight=float(weights[i]), inclusion_probability=float(pi[i]),
            aggregation_weight=float(w), real_norm=float(observed[i].norm()),
            proxy_geometry=None if proxies is None else c.geometry(proxies[i], observed[i]),
            history_geometry=None if history is None else c.geometry(history[i], observed[i])) for i, w in zip(selected, factors)],
        selected_sample_fraction=float(weights[selected].sum()), true_weight_sum=float(factors.sum()),
        aggregation_weight_sum=float(factors.sum()), update_norm=float(delta.norm()),
        selected_real_aggregate_norm=float(truth.norm()), baseline_norm=None if baseline is None else float(baseline.norm()),
        correction_norm=None if correction is None else float(correction.norm()),
        min_inclusion_probability=float(pi.min()), max_possible_weight=float((weights / pi).max()),
        max_selected_weight=float(factors.max()))
    # Compute the complete correction with old H before replacing any selected entry.
    if history is not None:
        for i in selected: history[i] = observed[i].detach().clone(); last_seen[i] = rnd
    return delta, row


def train(a, prepared):
    net, real, syn, weights, test, protocol = prepared
    folder = Path(a.output); folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder / 'protocol.json', protocol); sha = c.digest(folder / 'protocol.json')
    if (folder / 'COMPLETED.json').exists(): verify(folder, a.rounds); return True
    checkpoint = folder / 'resume.pt'; rows = []; elapsed = 0.; history = None; last_seen = None
    if checkpoint.exists():
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != sha: raise RuntimeError('Checkpoint protocol/fingerprint mismatch')
        rows = old['records']; elapsed = old['elapsed_seconds']
        if old['round'] != len(rows) or len(rows) > a.rounds or [r['round'] for r in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError('Checkpoint round coverage mismatch')
        net.load_state_dict(old['model']); c.restore_rng(old['rng'])
        history = old['history']; last_seen = old['last_seen']
        if history is not None: history = [h.to(c.DEVICE) for h in history]
    else:
        c.seed_all(base.SEED); history = initial_history(net, real, syn, a.method)
        if history is not None: last_seen = np.zeros(len(real), dtype=int)
    if a.method.startswith('history_'):
        width = sum(p.numel() for p in net.parameters())
        if history is None or len(history) != len(real) or any(h.shape != (width,) or not torch.isfinite(h).all() for h in history):
            raise RuntimeError('Invalid historical cache in checkpoint')
        if last_seen is None or np.shape(last_seen) != (len(real),) or np.any(last_seen < 0) or np.any(last_seen > len(rows)):
            raise RuntimeError('Invalid historical observation rounds')
    tick = time.monotonic()
    def persist(state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows, history=history,
            last_seen=last_seen, elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=sha))
        c.atomic_json(folder / 'rounds.json', rows); c.atomic_json(folder / 'summary.json', base.summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder / 'status.json', dict(state=state, round=len(rows), target=a.rounds,
            elapsed_seconds=duration, eta_seconds=float(np.mean(recent)) * (a.rounds - len(rows)) if recent else None))
    for rnd in range(len(rows) + 1, a.rounds + 1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        start = time.monotonic()
        delta, row = one_round(net, real, syn, weights, a.method, a.clusters, rnd, history, last_seen)
        c.add_delta(net, delta); row.update(fix.evaluate(net, *test))
        row.update(round_seconds=time.monotonic() - start, elapsed_seconds=elapsed + time.monotonic() - tick)
        rows.append(row)
        if rnd % 10 == 0 or rnd == a.rounds or c.expired(a.deadline): persist('training')
    persist('complete')
    c.atomic_json(folder / 'COMPLETED.json', dict(rounds=a.rounds, protocol_sha256=sha,
        rounds_sha256=c.digest(folder / 'rounds.json'), summary_sha256=c.digest(folder / 'summary.json')))
    verify(folder, a.rounds); return True


def verify(output, expected_rounds=1500):
    marker = base.verify(output, expected_rounds); folder = Path(output)
    protocol = c.read(folder / 'protocol.json'); rows = c.read(folder / 'rounds.json')
    if protocol != route_protocol(protocol, protocol['method'], (protocol['clustering'] or {}).get('k', 0)):
        raise RuntimeError('Extension settings or code fingerprints changed')
    seen = np.zeros(protocol['clients'], dtype=int)
    for row in rows:
        if protocol['history']:
            if row['history_last_seen_before'] != seen.tolist(): raise RuntimeError('History was updated before aggregation')
            seen[row['selected']] = row['round']
        if protocol['clustering']:
            d = row['clustering']; k = protocol['clustering']['k']; labels = np.asarray(d['labels'])
            if len(labels) != 32 or set(labels) != set(range(k)) or np.bincount(labels, minlength=k).tolist() != d['sizes']:
                raise RuntimeError('Invalid cluster membership')
            q = mixed_quotas(d['sizes'], d['spread'], .1)
            if not np.allclose(q, d['expected_quotas'], atol=1e-12, rtol=0): raise RuntimeError('Quota rule mismatch')
            selected, pi, counts = stratified_sample(labels, q, np.asarray(row['importance']), True,
                np.random.default_rng(base.SEED + row['round'] * 1000 + 400000), base.fixed_aocs_sample)
            if selected != row['selected'] or not np.allclose(pi, row['pi'], atol=1e-12, rtol=0) or counts.tolist() != d['actual_quotas']:
                raise RuntimeError('Incorrect quota/AOCS unconditional probabilities')
    return marker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--proxy-root', type=Path)
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alpha', choices=c.ALPHAS, required=True); p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--clusters', type=int, choices=(0, 2, 3), default=0)
    p.add_argument('--iteration', type=int, default=32000); p.add_argument('--gamma24', action='store_true')
    p.add_argument('--rounds', type=int, default=1500); p.add_argument('--deadline', type=float); p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.rounds < 1: p.error('Round target must be positive')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as scheduler
    with scheduler.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
