"""Old-loss 32000 proxy residual FL; past residual RMS or sample-weight selection."""
import argparse
import copy
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
import run_dirichlet_proxy_fl as base
import run_dirichlet_proxy_fl_fingerprintfix as fix

c = base.c
METHODS = ('ema', 'weight')
BETAS = (.05, .1, .2, .3, .5)


def route_protocol(original, method, beta):
    if method not in METHODS or (beta not in BETAS if method == 'ema' else beta is not None):
        raise ValueError('Invalid residual-selection method/beta')
    protocol = copy.deepcopy(original)
    protocol.update(version=2, method=method, beta=beta, clustering=None,
        initialization='cold; no extra real updates', renormalize=False, residual_alpha=1.,
        score='sqrt(EMA of observed squared R-P norm)' if method == 'ema' else '1',
        importance='w_i*score_i',
        sampling='first round uniform; thereafter fixed AOCS exact marginals' if method == 'ema'
                 else 'fixed AOCS exact marginals',
        aggregation='sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))',
        ema_update='first observation initializes e2; later (1-beta)*v+beta*e2; unselected unchanged',
        unseen_prior='global RMS of strictly past observed squared residual norms',
        diagnostic='double-precision selected R-P norm; no extra real updates; forecasts precede observation')
    file = Path(__file__).resolve()
    protocol['files_sha256']['code/' + file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    return protocol


def prepare(a):
    if a.iteration != 32000 or a.proxy_root is None:
        raise ValueError('This experiment requires original-loss 32000 proxies')
    mapped = copy.copy(a); mapped.method = 'residual_score'
    prepared = fix.prepare(mapped)
    if prepared[-1]['proxy_loss'] != 'original':
        raise RuntimeError('EMA selection uses only original-loss images')
    return (*prepared[:-1], route_protocol(prepared[-1], a.method, a.beta))


def initial_history(clients=32):
    return dict(v=np.full(clients, np.nan), visits=np.zeros(clients, dtype=int),
                last_seen=np.zeros(clients, dtype=int), squared_sum=0., observations=0)


def forecasts(history):
    prior = history['squared_sum'] / history['observations'] if history['observations'] else np.nan
    return np.sqrt(np.where(np.isnan(history['v']), prior, history['v']))


def observe(history, comparisons, rnd, beta):
    for record in comparisons:
        i, e2 = record['client'], record['residual_squared_norm']
        if beta is not None:
            v = history['v'][i]
            history['v'][i] = e2 if np.isnan(v) else (1 - beta) * v + beta * e2
        history['squared_sum'] += e2; history['observations'] += 1
        history['visits'][i] += 1; history['last_seen'][i] = rnd


def nullable(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def selection(weights, method, rnd, predictions):
    seed = base.SEED + rnd * 1000
    scores = predictions if method == 'ema' and rnd > 1 else np.ones(len(weights))
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise RuntimeError('Invalid past-only selection scores')
    if method == 'ema' and rnd == 1:
        selected = sorted(np.random.default_rng(seed + 17).choice(len(weights), 3, replace=False).tolist())
        pi = np.full(len(weights), 3 / len(weights))
    else:
        selected, pi = base.fixed_aocs_sample(weights * scores, 3, np.random.default_rng(seed + 400000))
    return selected, pi, scores


def one_round(net, real, syn, weights, method, beta, rnd, history, update_fn=None):
    if method not in METHODS or (beta not in BETAS if method == 'ema' else beta is not None):
        raise ValueError('Invalid residual-selection method/beta')
    weights = np.asarray(weights, dtype=float); n = len(real)
    if n != 32 or len(syn) != n or weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid 32-client weights/proxies')
    update_fn = update_fn or c.update; seed = base.SEED + rnd * 1000; net.train()
    predictions = forecasts(history) if method == 'ema' else np.full(n, np.nan)
    selected, pi, scores = selection(weights, method, rnd, predictions)
    if len(set(selected)) != 3 or not np.isfinite(pi).all() or np.any(pi <= 0) or np.any(pi > 1) or not np.isclose(pi.sum(), 3.):
        raise RuntimeError('Invalid inclusion probabilities')
    proxies = [update_fn(net, x, y, seed + 10000000 + i, cw) for i, (x, y, cw) in enumerate(syn)]
    actual = {i: update_fn(net, *real[i], seed + i) for i in selected}
    factors = weights[selected] / pi[selected]
    baseline = sum(float(w) * p for w, p in zip(weights, proxies))
    correction = sum(float(w) * (actual[i] - proxies[i]) for i, w in zip(selected, factors))
    delta = baseline + correction
    if not torch.isfinite(delta).all(): raise RuntimeError('Nonfinite federated update')
    comparisons = []
    for i, factor in zip(selected, factors):
        e2 = float((actual[i].double() - proxies[i].double()).square().sum())
        comparisons.append(dict(client=i, global_weight=float(weights[i]), inclusion_probability=float(pi[i]),
            aggregation_weight=float(factor), **c.geometry(proxies[i], actual[i]),
            residual_squared_norm=e2, residual_norm=float(np.sqrt(e2)),
            predicted_residual_norm=None if not np.isfinite(predictions[i]) else float(predictions[i]),
            visits_before=int(history['visits'][i]), last_seen_before=int(history['last_seen'][i]),
            prediction_source='not_used' if method == 'weight' else 'client_ema' if history['visits'][i]
                              else 'past_global_rms' if history['observations'] else 'no_history'))
    row = dict(round=rnd, selected=selected, pi=pi.tolist(), scores=scores.tolist(),
        importance=(weights * scores).tolist(), predictions=nullable(predictions),
        ema_squared_before=nullable(history['v']), visits_before=history['visits'].tolist(),
        last_seen_before=history['last_seen'].tolist(), global_squared_sum_before=history['squared_sum'],
        observations_before=history['observations'], selected_real=comparisons,
        proxy_norms=[float(p.double().norm()) for p in proxies],
        selected_sample_fraction=float(weights[selected].sum()), true_weight_sum=float(factors.sum()),
        aggregation_weight_sum=float(factors.sum()), update_norm=float(delta.norm()),
        selected_real_aggregate_norm=float(sum(float(w) * actual[i] for i, w in zip(selected, factors)).norm()),
        proxy_aggregate_norm=float(baseline.norm()), correction_norm=float(correction.norm()),
        min_inclusion_probability=float(pi.min()), max_possible_weight=float((weights / pi).max()),
        max_selected_weight=float(factors.max()))
    observe(history, comparisons, rnd, beta)  # Only after selection, aggregation and all forecasts.
    return delta, row


def replay(protocol, rows):
    """Rebuild strictly past-only history and exact fixed-size sampling from saved observations."""
    method, beta = protocol['method'], protocol['beta']; weights = np.asarray(protocol['weights'], dtype=float)
    if weights.shape != (32,) or np.any(weights <= 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
        raise RuntimeError('Invalid replay client weights')
    history = initial_history(protocol['clients'])
    for row in rows:
        predictions = forecasts(history) if method == 'ema' else np.full(len(weights), np.nan)
        selected, pi, scores = selection(weights, method, row['round'], predictions)
        for key, expected in dict(predictions=nullable(predictions), ema_squared_before=nullable(history['v']),
                visits_before=history['visits'].tolist(), last_seen_before=history['last_seen'].tolist(),
                global_squared_sum_before=history['squared_sum'], observations_before=history['observations']).items():
            if row[key] != expected: raise RuntimeError('Past-only history/forecast mismatch: ' + key)
        if selected != row['selected'] or not np.allclose(row['pi'], pi, rtol=0, atol=1e-12):
            raise RuntimeError('Selection or true inclusion probability mismatch')
        if not np.allclose(row['scores'], scores, rtol=0, atol=1e-12) or not np.allclose(row['importance'], weights * scores, rtol=0, atol=1e-12):
            raise RuntimeError('Incorrect EMA/sample-weight scores')
        if [r['client'] for r in row['selected_real']] != selected: raise RuntimeError('Missing selected-client observations')
        for r in row['selected_real']:
            i = r['client']; e2 = r['residual_squared_norm']; pn, rn = r['proxy_norm'], r['real_norm']
            if not np.isfinite([e2, pn, rn, r['cosine']]).all() or e2 < 0 or pn < 0 or rn < 0 or abs(r['cosine']) > 1 + 1e-12:
                raise RuntimeError('Invalid double-precision residual geometry')
            reconstructed = pn ** 2 + rn ** 2 - 2 * pn * rn * r['cosine']
            if not np.isclose(e2, reconstructed, rtol=1e-9, atol=1e-12) or not np.isclose(r['residual_norm'], np.sqrt(e2), rtol=1e-12, atol=1e-15):
                raise RuntimeError('Residual squared norm differs from R/P geometry')
            if not np.isclose(r['relative_error'], np.sqrt(e2) / max(rn, 1e-15), rtol=1e-12, atol=1e-15) or not np.isclose(row['proxy_norms'][i], pn, rtol=1e-12, atol=1e-15):
                raise RuntimeError('Residual/proxy diagnostic geometry mismatch')
            forecast = None if not np.isfinite(predictions[i]) else float(predictions[i])
            if r['predicted_residual_norm'] != forecast or r['visits_before'] != int(history['visits'][i]) or r['last_seen_before'] != int(history['last_seen'][i]):
                raise RuntimeError('Selected forecast used current or incorrect history')
            source = 'not_used' if method == 'weight' else 'client_ema' if history['visits'][i] else 'past_global_rms' if history['observations'] else 'no_history'
            if r['prediction_source'] != source: raise RuntimeError('Prediction source mismatch')
            if not np.isclose(r['global_weight'], weights[i]) or not np.isclose(r['inclusion_probability'], pi[i]) or not np.isclose(r['aggregation_weight'], weights[i] / pi[i]):
                raise RuntimeError('Residual aggregation weight mismatch')
        observe(history, row['selected_real'], row['round'], beta)
    return history


def train(a, prepared):
    net, real, syn, weights, test, protocol = prepared
    folder = Path(a.output); folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder / 'protocol.json', protocol); sha = c.digest(folder / 'protocol.json')
    if (folder / 'COMPLETED.json').exists(): verify(folder, a.rounds); return True
    checkpoint = folder / 'resume.pt'; rows = []; elapsed = 0.; history = initial_history(len(real))
    if checkpoint.exists():
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != sha: raise RuntimeError('Checkpoint protocol/fingerprint mismatch')
        rows = old['records']; elapsed = old['elapsed_seconds']
        if old['round'] != len(rows) or len(rows) > a.rounds or [r['round'] for r in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError('Checkpoint round coverage mismatch')
        history = replay(protocol, rows)
        for key, value in history.items():
            if not np.array_equal(value, old['history'][key], equal_nan=True):
                raise RuntimeError('Checkpoint EMA history differs from observations')
        net.load_state_dict(old['model']); c.restore_rng(old['rng'])
    else: c.seed_all(base.SEED)
    tick = time.monotonic()
    def persist(state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows, history=history,
            elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=sha))
        c.atomic_json(folder / 'rounds.json', rows); c.atomic_json(folder / 'summary.json', base.summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder / 'status.json', dict(state=state, round=len(rows), target=a.rounds,
            elapsed_seconds=duration, eta_seconds=float(np.mean(recent)) * (a.rounds - len(rows)) if recent else None))
    for rnd in range(len(rows) + 1, a.rounds + 1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        start = time.monotonic()
        delta, row = one_round(net, real, syn, weights, a.method, a.beta, rnd, history)
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
    protocol = c.read(folder / 'protocol.json')
    if protocol != route_protocol(protocol, protocol['method'], protocol['beta']):
        raise RuntimeError('EMA settings or code fingerprints changed')
    replay(protocol, c.read(folder / 'rounds.json'))
    return marker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--proxy-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alpha', choices=c.ALPHAS, required=True); p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--beta', type=float, choices=BETAS); p.add_argument('--iteration', type=int, default=32000)
    p.add_argument('--rounds', type=int, default=1500); p.add_argument('--deadline', type=float); p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.rounds < 1 or (a.beta is None if a.method == 'ema' else a.beta is not None): p.error('Invalid rounds/method/beta')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as scheduler
    with scheduler.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
