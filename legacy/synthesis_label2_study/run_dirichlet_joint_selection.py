"""Fixed-marginal proxy joint sampling; old-loss 32000 real/residual FL."""
import argparse
import copy
import json
import signal
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import joint_direction_sampling as joint
import run_dirichlet_proxy_fl as base
import run_dirichlet_proxy_fl_fingerprintfix as fix

c = base.c
TARGETS = ('representative', 'diversity')
AGGREGATIONS = ('real', 'residual')
ETAS = (.25, .5, .75, 1.)


def route_protocol(original, target, aggregation, eta):
    if target not in TARGETS or aggregation not in AGGREGATIONS or eta not in ETAS:
        raise ValueError('Invalid joint target/aggregation/eta')
    frozen = dict(seed=2025, clients=32, selected=3, local_steps=3, batch=256,
                  real_lr=.01, proxy_lr=.01, real_ce='ordinary',
                  proxy_ce=c.read(base.EXPECTED)['weighted_ce'],
                  augmentation=False, iteration=32000, proxy_loss='original')
    changed = [f'{key}: expected {value!r}, got {original.get(key)!r}'
               for key, value in frozen.items() if original.get(key) != value]
    if changed:
        raise RuntimeError('Frozen old-loss FL settings changed: ' + '; '.join(changed))
    if not 1 <= original.get('rounds', 0) <= 1500:
        raise RuntimeError('Invalid frozen round target')
    weights = np.asarray(original.get('weights'), dtype=float)
    if weights.shape != (32,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise RuntimeError('Invalid frozen client weights')
    protocol = copy.deepcopy(original)
    protocol.update(version=3, method='joint_' + aggregation, target=target, eta=eta,
        clustering=None, renormalize=False, residual_alpha=1.,
        score='norm(P)/(3*0.01)', importance='w_i*score_i',
        sampling='eta*optimal triple distribution+(1-eta)*original AOCS; fixed original marginals',
        aggregation=('sum_selected(w_i/pi_i*R_i)' if aggregation == 'real' else
                     'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))'),
        aggregation_kind=aggregation,
        joint_objective=('squared norm of selected HT proxy minus all-client weighted proxy'
                         if target == 'representative' else 'sum_pairs(max(0,cos(P_i,P_j))**2)'),
        joint_constraints='q>=0; sum(q)=1; sum_{S containing i}(q_S)=original_pi_i',
        joint_update='recompute all 32 P and original pi each round; enumerate all 4960 triples',
        zero_direction='zero proxy vector has zero normalized direction; no same-direction penalty',
        solver=dict(name='scipy.optimize.linprog/HiGHS', scipy=scipy.__version__,
                    method='highs-ds', options=dict(joint.SOLVER_OPTIONS),
                    formulation=joint.FORMULATION,
                    marginal_rtol=joint.MARGINAL_RTOL, certificate_tolerance=joint.CERTIFICATE_TOL,
                    threads=1, failure='stop; no fallback or marginal replacement'),
        diagnostic='proxy objectives and selected real geometry; no extra real updates')
    protocol['round_seeds'].update(joint_branch='+600000', joint_subset='+700000')
    for file in (Path(__file__).resolve(), Path(joint.__file__).resolve()):
        protocol['files_sha256']['code/' + file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    return protocol


def prepare(a):
    if a.iteration != 32000 or a.proxy_root is None:
        raise ValueError('Joint sampling requires old-loss 32000 proxies')
    mapped = copy.copy(a)
    mapped.method = 'score_real' if a.aggregation == 'real' else 'residual_score'
    prepared = fix.prepare(mapped)
    if prepared[-1]['proxy_loss'] != 'original':
        raise RuntimeError('Joint sampling uses only original-loss images')
    return (*prepared[:-1], route_protocol(prepared[-1], a.target, a.aggregation, a.eta))


def one_round(net, real, syn, weights, target, aggregation, eta, rnd, update_fn=None):
    if target not in TARGETS or aggregation not in AGGREGATIONS or eta not in ETAS:
        raise ValueError('Invalid joint target/aggregation/eta')
    weights = np.asarray(weights, dtype=float)
    if len(real) != 32 or len(syn) != 32 or weights.shape != (32,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid 32-client weights/proxies')
    update_fn = update_fn or c.update
    seed = base.SEED + rnd * 1000
    net.train()
    proxies = [update_fn(net, x, y, seed + 10000000 + i, cw)
               for i, (x, y, cw) in enumerate(syn)]
    selected, pi, details = joint.joint_sample(proxies, weights, target, eta, seed)
    actual = {i: update_fn(net, *real[i], seed + i) for i in selected}
    factors = weights[selected] / pi[selected]
    truth = sum(float(w) * actual[i] for i, w in zip(selected, factors))
    baseline = sum(float(w) * p for w, p in zip(weights, proxies))
    correction = sum(float(w) * (actual[i] - proxies[i]) for i, w in zip(selected, factors))
    delta = truth if aggregation == 'real' else baseline + correction
    if not torch.isfinite(delta).all():
        raise RuntimeError('Nonfinite federated update')
    comparisons = []
    for i, factor in zip(selected, factors):
        e2 = float((actual[i].double() - proxies[i].double()).square().sum())
        comparisons.append(dict(client=i, global_weight=float(weights[i]),
            inclusion_probability=float(pi[i]), aggregation_weight=float(factor),
            **c.geometry(proxies[i], actual[i]), residual_squared_norm=e2,
            residual_norm=float(np.sqrt(e2))))
    row = dict(round=rnd, target=target, aggregation_kind=aggregation, eta=eta,
        selected=selected, pi=pi.tolist(), scores=details['scores'],
        importance=details['importance'], proxy_norms=details['proxy_norms'], joint=details,
        selected_real=comparisons, selected_sample_fraction=float(weights[selected].sum()),
        true_weight_sum=float(factors.sum()), aggregation_weight_sum=float(factors.sum()),
        selected_real_aggregate_norm=float(truth.norm()), proxy_aggregate_norm=float(baseline.norm()),
        correction_norm=float(correction.norm()), update_norm=float(delta.norm()))
    return delta, row


def replay(protocol, rows):
    weights = np.asarray(protocol['weights'], dtype=float)
    for row in rows:
        target, aggregation, eta = protocol['target'], protocol['aggregation_kind'], protocol['eta']
        if (row['target'], row['aggregation_kind'], row['eta']) != (target, aggregation, eta):
            raise RuntimeError('Joint route definition differs from records')
        selected, pi = joint.replay_selection(row['joint'], weights,
            base.SEED + row['round'] * 1000, target, eta)
        if selected != row['selected'] or not np.allclose(row['pi'], pi, rtol=0, atol=1e-12):
            raise RuntimeError('Joint selection or original inclusion probabilities differ')
        for name in ('scores', 'importance', 'proxy_norms'):
            if row[name] != row['joint'][name]:
                raise RuntimeError('Joint score/geometry differs: ' + name)
        if [item['client'] for item in row['selected_real']] != selected:
            raise RuntimeError('Missing selected-client observations')
        factors = weights[selected] / pi[selected]
        for item, factor in zip(row['selected_real'], factors):
            i = item['client']
            expected = (weights[i], pi[i], factor)
            observed = (item['global_weight'], item['inclusion_probability'], item['aggregation_weight'])
            if not np.allclose(observed, expected, rtol=1e-12, atol=1e-15):
                raise RuntimeError('Changed HT aggregation weights')
            pn, rn, cosine, e2 = (item[k] for k in
                                  ('proxy_norm', 'real_norm', 'cosine', 'residual_squared_norm'))
            if not np.isfinite([pn, rn, cosine, e2]).all() or min(pn, rn, e2) < 0 or abs(cosine) > 1 + 1e-12:
                raise RuntimeError('Invalid selected real/proxy geometry')
            if not np.isclose(pn, row['proxy_norms'][i], rtol=joint.LEGACY_NORM_RTOL, atol=1e-20):
                raise RuntimeError('Selected geometry differs from current proxy norm')
            squared = pn**2 + rn**2 - 2*pn*rn*cosine
            if not np.isclose(e2, squared, rtol=1e-9, atol=1e-12) or not np.isclose(item['residual_norm'], np.sqrt(e2), rtol=1e-12, atol=1e-15):
                raise RuntimeError('Residual geometry differs from observations')
        for name in ('true_weight_sum', 'aggregation_weight_sum'):
            if not np.isclose(row[name], factors.sum(), rtol=1e-12, atol=1e-15):
                raise RuntimeError('Changed aggregate weight sum')


def train(a, prepared):
    net, real, syn, weights, test, protocol = prepared
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder / 'protocol.json', protocol)
    sha = c.digest(folder / 'protocol.json')
    if (folder / 'COMPLETED.json').exists():
        verify(folder, a.rounds)
        return True
    checkpoint = folder / 'resume.pt'
    rows, elapsed = [], 0.
    if checkpoint.exists():
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != sha:
            raise RuntimeError('Checkpoint protocol/fingerprint mismatch')
        rows, elapsed = old['records'], old['elapsed_seconds']
        if old['round'] != len(rows) or len(rows) > a.rounds or [r['round'] for r in rows] != list(range(1, len(rows)+1)):
            raise RuntimeError('Checkpoint round coverage mismatch')
        replay(protocol, rows)
        net.load_state_dict(old['model'])
        c.restore_rng(old['rng'])
    else:
        c.seed_all(base.SEED)
    tick = time.monotonic()

    def persist(state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows,
            elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=sha))
        c.atomic_json(folder / 'rounds.json', rows)
        c.atomic_json(folder / 'summary.json', base.summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder / 'status.json', dict(state=state, round=len(rows), target=a.rounds,
            elapsed_seconds=duration, eta_seconds=float(np.mean(recent))*(a.rounds-len(rows)) if recent else None))

    for rnd in range(len(rows)+1, a.rounds+1):
        if c.expired(a.deadline):
            persist('budget_exhausted')
            return False
        start = time.monotonic()
        delta, row = one_round(net, real, syn, weights, a.target, a.aggregation, a.eta, rnd)
        c.add_delta(net, delta)
        row.update(fix.evaluate(net, *test))
        row.update(round_seconds=time.monotonic()-start,
                   elapsed_seconds=elapsed+time.monotonic()-tick)
        rows.append(row)
        if rnd % 10 == 0 or rnd == a.rounds or c.expired(a.deadline):
            persist('training')
    persist('complete')
    c.atomic_json(folder / 'COMPLETED.json', dict(rounds=a.rounds, protocol_sha256=sha,
        rounds_sha256=c.digest(folder / 'rounds.json'), summary_sha256=c.digest(folder / 'summary.json')))
    verify(folder, a.rounds)
    return True


def verify(output, expected_rounds=1500):
    marker = base.verify(output, expected_rounds)
    folder = Path(output)
    protocol = c.read(folder / 'protocol.json')
    if protocol != route_protocol(protocol, protocol['target'], protocol['aggregation_kind'], protocol['eta']):
        raise RuntimeError('Joint settings or code fingerprints changed')
    replay(protocol, c.read(folder / 'rounds.json'))
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'proxy-root', 'data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--alpha', choices=c.ALPHAS, required=True)
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--aggregation', choices=AGGREGATIONS, required=True)
    parser.add_argument('--eta', type=float, choices=ETAS, required=True)
    parser.add_argument('--iteration', type=int, default=32000)
    parser.add_argument('--rounds', type=int, default=1500)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--run', action='store_true')
    a = parser.parse_args()
    if not 1 <= a.rounds <= 1500:
        parser.error('Invalid round count')
    if a.run and not torch.cuda.is_available():
        raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1])))
        return

    def stop(*_):
        a.deadline = 0.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output):
        raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__':
    main()
