"""Uniform equal/sample-weighted SCAFFOLD with current corrected synthetic updates."""
import argparse
import copy
import json
import signal
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import run_dirichlet_control_variates as cv

c, base, fix = cv.c, cv.base, cv.fix
METHODS = ('scaffold_proxy_equal', 'scaffold_proxy_weighted')
SEED, CLIENTS, SELECTED, LR, STEPS = cv.SEED, cv.CLIENTS, cv.SELECTED, cv.LR, cv.STEPS


def route_protocol(original, method):
    if method not in METHODS: raise ValueError('Unknown proxy-SCAFFOLD method')
    protocol = cv.route_protocol(original, 'proxy_equal')
    weighted = method.endswith('_weighted'); a = 'w_i' if weighted else '1/32'
    protocol.update(version=1, method=method, family='scaffold_proxy',
        weights_mode='samples' if weighted else 'equal',
        aggregation=f'sum_all({a}*Q_i_SC)+sum_selected({a}/Pi_i*(R_i_SC-Q_i_SC))',
        reference='32 current synthetic SCAFFOLD three-step updates at the same global model and OLD c_i,c',
        local_correction='every real/synthetic SGD step: gradient-c_i_old+c_old; no posthoc correction',
        control_update=f'c_i_new=c_i_old-c_old-R_i_SC/(3*0.01); c_new=c_old+sum_selected({a}*delta_c_i)',
        control_source='selected REAL corrected updates only; synthetic Q never refreshes controls',
        computation_order='all 32 Q with OLD controls; selected 3 R with OLD controls; aggregate; replace selected controls',
        initialization='all controls zero; first round uniform 3; no all-client warmup',
        control_invariant=f'c=sum_all({a}*c_i)', proxy_corrected=True,
        sampling='uniform without replacement; exact inclusion probability 3/32')
    path = Path(__file__).resolve()
    protocol['files_sha256']['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    return protocol


def prepare(a):
    mapped = copy.copy(a); mapped.method = 'proxy_equal'
    prepared = cv.prepare(mapped)
    return (*prepared[:-1], route_protocol(prepared[-1], a.method))


def corrected_proxy_update(net, x, y, seed, class_weights, client_control, global_control):
    """Class-weighted synthetic CE, with fixed OLD controls at every local step."""
    params = OrderedDict((name, p.detach().clone().requires_grad_(True)) for name, p in net.named_parameters())
    initial = list(params.values()); buffers = OrderedDict((name, b.detach().clone()) for name, b in net.named_buffers())
    correction = global_control-client_control; pieces = []; offset = 0
    for p in params.values():
        pieces.append(correction[offset:offset+p.numel()].view_as(p)); offset += p.numel()
    if offset != correction.numel(): raise ValueError('Control vector size mismatch')
    for batch in c.batches(len(y), seed):
        logits = torch.func.functional_call(net, (params, buffers), (x[batch].to(c.DEVICE),))
        gradients = torch.autograd.grad(c.ce_loss(logits, y[batch].to(c.DEVICE), class_weights), tuple(params.values()))
        params = OrderedDict((name, p-LR*(g+d)) for (name, p), g, d in zip(params.items(), gradients, pieces))
    return torch.cat([(p-old).reshape(-1) for p, old in zip(params.values(), initial)]).detach()


def aggregate(references, observed, weights, pi):
    """Difference estimator; also used to verify full coverage and finite-population unbiasedness."""
    weights = np.asarray(weights, dtype=float); pi = np.asarray(pi, dtype=float); n = len(references)
    if weights.shape != (n,) or pi.shape != (n,) or not observed or any(not 0 <= i < n for i in observed):
        raise ValueError('Invalid difference estimator dimensions')
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.) or not np.isfinite(pi).all() or np.any(pi <= 0) or np.any(pi > 1):
        raise ValueError('Invalid weights or inclusion probabilities')
    baseline = sum(float(a)*q for a, q in zip(weights, references))
    correction = sum(float(weights[i]/pi[i])*(r-references[i]) for i, r in observed.items())
    truth = sum(float(weights[i]/pi[i])*r for i, r in observed.items())
    delta = baseline+correction
    if any(not torch.isfinite(t).all() for t in (baseline, correction, truth, delta)):
        raise RuntimeError('Nonfinite proxy-SCAFFOLD aggregate')
    return delta, baseline, correction, truth


def initial_state(net, method):
    if method not in METHODS: raise ValueError('Unknown proxy-SCAFFOLD method')
    return cv.initial_state(net, 'scaffold_weighted' if method.endswith('_weighted') else 'scaffold_equal')


def selected_geometry(vectors, selected):
    points = torch.stack([vectors[i].detach().double() for i in selected])
    gram = points @ points.T; norms = gram.diag().clamp_min(0).sqrt()
    denominator = norms[:, None]*norms[None, :]
    cosine = torch.where(denominator > 0, gram/denominator.clamp_min(1e-300), torch.zeros_like(gram)).clamp(-1, 1)
    return dict(gram=gram.cpu().tolist(), cosine=cosine.cpu().tolist())


def one_round(net, real, syn, weights, method, rnd, state, scaffold_fn=None, proxy_fn=None):
    if method not in METHODS: raise ValueError('Unknown proxy-SCAFFOLD method')
    weights = np.asarray(weights, dtype=float)
    if len(real) != CLIENTS or len(syn or []) != CLIENTS or weights.shape != (CLIENTS,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid 32-client inputs/global weights')
    a = weights if method.endswith('_weighted') else np.full(CLIENTS, 1/CLIENTS)
    scaffold_fn = scaffold_fn or cv.scaffold_update; proxy_fn = proxy_fn or corrected_proxy_update
    cv.control_error(state, a); old_control = state['control'].detach().clone()
    old_seen = state['last_seen'].tolist(); before_hash = cv.cache_fingerprint(state['cache'])
    control_before = cv.tensor_fingerprint({'c':old_control}); seed = SEED+rnd*1000; net.train()
    # None of this round's real observations enters a synthetic trajectory or a control.
    reference = [proxy_fn(net, x, y, seed+10000000+i, w, state['cache'][i], old_control)
                 for i, (x, y, w) in enumerate(syn)]
    if any(not torch.isfinite(q).all() for q in reference): raise RuntimeError('Nonfinite corrected proxy')
    selected = sorted(np.random.default_rng(seed+17).choice(CLIENTS, SELECTED, replace=False).tolist())
    pi = np.full(CLIENTS, SELECTED/CLIENTS); factors = a[selected]/pi[selected]
    observed = {i:scaffold_fn(net, *real[i], seed+i, state['cache'][i], old_control) for i in selected}
    delta, baseline, correction, truth = aggregate(reference, observed, a, pi)
    residual = {i:observed[i]-reference[i] for i in selected}
    diagnostics = [dict(client=i, global_weight=float(weights[i]), objective_weight=float(a[i]),
        inclusion_probability=float(pi[i]), aggregation_weight=float(w), real_norm=float(observed[i].norm()),
        reference_norm=float(reference[i].norm()), residual_norm=float(residual[i].norm())) for i, w in zip(selected, factors)]
    geometry = dict(proxy=selected_geometry(reference, selected), real=selected_geometry(observed, selected),
                    residual=selected_geometry(residual, selected))
    # Option II uses the REAL corrected update, not its difference from Q.
    differences = {i:-old_control-observed[i]/(STEPS*LR) for i in selected}
    state['control'] = old_control+sum(float(a[i])*differences[i] for i in selected)
    for i in selected: state['cache'][i] = (state['cache'][i]+differences[i]).detach().clone()
    state['last_seen'][selected] = rnd
    error = cv.control_error(state, a)
    row = dict(round=rnd, selected=selected, pi=pi.tolist(), selected_real=diagnostics,
        selected_sample_fraction=float(weights[selected].sum()), aggregation_weight_sum=float(factors.sum()),
        true_weight_sum=float(factors.sum()), update_norm=float(delta.norm()), selected_real_aggregate_norm=float(truth.norm()),
        baseline_norm=float(baseline.norm()), proxy_aggregate_norm=float(baseline.norm()), correction_norm=float(correction.norm()),
        reference_norms=[float(q.norm()) for q in reference], selected_geometry=geometry,
        last_seen_before=old_seen, cache_sha256_before=before_hash, cache_sha256_after=cv.cache_fingerprint(state['cache']),
        control_sha256_before=control_before, control_sha256_after=cv.tensor_fingerprint({'c':state['control']}),
        control_mean_error=error, control_norm_before=float(old_control.norm()), control_norm=float(state['control'].norm()))
    return delta, row


def verify_records(protocol, rows):
    cv.verify_records(protocol, rows)
    previous_control = None
    for row in rows:
        if len(row['reference_norms']) != CLIENTS or not np.isfinite(row['reference_norms']).all() or min(row['reference_norms']) < 0:
            raise RuntimeError('Invalid corrected proxy diagnostics')
        if previous_control is not None and previous_control != row['control_sha256_before']:
            raise RuntimeError('Global control continuity mismatch')
        previous_control = row['control_sha256_after']
        if any(not np.isfinite(row[k]) or row[k] < 0 for k in ('baseline_norm', 'correction_norm', 'control_norm_before')):
            raise RuntimeError('Nonfinite proxy/control diagnostics')
        for family, norm_key in (('proxy','reference_norm'), ('real','real_norm'), ('residual','residual_norm')):
            geometry = row['selected_geometry'][family]; gram = np.asarray(geometry['gram']); cosine = np.asarray(geometry['cosine'])
            if gram.shape != (SELECTED, SELECTED) or cosine.shape != gram.shape or not np.isfinite(gram).all() or not np.isfinite(cosine).all():
                raise RuntimeError('Invalid selected update geometry')
            norms = np.array([d[norm_key] for d in row['selected_real']])
            denominator = norms[:, None]*norms[None, :]
            wanted = np.clip(np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0), -1, 1)
            if not np.allclose(gram, gram.T, atol=1e-12, rtol=1e-10) or not np.allclose(gram.diagonal(), norms**2, atol=1e-10, rtol=2e-6) or not np.allclose(cosine, wanted, atol=2e-6, rtol=2e-6):
                raise RuntimeError('Selected geometry differs from recorded norms')


def train(a, prepared):
    net, real, syn, weights, test, protocol = prepared
    folder = Path(a.output); folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder/'protocol.json', protocol); sha = c.digest(folder/'protocol.json')
    if (folder/'COMPLETED.json').exists(): verify(folder, a.rounds); return True
    checkpoint = folder/'resume.pt'; rows = []; elapsed = 0.; state = initial_state(net, a.method)
    if checkpoint.exists():
        integrity = c.read(folder/'checkpoint_integrity.json')
        if integrity['checkpoint_sha256'] != c.digest(checkpoint) or integrity['protocol_sha256'] != sha:
            raise RuntimeError('Checkpoint integrity/protocol mismatch')
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != sha: raise RuntimeError('Checkpoint protocol mismatch')
        rows = old['records']; elapsed = old['elapsed_seconds']; state = old['state']
        if old['round'] != len(rows) or len(rows) > a.rounds or [r['round'] for r in rows] != list(range(1, len(rows)+1)):
            raise RuntimeError('Checkpoint round coverage mismatch')
        verify_records(protocol, rows); net.load_state_dict(old['model']); c.restore_rng(old['rng'])
        state['cache'] = [h.to(c.DEVICE) for h in state['cache']]; state['control'] = state['control'].to(c.DEVICE)
    else: c.seed_all(SEED)
    cv._validate_state(net, state, a.method, rows, weights); tick = time.monotonic()
    def persist(status):
        duration = elapsed+time.monotonic()-tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows, state=state,
            elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=sha))
        c.atomic_json(folder/'checkpoint_integrity.json', dict(checkpoint_sha256=c.digest(checkpoint), protocol_sha256=sha, round=len(rows)))
        c.atomic_json(folder/'rounds.json', rows); c.atomic_json(folder/'summary.json', base.summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder/'status.json', dict(state=status, round=len(rows), target=a.rounds, elapsed_seconds=duration,
            eta_seconds=float(np.mean(recent))*(a.rounds-len(rows)) if recent else None))
    for rnd in range(len(rows)+1, a.rounds+1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        start = time.monotonic(); delta, row = one_round(net, real, syn, weights, a.method, rnd, state)
        c.add_delta(net, delta); row.update(fix.evaluate(net, *test))
        row.update(round_seconds=time.monotonic()-start, elapsed_seconds=elapsed+time.monotonic()-tick); rows.append(row)
        if rnd % 10 == 0 or rnd == a.rounds or c.expired(a.deadline): persist('training')
    persist('verifying')
    c.atomic_json(folder/'COMPLETED.json', dict(rounds=a.rounds, protocol_sha256=sha,
        rounds_sha256=c.digest(folder/'rounds.json'), summary_sha256=c.digest(folder/'summary.json')))
    verify(folder, a.rounds)
    status = c.read(folder/'status.json'); status['state'] = 'complete'; c.atomic_json(folder/'status.json', status)
    return True


def verify(output, expected_rounds=1500):
    marker = base.verify(output, expected_rounds); folder = Path(output); protocol = c.read(folder/'protocol.json')
    if protocol != route_protocol(protocol, protocol['method']): raise RuntimeError('Proxy-SCAFFOLD settings/code changed')
    for key, wanted in protocol['files_sha256'].items():
        if key.startswith('code/') and c.digest(c.ROOT/key[5:]) != wanted: raise RuntimeError('Code fingerprint changed: '+key)
    rows = c.read(folder/'rounds.json'); verify_records(protocol, rows); checkpoint = folder/'resume.pt'
    if checkpoint.exists():
        integrity = c.read(folder/'checkpoint_integrity.json')
        if integrity['checkpoint_sha256'] != c.digest(checkpoint): raise RuntimeError('Completed checkpoint checksum mismatch')
        saved = c.load(checkpoint)
        if saved['round'] != expected_rounds or saved['protocol_sha256'] != marker['protocol_sha256'] or saved['records'] != rows:
            raise RuntimeError('Completed checkpoint record mismatch')
        cv._validate_state(None, saved['state'], protocol['method'], rows, protocol['weights'], protocol['model_parameters'])
        if rows[-1]['control_sha256_after'] != cv.tensor_fingerprint({'c':saved['state']['control']}):
            raise RuntimeError('Completed global control fingerprint mismatch')
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True); parser.add_argument('--proxy-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', choices=c.ALPHAS, required=True); parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--rounds', type=int, default=1500); parser.add_argument('--deadline', type=float); parser.add_argument('--run', action='store_true')
    a = parser.parse_args()
    if a.rounds < 1: parser.error('Round target must be positive')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as scheduler
    with scheduler.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
