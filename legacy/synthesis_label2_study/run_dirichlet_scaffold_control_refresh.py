"""Refresh all SCAFFOLD controls from selected real and unselected current proxy trajectories."""
import argparse
import copy
import json
import signal
from pathlib import Path
from types import FunctionType

import numpy as np
import torch
import run_dirichlet_proxy_scaffold as original

cv, c, base, fix = original.cv, original.c, original.base, original.fix
SEED, CLIENTS, SELECTED, LR, STEPS = original.SEED, original.CLIENTS, original.SELECTED, original.LR, original.STEPS
METHODS = ('scaffold_refresh_real_equal', 'scaffold_refresh_real_weighted',
           'scaffold_refresh_residual_equal', 'scaffold_refresh_residual_weighted')
corrected_proxy_update = original.corrected_proxy_update


def route_protocol(template, method):
    if method not in METHODS: raise ValueError('Unknown refreshed-control method')
    protocol = original.route_protocol(template, 'scaffold_proxy_equal')
    weighted = method.endswith('_weighted'); a = 'w_i' if weighted else '1/32'
    residual = '_residual_' in method
    protocol.update(method=method, family='scaffold_control_refresh', weights_mode='samples' if weighted else 'equal',
        aggregation=f'sum_all({a}*Q_i_SC)+sum_selected({a}/Pi_i*(R_i_SC-Q_i_SC))' if residual else
                    f'sum_selected({a}/Pi_i*R_i_SC)',
        model_aggregation='proxy_residual' if residual else 'real_ht',
        control_update=f'selected c_i_new=c_i_old-c_old-R_i_SC/(3*0.01); unselected c_i_new=c_i_old-c_old-Q_i_SC/(3*0.01); c_new=sum_all({a}*c_i_new)',
        control_source='selected REAL corrected trajectories; unselected corrected SYNTHETIC trajectories',
        computation_order='all Q and selected R use OLD theta,c_i,c; aggregate once; replace ALL controls for NEXT round',
        control_refresh='all32 every round; no Pi in control mean; real last_seen changes only for selected3',
        diagnostic_target='selected real Option II control = mean ordinary-CE gradient along its corrected three-step trajectory; no extra real computation',
        diagnostic_prediction='old c_i and current synthetic Option II control compared against selected real Option II control')
    path = Path(__file__).resolve()
    protocol['files_sha256']['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    return protocol


def prepare(a):
    mapped = copy.copy(a); mapped.method = 'scaffold_proxy_equal'
    prepared = original.prepare(mapped)
    return (*prepared[:-1], route_protocol(prepared[-1], a.method))


def initial_state(net, method):
    if method not in METHODS: raise ValueError('Unknown refreshed-control method')
    return original.initial_state(net, 'scaffold_proxy_weighted' if method.endswith('_weighted') else 'scaffold_proxy_equal')


def _prediction_error(prediction, truth):
    error = float((prediction-truth).norm()); pn, tn = float(prediction.norm()), float(truth.norm())
    cosine = float(torch.dot(prediction.double(), truth.double())/(pn*tn)) if pn > 0 and tn > 0 else None
    return dict(error_norm=error, relative_error=None if tn == 0 else error/tn,
                cosine=None if cosine is None else max(-1., min(1., cosine)))


def one_round(net, real, syn, weights, method, rnd, state, scaffold_fn=None, proxy_fn=None):
    if method not in METHODS: raise ValueError('Unknown refreshed-control method')
    weights = np.asarray(weights, dtype=float)
    if len(real) != CLIENTS or len(syn or []) != CLIENTS or weights.shape != (CLIENTS,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid 32-client inputs/global weights')
    a = weights if method.endswith('_weighted') else np.full(CLIENTS, 1/CLIENTS)
    scaffold_fn = scaffold_fn or cv.scaffold_update; proxy_fn = proxy_fn or corrected_proxy_update
    cv.control_error(state, a); old_control = state['control'].detach().clone()
    old_seen = state['last_seen'].tolist(); before_hash = cv.cache_fingerprint(state['cache'])
    control_before = cv.tensor_fingerprint({'c':old_control}); seed = SEED+rnd*1000; net.train()
    reference = [proxy_fn(net, x, y, seed+10000000+i, w, state['cache'][i], old_control)
                 for i, (x, y, w) in enumerate(syn)]
    selected = sorted(np.random.default_rng(seed+17).choice(CLIENTS, SELECTED, replace=False).tolist())
    pi = np.full(CLIENTS, SELECTED/CLIENTS); factors = a[selected]/pi[selected]
    observed = {i:scaffold_fn(net, *real[i], seed+i, state['cache'][i], old_control) for i in selected}
    residual_delta, baseline, correction, truth = original.aggregate(reference, observed, a, pi)
    delta = residual_delta if '_residual_' in method else truth
    # Compute both candidate controls before changing ANY old control.
    proxy_controls = [ci-old_control-q/(STEPS*LR) for ci, q in zip(state['cache'], reference)]
    real_controls = {i:state['cache'][i]-old_control-observed[i]/(STEPS*LR) for i in selected}
    residual = {i:observed[i]-reference[i] for i in selected}
    diagnostics = [dict(client=i, global_weight=float(weights[i]), objective_weight=float(a[i]),
        inclusion_probability=float(pi[i]), aggregation_weight=float(w), real_norm=float(observed[i].norm()),
        reference_norm=float(reference[i].norm()), residual_norm=float(residual[i].norm()),
        real_control_norm=float(real_controls[i].norm()), proxy_control_norm=float(proxy_controls[i].norm()),
        old_control_prediction=_prediction_error(state['cache'][i], real_controls[i]),
        proxy_control_prediction=_prediction_error(proxy_controls[i], real_controls[i])) for i, w in zip(selected, factors)]
    geometry = dict(proxy=original.selected_geometry(reference, selected), real=original.selected_geometry(observed, selected),
                    residual=original.selected_geometry(residual, selected))
    refreshed = [real_controls[i] if i in observed else proxy_controls[i] for i in range(CLIENTS)]
    if any(not torch.isfinite(ci).all() for ci in refreshed): raise RuntimeError('Nonfinite refreshed control')
    state['cache'] = [ci.detach().clone() for ci in refreshed]
    state['control'] = sum(float(w)*ci for w, ci in zip(a, state['cache'])).detach()
    state['last_seen'][selected] = rnd
    row = dict(round=rnd, selected=selected, pi=pi.tolist(), selected_real=diagnostics,
        selected_sample_fraction=float(weights[selected].sum()), aggregation_weight_sum=float(factors.sum()),
        true_weight_sum=float(factors.sum()), update_norm=float(delta.norm()), selected_real_aggregate_norm=float(truth.norm()),
        baseline_norm=float(baseline.norm()), proxy_aggregate_norm=float(baseline.norm()), correction_norm=float(correction.norm()),
        reference_norms=[float(q.norm()) for q in reference], selected_geometry=geometry,
        last_seen_before=old_seen, cache_sha256_before=before_hash, cache_sha256_after=cv.cache_fingerprint(state['cache']),
        control_sha256_before=control_before, control_sha256_after=cv.tensor_fingerprint({'c':state['control']}),
        control_mean_error=cv.control_error(state, a), control_norm_before=float(old_control.norm()), control_norm=float(state['control'].norm()),
        control_sources=['real' if i in observed else 'proxy' for i in range(CLIENTS)], control_refresh_count=CLIENTS,
        real_update_count=SELECTED, proxy_update_count=CLIENTS, real_step_count=STEPS*SELECTED, proxy_step_count=STEPS*CLIENTS,
        control_norms_after=[float(ci.norm()) for ci in state['cache']], model_aggregation='proxy_residual' if '_residual_' in method else 'real_ht')
    return delta, row


def verify_records(protocol, rows):
    original.verify_records(protocol, rows)
    for row in rows:
        selected = row['selected']
        expected = ['real' if i in selected else 'proxy' for i in range(CLIENTS)]
        counts = dict(control_refresh_count=CLIENTS, real_update_count=SELECTED, proxy_update_count=CLIENTS,
                      real_step_count=STEPS*SELECTED, proxy_step_count=STEPS*CLIENTS)
        if row['control_sources'] != expected or any(row[k] != v for k, v in counts.items()) or row['model_aggregation'] != protocol['model_aggregation']:
            raise RuntimeError('Control source/compute-count/model aggregation mismatch')
        norms = np.asarray(row['control_norms_after'])
        if norms.shape != (CLIENTS,) or not np.isfinite(norms).all() or np.any(norms < 0): raise RuntimeError('Invalid refreshed-control norms')
        if protocol['model_aggregation'] == 'real_ht' and not np.isclose(row['update_norm'], row['selected_real_aggregate_norm'], rtol=1e-12, atol=0):
            raise RuntimeError('Real HT route applied a synthetic model aggregate')
        for d in row['selected_real']:
            if not np.isclose(d['real_control_norm'], norms[d['client']], rtol=2e-6, atol=1e-9):
                raise RuntimeError('Selected control differs from real refresh')
            for key in ('old_control_prediction', 'proxy_control_prediction'):
                item = d[key]
                if not np.isfinite(item['error_norm']) or item['error_norm'] < 0 or (item['relative_error'] is not None and (not np.isfinite(item['relative_error']) or item['relative_error'] < 0)) or (item['cosine'] is not None and (not np.isfinite(item['cosine']) or not -1 <= item['cosine'] <= 1)):
                    raise RuntimeError('Invalid selected-control prediction diagnostic')


# Freeze the old persistence logic with a separate globals dictionary. The
# existing worker and its protocols stay unchanged; only this route is bound.
_globals = dict(original.train.__globals__, route_protocol=route_protocol, one_round=one_round,
                initial_state=initial_state, verify_records=verify_records)
verify = FunctionType(original.verify.__code__, _globals, name='verify', argdefs=original.verify.__defaults__, closure=original.verify.__closure__)
_globals['verify'] = verify
train = FunctionType(original.train.__code__, _globals, name='train', argdefs=original.train.__defaults__, closure=original.train.__closure__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'proxy-root', 'data-root', 'output'): parser.add_argument('--'+name, type=Path, required=True)
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
