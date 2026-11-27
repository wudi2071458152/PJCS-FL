"""Equal/sample-weighted SCAFFOLD and historical/current-proxy residual pilot."""
import argparse
import copy
import hashlib
import json
import signal
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import run_dirichlet_proxy_fl_fingerprintfix as fix

c, base = fix.p.c, fix.p
METHODS = ('scaffold_equal', 'scaffold_weighted', 'history_equal', 'history_weighted',
           'proxy_equal', 'proxy_weighted')
SEED, CLIENTS, SELECTED, LR, STEPS = 2025, 32, 3, .01, 3


def tensor_fingerprint(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode()); digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def cache_fingerprint(cache):
    return tensor_fingerprint({str(i): value for i, value in enumerate(cache)})


def route_protocol(original, method):
    if method not in METHODS: raise ValueError('Unknown control-variate method')
    frozen = dict(seed=SEED, clients=CLIENTS, selected=SELECTED, local_steps=STEPS,
                  batch=256, real_lr=LR, real_ce='ordinary', augmentation=False)
    if original.get('alpha') not in c.ALPHAS or any(original.get(k) != v for k, v in frozen.items()):
        raise RuntimeError('Frozen real FL settings changed')
    proxy = method.startswith('proxy_'); weighted = method.endswith('_weighted')
    if proxy and (original.get('iteration') != 32000 or original.get('proxy_loss') != 'original' or original.get('proxy_lr') != LR):
        raise RuntimeError('Current-proxy routes require original-loss 32000 images')
    protocol = copy.deepcopy(original); files = protocol['files_sha256']
    legacy = 'code/' + base.EXPECTED.name
    canonical = 'code/' + base.EXPECTED.resolve().relative_to(c.ROOT).as_posix()
    if legacy in files:
        if canonical in files and files[canonical] != files[legacy]:
            raise RuntimeError('Conflicting source protocol fingerprints')
        files[canonical] = files.pop(legacy)
    if not proxy:
        for key in list(files):
            if key.startswith(('proxy/', 'source/client_', 'source/calibration/')): files.pop(key)
    for key in ('sampler_ast_sha256', 'sampler_copied_from'): protocol.pop(key, None)
    a = 'w_i' if weighted else '1/32'
    family = method.split('_')[0]
    protocol.update(version=1, method=method, family=family, weights_mode='samples' if weighted else 'equal',
        iteration=32000 if proxy else 0, proxy_loss='original' if proxy else None,
        proxy_lr=LR if proxy else None, proxy_ce=original.get('proxy_ce') if proxy else None,
        proxy_required=proxy, score=None, importance=None, residual_alpha=1., renormalize=False,
        sampling='uniform without replacement; exact inclusion probability 3/32',
        aggregation=f'sum_selected({a}/Pi_i*R_i)' if family == 'scaffold' else
            f'sum_all({a}*V_i)+sum_selected({a}/Pi_i*(R_i-V_i))',
        reference='current P_i recomputed at each round global model' if proxy else
            'latest selected R_i; zero initial; replace AFTER aggregate' if family == 'history' else
            'per-step gradient-c_i_old+c_old; Option II; c=sum_all(a_i*c_i)',
        control_update='c_i_new=c_i_old-c_old-R_i/(3*0.01); c_new=c_old+sum_selected(a_i*delta_c_i)'
            if family == 'scaffold' else None,
        initialization='all controls/caches zero; first round uniform 3; no all-client warmup',
        server_lr=1., server_momentum=0., local_momentum=0., weight_decay=0., ema=False,
        round_seeds=dict(base='2025+round*1000', uniform='+17', real='+client', proxy='+10000000+client'),
        initial_model_state_sha256=original.get('initial_model_state_sha256'))
    for module in (fix, base, c):
        path = Path(module.__file__).resolve()
        files['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    path = Path(__file__).resolve()
    files['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    return protocol


def prepare(a):
    mapped = copy.copy(a); mapped.iteration = 32000 if a.method.startswith('proxy_') else 0
    mapped.method = 'residual_uniform' if a.method.startswith('proxy_') else 'fedavg'
    if mapped.method == 'fedavg': mapped.proxy_root = None
    prepared = fix.prepare(mapped)
    prepared[-1]['initial_model_state_sha256'] = tensor_fingerprint(prepared[0].state_dict())
    prepared[-1]['model_parameters'] = sum(p.numel() for p in prepared[0].parameters())
    return (*prepared[:-1], route_protocol(prepared[-1], a.method))


def scaffold_update(net, x, y, seed, client_control, global_control):
    """Correct every SGD step at its own parameter state; controls stay fixed."""
    params = OrderedDict((name, p.detach().clone().requires_grad_(True)) for name, p in net.named_parameters())
    initial = list(params.values()); buffers = OrderedDict((name, b.detach().clone()) for name, b in net.named_buffers())
    correction = global_control - client_control; offset = 0; pieces = []
    for p in params.values():
        pieces.append(correction[offset:offset+p.numel()].view_as(p)); offset += p.numel()
    if offset != correction.numel(): raise ValueError('Control vector size mismatch')
    for batch in c.batches(len(y), seed):
        logits = torch.func.functional_call(net, (params, buffers), (x[batch].to(c.DEVICE),))
        gradients = torch.autograd.grad(c.ce_loss(logits, y[batch].to(c.DEVICE)), tuple(params.values()))
        params = OrderedDict((name, p - LR * (g + d))
                             for (name, p), g, d in zip(params.items(), gradients, pieces))
    return torch.cat([(p - old).reshape(-1) for p, old in zip(params.values(), initial)]).detach()


def initial_state(net, method):
    if method not in METHODS: raise ValueError('Unknown control-variate method')
    if method.startswith('proxy_'): return None
    zero = torch.cat([torch.zeros_like(p).reshape(-1) for p in net.parameters()])
    return dict(cache=[zero.clone() for _ in range(CLIENTS)], last_seen=np.zeros(CLIENTS, dtype=int),
                control=zero.clone() if method.startswith('scaffold_') else None)


def control_error(state, a):
    average = sum(float(w) * h for w, h in zip(a, state['cache']))
    if not torch.isfinite(average).all() or not torch.isfinite(state['control']).all():
        raise RuntimeError('Nonfinite global control/cache mean')
    error = float((state['control'] - average).norm())
    if error > 1e-6 + 1e-5 * float(average.norm()): raise RuntimeError('Global control/cache weighted mean mismatch')
    return error


def one_round(net, real, syn, weights, method, rnd, state=None, update_fn=None, scaffold_fn=None):
    if method not in METHODS: raise ValueError('Unknown control-variate method')
    weights = np.asarray(weights, dtype=float)
    if len(real) != CLIENTS or weights.shape != (CLIENTS,) or np.any(weights <= 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid 32-client global weights')
    a = weights if method.endswith('_weighted') else np.full(CLIENTS, 1/CLIENTS)
    update_fn = update_fn or c.update; scaffold_fn = scaffold_fn or scaffold_update
    seed = SEED + rnd * 1000; net.train()
    selected = sorted(np.random.default_rng(seed + 17).choice(CLIENTS, SELECTED, replace=False).tolist())
    pi = np.full(CLIENTS, SELECTED/CLIENTS); factors = a[selected] / pi[selected]
    family = method.split('_')[0]; old_seen = None; before_hash = None; old_control = None
    if family != 'proxy':
        if state is None or len(state['cache']) != CLIENTS: raise RuntimeError('Missing historical/control cache')
        old_seen = state['last_seen'].tolist(); before_hash = cache_fingerprint(state['cache'])
    if family == 'scaffold':
        control_error(state, a); old_control = state['control'].detach().clone()
        observed = {i: scaffold_fn(net, *real[i], seed+i, state['cache'][i], old_control) for i in selected}
        reference = None
    else:
        reference = state['cache'] if family == 'history' else [
            update_fn(net, x, y, seed+10000000+i, w) for i, (x, y, w) in enumerate(syn or [])]
        if len(reference) != CLIENTS: raise RuntimeError('Missing 32 current proxy updates')
        observed = {i: update_fn(net, *real[i], seed+i) for i in selected}
    truth = sum(float(w)*observed[i] for i, w in zip(selected, factors))
    baseline = None if reference is None else sum(float(w)*v for w, v in zip(a, reference))
    correction = None if reference is None else sum(float(w)*(observed[i]-reference[i]) for i, w in zip(selected, factors))
    delta = truth if reference is None else baseline+correction
    if not torch.isfinite(delta).all(): raise RuntimeError('Nonfinite federated aggregate')
    diagnostics = [dict(client=i, global_weight=float(weights[i]), objective_weight=float(a[i]),
        inclusion_probability=float(pi[i]), aggregation_weight=float(w), real_norm=float(observed[i].norm()),
        reference_norm=None if reference is None else float(reference[i].norm())) for i, w in zip(selected, factors)]
    reference_norms = None if reference is None else [float(v.norm()) for v in reference]
    if family == 'scaffold':
        differences = {i: -old_control-observed[i]/(STEPS*LR) for i in selected}
        state['control'] = old_control + sum(float(a[i])*differences[i] for i in selected)
        for i in selected: state['cache'][i] = (state['cache'][i]+differences[i]).detach().clone()
    elif family == 'history':
        for i in selected: state['cache'][i] = observed[i].detach().clone()
    if state is not None:
        state['last_seen'][selected] = rnd
        if any(not torch.isfinite(h).all() for h in state['cache']): raise RuntimeError('Nonfinite cache')
    row = dict(round=rnd, selected=selected, pi=pi.tolist(), selected_real=diagnostics,
        selected_sample_fraction=float(weights[selected].sum()), aggregation_weight_sum=float(factors.sum()),
        true_weight_sum=float(factors.sum()), update_norm=float(delta.norm()), selected_real_aggregate_norm=float(truth.norm()),
        baseline_norm=None if baseline is None else float(baseline.norm()),
        proxy_aggregate_norm=float(baseline.norm()) if family == 'proxy' else None,
        correction_norm=None if correction is None else float(correction.norm()), reference_norms=reference_norms,
        last_seen_before=old_seen, cache_sha256_before=before_hash,
        cache_sha256_after=None if state is None else cache_fingerprint(state['cache']),
        control_mean_error=None if family != 'scaffold' else control_error(state, a),
        control_norm=None if family != 'scaffold' else float(state['control'].norm()))
    return delta, row


def _validate_state(net, state, method, rows, weights, width=None):
    if method.startswith('proxy_'):
        if state is not None: raise RuntimeError('Unexpected proxy cache')
        return
    width = sum(p.numel() for p in net.parameters()) if width is None else width
    seen = np.zeros(CLIENTS, dtype=int)
    for row in rows: seen[row['selected']] = row['round']
    if state is None or len(state['cache']) != CLIENTS or any(h.shape != (width,) or not torch.isfinite(h).all() for h in state['cache']):
        raise RuntimeError('Invalid historical/control checkpoint cache')
    if not np.array_equal(state['last_seen'], seen): raise RuntimeError('Cache last observation coverage mismatch')
    if rows and rows[-1]['cache_sha256_after'] != cache_fingerprint(state['cache']): raise RuntimeError('Checkpoint cache differs from records')
    if method.startswith('scaffold_'):
        if state['control'].shape != (width,) or not torch.isfinite(state['control']).all(): raise RuntimeError('Invalid global control')
        control_error(state, weights if method.endswith('_weighted') else np.full(CLIENTS, 1/CLIENTS))
    elif state['control'] is not None: raise RuntimeError('Unexpected historical global control')


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
        verify_records(protocol, rows)
        net.load_state_dict(old['model']); c.restore_rng(old['rng'])
        if state is not None:
            state['cache'] = [h.to(c.DEVICE) for h in state['cache']]
            if state['control'] is not None: state['control'] = state['control'].to(c.DEVICE)
    else: c.seed_all(SEED)
    _validate_state(net, state, a.method, rows, weights); tick = time.monotonic()
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


def verify_records(protocol, rows):
    method = protocol['method']; a = np.asarray(protocol['weights']) if method.endswith('_weighted') else np.full(CLIENTS, 1/CLIENTS)
    seen = np.zeros(CLIENTS, dtype=int); previous = None
    for row in rows:
        rnd = row['round']; selected = sorted(np.random.default_rng(SEED+rnd*1000+17).choice(CLIENTS, SELECTED, replace=False).tolist())
        if row['selected'] != selected or not np.array_equal(row['pi'], np.full(CLIENTS, SELECTED/CLIENTS)):
            raise RuntimeError('Uniform selection/inclusion replay mismatch')
        if [d['client'] for d in row['selected_real']] != selected: raise RuntimeError('Selected diagnostics order mismatch')
        for i, d in zip(selected, row['selected_real']):
            for key, wanted in (('global_weight', protocol['weights'][i]), ('objective_weight', a[i]),
                                ('inclusion_probability', SELECTED/CLIENTS), ('aggregation_weight', a[i]/(SELECTED/CLIENTS))):
                if not np.isclose(d[key], wanted, atol=1e-12, rtol=0): raise RuntimeError('Aggregation coefficient mismatch')
        if not method.startswith('proxy_'):
            if row['last_seen_before'] != seen.tolist(): raise RuntimeError('Cache updated before aggregation')
            if previous is not None and previous != row['cache_sha256_before']: raise RuntimeError('Cache continuity mismatch')
            previous = row['cache_sha256_after']; seen[selected] = rnd
        if method.startswith('scaffold_') and (not np.isfinite(row['control_mean_error']) or row['control_mean_error'] > 1e-6+1e-5*row['control_norm']):
            raise RuntimeError('Global control invariant failed')
        for key in ('aggregation_weight_sum', 'true_weight_sum'):
            if not np.isclose(row[key], sum(a[selected]/(SELECTED/CLIENTS)), atol=1e-12, rtol=0): raise RuntimeError('Aggregation mass mismatch')
        if any(not np.isfinite(row[k]) for k in ('test_accuracy', 'test_loss', 'update_norm')): raise RuntimeError('Nonfinite result')


def verify(output, expected_rounds=1500):
    marker = base.verify(output, expected_rounds); folder = Path(output); protocol = c.read(folder/'protocol.json')
    if protocol != route_protocol(protocol, protocol['method']): raise RuntimeError('Control-variate settings/code changed')
    for key, wanted in protocol['files_sha256'].items():
        if key.startswith('code/') and c.digest(c.ROOT/key[5:]) != wanted: raise RuntimeError('Code fingerprint changed: '+key)
    rows = c.read(folder/'rounds.json'); verify_records(protocol, rows)
    checkpoint = folder/'resume.pt'
    if checkpoint.exists():
        integrity = c.read(folder/'checkpoint_integrity.json')
        if integrity['checkpoint_sha256'] != c.digest(checkpoint): raise RuntimeError('Completed checkpoint checksum mismatch')
        saved = c.load(checkpoint)
        if saved['round'] != expected_rounds or saved['protocol_sha256'] != marker['protocol_sha256'] or saved['records'] != rows:
            raise RuntimeError('Completed checkpoint record mismatch')
        _validate_state(None, saved['state'], protocol['method'], rows, protocol['weights'], protocol['model_parameters'])
    return marker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True); p.add_argument('--proxy-root', type=Path)
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alpha', choices=c.ALPHAS, required=True); p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--rounds', type=int, default=1500); p.add_argument('--deadline', type=float); p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.rounds < 1: p.error('Round target must be positive')
    if a.method.startswith('proxy_') and a.proxy_root is None: p.error('Current proxy routes require --proxy-root')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as scheduler
    with scheduler.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
