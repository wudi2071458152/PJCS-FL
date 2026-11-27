"""Confirmed PoC/IS-HT/DELTA-HT/DivFL kernels; this module never starts training."""
import time
from collections import OrderedDict
import numpy as np
import torch
from scipy.spatial.distance import cdist
import dirichlet_screen_common as c
from run_dirichlet_proxy_fl import fixed_aocs_sample
from pracis_sampling_comparison import pracis_feedback, ht_coefficients

ROUTES = ('poc', 'is_ht', 'delta_ht', 'divfl')
SOURCES = {
    'poc': dict(url='https://proceedings.mlr.press/v151/jee-cho22a/jee-cho22a.pdf',
                section='Section 4: sample-size weighted candidates without replacement, random loss ties',
                implementation='new common-protocol implementation of the paper; not author source'),
    'is_ht': dict(url='https://github.com/T-Lab-CUHKSZ/DELTA_FL',
                  revision='bc8e3a7aed6691ea6a954837656437721edf81e7',
                  source='archive/test_runs/practical_delta_fresh_review_20261013/pro_comm_helpers.py',
                  sha256='f43dd6886dc99b8ec5d1dda381c0986c137fc8af687079c83fd9a2aed4487078'),
    'divfl': dict(url='https://github.com/melodi-lab/divfl',
                 revision='4a5b80df85a72b97751ce1130ee5c7d76a2c1540',
                 source='archive/test_runs/divfl_fresh_review_20261013/flearn_trainers_fedbase.py',
                 sha256='223a7aca6fb1bf68451d2135c75739886ec7916f388f0f64f10b8839eeee22e0')}
SOURCES['delta_ht'] = SOURCES['is_ht'].copy()


def method_protocol(route):
    if route not in ROUTES: raise ValueError('Unknown baseline route')
    definition = {
        'poc': dict(selection='6 sample-size weighted WOR candidates, top3 current full-data mean ordinary CE; random exact ties',
                    aggregation='sum_selected(R_i/3)', feedback=None),
        'is_ht': dict(selection='fixed_aocs_sample(old u), true inclusion Pi',
                      aggregation='sum_selected(w_i/Pi_i*R_i), no normalization',
                      score='w_i*norm(mean 3 trajectory gradients)', beta=0.,
                      feedback='redistribute selected old u mass AFTER current aggregation; all-zero retains old u'),
        'delta_ht': dict(selection='fixed_aocs_sample(old u), true inclusion Pi',
                         aggregation='sum_selected(w_i/Pi_i*R_i), no normalization',
                         score='w_i*(0.5*norm(gbar_i)+0.5*norm(gbar_i-mean_selected_gbar))', beta=.5,
                         feedback='same as IS-HT; equal selected-client center, no HT center'),
        'divfl': dict(selection='author lazy greedy, unweighted facility location, full-parameter Euclidean distance',
                      aggregation='sum_selected(w_i*R_i)/sum_selected(w_i)',
                      feedback='theta0 all32 full-data gradient queries; selected3 queried before local SGD and cache replaced for NEXT round',
                      first_round_query_calls=35, gradient='fixed current global model, ordinary mean CE over all local real data',
                      distance='unsquared Euclidean, no vector normalization or sample weights')}
    return dict(definition[route], source=SOURCES[route], local_lr=.01, local_steps=3, batch_size=256,
                rng='seed+round*1000+client for real batches; selection +400000; query preserves training RNG')


def _weights(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) < 6 or np.any(w <= 0) or not np.isfinite(w).all() or not np.isclose(w.sum(), 1.):
        raise ValueError('Positive global sample fractions are required')
    return w


def initial_state(net, route, weights):
    n = len(_weights(weights))
    if route in ('is_ht', 'delta_ht'): return dict(u=np.full(n, 1/n))
    if route == 'divfl': return dict(gradients=[None]*n, updated_round=[-1]*n, distance_matrix=None)
    if route == 'poc': return {}
    raise ValueError('Unknown baseline route')


def _sync(net):
    device = next(net.parameters()).device
    if device.type == 'cuda': torch.cuda.synchronize(device)


def _clock(net):
    _sync(net)
    return time.perf_counter()


def full_data_query(net, x, y, gradient=False):
    """No optimizer step, no training-batch RNG consumption, no parameter/buffer mutation."""
    if len(y) == 0: raise ValueError('An empty client cannot answer a full-data query')
    rng = c.rng_state(); device = next(net.parameters()).device
    params = OrderedDict((name, p.detach().clone().requires_grad_(gradient)) for name, p in net.named_parameters())
    buffers = OrderedDict((name, b.detach().clone()) for name, b in net.named_buffers())
    totals = [torch.zeros_like(p) for p in params.values()] if gradient else None
    loss_sum = 0.
    try:
        with torch.set_grad_enabled(gradient):
            for start in range(0, len(y), 256):
                xb, yb = x[start:start+256].to(device), y[start:start+256].to(device)
                loss = c.ce_loss(torch.func.functional_call(net, (params, buffers), (xb,)), yb)
                loss_sum += float(loss.detach())*len(yb)
                if gradient:
                    grads = torch.autograd.grad(loss, tuple(params.values()))
                    for total, grad in zip(totals, grads): total.add_(grad.detach(), alpha=len(yb)/len(y))
        return torch.cat([g.reshape(-1) for g in totals]).detach() if gradient else loss_sum/len(y)
    finally:
        c.restore_rng(rng)


def trajectory_observation(net, x, y, seed):
    """Identical three continuous SGD steps, retaining their actual average gradient."""
    device = next(net.parameters()).device
    params = OrderedDict((name, p.detach().clone().requires_grad_(True)) for name, p in net.named_parameters())
    initial = list(params.values())
    buffers = OrderedDict((name, b.detach().clone()) for name, b in net.named_buffers())
    totals = [torch.zeros_like(p) for p in initial]
    for batch in c.batches(len(y), seed):
        loss = c.ce_loss(torch.func.functional_call(net, (params, buffers), (x[batch].to(device),)), y[batch].to(device))
        grads = torch.autograd.grad(loss, tuple(params.values()))
        for total, grad in zip(totals, grads): total.add_(grad.detach())
        params = OrderedDict((name, p-.01*g) for (name, p), g in zip(params.items(), grads))
    delta = torch.cat([(p-old).reshape(-1) for p, old in zip(params.values(), initial)]).detach()
    return delta, torch.cat([g.reshape(-1) for g in totals]).div_(3).detach()


def feedback_scores(gradients, selected_weights, beta):
    if not 0 <= beta <= 1: raise ValueError('Invalid DELTA beta')
    g = torch.stack(gradients).double()
    gram = (g@g.T).detach().cpu().numpy()
    norms = np.sqrt(np.maximum(np.diag(gram), 0.))
    deviation = np.sqrt(np.maximum(np.diag(gram)-2*gram.mean(1)+gram.mean(), 0.))
    return np.asarray(selected_weights)*((1-beta)*norms+beta*deviation), gram, norms, deviation


def poc_selection(weights, seed, candidate_losses=None):
    rng = np.random.default_rng(seed)
    candidates = rng.choice(len(weights), 6, replace=False, p=weights).tolist()
    priority = rng.random(6)  # Independent random priority for exact loss ties.
    if candidate_losses is None: return candidates, priority
    selected = sorted(range(6), key=lambda j: (-candidate_losses[j], priority[j]))[:3]
    return sorted(candidates[j] for j in selected)


def lazy_greedy(distance, m=3):
    """Port of author's fedbase.py lazy_greedy (including its NumPy tie order)."""
    d = np.asarray(distance, dtype=float)
    if d.shape != (len(d), len(d)) or not np.isfinite(d).all() or np.any(d < 0): raise ValueError('Invalid distances')
    chosen = set(); utility = d.sum(0); first = int(utility.argmin())
    constant = 2*utility.max(); utility = constant-utility
    nearest = d[:, first]; chosen.add(first); current = utility[first]; utility[first] = -1.
    while len(chosen) < m:
        order = np.argsort(utility)[len(chosen):]
        for pos in range(len(order)):
            i = int(order[-pos-1]); chosen.add(i)
            candidate_nearest = np.minimum(nearest, d[:, i]); total = constant-candidate_nearest.sum()
            utility[i] = total-current
            if pos and utility[i] < utility[previous]:
                if pos == len(order)-1 or utility[previous] >= utility[order[-pos-2]]:
                    current += utility[previous]; chosen.remove(i); chosen.add(previous)
                    utility[previous] = -1.; nearest = previous_nearest.copy(); break
                chosen.remove(i)
            elif pos:
                if pos == len(order)-1 or utility[i] >= utility[order[-pos-2]]:
                    current = total; utility[i] = -1.; nearest = candidate_nearest.copy(); break
                previous = i; chosen.remove(i); previous_nearest = candidate_nearest.copy()
            elif utility[i] >= utility[order[-pos-2]]:
                current = total; utility[i] = -1.; nearest = candidate_nearest.copy(); break
            else:
                previous = i; chosen.remove(i); previous_nearest = candidate_nearest.copy()
    return sorted(chosen)


def _distances(gradients, selected=None):
    values = torch.stack(gradients).double().cpu().numpy()
    result = cdist(values if selected is None else values[selected], values, metric='euclidean')
    if selected is None: np.fill_diagonal(result, 0.)
    return result


def one_round(net, real, weights, route, seed, rnd, state, update_fn=None, query_fn=None):
    if route not in ROUTES: raise ValueError('Unknown baseline route')
    w = _weights(weights); n = len(w); base = seed+rnd*1000
    if len(real) != n: raise ValueError('Client count differs')
    query_fn = query_fn or full_data_query; net.train()
    costs = dict(client_train_seconds=0., query_seconds=0., selection_seconds=0., aggregation_seconds=0.,
                 geometry_seconds=0., feedback_seconds=0., client_train_calls=0, client_train_steps=0,
                 client_train_examples=0, query_calls=0, query_clients=0, query_examples=0, query_forward_examples=0, query_backward_examples=0)
    row = dict(round=rnd, route=route, query_client_ids=[]); queried = set()
    def query(i, gradient):
        start = _clock(net); value = query_fn(net, *real[i], gradient=gradient); _sync(net)
        costs['query_seconds'] += time.perf_counter()-start
        costs['query_calls'] += 1; costs['query_examples'] += len(real[i][1])
        row['query_client_ids'].append(int(i))
        queried.add(i); costs['query_clients'] = len(queried)
        costs['query_forward_examples'] += len(real[i][1]); costs['query_backward_examples'] += len(real[i][1]) if gradient else 0
        return value
    if route == 'poc':
        start = _clock(net); candidates, priorities = poc_selection(w, base+400000)
        costs['selection_seconds'] += _clock(net)-start
        losses = [float(query(i, False)) for i in candidates]
        if not np.isfinite(losses).all() or np.any(np.asarray(losses) < 0): raise RuntimeError('Invalid candidate mean CE')
        start = _clock(net); selected = poc_selection(w, base+400000, losses)
        costs['selection_seconds'] += _clock(net)-start; pi = None; factors = np.full(3, 1/3)
        row.update(candidates=candidates, candidate_losses=losses, tie_priorities=priorities.tolist())
    elif route in ('is_ht', 'delta_ht'):
        start = _clock(net); old = np.asarray(state['u'], dtype=float).copy()
        selected, pi = fixed_aocs_sample(old, 3, np.random.default_rng(base+400000))
        factors = ht_coefficients(w, selected, pi); costs['selection_seconds'] += _clock(net)-start
        row['sampling_weights'] = old.tolist()
    else:
        if state['distance_matrix'] is None:
            state['gradients'] = [query(i, True).detach().cpu() for i in range(n)]
            state['updated_round'] = [0]*n
            start = _clock(net); state['distance_matrix'] = _distances(state['gradients'])
            costs['geometry_seconds'] += _clock(net)-start
        start = _clock(net); distance = state['distance_matrix']; selected = lazy_greedy(distance)
        costs['selection_seconds'] += _clock(net)-start; pi = None; factors = w[selected]/w[selected].sum()
        row.update(cache_rounds_before=list(state['updated_round']), distance_matrix_before=distance.tolist(),
                   facility_cost_before=float(distance.min(1).sum()), facility_cost_selected=float(distance[:, selected].min(1).sum()))
    observed = []; gradients = []
    for i in selected:
        if route == 'divfl':
            state['gradients'][i] = query(i, True).detach().cpu(); state['updated_round'][i] = rnd
        start = _clock(net)
        if route in ('is_ht', 'delta_ht'):
            observation = (update_fn or trajectory_observation)(net, *real[i], base+i)
            delta, gradient = observation if isinstance(observation, tuple) else (observation, -observation/.03)
            gradients.append(gradient.detach())
        else: delta = (update_fn or c.update)(net, *real[i], base+i)
        observed.append(delta.detach()); costs['client_train_seconds'] += _clock(net)-start
        costs['client_train_calls'] += 1; costs['client_train_steps'] += 3
        costs['client_train_examples'] += sum(len(b) for b in c.batches(len(real[i][1]), base+i))
    start = _clock(net); delta = sum(float(f)*r for f, r in zip(factors, observed))
    costs['aggregation_seconds'] += _clock(net)-start
    if not torch.isfinite(delta).all(): raise RuntimeError('Nonfinite baseline update')
    if route in ('is_ht', 'delta_ht'):
        start = _clock(net); scores, gram, norms, deviation = feedback_scores(gradients, w[selected], .5 if route == 'delta_ht' else 0.)
        state['u'] = pracis_feedback(old, selected, scores)
        row.update(selected_feedback_scores=scores.tolist(), selected_gradient_gram=gram.tolist(),
                   trajectory_gradient_norms=norms.tolist(), gradient_deviation_norms=deviation.tolist(),
                   sampling_weights_after=state['u'].tolist())
        costs['feedback_seconds'] += _clock(net)-start
    if route == 'divfl':
        start = _clock(net); changed = _distances(state['gradients'], selected)
        state['distance_matrix'][selected] = changed; state['distance_matrix'][:, selected] = changed.T
        np.fill_diagonal(state['distance_matrix'], 0.)
        costs['geometry_seconds'] += _clock(net)-start
        row.update(cache_rounds_after=list(state['updated_round']), distance_matrix_after=state['distance_matrix'].tolist())
    row.update(selected=selected, pi=None if pi is None else pi.tolist(), aggregation_weights=factors.tolist(),
               aggregation_weight_sum=float(factors.sum()), update_norm=float(delta.norm()),
               selected_real_norms=[float(v.norm()) for v in observed], selected_sample_fraction=float(w[selected].sum()), costs=costs)
    return delta, row, state


def replay_rows(rows, weights, route, seed):
    """Audit sampling/feedback/cache chronology from saved scalar/geometry diagnostics."""
    w = _weights(weights); historic = np.full(len(w), 1/len(w)); timestamps = [0]*len(w); previous_distance = None
    for rnd, row in enumerate(rows, 1):
        if row['round'] != rnd or row['route'] != route: raise RuntimeError('Round/route coverage differs')
        base = seed+rnd*1000
        if route == 'poc':
            candidates, priorities = poc_selection(w, base+400000)
            if candidates != row['candidates'] or not np.array_equal(priorities, row['tie_priorities']): raise RuntimeError('PoC candidates differ')
            selected = poc_selection(w, base+400000, row['candidate_losses']); factors = np.full(3, 1/3); pi = None
        elif route in ('is_ht', 'delta_ht'):
            selected, pi = fixed_aocs_sample(historic, 3, np.random.default_rng(base+400000)); factors = ht_coefficients(w, selected, pi)
            if not np.allclose(historic, row['sampling_weights'], rtol=1e-12, atol=1e-15): raise RuntimeError('Historical sampling mass differs')
            gram = np.asarray(row['selected_gradient_gram']); norms = np.sqrt(np.maximum(gram.diagonal(), 0.))
            if gram.shape != (3, 3) or not np.isfinite(gram).all() or not np.allclose(gram, gram.T, rtol=0, atol=1e-12):
                raise RuntimeError('Invalid selected-gradient Gram')
            deviation = np.sqrt(np.maximum(gram.diagonal()-2*gram.mean(1)+gram.mean(), 0.))
            score = w[selected]*(norms if route == 'is_ht' else .5*(norms+deviation))
            historic = pracis_feedback(historic, selected, score)
            for expected, actual in ((norms, row['trajectory_gradient_norms']), (deviation, row['gradient_deviation_norms']),
                                     (score, row['selected_feedback_scores']), (historic, row['sampling_weights_after'])):
                if not np.allclose(expected, actual, rtol=1e-12, atol=1e-15): raise RuntimeError('Feedback score/state differs')
        else:
            distance = np.asarray(row['distance_matrix_before'])
            after = np.asarray(row['distance_matrix_after'])
            for matrix in (distance, after):
                if matrix.shape != (len(w), len(w)) or not np.isfinite(matrix).all() or np.any(matrix < 0) or not np.array_equal(matrix, matrix.T) or np.any(matrix.diagonal()):
                    raise RuntimeError('Invalid cached Euclidean geometry')
            if row['cache_rounds_before'] != timestamps or (previous_distance is not None and not np.array_equal(previous_distance, distance)):
                raise RuntimeError('DivFL cache used out of order')
            selected = lazy_greedy(distance); factors = w[selected]/w[selected].sum(); pi = None
            if not np.isclose(row['facility_cost_selected'], distance[:, selected].min(1).sum(), rtol=1e-12, atol=1e-15):
                raise RuntimeError('DivFL facility objective differs')
            previous_distance = np.asarray(row['distance_matrix_after'])
            unselected = np.setdiff1d(np.arange(len(w)), selected)
            if not np.array_equal(previous_distance[np.ix_(unselected, unselected)], distance[np.ix_(unselected, unselected)]):
                raise RuntimeError('Unselected DivFL geometry changed')
            for i in selected: timestamps[i] = rnd
            if row['cache_rounds_after'] != timestamps: raise RuntimeError('DivFL cache replacement differs')
        if row['selected'] != selected or not np.allclose(factors, row['aggregation_weights'], rtol=1e-12, atol=1e-15):
            raise RuntimeError('Selected clients/aggregation differs')
        if (pi is None and row['pi'] is not None) or (pi is not None and not np.allclose(pi, row['pi'], rtol=0, atol=1e-12)):
            raise RuntimeError('Inclusion probabilities differ')
        if not np.isclose(w[selected].sum(), row['selected_sample_fraction'], rtol=1e-12, atol=1e-15): raise RuntimeError('Sample fraction differs')
        if not np.isclose(factors.sum(), row['aggregation_weight_sum'], rtol=1e-12, atol=1e-15): raise RuntimeError('Aggregation coefficient sum differs')
        queries = (row['candidates'] if route == 'poc' else
                   list(range(len(w)))+selected if route == 'divfl' and rnd == 1 else
                   selected if route == 'divfl' else [])
        if row['query_client_ids'] != queries or row['costs']['query_calls'] != len(queries) or row['costs']['query_clients'] != len(set(queries)):
            raise RuntimeError('Observed full-data query coverage differs')
        if row['costs']['client_train_calls'] != 3 or row['costs']['client_train_steps'] != 9:
            raise RuntimeError('Observed local training work differs')
        if any(not np.isfinite(v) or v < 0 for v in row['costs'].values()): raise RuntimeError('Invalid cost record')
    return dict(u=historic, updated_round=timestamps)
