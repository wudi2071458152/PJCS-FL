"""FMNIST 18-method, five-seed FL; immutable inputs and unchanged numerical kernels."""
import argparse
import copy
import json
import signal
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import fmnist_common as common
import run_dirichlet_baseline191 as legacy

c, costs, multi = legacy.c, legacy.costs, legacy.multi
ALPHAS, SEEDS = common.ALPHAS, common.FL_SEEDS
ROUTES = ('fedavg', 'uniform_real', 'score_real', 'cluster_score_real_k2',
          'cluster_score_real_k3', 'cluster_score_real_k4', 'representative_real_eta1p0',
          'poc', 'is_ht', 'delta_ht', 'divfl', 'hcs_static_gram_equal', 'fedvarp',
          'residual_uniform', 'residual_score', 'weight',
          'scaffold_weighted', 'scaffold_refresh_real_weighted')
SC = legacy.SC
PROXY = ('score_real', 'cluster_score_real_k2', 'cluster_score_real_k3',
         'cluster_score_real_k4', 'representative_real_eta1p0', 'residual_uniform',
         'residual_score', 'weight', SC[1])
SUITE = 'fmnist_20261016'


def route_definition(route):
    if route not in ROUTES: raise ValueError('Unknown FMNIST route')
    if route in legacy.ROUTES: return legacy.route_definition(route)
    result = dict(renormalize=route == 'fedavg', residual_alpha=1.,
        score='1' if route == 'weight' else 'norm(P)/(3*0.01)', importance='w_i*score_i',
        sampling='fixed_aocs_sample, dependent rounding; exact marginals' if route in ('score_real', 'residual_score', 'weight') else 'uniform without replacement',
        aggregation='sum_selected(n_i/sum_selected(n_j)*R_i)' if route == 'fedavg' else
                    'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))' if route in ('residual_uniform', 'residual_score', 'weight') else
                    'sum_selected(w_i/pi_i*R_i)')
    if route.startswith('cluster_'):
        result.update(clustering=dict(multi.cluster.CLUSTERING, k=int(route[-1])),
            sampling='dependent quota rounding then within-cluster AOCS; unconditional exact marginals')
    if route.startswith('representative_'):
        result.update(target='representative', eta=1., aggregation_kind='real',
            sampling='PRJS optimal triple distribution; original AOCS marginals fixed each round',
            joint_objective='squared norm of selected HT proxy minus all-client weighted proxy',
            joint_constraints='q>=0; sum(q)=1; sum_{S containing i}(q_S)=original_pi_i',
            joint_numerical_policy=dict(bound_tolerance=multi.precision.previous.BOUND_TOL,
                retry_dual_tolerance=multi.precision.DUAL_TOL,
                certificate_tolerance=multi.precision.legacy.CERTIFICATE_TOL,
                policy='unchanged bounded cleanup and identical-LP strict retry; failure stops'))
    if route == 'weight': result.update(beta=None)
    return result


@lru_cache(maxsize=3)
def _prepared_data(inputs_root, source_root, data_root, alpha, source_sha):
    inputs, source, data = map(Path, (inputs_root, source_root, data_root))
    expected = json.loads(json.dumps(common.protocol(inputs)))
    if c.read(source/'protocol.json') != expected or c.digest(source/'protocol.json') != source_sha:
        raise RuntimeError('FMNIST synthesis source protocol changed')
    common.preflight(data, inputs)
    files, paths = {}, {}
    def fingerprint(path, role, wanted=None):
        path = Path(path).resolve(); actual = c.digest(path)
        if wanted is not None and actual != wanted: raise RuntimeError('Immutable input changed: '+role)
        files[role], paths[role] = actual, str(path)
    fingerprint(source/'protocol.json', 'source/protocol.json')
    for name in ('manifest.json', 'counts.json'): fingerprint(inputs/name, 'inputs/'+name)
    for rel, sha in expected['inputs']['splits'].items(): fingerprint(inputs/rel, 'inputs/'+rel, sha)
    for rel, sha in expected['inputs']['data'].items(): fingerprint(data/rel, 'data/'+rel, sha)
    for rel, sha in expected['code_sha256'].items(): fingerprint(c.ROOT/rel, 'code/'+rel, sha)
    with common.context(inputs):
        indices = c.split(alpha); x, y = c.dataset(data)
        table = expected['budgets'][alpha]
        if len(indices) != 32 or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(60000)):
            raise RuntimeError('FMNIST partition must cover 60000 samples exactly')
        counts = [torch.bincount(y[row], minlength=10).tolist() for row in indices]
        if counts != table['real_counts'] or sum(table['synthetic_sizes']) != 12000:
            raise RuntimeError('FMNIST real counts or 20% capacity changed')
        model_path = c.model_path(source, alpha, 2025, 0)
        marker = model_path.parent.parent/'COMPLETED.json'
        fingerprint(marker, 'source/reference/COMPLETED.json')
        fingerprint(model_path, 'source/reference/round_0000.pt', c.read(marker)['model_sha256']['0'])
        saved, regenerated = c.reference(source, alpha, 2025, 0), c.model(2025)
        if multi.state_sha256(saved) != multi.state_sha256(regenerated):
            raise RuntimeError('FMNIST saved theta0 differs from same-seed cold initialization')
        fingerprint(source/'calibration'/alpha/'coefficients.json', 'proxy/calibration/coefficients.json')
        syn = []
        for i in range(32):
            path = c.validate_snapshot(source, alpha, common.CONFIG, i, 8000)
            folder = c.route(source, alpha, common.CONFIG, i)
            fingerprint(path, f'proxy/client_{i:02d}/snapshot.pt')
            fingerprint(folder/'config.json', f'proxy/client_{i:02d}/config.json')
            fingerprint(folder/'COMPLETED_8000.json', f'proxy/client_{i:02d}/COMPLETED.json')
            item = c.load(path)
            syn.append((item['images'], item['labels'], c.class_weights(table['real_counts'][i], table['synthetic_counts'][i])))
        test = c.dataset(data, train=False)
    weights = np.asarray([len(row) for row in indices], dtype=float)/60000
    return [(x[row], y[row]) for row in indices], syn, weights, test, files, paths, expected


def release_prepared_cache():
    _prepared_data.cache_clear()


def prepare(a):
    if a.alpha not in ALPHAS or a.seed not in SEEDS or a.route not in ROUTES or not 1 <= a.rounds <= 1500:
        raise ValueError('Unsupported FMNIST alpha, seed, route or round count')
    inputs, source, data, output = (Path(getattr(a, name)).resolve() for name in ('inputs_root', 'source_root', 'data_root', 'output'))
    for path in (inputs, source, data):
        if output.is_relative_to(path) or path.is_relative_to(output): raise RuntimeError('FL output overlaps immutable inputs')
    real, all_syn, weights, test, files, paths, expected = _prepared_data(
        str(inputs), str(source), str(data), a.alpha, c.digest(source/'protocol.json'))
    files, paths = copy.deepcopy(files), copy.deepcopy(paths)
    net = common.model(a.seed); syn = all_syn if a.route in PROXY else None
    # Numerical dependencies are fingerprinted without loading any CIFAR source/data.
    modules = (common, c, legacy, legacy.algorithms, costs, legacy.cv, legacy.hcs,
        legacy.hcs.sampling, legacy.hcs.cluster_sampling, legacy.refresh, legacy.refresh.original,
        legacy.importance_helpers, multi, legacy.a007, multi.precision.legacy,
        sys.modules[multi.cluster.full_direction_features.__module__],
        sys.modules[multi.cluster.stratified_sample.__module__], *multi.MODULES)
    for module in modules:
        path = Path(module.__file__).resolve(); role = 'code/'+path.relative_to(c.ROOT).as_posix()
        files[role], paths[role] = c.digest(path), str(path)
    path = Path(__file__).resolve(); role = 'code/'+path.relative_to(c.ROOT).as_posix()
    files[role], paths[role] = c.digest(path), str(path)
    protocol = dict(version=1, suite=SUITE, dataset='FashionMNIST', alpha=a.alpha, route=a.route,
        method=a.route, seed=a.seed, rounds=a.rounds, clients=32, selected=3, local_steps=3,
        batch=256, real_lr=.01, proxy_lr=.01 if syn is not None else None,
        real_ce='ordinary', proxy_ce=expected['weighted_ce'] if syn is not None else None,
        proxy_loss='original' if syn is not None else None, iteration=8000 if syn is not None else 0,
        input_shape=[1, 28, 28], partition_seed=35, budget_ratio=.2, budget_total=12000,
        train_samples=60000, augmentation=False, weights=weights.tolist(),
        real_sizes=[len(y) for _, y in real], synthetic_sizes=[len(y) for _, y, _ in all_syn],
        synthetic_updates_per_round=32 if syn is not None else 0,
        model_parameters=sum(p.numel() for p in net.parameters()),
        initial_model_state_sha256=legacy.cv.tensor_fingerprint(net.state_dict()),
        fl_initialization=dict(seed=a.seed, state_sha256=multi.state_sha256(net),
            initializer='fmnist_common.model(seed): ConvNet(1,10,128,3,ReLU,InstanceNorm,AvgPool,28x28)',
            shared_across_methods=True),
        round_seeds=dict(base=f'{a.seed}+round*1000', real='+client', uniform='+17', score='+400000',
            proxy='+10000000+client', joint_branch='+600000', joint_subset='+700000'),
        files_sha256=files, files_paths=paths, input_roots=dict(inputs_root=str(inputs), source_root=str(source)),
        source_protocol_sha256=files['source/protocol.json'], communication=costs.policy(net, syn),
        cost_policy=dict(client='real SGD and full-data queries', server='proxy, selection, geometry and aggregate',
            combined='unchanged legacy proxy kernels: server selection/geometry/aggregation/diagnostics combined; unavailable individual splits null',
            missing='null; genuine non-occurrence zero', evaluation_and_checkpoint_io='separate from algorithm timers'),
        software=dict(torch=torch.__version__, numpy=np.__version__))
    protocol.update(route_definition(a.route))
    return net, real, syn, weights, test, protocol


def initial_state(net, route, weights):
    if route in legacy.ROUTES: return legacy.initial_state(net, route, weights)
    return multi.weight.initial_history(32) if route == 'weight' else None


def one_round(net, real, syn, weights, route, seed, rnd, state, update_fn=None, scaffold_fn=None, proxy_fn=None, query_fn=None):
    if route in legacy.ROUTES:
        return legacy.one_round(net, real, syn, weights, route, seed, rnd, state, update_fn, scaffold_fn, proxy_fn, query_fn)
    if route not in ROUTES: raise ValueError('Unknown FMNIST route')
    measured = dict(client_train_calls=0, client_train_steps=0, client_train_examples=0, client_train_seconds=0.,
        server_proxy_calls=0, server_proxy_steps=0, server_proxy_examples=0, server_proxy_seconds=0.,
        query_calls=0, query_clients=0, query_examples=0, query_forward_examples=0, query_backward_examples=0, query_seconds=0.)
    def update(net, x, y, update_seed, *args):
        value, duration = costs.timed(update_fn or c.update, net, x, y, update_seed, *args)
        key = 'server_proxy' if args else 'client_train'
        measured[key+'_calls'] += 1; measured[key+'_steps'] += 3
        measured[key+'_examples'] += sum(len(batch) for batch in c.batches(len(y), update_seed))
        measured[key+'_seconds'] += duration
        return value
    start = time.perf_counter()
    delta, row = multi.one_round(net, real, syn, weights, route, seed, rnd, history=state, update_fn=update)
    costs.synchronize(); duration = time.perf_counter()-start
    measured.update(selection_seconds=None, aggregation_seconds=None, geometry_seconds=None,
        solver_seconds=row.get('joint', {}).get('solve_seconds', 0.))
    measured['server_total_seconds'] = max(0., duration-measured['client_train_seconds'])
    measured['server_selection_geometry_aggregation_diagnostics_seconds'] = max(0., measured['server_total_seconds']-measured['server_proxy_seconds'])
    row['costs'] = measured
    row['communication'] = costs.communication(route, row, sum(p.numel() for p in net.parameters()))
    return delta, row, state


def verify_records(protocol, rows):
    route = protocol['route']
    if route in legacy.ROUTES: return legacy.verify_records(protocol, rows)
    if [row['round'] for row in rows] != list(range(1, len(rows)+1)): raise RuntimeError('Round coverage mismatch')
    multi.replay(protocol, rows)
    for row in rows:
        costs.validate(row, route, protocol['model_parameters'])
        if any(not np.isfinite(row[k]) for k in ('test_accuracy', 'test_loss', 'update_norm', 'round_seconds', 'elapsed_seconds')):
            raise RuntimeError('Nonfinite training record')
        expected_proxy = 32 if route in PROXY else 0
        if row['costs']['client_train_calls'] != 3 or row['costs']['server_proxy_calls'] != expected_proxy:
            raise RuntimeError('Observed client/proxy calls differ from route')


def validate_state(net, state, protocol, rows):
    if protocol['route'] in legacy.ROUTES: return legacy.validate_state(net, state, protocol, rows)
    if protocol['route'] == 'weight':
        with multi.seed_context(protocol['seed']): rebuilt = multi.weight.replay(protocol, rows)
        if state.keys() != rebuilt.keys() or any(not np.array_equal(value, rebuilt[key], equal_nan=True) for key, value in state.items()):
            raise RuntimeError('Weight observation history changed')
    elif state is not None: raise RuntimeError('Unexpected state for memoryless route')


def verify_protocol(folder):
    protocol = c.read(Path(folder)/'protocol.json')
    if protocol.get('suite') != SUITE or protocol.get('dataset') != 'FashionMNIST' or protocol['route'] not in ROUTES or protocol['alpha'] not in ALPHAS or protocol['seed'] not in SEEDS:
        raise RuntimeError('Unexpected FMNIST result identity')
    frozen = dict(clients=32, selected=3, local_steps=3, batch=256, real_lr=.01, real_ce='ordinary',
        augmentation=False, input_shape=[1, 28, 28], partition_seed=35, budget_total=12000, train_samples=60000)
    frozen.update(route_definition(protocol['route']))
    if any(protocol.get(key) != value for key, value in frozen.items()): raise RuntimeError('Frozen FMNIST numerical definition changed')
    proxy = protocol['route'] in PROXY
    if (protocol['iteration'], protocol['proxy_lr'], protocol['proxy_loss']) != (8000 if proxy else 0, .01 if proxy else None, 'original' if proxy else None):
        raise RuntimeError('Frozen FMNIST image/lr/loss changed')
    for role, wanted in protocol['files_sha256'].items():
        path = Path(protocol['files_paths'][role])
        if not path.is_file() or c.digest(path) != wanted: raise RuntimeError('Bound input/code changed: '+role)
    return protocol


def verify(output, expected_rounds=1500):
    kernel = legacy.clone(legacy.verify, verify_protocol=verify_protocol,
        verify_records=verify_records, validate_state=validate_state)
    return kernel(output, expected_rounds)


def train(a, prepared):
    kernel = legacy.clone(legacy.train, initial_state=initial_state, one_round=one_round,
        verify_records=verify_records, validate_state=validate_state, verify=verify)
    return kernel(a, prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs-root', 'source-root', 'data-root', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--alpha', choices=ALPHAS, required=True)
    parser.add_argument('--seed', choices=SEEDS, type=int, required=True)
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--rounds', type=int, default=1500); parser.add_argument('--deadline', type=float)
    parser.add_argument('--run', action='store_true'); a = parser.parse_args()
    if a.rounds != 1500: parser.error('Formal FMNIST FL requires 1500 rounds')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
