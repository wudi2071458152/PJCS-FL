"""Alpha=.07 fixed-image FL: 11 methods, five seeds, 90 independent routes."""
import argparse
import copy
import json
import signal
from contextlib import nullcontext
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import torch
import run_dirichlet_multiseed as multi
import run_dirichlet_control_variates as cv
import run_dirichlet_proxy_scaffold as proxy
import run_dirichlet_scaffold_control_refresh as refresh
import dirichlet_a007_common as common

c = cv.c
SEEDS = common.FL_SEEDS
ROUTES = ('fedavg', 'uniform_real', 'score_real', 'cluster_score_real_k3',
          'representative_real_eta1p0', 'residual_uniform', 'residual_score', 'weight',
          'history_uniform_zero', 'scaffold_weighted', 'scaffold_refresh_real_weighted')
SHARED = ('fedavg', 'uniform_real', 'history_uniform_zero', 'scaffold_weighted')
SC = ('scaffold_weighted', 'scaffold_refresh_real_weighted')


def _clone(function, **overrides):
    return FunctionType(function.__code__, dict(function.__globals__, **overrides),
                        function.__name__, function.__defaults__, function.__closure__)


def route_jobs():
    return [dict(seed=seed, route=route, iteration=iteration)
            for seed in SEEDS for route in ROUTES
            for iteration in ((0,) if route in SHARED else (8000, 32000))]


def _scoped(seed):
    """Bind seeds in separate globals; never change the old modules' SEED."""
    local_base = SimpleNamespace(**dict(vars(multi.base), SEED=seed))
    selection = _clone(multi.weight.selection, base=local_base)
    weight_replay = _clone(multi.weight.replay, selection=selection)
    weighted = SimpleNamespace(**dict(vars(multi.weight), selection=selection,
        replay=weight_replay, one_round=_clone(multi.weight.one_round, base=local_base, selection=selection)))
    joint = SimpleNamespace(**dict(vars(multi.joint),
        one_round=_clone(multi.joint.one_round, base=local_base),
        replay=_clone(multi.joint.replay, base=local_base)))
    cluster = SimpleNamespace(**dict(vars(multi.cluster),
        one_round=_clone(multi.cluster.one_round, base=local_base)))
    replay = _clone(multi.replay, base=local_base, weight=weighted, joint=joint,
                    seed_context=lambda _: nullcontext())
    cv_replay = _clone(cv.verify_records, SEED=seed)
    local_cv = SimpleNamespace(**dict(vars(cv), verify_records=cv_replay))
    proxy_replay = _clone(proxy.verify_records, cv=local_cv)
    local_proxy = SimpleNamespace(**dict(vars(proxy), verify_records=proxy_replay))
    return SimpleNamespace(weight=weighted, joint=joint, cluster=cluster, replay=replay,
        sc_round=_clone(cv.one_round, SEED=seed), sc_replay=cv_replay,
        refresh_round=_clone(refresh.one_round, SEED=seed),
        refresh_replay=_clone(refresh.verify_records, original=local_proxy))


def route_protocol(template, route, seed, iteration):
    if route not in ROUTES or seed not in SEEDS or iteration not in ((0,) if route in SHARED else (8000, 32000)):
        raise ValueError('Unsupported alpha=.07 route, seed or image iteration')
    frozen = dict(alpha=common.ALPHA, clients=32, selected=3, local_steps=3, batch=256,
                  real_lr=.01, real_ce='ordinary', augmentation=False, budget_ratio=.2, budget_total=10000)
    if any(template.get(k) != v for k, v in frozen.items()):
        raise RuntimeError('Frozen alpha=.07 FL settings changed')
    if template.get('suite') == 'dirichlet_a007_20261014' and (template.get('route'), template.get('seed'), template.get('iteration')) != (route, seed, iteration):
        raise RuntimeError('Prepared route, seed or image iteration changed')
    if not 1 <= template.get('rounds', 0) <= 1500: raise ValueError('Invalid round target')
    if route not in SHARED and (template.get('proxy_lr') != .01 or template.get('proxy_loss') != 'original'):
        raise RuntimeError('Frozen original proxy settings changed')
    original = copy.deepcopy(template)
    original.update(seed=2025, iteration=32000, proxy_lr=.01, proxy_loss='original',
                    proxy_ce=c.read(multi.base.EXPECTED)['weighted_ce'])
    if route in SC:
        original['seed'] = seed
        alpha_common = SimpleNamespace(**dict(vars(c), ALPHAS=(common.ALPHA,)))
        sc_protocol = _clone(cv.route_protocol, c=alpha_common, SEED=seed)
        if route == SC[0]: protocol = sc_protocol(original, route)
        else:
            proxy_protocol = _clone(proxy.route_protocol, cv=SimpleNamespace(route_protocol=sc_protocol))
            protocol = _clone(refresh.route_protocol,
                original=SimpleNamespace(route_protocol=proxy_protocol))(original, route)
        protocol['fl_initialization'] = dict(seed=seed, initializer='dirichlet_screen_common.model(seed)',
            state_sha256=multi.initialization_sha256(seed), source_reference_seed=2025,
            source_reference_role='frozen synthesis lineage; independent cold FL initialization',
            source2025_initializer_parity='exact state_dict equality checked during prepare')
    else:
        protocol = multi.route_protocol(original, route, seed)
    protocol.update(suite='dirichlet_a007_20261014', route=route, seed=seed, iteration=iteration,
        proxy_loss=None if route in SHARED else 'original',
        synthetic_updates_per_round=0 if route in SHARED else 32,
        image_comparison='cold initialization per image iteration; no FL model transferred between 8k and 32k')
    protocol['round_seeds']['base'] = f'{seed}+round*1000'
    if 'history_initial_proxy' in protocol['round_seeds']:
        protocol['round_seeds']['history_initial_proxy'] = f'{seed}+10000000+client at theta0'
    for module in (common, cv, proxy, refresh, multi, *multi.MODULES):
        path = Path(module.__file__).resolve()
        protocol['files_sha256']['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    path = multi.base.EXPECTED.resolve()
    protocol['files_sha256']['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    path = Path(__file__).resolve()
    protocol['files_sha256']['code/' + path.relative_to(c.ROOT).as_posix()] = c.digest(path)
    return protocol


def prepare(a):
    source, inputs = Path(a.source_root).resolve(), Path(a.inputs_root).resolve()
    output = Path(a.output).resolve()
    for path in (source, inputs, Path(a.data_root).resolve()):
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise RuntimeError('FL output must be disjoint from source, inputs and CIFAR data')
    if (a.route, a.seed, a.iteration) not in [(j['route'], j['seed'], j['iteration']) for j in route_jobs()]:
        raise ValueError('Unsupported alpha=.07 FL route')
    expected = common.protocol(inputs)
    if c.read(source/'protocol.json') != expected: raise RuntimeError('Alpha=.07 synthesis source protocol changed')
    files, paths = {}, {}
    def fingerprint(path, role):
        path = Path(path).resolve(); files[role] = c.digest(path); paths[role] = str(path)
        return files[role]
    fingerprint(source/'protocol.json', 'source/protocol.json')
    for name in ('manifest.json', 'counts.json'): fingerprint(inputs/name, 'inputs/'+name)
    for rel, wanted in expected['inputs']['splits'].items():
        if fingerprint(inputs/rel, 'inputs/'+rel) != wanted: raise RuntimeError('Split checksum changed: '+rel)
    for rel, wanted in expected['inputs']['data'].items():
        if fingerprint(Path(a.data_root)/rel, 'data/'+rel) != wanted: raise RuntimeError('CIFAR checksum changed: '+rel)
    for rel, wanted in expected['code_sha256'].items():
        if fingerprint(c.ROOT/rel, 'code/'+rel) != wanted: raise RuntimeError('Synthesis numerical code changed: '+rel)
    with common.context(inputs):
        indices = c.split(common.ALPHA); x, y = c.dataset(a.data_root)
        table = expected['budgets'][common.ALPHA]
        if len(indices) != 32 or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(50000)):
            raise RuntimeError('Invalid alpha=.07 partition')
        counts = [torch.bincount(y[row], minlength=10).tolist() for row in indices]
        if counts != table['real_counts'] or sum(table['synthetic_sizes']) != 10000:
            raise RuntimeError('Real counts or fixed 20% budget changed')
        model_file = c.model_path(source, common.ALPHA, 2025, 0)
        sha = fingerprint(model_file, 'source/reference/round_0000.pt')
        marker = model_file.parent.parent/'COMPLETED.json'
        fingerprint(marker, 'source/reference/COMPLETED.json')
        if c.read(marker)['model_sha256']['0'] != sha: raise RuntimeError('Reference initial model changed')
        first, second = c.reference(source, common.ALPHA, 2025, 0), c.model(2025)
        if first.state_dict().keys() != second.state_dict().keys() or any(not torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items()):
            raise RuntimeError('Source seed2025 initializer parity failed')
        syn = None
        if a.route not in SHARED:
            syn = []
            coefficient = source/'calibration'/common.ALPHA/'coefficients.json'
            fingerprint(coefficient, 'proxy/calibration/coefficients.json')
            for i in range(32):
                path = c.validate_snapshot(source, common.ALPHA, common.CONFIG, i, a.iteration)
                fingerprint(path, f'proxy/client_{i:02d}/snapshot.pt')
                folder = c.route(source, common.ALPHA, common.CONFIG, i)
                fingerprint(folder/'config.json', f'proxy/client_{i:02d}/config.json')
                completed = folder/f'COMPLETED_{a.iteration}.json'
                if not completed.is_file(): raise RuntimeError('Missing image completion marker: '+str(completed))
                fingerprint(completed, f'proxy/client_{i:02d}/COMPLETED.json')
                item = c.load(path)
                syn.append((item['images'], item['labels'], c.class_weights(table['real_counts'][i], table['synthetic_counts'][i])))
        net, test = c.model(a.seed), c.dataset(a.data_root, train=False)
    weights = np.asarray([len(row) for row in indices], dtype=float)/50000
    template = dict(alpha=common.ALPHA, budget_ratio=.2, budget_total=10000, clients=32, selected=3,
        local_steps=3, batch=256, real_lr=.01, proxy_lr=.01, real_ce='ordinary',
        proxy_ce=expected['weighted_ce'], augmentation=False, seed=a.seed, rounds=a.rounds,
        iteration=a.iteration, proxy_loss='original', weights=weights.tolist(),
        files_sha256=files, files_paths=paths, inputs_root=str(inputs), source_root=str(source),
        model_parameters=sum(p.numel() for p in net.parameters()),
        initial_model_state_sha256=cv.tensor_fingerprint(net.state_dict()))
    protocol = route_protocol(template, a.route, a.seed, a.iteration)
    if multi.state_sha256(net) != protocol['fl_initialization']['state_sha256']:
        raise RuntimeError('Generated initialization fingerprint differs')
    return net, [(x[row], y[row]) for row in indices], syn, weights, test, protocol


def initial_state(net, route):
    if route in SC: return cv.initial_state(net, 'scaffold_weighted')
    if route == 'weight': return multi.weight.initial_history(32)
    if route == 'history_uniform_zero': return multi.cluster.initial_history(net, [None]*32, None, route)
    return None


def one_round(net, real, syn, weights, route, seed, rnd, history=None, last_seen=None,
              update_fn=None, scaffold_fn=None, proxy_fn=None):
    if route not in ROUTES or seed not in SEEDS: raise ValueError('Unknown route/seed')
    scoped = _scoped(seed)
    if route in multi.base.METHODS:
        return multi.base.one_round(net, real, syn, weights, route, seed, rnd, update_fn)
    if route == 'weight': return scoped.weight.one_round(net, real, syn, weights, route, None, rnd, history, update_fn)
    if route == 'history_uniform_zero': return scoped.cluster.one_round(net, real, syn, weights, route, 0, rnd, history, last_seen, update_fn)
    if route == 'cluster_score_real_k3': return scoped.cluster.one_round(net, real, syn, weights, 'cluster_score_real', 3, rnd, update_fn=update_fn)
    if route == 'representative_real_eta1p0': return scoped.joint.one_round(net, real, syn, weights, 'representative', 'real', 1., rnd, update_fn)
    if route == SC[0]: return scoped.sc_round(net, real, syn, weights, route, rnd, history, scaffold_fn=scaffold_fn)
    return scoped.refresh_round(net, real, syn, weights, route, rnd, history, scaffold_fn=scaffold_fn, proxy_fn=proxy_fn)


def replay(protocol, rows):
    scoped = _scoped(protocol['seed'])
    if protocol['route'] in SC:
        (scoped.sc_replay if protocol['route'] == SC[0] else scoped.refresh_replay)(protocol, rows)
        return None, None
    return scoped.replay(protocol, rows)


def verify_protocol(folder):
    protocol = c.read(Path(folder)/'protocol.json')
    if protocol != route_protocol(protocol, protocol['route'], protocol['seed'], protocol['iteration']):
        raise RuntimeError('Alpha=.07 protocol or numerical code changed')
    for role, wanted in protocol['files_sha256'].items():
        if role.startswith('code/'): path = c.ROOT/role[5:]
        elif role in protocol.get('files_paths', {}): path = Path(protocol['files_paths'][role])
        else: continue
        if not path.is_file() or c.digest(path) != wanted: raise RuntimeError('Input/code fingerprint changed: '+role)
    return protocol


def verify(output, expected_rounds=1500):
    folder = Path(output); marker = multi.base.verify(folder, expected_rounds)
    protocol = verify_protocol(folder); rows = c.read(folder/'rounds.json')
    rebuilt, seen = replay(protocol, rows)
    if (folder/'resume.pt').exists():
        integrity = c.read(folder/'checkpoint_integrity.json')
        if integrity.get('checkpoint_sha256') != c.digest(folder/'resume.pt') or integrity.get('protocol_sha256') != marker['protocol_sha256']:
            raise RuntimeError('Checkpoint integrity mismatch')
        saved = c.load(folder/'resume.pt')
        if saved['round'] != expected_rounds or saved['records'] != rows or saved['protocol_sha256'] != marker['protocol_sha256']:
            raise RuntimeError('Checkpoint round/record/protocol mismatch')
        if any(not torch.isfinite(v).all() for v in saved['model'].values()): raise RuntimeError('Nonfinite checkpoint model')
        if protocol['route'] in SC:
            cv._validate_state(None, saved['state'], protocol['route'], rows, protocol['weights'], protocol['model_parameters'])
            if protocol['route'] == SC[1] and rows[-1]['control_sha256_after'] != cv.tensor_fingerprint({'c': saved['state']['control']}):
                raise RuntimeError('Refreshed global control fingerprint changed')
        elif protocol['route'] == 'weight':
            if any(not np.array_equal(v, saved['history'][k], equal_nan=True) for k, v in rebuilt.items()):
                raise RuntimeError('Weight observation cache changed')
        elif protocol['route'] == 'history_uniform_zero':
            cache = saved['history']
            if len(cache) != 32 or any(h.shape != (protocol['model_parameters'],) or not torch.isfinite(h).all() for h in cache) or not np.array_equal(seen, saved['last_seen']):
                raise RuntimeError('Historical cache/coverage changed')
            if saved['history_state_sha256'] != multi.state_fingerprint({str(i): h for i, h in enumerate(cache)}):
                raise RuntimeError('Historical vector fingerprint changed')
    return marker


def train(a, prepared):
    protocol = prepared[-1]
    if (a.route, a.seed, a.iteration, a.rounds) != (protocol['route'], protocol['seed'], protocol['iteration'], protocol['rounds']):
        raise RuntimeError('Training arguments differ from prepared route')
    if multi.state_sha256(prepared[0]) != protocol['fl_initialization']['state_sha256']:
        raise RuntimeError('Cold initial model changed')
    # Add an integrity record to the unchanged multi-seed persistence loop.
    checkpoint = Path(a.output)/'resume.pt'
    def save(path, value):
        c.save(path, value)
        if Path(path) == checkpoint:
            c.atomic_json(checkpoint.parent/'checkpoint_integrity.json', dict(
                checkpoint_sha256=c.digest(checkpoint), protocol_sha256=value['protocol_sha256'], round=value['round']))
    def load(path, *args, **kwargs):
        if Path(path) == checkpoint:
            integrity = c.read(checkpoint.parent/'checkpoint_integrity.json')
            if integrity['checkpoint_sha256'] != c.digest(checkpoint): raise RuntimeError('Checkpoint integrity mismatch')
        return c.load(path, *args, **kwargs)
    local_c = SimpleNamespace(**dict(vars(c), save=save, load=load))
    if a.route in SC:
        scoped = _scoped(a.seed)
        def sc_round(net, real, syn, weights, method, rnd, state):
            return one_round(net, real, syn, weights, method, a.seed, rnd, state)
        kernel = _clone(cv.train, c=local_c, SEED=a.seed, one_round=sc_round,
            initial_state=initial_state, verify_records=lambda p, r: replay(p, r), verify=verify)
        mapped = copy.copy(a); mapped.method = a.route
    else:
        kernel = _clone(multi.train, c=local_c, one_round=one_round, replay=replay, verify=verify)
        mapped = a
    return kernel(mapped, prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs-root', 'source-root', 'data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--iteration', type=int, choices=(0, 8000, 32000), required=True)
    parser.add_argument('--rounds', type=int, default=1500)
    parser.add_argument('--deadline', type=float); parser.add_argument('--run', action='store_true')
    a = parser.parse_args()
    if a.rounds != 1500: parser.error('Formal FL requires 1500 rounds')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
