"""Independent HCS cache/compression pilot; preparing does not train models."""
import argparse
import copy
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
import sklearn
import run_dirichlet_proxy_fl_fingerprintfix as fix
import cluster_sampling
import hcs_pilot_sampling as sampling

c, base = fix.p.c, fix.p
REFRESH = ('static', 'cache', 'full')
REPRESENTATIONS = ('gram', 'countsketch')
AGGREGATIONS = ('equal', 'datasize')
SEED = 2025


def tensor_fingerprint(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cache_fingerprint(cache):
    return None if cache is None else tensor_fingerprint({str(i): value for i, value in enumerate(cache)})


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def route_protocol(original, refresh, representation, aggregation):
    if refresh not in REFRESH or representation not in REPRESENTATIONS or aggregation not in AGGREGATIONS:
        raise ValueError('Invalid HCS pilot route')
    if any(original.get(key) != value for key, value in dict(alpha='0p1', seed=2025, clients=32,
            selected=3, local_steps=3, batch=256, real_lr=.01, real_ce='ordinary', augmentation=False).items()):
        raise RuntimeError('Frozen real FL settings changed')
    protocol = copy.deepcopy(original)
    files = protocol['files_sha256']
    legacy = 'code/' + base.EXPECTED.name
    if legacy in files:
        canonical = 'code/' + base.EXPECTED.resolve().relative_to(c.ROOT).as_posix()
        if canonical in files and files[canonical] != files[legacy]:
            raise RuntimeError('Conflicting source protocol fingerprints')
        files[canonical] = files.pop(legacy)
    for key in ('sampler_ast_sha256', 'sampler_copied_from', 'proxy_ce', 'proxy_lr', 'proxy_loss', 'residual_alpha'):
        protocol.pop(key, None)
    protocol.update(version=1, method='hcs_pilot', refresh=refresh, representation=representation,
        aggregation_mode=aggregation, iteration=0, proxy_required=False, server_momentum=0.,
        vector='three-step real R_i=theta_local-theta_global; no unit normalization',
        sampling='joint expected-quota rounding; within-cluster norm weighted sequential without replacement',
        score='norm(represented R_i); no sample weight in selection', importance='cluster norm score',
        aggregation='sum_selected(R_i)/len(selected)' if aggregation == 'equal' else
                    'sum_selected(n_i*R_i)/sum_selected(n_i)', renormalize=True,
        warmup=dict(round=1, clients=32, counted_in_target=True, aggregate=aggregation),
        clustering=dict(clusters=3, seed=502025, norm='Euclidean; unnormalized',
            initialization='fixed client indices; Lloyd KMeans',
            dispersion='sum ordered pairwise squared distances/(N_h-1+1e-6)',
            quota='m*N_h*dispersion_h/sum(N_h*dispersion_h)',
            capacities='cap then proportional redistribution; all-zero fallback proportional to client count',
            rounding='dependent rounding preserves expected quota and fixed total=3',
            empty_cluster='allow fewer effective clusters', singletons='zero dispersion',
            zero_scores='uniform within cluster; positive scores exhausted before zero scores'),
        compression=dict(kind=representation, dimension=32 if representation == 'gram' else 256,
            seed=None if representation == 'gram' else 602025,
            exact_geometry=representation == 'gram', communication_reduction_claim=False,
            shared_fixed_sketch=representation == 'countsketch'),
        cache_policy='initial descriptors frozen' if refresh == 'static' else
            'selected-only replacement after computing aggregate; re-cluster next round' if refresh == 'cache' else
            'all-current descriptions each round; reuse selected real updates',
        round_seeds=dict(base='2025+round*1000', real='+client', sample='+400000'),
        initial_model_state_sha256=original.get('initial_model_state_sha256'))
    protocol.setdefault('software', {})['sklearn'] = sklearn.__version__
    for module in (sampling, cluster_sampling, fix, base, c):
        file = Path(module.__file__).resolve()
        protocol['files_sha256']['code/' + file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    file = Path(__file__).resolve()
    protocol['files_sha256']['code/' + file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    return protocol


def prepare(a):
    if getattr(a, 'alpha', '0p1') != '0p1' or a.rounds < 1:
        raise ValueError('HCS pilot requires alpha=0.1 and a positive round target')
    mapped = copy.copy(a); mapped.alpha = '0p1'; mapped.method = 'fedavg'
    mapped.iteration = 0; mapped.proxy_root = None
    prepared = fix.prepare(mapped)
    prepared[-1]['initial_model_state_sha256'] = tensor_fingerprint(prepared[0].state_dict())
    return (*prepared[:-1], route_protocol(prepared[-1], a.refresh, a.representation, a.aggregation))


def one_round(net, real, weights, refresh, representation, aggregation, rnd,
              cache=None, last_seen=None, static_geometry=None, sketch=None, update_fn=None):
    """Only full refresh computes unselected current updates after the warm-up."""
    update_fn = update_fn or c.update
    if len(real) != 32 or rnd < 1: raise ValueError('Invalid HCS pilot clients/round')
    before = cache_fingerprint(cache)
    old_seen = np.zeros(32, dtype=int) if last_seen is None else np.asarray(last_seen).copy()
    observed = {}; base_seed = SEED + rnd * 1000; net.train()
    update_tick = time.monotonic()
    if rnd == 1 or refresh == 'full':
        observed = {i: update_fn(net, *real[i], base_seed + i) for i in range(32)}
        cache = [observed[i].detach().cpu().clone() for i in range(32)]
        description_seen = np.full(32, rnd, dtype=int)
    else:
        if cache is None or len(cache) != 32 or last_seen is None: raise RuntimeError('Missing descriptor cache')
        description_seen = old_seen.copy()
    update_seconds = time.monotonic() - update_tick
    geometry_tick = time.monotonic()
    if refresh == 'static' and static_geometry is not None:
        geometry = copy.deepcopy(static_geometry)
    else:
        geometry = sampling.geometry(cache, representation, sketch=sketch, seed=502025)
    if refresh == 'static' and static_geometry is None: static_geometry = copy.deepcopy(geometry)
    geometry_seconds = time.monotonic() - geometry_tick
    selection_tick = time.monotonic()
    if rnd == 1:
        selected = list(range(32)); details = dict(actual_quotas=None, zero_score_fallback=False)
    else:
        selected, details = sampling.sample(geometry, base_seed + 400000)
    selection_seconds = time.monotonic() - selection_tick
    update_tick = time.monotonic()
    if not observed:
        observed = {i: update_fn(net, *real[i], base_seed + i) for i in selected}
    update_seconds += time.monotonic() - update_tick
    factors = sampling.aggregation_factors(weights, selected, aggregation)
    aggregate_tick = time.monotonic()
    delta = sum(float(factor) * observed[i] for i, factor in zip(selected, factors))
    if not torch.isfinite(delta).all(): raise RuntimeError('Nonfinite HCS aggregate')
    aggregate_seconds = time.monotonic() - aggregate_tick
    if refresh == 'cache' and rnd > 1:
        for i in selected: cache[i] = observed[i].detach().cpu().clone()
        last_seen = old_seen.copy(); last_seen[selected] = rnd
    else:
        last_seen = description_seen.copy()
    record = dict(round=rnd, selected=selected, warmup=rnd == 1,
        aggregation_weights=np.asarray(factors).tolist(), aggregation_weight_sum=float(np.sum(factors)),
        selected_sample_fraction=float(np.asarray(weights)[selected].sum()), geometry=geometry,
        actual_quotas=details['actual_quotas'], zero_score_fallback=details['zero_score_fallback'],
        cache_last_seen_before=old_seen.tolist(), description_last_seen_before=description_seen.tolist(),
        cache_last_seen_after=last_seen.tolist(), cache_sha256_before=before, cache_sha256_after=cache_fingerprint(cache),
        geometry_sha256=json_fingerprint(geometry), real_update_count=len(observed), real_step_count=3 * len(observed),
        selected_real_norms=[float(observed[i].double().norm()) for i in selected], update_norm=float(delta.double().norm()),
        real_update_seconds=update_seconds, geometry_seconds=geometry_seconds,
        selection_seconds=selection_seconds, aggregation_seconds=aggregate_seconds)
    return delta, record, cache, last_seen, static_geometry


def replay(protocol, rows):
    """Audit selections, cached-information timing and actual compute counts."""
    weights = np.asarray(protocol['weights']); seen = np.zeros(32, dtype=int)
    previous = None; static_sha = None
    for rnd, row in enumerate(rows, 1):
        expected_count = 32 if rnd == 1 or protocol['refresh'] == 'full' else 3
        if row['round'] != rnd or row['warmup'] != (rnd == 1) or row['real_update_count'] != expected_count or row['real_step_count'] != 3 * expected_count:
            raise RuntimeError('HCS round or compute-count mismatch')
        geometry = row['geometry']; selected = row['selected']
        if json_fingerprint(geometry) != row['geometry_sha256']: raise RuntimeError('HCS geometry changed')
        if hasattr(sampling, 'validate_geometry'): sampling.validate_geometry(geometry)
        if rnd == 1:
            expected = list(range(32)); quotas = None; zero = False
        else:
            expected, detail = sampling.sample(geometry, SEED + rnd * 1000 + 400000)
            quotas, zero = detail['actual_quotas'], detail['zero_score_fallback']
        if selected != expected or row['actual_quotas'] != quotas or row['zero_score_fallback'] != zero:
            raise RuntimeError('HCS selection/quota replay differs')
        factors = sampling.aggregation_factors(weights, selected, protocol['aggregation_mode'])
        if not np.allclose(factors, row['aggregation_weights'], rtol=1e-12, atol=1e-15) or not np.isclose(row['aggregation_weight_sum'], 1.):
            raise RuntimeError('HCS aggregation weights changed')
        if not np.isclose(row['selected_sample_fraction'], weights[selected].sum(), rtol=1e-12, atol=1e-15):
            raise RuntimeError('HCS selected sample mass changed')
        current = np.full(32, rnd, dtype=int) if rnd == 1 or protocol['refresh'] == 'full' else seen.copy()
        if row['cache_last_seen_before'] != seen.tolist() or row['description_last_seen_before'] != current.tolist() or row['cache_sha256_before'] != previous:
            raise RuntimeError('HCS cache timing changed')
        if protocol['refresh'] == 'cache' and rnd > 1: current[selected] = rnd
        if row['cache_last_seen_after'] != current.tolist(): raise RuntimeError('HCS cache replacement changed')
        if protocol['refresh'] == 'static':
            if static_sha is not None and (row['geometry_sha256'] != static_sha or row['cache_sha256_after'] != previous):
                raise RuntimeError('Frozen HCS descriptors changed')
            static_sha = row['geometry_sha256']
        seen, previous = current, row['cache_sha256_after']
        scalar_keys = ('update_norm', 'aggregation_weight_sum', 'selected_sample_fraction', 'real_update_seconds',
                       'geometry_seconds', 'selection_seconds', 'aggregation_seconds')
        if not np.isfinite([row[key] for key in scalar_keys]).all() or any(row[key] < 0 for key in scalar_keys):
            raise RuntimeError('Nonfinite HCS metrics')
        norms = np.asarray(row['selected_real_norms'])
        if norms.shape != (len(selected),) or not np.isfinite(norms).all() or np.any(norms < 0):
            raise RuntimeError('Invalid HCS real update norms')
    return seen, previous, static_sha


def summarize(rows):
    summary = base.summarize(rows)
    summary['total_real_updates'] = sum(row['real_update_count'] for row in rows)
    summary['total_real_steps'] = sum(row['real_step_count'] for row in rows)
    return summary


def verify(output, expected_rounds=1500):
    folder = Path(output); protocol = c.read(folder / 'protocol.json')
    marker = c.read(folder / 'COMPLETED.json'); rows = c.read(folder / 'rounds.json')
    if protocol != route_protocol(protocol, protocol['refresh'], protocol['representation'], protocol['aggregation_mode']):
        raise RuntimeError('HCS protocol or code changed')
    for name, sha in protocol['files_sha256'].items():
        if name.startswith('code/') and c.digest(c.ROOT / name[5:]) != sha:
            raise RuntimeError('Numerical code fingerprint changed: ' + name)
    if protocol['rounds'] != expected_rounds or marker.get('rounds') != expected_rounds or len(rows) != expected_rounds:
        raise RuntimeError('Missing or unexpected HCS rounds')
    for name in ('protocol', 'rounds', 'summary'):
        if marker.get(name + '_sha256') != c.digest(folder / (name + '.json')):
            raise RuntimeError('HCS result checksum mismatch: ' + name)
    replay(protocol, rows)
    for row in rows:
        if not np.isfinite([row[key] for key in ('test_accuracy', 'test_loss', 'model_norm', 'round_seconds', 'elapsed_seconds')]).all():
            raise RuntimeError('Nonfinite HCS training result')
    if c.read(folder / 'summary.json') != summarize(rows): raise RuntimeError('HCS summary differs from records')
    return marker


def train(a, prepared):
    net, real, _syn, weights, test, protocol = prepared
    if (a.refresh, a.representation, a.aggregation, a.rounds) != (protocol['refresh'], protocol['representation'], protocol['aggregation_mode'], protocol['rounds']):
        raise RuntimeError('Prepared HCS route differs from arguments')
    folder = Path(a.output); folder.mkdir(parents=True, exist_ok=True)
    c.bind(folder / 'protocol.json', protocol); sha = c.digest(folder / 'protocol.json')
    if (folder / 'COMPLETED.json').exists(): verify(folder, a.rounds); return True
    checkpoint = folder / 'resume.pt'; rows, elapsed, cache, last_seen, frozen = [], 0., None, None, None
    if checkpoint.exists():
        old = c.load(checkpoint)
        if old.get('protocol_sha256') != sha or old['round'] != len(old['records']) or old['round'] > a.rounds:
            raise RuntimeError('Checkpoint protocol/round mismatch')
        rows, elapsed, cache, last_seen, frozen = old['records'], old['elapsed_seconds'], old['cache'], old['last_seen'], old['static_geometry']
        seen, cache_sha, geometry_sha = replay(protocol, rows)
        if rows:
            width = sum(p.numel() for p in net.parameters())
            if cache is None or len(cache) != 32 or any(v.shape != (width,) or not torch.isfinite(v).all() for v in cache):
                raise RuntimeError('Invalid HCS checkpoint descriptor cache')
            if cache_fingerprint(cache) != cache_sha or seen.tolist() != last_seen.tolist(): raise RuntimeError('HCS checkpoint cache changed')
            if a.refresh == 'static' and (frozen is None or json_fingerprint(frozen) != geometry_sha): raise RuntimeError('HCS static geometry changed')
        net.load_state_dict(old['model']); c.restore_rng(old['rng'])
    else:
        if (folder / 'rounds.json').exists() and c.read(folder / 'rounds.json'):
            raise RuntimeError('Existing partial results have no resume checkpoint; refusing to restart')
        if tensor_fingerprint(net.state_dict()) != protocol['initial_model_state_sha256']: raise RuntimeError('Initial HCS model changed')
        c.seed_all(SEED)
    width = sum(p.numel() for p in net.parameters())
    sketch = sampling.make_sketch(width, seed=602025) if a.representation == 'countsketch' else None
    tick = time.monotonic()
    def persist(state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=len(rows), model=net.state_dict(), records=rows, cache=cache,
            last_seen=last_seen, static_geometry=frozen, elapsed_seconds=duration, rng=c.rng_state(), protocol_sha256=sha))
        c.atomic_json(folder / 'rounds.json', rows); c.atomic_json(folder / 'summary.json', summarize(rows))
        recent = [r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder / 'status.json', dict(state=state, round=len(rows), target=a.rounds,
            elapsed_seconds=duration, eta_seconds=float(np.mean(recent)) * (a.rounds - len(rows)) if recent else None))
    for rnd in range(len(rows) + 1, a.rounds + 1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        start = time.monotonic()
        delta, row, cache, last_seen, frozen = one_round(net, real, weights, a.refresh, a.representation,
            a.aggregation, rnd, cache, last_seen, frozen, sketch)
        c.add_delta(net, delta); row.update(fix.evaluate(net, *test))
        row.update(model_norm=float(torch.cat([p.detach().reshape(-1) for p in net.parameters()]).double().norm()),
            round_seconds=time.monotonic() - start, elapsed_seconds=elapsed + time.monotonic() - tick)
        rows.append(row)
        if rnd % 10 == 0 or rnd == a.rounds or c.expired(a.deadline): persist('training')
    persist('verifying')
    c.atomic_json(folder / 'COMPLETED.json', dict(rounds=a.rounds, protocol_sha256=sha,
        rounds_sha256=c.digest(folder / 'rounds.json'), summary_sha256=c.digest(folder / 'summary.json')))
    verify(folder, a.rounds)
    c.atomic_json(folder / 'status.json', dict(state='complete', round=a.rounds, target=a.rounds))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-root', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--alpha', choices=('0p1',), default='0p1')
    parser.add_argument('--refresh', choices=REFRESH, required=True)
    parser.add_argument('--representation', choices=REPRESENTATIONS, required=True)
    parser.add_argument('--aggregation', choices=AGGREGATIONS, required=True)
    parser.add_argument('--rounds', type=int, default=1500); parser.add_argument('--deadline', type=float)
    parser.add_argument('--run', action='store_true'); a = parser.parse_args()
    if a.rounds < 1: parser.error('--rounds must be positive')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]), ensure_ascii=False)); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
