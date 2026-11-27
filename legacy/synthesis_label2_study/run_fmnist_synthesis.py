"""Fashion-MNIST reference, calibrated old-loss synthesis and vector diagnostics."""
import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import fmnist_common as common
import run_dirichlet_reference as reference
import run_dirichlet_screen as screen
import dirichlet_dsa4000_worker as frozen

c = common.c
STAGES = common.STAGES


def folder(a):
    if a.phase == 'reference': return a.root / 'reference' / a.alpha / str(a.seed)
    if a.phase == 'calibrate': return a.root / 'calibration' / a.alpha
    if a.phase == 'synthesize': return c.route(a.root, a.alpha, common.CONFIG, a.client)
    return a.root / 'evaluation' / f'iter_{a.iteration:04d}' / a.alpha / 'vectors'


def marker(a):
    return folder(a) / ('COMPLETED_8000.json' if a.phase == 'synthesize' else 'COMPLETED.json')


def _args(job, inputs_root):
    return SimpleNamespace(phase=job['phase'], root=Path(job['root']), inputs_root=Path(inputs_root),
        alpha=job['alpha'], config=common.CONFIG, seed=job.get('seed'), client=job.get('client'),
        iteration=job.get('iteration', 8000), deadline=None)


def _binding(a):
    expected = common.protocol(a.inputs_root)
    if c.read(a.root / 'protocol.json') != json.loads(json.dumps(expected)):
        raise RuntimeError('Synthesis source protocol/input/code fingerprint changed')
    dependencies = {}
    rounds = {'calibrate': c.CAL_ROUNDS, 'synthesize': tuple(r for group in c.TRAIN_POOL for r in group),
              'vectors': c.EVAL_ROUNDS}.get(a.phase, ())
    for seed in c.SEEDS if rounds else ():
        done = c.read(a.root / 'reference' / a.alpha / str(seed) / 'COMPLETED.json')
        for rnd in rounds:
            path = c.model_path(a.root, a.alpha, seed, rnd)
            sha = c.digest(path)
            if done['model_sha256'].get(str(rnd)) != sha:
                raise RuntimeError('Reference model dependency changed: ' + str(path))
            dependencies[path.relative_to(a.root).as_posix()] = sha
    if a.phase == 'synthesize':
        path = a.root / 'calibration' / a.alpha / 'coefficients.json'
        done = c.read(path.parent / 'COMPLETED.json')
        if done['records'] != 384 or c.digest(path) != done['coefficients_sha256']:
            raise RuntimeError('DM calibration dependency changed')
        dependencies[path.relative_to(a.root).as_posix()] = c.digest(path)
    if a.phase == 'vectors':
        for client in range(32):
            path = c.validate_snapshot(a.root, a.alpha, common.CONFIG, client, a.iteration)
            done = path.parent.parent / f'COMPLETED_{a.iteration}.json'
            if not done.is_file(): raise RuntimeError('Missing bound diagnostic image marker: ' + str(done))
            dependencies[path.relative_to(a.root).as_posix()] = c.digest(path)
    return dict(suite=common.SUITE, phase=a.phase, alpha=a.alpha,
        seed=a.seed if a.phase == 'reference' else None, client=a.client if a.phase == 'synthesize' else None,
        iteration=a.iteration if a.phase == 'vectors' else None, synthesis_stages=list(STAGES),
        source_protocol_sha256=c.digest(a.root / 'protocol.json'),
        worker_sha256=c.digest(Path(__file__)), dependencies=dependencies)


def bind_job(a):
    path = folder(a) / 'job_binding.json'
    if not path.exists() and folder(a).exists() and any(folder(a).iterdir()):
        raise RuntimeError('Unbound existing synthesis/reference/diagnostic progress: ' + str(folder(a)))
    c.bind(path, _binding(a))


def _verify_resume(a):
    path = folder(a) / 'resume.pt'
    if not path.exists(): return
    integrity = folder(a) / 'resume_integrity.json'
    expected = dict(checkpoint_sha256=c.digest(path), binding_sha256=c.digest(folder(a) / 'job_binding.json'))
    if not integrity.exists() or c.read(integrity) != expected:
        raise RuntimeError('Unbound or changed numerical checkpoint: ' + str(path))
    item = c.load(path)
    key = 'round' if a.phase == 'reference' else 'iteration'
    target = 1500 if a.phase == 'reference' else 8000
    progress = item[key]
    if not isinstance(progress, int) or not 0 <= progress <= target:
        raise RuntimeError('Invalid saved numerical progress')
    if not np.isfinite(item['elapsed_seconds']) or item['elapsed_seconds'] < 0:
        raise RuntimeError('Invalid saved elapsed time')
    if a.phase == 'synthesize':
        if not torch.isfinite(item['images']).all() or (progress > 0 and not item['optimizer'].get('state')):
            raise RuntimeError('Saved image values or SGD momentum missing')
        for state in item['optimizer']['state'].values():
            if 'momentum_buffer' not in state or not torch.isfinite(state['momentum_buffer']).all():
                raise RuntimeError('Saved image SGD momentum missing/nonfinite')


@contextmanager
def checkpoint_hashes(a):
    """Seal unchanged frozen-kernel checkpoints immediately after their atomic save."""
    original = c.save
    def save(path, value):
        original(path, value)
        if Path(path) == folder(a) / 'resume.pt':
            c.atomic_json(folder(a) / 'resume_integrity.json', dict(checkpoint_sha256=c.digest(path),
                binding_sha256=c.digest(folder(a) / 'job_binding.json')))
            if a.phase == 'synthesize' and value['iteration'] in STAGES:
                milestone = folder(a) / 'checkpoints' / f"iter_{value['iteration']:04d}.pt"
                original(milestone, value)
                integrity = folder(a) / 'milestone_integrity.json'
                hashes = c.read(integrity) if integrity.exists() else {}
                hashes[milestone.name] = c.digest(milestone)
                c.atomic_json(integrity, hashes)
    c.save = save
    try: yield
    finally: c.save = original


@contextmanager
def progress_metadata(a):
    """Attach cumulative timing to calibration/diagnostics without altering kernels."""
    original = c.atomic_json; path = folder(a) / 'status.json'; tick = time.monotonic()
    elapsed = c.read(path).get('elapsed_seconds', 0.) if path.exists() else 0.
    integrity = folder(a) / 'progress_integrity.json'
    hashes = c.read(integrity)['files_sha256'] if integrity.exists() else {}
    def atomic_json(target, value):
        if Path(target) == path:
            value = dict(value)
            if a.phase in ('calibrate', 'vectors'):
                value['elapsed_seconds'] = elapsed + time.monotonic() - tick
                value['target'] = 384
            if a.phase == 'synthesize':
                value['stage_target'] = value.get('target'); value['target'] = 8000
        original(target, value)
        target = Path(target)
        if target.parent == folder(a) and ((a.phase == 'calibrate' and target.name == 'records.json') or
                                          (a.phase == 'vectors' and target.name.startswith('client_'))):
            hashes[target.name] = c.digest(target)
            original(integrity, dict(binding_sha256=c.digest(folder(a) / 'job_binding.json'), files_sha256=hashes))
    c.atomic_json = atomic_json
    try: yield
    finally: c.atomic_json = original


def _verify_partial(a):
    paths = []
    if a.phase == 'calibrate' and (folder(a) / 'records.json').exists(): paths = [folder(a) / 'records.json']
    if a.phase == 'vectors': paths = list(folder(a).glob('client_*.json'))
    if not paths: return
    expected = dict(binding_sha256=c.digest(folder(a) / 'job_binding.json'),
                    files_sha256={p.name: c.digest(p) for p in paths})
    path = folder(a) / 'progress_integrity.json'
    if not path.exists() or c.read(path) != expected:
        raise RuntimeError('Unbound or changed calibration/vector progress fingerprint')


def synthesize(a, stages=STAGES):
    """One queued client retains the same images and optimizer across both milestones."""
    for iteration in stages:
        item = SimpleNamespace(**vars(a)); item.iteration = iteration
        if not frozen.synthesize(item): return False
    return True


def _vector_names():
    return {f'client_{i:02d}_seed{s}_round{r:04d}.json' for i in range(32)
            for s in c.SEEDS for r in c.EVAL_ROUNDS}


def _verify_numeric(a):
    done = c.read(marker(a)); path = folder(a)
    if a.phase == 'reference':
        rows = c.read(path / 'rounds.json')
        if done['rounds'] != 1500 or [r['round'] for r in rows] != list(range(1, 1501)):
            raise RuntimeError('Reference pool trajectory is incomplete')
        if set(done['model_sha256']) != {str(r) for r in c.MODEL_ROUNDS}:
            raise RuntimeError('Reference heldout/training model coverage mismatch')
        sizes = np.asarray(c.read(a.root / 'protocol.json')['budgets'][a.alpha]['real_sizes'])
        for row in rows:
            selected = np.random.default_rng(a.seed + row['round'] * 1000).choice(32, 3, replace=False)
            weights = sizes[selected] / sizes[selected].sum()
            if row['selected'] != selected.tolist() or row['weights'] != weights.tolist() or not np.isfinite(row['update_norm']):
                raise RuntimeError('Reference sampling/FedAvg weights changed')
        for rnd, sha in done['model_sha256'].items():
            if c.digest(c.model_path(a.root, a.alpha, a.seed, int(rnd))) != sha:
                raise RuntimeError('Reference model fingerprint changed')
    elif a.phase == 'calibrate':
        rows = c.read(path / 'records.json')
        keys = {(i, s, r, k) for i in range(32) for s in c.SEEDS for r in c.CAL_ROUNDS for k in range(2)}
        if len(rows) != 384 or {(r['client'], r['model_seed'], r['round'], r['repeat']) for r in rows} != keys or done['records'] != 384:
            raise RuntimeError('DM calibration contexts must cover all 384 points exactly once')
        expected = {}
        for ce in ('ordinary', 'weighted'):
            for mode in ('weak', 'strong'):
                values = [r['update_gradient_norm'][ce] / r['dm_gradient_norm'][mode] for r in rows]
                if not all(np.isfinite(v) and v > 0 for v in values): raise RuntimeError('Invalid DM gradient ratio')
                expected[f'{ce}_{mode}'] = dict(r=float(np.median(values)), q25=float(np.quantile(values, .25)),
                                              q75=float(np.quantile(values, .75)), n=len(values))
        if c.read(path / 'coefficients.json') != expected or done['coefficients_sha256'] != c.digest(path / 'coefficients.json'):
            raise RuntimeError('DM coefficient recomputation/fingerprint mismatch')
    elif a.phase == 'synthesize':
        c.validate_snapshot(a.root, a.alpha, common.CONFIG, a.client, 0)
        hashes = c.read(path / 'milestone_integrity.json')
        if set(hashes) != {f'iter_{r:04d}.pt' for r in STAGES}:
            raise RuntimeError('Image optimizer milestone coverage changed')
        for iteration in STAGES:
            snap = c.validate_snapshot(a.root, a.alpha, common.CONFIG, a.client, iteration)
            completion = path / f'COMPLETED_{iteration}.json'
            if c.read(completion) != dict(iteration=iteration, snapshot_sha256=c.digest(snap)):
                raise RuntimeError('Image milestone marker mismatch')
            saved = path / 'checkpoints' / f'iter_{iteration:04d}.pt'
            if c.digest(saved) != hashes[saved.name]: raise RuntimeError('Image optimizer milestone changed')
            state = c.load(saved); image = c.load(snap)
            if state['iteration'] != iteration or not torch.equal(state['images'], image['images']) or not torch.equal(state['labels'], image['labels']):
                raise RuntimeError('Image/optimizer milestone does not match image snapshot')
            groups = state['optimizer']['param_groups']
            if len(groups) != 1 or groups[0]['lr'] != 1. or groups[0]['momentum'] != .5 or not state['optimizer']['state']:
                raise RuntimeError('Frozen image SGD state missing or changed')
            for entry in state['optimizer']['state'].values():
                if 'momentum_buffer' not in entry or not torch.isfinite(entry['momentum_buffer']).all():
                    raise RuntimeError('Image milestone has invalid SGD momentum')
        _verify_resume(a)
        item = c.load(path / 'resume.pt'); snap = c.load(c.snapshot(a.root, a.alpha, common.CONFIG, a.client, 8000))
        if item['iteration'] != 8000 or not torch.equal(item['images'], snap['images']) or not torch.equal(item['labels'], snap['labels']):
            raise RuntimeError('Final image checkpoint and snapshot differ')
        rows = c.read(path / 'metrics.json')
        if rows != item['records'] or [r['iteration'] for r in rows] != list(range(10, STAGES[-1] + 1, 10)):
            raise RuntimeError('Image optimization records incomplete or changed')
    elif a.phase == 'vectors':
        if done != dict(points=384, repeats=30, configs=1, files_sha256={n: c.digest(path / n) for n in sorted(_vector_names())}):
            raise RuntimeError('Vector diagnostic coverage/fingerprint mismatch')
        for name in _vector_names():
            point = c.read(path / name)
            expected_name = f"client_{point['client']:02d}_seed{point['model_seed']}_round{point['round']:04d}.json"
            if name != expected_name or point['repeats'] != 30 or [r['config'] for r in point['records']] != [common.CONFIG]:
                raise RuntimeError('Vector point identity/configuration mismatch')
            if len(point['records'][0]['individual_geometry']) != 30:
                raise RuntimeError('Vector diagnostic repeat count changed')


def seal(a):
    files = [p for p in folder(a).glob('*.json') if p.name != 'job_integrity.json']
    c.atomic_json(folder(a) / 'job_integrity.json', dict(binding_sha256=c.digest(folder(a) / 'job_binding.json'),
        files_sha256={p.name: c.digest(p) for p in sorted(files)}))


def _construction_cost(a):
    status = c.read(folder(a) / 'status.json')
    elapsed = status.get('elapsed_seconds')
    if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError('Construction elapsed time missing/invalid')
    calls = dict(reference=dict(real_three_step_calls=4500, proxy_three_step_calls=0),
        calibrate=dict(real_three_step_calls=384, proxy_three_step_calls=768,
                       dm_gradient_calls=768),
        synthesize=dict(real_three_step_calls=8000, proxy_three_step_calls=8000,
                        dm_gradient_calls=8000),
        vectors=dict(real_three_step_calls=11520, proxy_three_step_calls=11520))[a.phase]
    return dict(phase=a.phase, elapsed_seconds=elapsed,
                timing_scope='worker numerical phase; cumulative persisted resume wall time; work lost before last checkpoint/status is excluded',
                call_counts_scope='nominal completed target workload; interrupted work repeated before persistence is excluded',
                local_steps=3, accounting='offline construction; excluded from online FL communication', **calls)


def verify_job(job, inputs_root):
    a = _args(job, inputs_root)
    if not marker(a).is_file(): return False
    if job.get('alpha') != a.alpha or Path(job['artifact']) != marker(a):
        raise RuntimeError('Synthesis worker job identity/marker path changed')
    with common.context(a.inputs_root):
        if c.read(folder(a) / 'job_binding.json') != _binding(a): raise RuntimeError('Synthesis job binding changed')
        _verify_resume(a)
        _verify_numeric(a)
        _verify_partial(a)
        item = c.read(folder(a) / 'job_integrity.json')
        if item['binding_sha256'] != c.digest(folder(a) / 'job_binding.json'):
            raise RuntimeError('Synthesis job integrity binding mismatch')
        files = {p.name: c.digest(p) for p in folder(a).glob('*.json') if p.name != 'job_integrity.json'}
        if item['files_sha256'] != files: raise RuntimeError('Synthesis result JSON fingerprint changed')
    return True


def execute(a):
    with common.context(a.inputs_root):
        c.bind(a.root / 'protocol.json', common.protocol(a.inputs_root))
        bind_job(a); _verify_resume(a); _verify_partial(a)
        if marker(a).exists():
            if not (folder(a) / 'job_integrity.json').exists():
                _verify_numeric(a)
                c.atomic_json(folder(a) / 'construction_cost.json', _construction_cost(a))
                seal(a)
            return verify_job(dict(vars(a), artifact=str(marker(a))), a.inputs_root)
        with checkpoint_hashes(a), progress_metadata(a):
            if a.phase == 'reference': result = reference.train(a.root, a.data_root, a.alpha, a.seed, a.deadline)
            elif a.phase == 'calibrate': result = screen.calibrate(a)
            elif a.phase == 'synthesize': result = synthesize(a)
            else:
                a.clients = list(range(32)); a.configs = (common.CONFIG,)
                result = frozen.vectors(a)
            if result and a.phase in ('calibrate', 'vectors'):
                status = folder(a) / 'status.json'
                c.atomic_json(status, dict(c.read(status), state='complete', completed=384, target=384))
        if result:
            _verify_numeric(a)
            c.atomic_json(folder(a) / 'construction_cost.json', _construction_cost(a))
            seal(a)
            verify_job(dict(vars(a), artifact=str(marker(a))), a.inputs_root)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'reference', 'calibrate', 'synthesize', 'vectors'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--inputs-root', type=Path, default=common.INPUTS)
    p.add_argument('--alpha', choices=common.ALPHAS); p.add_argument('--download', action='store_true')
    p.add_argument('--seed', type=int, choices=common.SEEDS); p.add_argument('--client', type=int, choices=range(32))
    p.add_argument('--iteration', type=int, choices=STAGES, default=8000)
    p.add_argument('--deadline', type=float); p.add_argument('--run', action='store_true')
    a = p.parse_args(); a.root = a.root.resolve(); a.inputs_root = a.inputs_root.resolve()
    a.config = common.CONFIG
    if a.phase == 'prepare':
        print(json.dumps(common.prepare_inputs(a.data_root, a.inputs_root, a.download), indent=2)); return
    if a.alpha is None: p.error('--alpha required')
    if a.download: p.error('--download only applies to prepare')
    if a.phase == 'reference' and a.seed is None: p.error('reference requires --seed')
    if a.phase == 'synthesize' and a.client is None: p.error('synthesize requires --client')
    if not a.run:
        print(json.dumps(common.preflight(a.data_root, a.inputs_root), indent=2)); return
    common.preflight(a.data_root, a.inputs_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError('Formal Fashion-MNIST tasks require a visible CUDA device')
    raise SystemExit(0 if execute(a) else 75)


if __name__ == '__main__': main()
