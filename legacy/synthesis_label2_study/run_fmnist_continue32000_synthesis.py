"""Exact 8k-to-32k image-SGD continuation and matched 32k vector diagnostics."""
import run_fmnist_synthesis as _original
from fmnist_extension_adapter import functions

functions(_original, globals())
import fmnist_continue32000_common as common

STAGES = common.STAGES
_base_binding, _base_partial, _base_numeric = _binding, _verify_partial, _verify_numeric
_base_cost = _construction_cost


def synthesize(a, stages=STAGES):
    for iteration in stages:
        item = SimpleNamespace(**vars(a)); item.iteration = iteration
        if not frozen.synthesize(item): return False
    return True


def marker(a):
    return folder(a) / ('COMPLETED_32000.json' if a.phase == 'synthesize' else 'COMPLETED.json')


def _args(job, inputs_root):
    return SimpleNamespace(phase=job['phase'], root=Path(job['root']), inputs_root=Path(inputs_root),
        alpha=job['alpha'], config=common.CONFIG, seed=None, client=job.get('client'),
        iteration=job.get('iteration', 32000), deadline=None)


def _binding(a):
    value = _base_binding(a)
    if a.phase == 'synthesize':
        provenance = folder(a) / 'continuation_import.json'
        item = c.read(provenance)
        state = folder(a) / 'checkpoints' / 'iter_8000.pt'
        if item['start_iteration'] != 8000 or c.digest(state) != item['checkpoint_sha256']:
            raise RuntimeError('Original continuation checkpoint/import fingerprint changed')
        value.update(continuation_import_sha256=c.digest(provenance),
                     start_iteration=8000, required_optimizer='SGD lr=1 momentum=0.5')
    return value


def bind_job(a):
    path = folder(a) / 'job_binding.json'
    if a.phase == 'synthesize':
        # Only the explicitly prepared, hash-bound 8k import may have pre-existing progress.
        imported = c.read(folder(a) / 'continuation_import.json')
        new = not path.exists()
        if new and c.digest(folder(a) / 'resume.pt') != imported['checkpoint_sha256']:
            raise RuntimeError('New continuation must start from its imported 8k optimizer checkpoint')
        c.bind(path, _binding(a))
        integrity = folder(a) / 'resume_integrity.json'
        if new or not integrity.exists():
            if c.digest(folder(a) / 'resume.pt') != imported['checkpoint_sha256']:
                raise RuntimeError('Missing continuation integrity beyond the verified 8k import')
            c.atomic_json(integrity, dict(checkpoint_sha256=c.digest(folder(a) / 'resume.pt'),
                binding_sha256=c.digest(path)))
    else:
        if not path.exists() and folder(a).exists() and any(folder(a).iterdir()):
            raise RuntimeError('Unbound existing continuation diagnostic progress')
        c.bind(path, _binding(a))


def _verify_resume(a):
    path = folder(a) / 'resume.pt'
    if a.phase != 'synthesize': return
    expected = dict(checkpoint_sha256=c.digest(path), binding_sha256=c.digest(folder(a) / 'job_binding.json'))
    if c.read(folder(a) / 'resume_integrity.json') != expected:
        raise RuntimeError('Unbound or changed continuation checkpoint')
    item = c.load(path); iteration = item.get('iteration')
    if not isinstance(iteration, int) or not 8000 <= iteration <= 32000:
        raise RuntimeError('Continuation progress outside 8000..32000')
    # An interrupted save may be before a multiple of ten; its records end at that multiple.
    common.validate_optimizer(item, iteration)


@contextmanager
def progress_metadata(a):
    original = c.atomic_json; path = folder(a) / 'status.json'; tick = time.monotonic()
    old = c.read(path) if path.exists() else {}
    elapsed = old.get('elapsed_seconds', 0.)
    base = c.read(folder(a) / 'continuation_import.json')['elapsed_seconds_before'] if a.phase == 'synthesize' else 0.
    integrity = folder(a) / 'progress_integrity.json'
    hashes = c.read(integrity)['files_sha256'] if integrity.exists() else {}
    def atomic_json(target, value):
        if Path(target) == path:
            value = dict(value)
            if a.phase == 'vectors':
                value.update(elapsed_seconds=elapsed + time.monotonic() - tick, target=384)
            else:
                value.update(stage_target=value.get('target'), target=32000,
                    additional_elapsed_seconds=max(0., value.get('elapsed_seconds', 0.) - base))
        original(target, value); target = Path(target)
        if a.phase == 'vectors' and target.parent == folder(a) and target.name.startswith('client_'):
            hashes[target.name] = c.digest(target)
            original(integrity, dict(binding_sha256=c.digest(folder(a) / 'job_binding.json'), files_sha256=hashes))
    c.atomic_json = atomic_json
    try: yield
    finally: c.atomic_json = original


def _verify_numeric(a):
    if a.phase == 'vectors': return _base_numeric(a)
    path = folder(a); _verify_resume(a)
    hashes = c.read(path / 'milestone_integrity.json')
    if set(hashes) != {f'iter_{r:04d}.pt' for r in (8000, *STAGES)}:
        raise RuntimeError('Continuation optimizer milestone coverage changed')
    for iteration in (8000, *STAGES):
        snap = common.validate_snapshot(a.root, a.alpha, common.CONFIG, a.client, iteration)
        if c.read(path / f'COMPLETED_{iteration}.json') != dict(iteration=iteration, snapshot_sha256=c.digest(snap)):
            raise RuntimeError('Continuation snapshot marker changed')
        saved = path / 'checkpoints' / f'iter_{iteration:04d}.pt'
        if c.digest(saved) != hashes[saved.name]: raise RuntimeError('Continuation optimizer milestone changed')
        common.validate_optimizer(c.load(saved), iteration, c.load(snap))
    final = c.load(path / 'resume.pt')
    common.validate_optimizer(final, 32000, c.load(c.snapshot(a.root, a.alpha, common.CONFIG, a.client, 32000)))
    if c.read(path / 'metrics.json') != final['records']:
        raise RuntimeError('Continuation metrics differ from saved optimizer records')
    original = c.load(path / 'checkpoints' / 'iter_8000.pt')
    if final['records'][:800] != original['records']:
        raise RuntimeError('Original 8k optimization history changed')


def _construction_cost(a):
    result = _base_cost(a)
    if a.phase == 'synthesize':
        before = c.read(folder(a) / 'continuation_import.json')['elapsed_seconds_before']
        result.update(elapsed_seconds=result['elapsed_seconds'] - before,
            elapsed_seconds_before=before, start_iteration=8000, final_iteration=32000,
            real_three_step_calls=24000, proxy_three_step_calls=24000, dm_gradient_calls=24000,
            timing_scope='additional image-SGD continuation only; excludes the imported first 8000 iterations')
        if result['elapsed_seconds'] < 0: raise RuntimeError('Continuation elapsed time decreased')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('synthesize', 'vectors'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--inputs-root', type=Path, default=common.INPUTS)
    p.add_argument('--alpha', choices=common.ALPHAS, required=True)
    p.add_argument('--client', type=int, choices=range(32)); p.add_argument('--iteration', type=int, choices=(32000,), default=32000)
    p.add_argument('--deadline', type=float); p.add_argument('--run', action='store_true')
    a = p.parse_args(); a.root = a.root.resolve(); a.inputs_root = a.inputs_root.resolve()
    a.config, a.seed = common.CONFIG, None
    if a.phase == 'synthesize' and a.client is None: p.error('synthesize requires --client')
    common.preflight(a.data_root, a.inputs_root)
    if not a.run:
        print(json.dumps(common.verify_prepared_source(a.root, a.inputs_root), indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError('Formal Fashion-MNIST continuation requires CUDA')
    raise SystemExit(0 if execute(a) else 75)


if __name__ == '__main__': main()
