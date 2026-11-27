"""Read-only import of the five completed 8k suites for exact SGD continuation."""
import copy
import shutil
import fmnist_common as _original
import fmnist_extension_common as _extension
from fmnist_extension_adapter import functions

functions(_original, globals())
SUITE = 'fmnist_continue32000_20261018'
INPUTS = ROOT / 'inputs' / SUITE
ALPHAS = ('0p05', '0p07', '0p1', '0p2', '0p3')
START = 8000
STAGES = (16000, 24000, 32000)
_base_protocol, _base_preflight = protocol, preflight
_base_context, _image_initializer = context, c.initial_images


def initial_images(x, y, indices, counts, alpha, client):
    # The frozen initializer includes ALPHAS.index(alpha); preserve each old suite's index.
    from fmnist_extension_adapter import clone
    order = _original.ALPHAS if alpha in _original.ALPHAS else _extension.ALPHAS
    return clone(_image_initializer, dict(vars(c), ALPHAS=order))(x, y, indices, counts, alpha, client)


@contextmanager
def context(inputs_root=INPUTS, bind_protocol=True):
    with _base_context(inputs_root, bind_protocol):
        old = c.initial_images; c.initial_images = initial_images
        try: yield
        finally: c.initial_images = old


def expected_config():
    value = copy.deepcopy(_original.validate_config())
    value.update(suite=SUITE, alphas=list(ALPHAS), synthesis_iterations=32000,
        snapshot_iterations=[0, 8000, *STAGES], run_order=['synthesize', 'vectors', 'first_seed', 'remaining_seeds'],
        first_seed_routes=45, remaining_seed_routes=180, total_fl_routes=225,
        reused_fl_routes=225, complete_matrix_routes=450, physical_jobs=390)
    value['diagnostics']['iterations'] = [32000]
    value['synthesis'].update(continuation_from=START,
        continuation='same images and exact saved SGD momentum; absolute-iteration random seeds')
    return value


def validate_config(path=None):
    value = c.read(Path(path) if path is not None else INPUTS / 'config.json')
    if value != expected_config():
        raise RuntimeError('Frozen FMNIST 32k continuation configuration changed')
    return value


def protocol(inputs_root=INPUTS):
    value = _base_protocol(inputs_root)
    value.update(synthesis_order='read-only 8000 import, then 16000/24000/32000 with retained SGD momentum',
        continuation=dict(start=START, final=STAGES[-1], old_loss=True,
            source_map_sha256=c.digest(Path(inputs_root) / 'source_map.json')))
    for name in ('fmnist_extension_adapter.py', 'fmnist_continue32000_common.py',
                 'run_fmnist_continue32000_synthesis.py', 'run_fmnist_fl.py',
                 'run_fmnist_continue32000_fl.py'):
        rel = 'synthesis_label2_study/' + name
        value['code_sha256'][rel] = c.digest(ROOT / rel)
    return json.loads(json.dumps(value))


def preflight(data_root, inputs_root=INPUTS):
    value = _base_preflight(data_root, inputs_root)
    value.update(reference_jobs=0, calibration_jobs=0, synthesis_jobs=160, vector_jobs=5,
                 fl_jobs=225, reused_fl_jobs=225, start_iteration=START, final_iteration=STAGES[-1])
    return value


def _copy_bound(source, target, dependencies, copied):
    source, target = Path(source), Path(target)
    sha = c.digest(source); dependencies[str(source.resolve())] = sha
    if target.exists():
        if c.digest(target) != sha: raise RuntimeError('Imported immutable file changed: ' + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.importing')
        shutil.copyfile(source, temporary)
        if c.digest(temporary) != sha: raise RuntimeError('Source changed during import: ' + str(source))
        os.replace(temporary, target)
    copied[str(target.resolve())] = sha


def validate_optimizer(state, iteration, snapshot=None):
    """An image snapshot is insufficient: continuation requires the bound momentum."""
    if state.get('iteration') != iteration or not torch.isfinite(state['images']).all():
        raise RuntimeError('Invalid continuation checkpoint progress/images')
    opt = state.get('optimizer', {}); groups = opt.get('param_groups', [])
    if len(groups) != 1 or groups[0]['lr'] != 1. or groups[0]['momentum'] != .5 or not opt.get('state'):
        raise RuntimeError('Continuation requires unchanged image SGD and saved momentum')
    if groups[0].get('weight_decay', 0.) != 0. or groups[0].get('dampening', 0.) != 0. or groups[0].get('nesterov', False):
        raise RuntimeError('Image SGD settings changed')
    if len(groups[0].get('params', [])) != 1 or set(opt['state']) != set(groups[0]['params']) or any('momentum_buffer' not in v or
        v['momentum_buffer'].shape != state['images'].shape or
        v['momentum_buffer'].dtype != state['images'].dtype or
        not torch.isfinite(v['momentum_buffer']).all() for v in opt['state'].values()):
        raise RuntimeError('Continuation checkpoint momentum missing/nonfinite')
    if not isinstance(state.get('elapsed_seconds'), (float, int)) or not np.isfinite(state['elapsed_seconds']) or state['elapsed_seconds'] < 0:
        raise RuntimeError('Invalid continuation checkpoint elapsed time')
    if [r['iteration'] for r in state['records']] != list(range(10, iteration + 1, 10)):
        raise RuntimeError('Continuation checkpoint optimization records incomplete')
    if snapshot is not None and (not torch.equal(state['images'], snapshot['images']) or not torch.equal(state['labels'], snapshot['labels'])):
        raise RuntimeError('Continuation checkpoint and 8k image snapshot differ')


def prepare_inputs(data_root, inputs_root, source_root, old_output, extension_output,
                   old_inputs, extension_inputs):
    inputs_root, source_root = Path(inputs_root).resolve(), Path(source_root).resolve()
    old_roots = [Path(old_output).resolve(), Path(extension_output).resolve()]
    if any(source_root == p or p in source_root.parents or source_root in p.parents for p in old_roots):
        raise RuntimeError('Continuation output must be independent of both old output directories')
    old_input_paths = [Path(p).resolve() for p in (old_inputs, extension_inputs)]
    if any(inputs_root == p or p in inputs_root.parents or inputs_root in p.parents for p in old_input_paths):
        raise RuntimeError('Continuation inputs must not overwrite old inputs')
    c.bind(inputs_root / 'config.json', expected_config())
    specifications = ((_original, old_roots[0], Path(old_inputs).resolve()),
                      (_extension, old_roots[1], Path(extension_inputs).resolve()))
    dependencies, copied, scenarios, splits, data_hashes, origins = {}, {}, [], {}, None, {}
    for module, output, old_input in specifications:
        module.preflight(data_root, old_input)
        source = output / 'synthesis_source'
        expected = module.protocol(old_input)
        if c.read(source / 'protocol.json') != expected:
            raise RuntimeError('Old synthesis protocol differs from verified old inputs/code: ' + str(source))
        dependencies[str((source / 'protocol.json').resolve())] = c.digest(source / 'protocol.json')
        manifest = c.read(old_input / 'manifest.json')
        if data_hashes is not None and data_hashes != manifest['data']:
            raise RuntimeError('Old FMNIST suites used different dataset bytes')
        data_hashes = manifest['data']
        for name in ('manifest.json', 'counts.json', 'config.json'):
            dependencies[str(old_input / name)] = c.digest(old_input / name)
        scenarios.extend(c.read(old_input / 'counts.json')['scenarios'])
        for alpha in module.ALPHAS:
            rel = 'splits/alpha' + alpha + '.pt'
            _copy_bound(old_input / rel, inputs_root / rel, dependencies, copied)
            splits[rel] = c.digest(inputs_root / rel)
            origins[alpha] = dict(old_output=str(output), old_inputs=str(old_input),
                old_source=str(source), old_suite=module.SUITE)
            for seed in SEEDS:
                ref = source / 'reference' / alpha / str(seed)
                done = c.read(ref / 'COMPLETED.json')
                if done['rounds'] != 1500 or set(done['model_sha256']) != {str(r) for r in c.MODEL_ROUNDS}:
                    raise RuntimeError('Incomplete original reference model pool')
                for rnd in c.MODEL_ROUNDS:
                    path = c.model_path(source, alpha, seed, rnd)
                    if c.digest(path) != done['model_sha256'][str(rnd)]:
                        raise RuntimeError('Original reference model fingerprint changed')
                    _copy_bound(path, source_root / path.relative_to(source), dependencies, copied)
                _copy_bound(ref / 'COMPLETED.json', source_root / ref.relative_to(source) / 'COMPLETED.json', dependencies, copied)
            calibration = source / 'calibration' / alpha
            done = c.read(calibration / 'COMPLETED.json')
            if done['records'] != 384 or c.digest(calibration / 'coefficients.json') != done['coefficients_sha256']:
                raise RuntimeError('Original DM calibration fingerprint changed')
            for name in ('coefficients.json', 'COMPLETED.json'):
                _copy_bound(calibration / name, source_root / 'calibration' / alpha / name, dependencies, copied)
            for client in range(32):
                old = c.route(source, alpha, CONFIG, client); target = c.route(source_root, alpha, CONFIG, client)
                image = module.validate_snapshot(source, alpha, CONFIG, client, START)
                if c.read(old / 'COMPLETED_8000.json') != dict(iteration=START, snapshot_sha256=c.digest(image)):
                    raise RuntimeError('Original 8k snapshot is not complete')
                checkpoint = old / 'checkpoints' / 'iter_8000.pt'
                if c.digest(checkpoint) != c.read(old / 'milestone_integrity.json').get(checkpoint.name):
                    raise RuntimeError('Original 8k optimizer checkpoint fingerprint changed')
                state = c.load(checkpoint); validate_optimizer(state, START, c.load(image))
                if c.read(old / 'metrics.json') != state['records']:
                    raise RuntimeError('Original 8k optimization records changed')
                for rel in ('config.json', 'snapshots/iter_0000.pt', 'snapshots/iter_8000.pt',
                            'checkpoints/iter_8000.pt', 'COMPLETED_8000.json'):
                    _copy_bound(old / rel, target / rel, dependencies, copied)
                c.bind(target / 'continuation_import.json', dict(start_iteration=START,
                    checkpoint_sha256=c.digest(checkpoint), elapsed_seconds_before=state['elapsed_seconds'],
                    original_folder=str(old), original_protocol_sha256=c.digest(source / 'protocol.json')))
                if not (target / 'job_binding.json').exists():
                    if (target / 'resume.pt').exists() and c.digest(target / 'resume.pt') != c.digest(checkpoint):
                        raise RuntimeError('Unbound continuation resume differs from imported 8k checkpoint')
                    if not (target / 'resume.pt').exists(): shutil.copyfile(checkpoint, target / 'resume.pt')
                    c.bind(target / 'metrics.json', state['records'])
                    c.bind(target / 'status.json', dict(state='prepared', iteration=START,
                        target=STAGES[-1], elapsed_seconds=state['elapsed_seconds'], additional_elapsed_seconds=0.))
                    c.bind(target / 'milestone_integrity.json', {checkpoint.name: c.digest(checkpoint)})
                elif not (target / 'resume.pt').exists():
                    raise RuntimeError('Bound continuation resume checkpoint disappeared')
    c.bind(inputs_root / 'counts.json', dict(dataset='FashionMNIST', scenarios=scenarios))
    c.bind(inputs_root / 'manifest.json', dict(dataset='FashionMNIST', data=data_hashes, splits=splits,
        counts_sha256=c.digest(inputs_root / 'counts.json'), partition=dict(seed=35, clients=32, min_per_client=1)))
    c.bind(inputs_root / 'source_map.json', dict(source_root=str(source_root), origins=origins,
        original_dependencies_sha256=dependencies, copied_files_sha256=copied))
    c.bind(source_root / 'protocol.json', protocol(inputs_root))
    # The worker creates the new binding around the imported 8k resume, without a training call.
    import run_fmnist_continue32000_synthesis as worker
    for alpha in ALPHAS:
        for client in range(32):
            a = worker._args(dict(phase='synthesize', root=str(source_root), alpha=alpha, client=client), inputs_root)
            with context(inputs_root): worker.bind_job(a)
    preflight(data_root, inputs_root)
    return verify_prepared_source(source_root, inputs_root)


def verify_prepared_source(source_root, inputs_root=INPUTS):
    source_root, inputs_root = Path(source_root).resolve(), Path(inputs_root)
    source_map = c.read(inputs_root / 'source_map.json')
    if source_map['source_root'] != str(source_root) or c.read(source_root / 'protocol.json') != protocol(inputs_root):
        raise RuntimeError('Prepared continuation source/protocol changed')
    for path, sha in source_map['copied_files_sha256'].items():
        if c.digest(path) != sha: raise RuntimeError('Prepared immutable continuation dependency changed: ' + path)
    import run_fmnist_continue32000_synthesis as worker
    with context(inputs_root):
        for alpha in ALPHAS:
            for client in range(32):
                a = worker._args(dict(phase='synthesize', root=str(source_root), alpha=alpha, client=client), inputs_root)
                if c.read(worker.folder(a) / 'job_binding.json') != worker._binding(a):
                    raise RuntimeError('Prepared continuation client binding changed')
                worker._verify_resume(a)
    return dict(state='preflight_ok', suite=SUITE, source_root=str(source_root),
        start_iteration=START, final_iteration=STAGES[-1], synthesis_jobs=160, vector_jobs=5,
        fl_jobs=225, reused_fl_jobs=225, formal_training_started=False)


def validate_snapshot(root, alpha, config, client, iteration):
    # Bind the original validation body to the expanded identities; all tensor checks remain exact.
    from fmnist_extension_adapter import clone
    namespace = dict(globals(), STAGES=(START, *STAGES))
    return clone(_original.validate_snapshot, namespace)(root, alpha, config, client, iteration)
