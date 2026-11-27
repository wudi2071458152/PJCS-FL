"""Five FL seeds on migrated alpha=.05 old-loss 8k/32k images; no synthesis."""
import argparse
import copy
import json
import signal
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import run_dirichlet_multiseed as multi
import dirichlet_oldserver_common as common
import run_dirichlet_oldserver as old

c = multi.c
ROOT = c.ROOT
CASE, ALPHA = 'a0p05_b20', '0p05'
SEEDS = (2025, *multi.NEW_SEEDS)
ROUTES = multi.ROUTES
SHARED = ('fedavg', 'uniform_real', 'history_uniform_zero')
DEFAULT_MIGRATED_ROOT = Path('/home/wd/project/Cifar10-experiments/oldserver_import_20261012/dirichlet_oldserver_20261009')
SOURCE_PROTOCOL_SHA256 = '9edc8840fdb432b50b59c5682a86fd57a9b599cc2fd137f1a4b7313187cd4d5d'
ORIGINAL_PROTOCOL_SHA256 = '238945d433178b7e261565312327ec2d82cc199f8f22b620a2905829885be222'


def route_jobs():
    return [dict(seed=seed, route=route, iteration=iteration)
            for seed in SEEDS for route in ROUTES
            for iteration in ((0,) if route in SHARED else (8000, 32000))]


def paths(migrated_root):
    root = Path(migrated_root).resolve()
    case = root / 'outputs/dirichlet_oldserver_20261009' / CASE
    return root, case / 'source24000', case / 'original', root / 'inputs/dirichlet_oldserver_20261009' / CASE


def migration_binding(migrated_root):
    """Map old relative paths explicitly; preserve both saved protocols byte-for-byte."""
    root, source, original, inputs = paths(migrated_root)
    files = {}
    def check(file, expected=None):
        sha = c.digest(file)
        if expected is not None and sha != expected:
            raise RuntimeError('Migrated input fingerprint mismatch: ' + str(file))
        files[file.relative_to(root).as_posix()] = sha
        return sha
    check(source / 'protocol.json', SOURCE_PROTOCOL_SHA256)
    check(original / 'protocol.json', ORIGINAL_PROTOCOL_SHA256)
    expected = c.read(source / 'protocol.json')
    if expected['case'] != CASE or expected['budget_ratio'] != .2 or expected['budget_total'] != 10000:
        raise RuntimeError('Only alpha=.05 budget20% is supported')
    for rel, sha in expected['code_sha256'].items():
        check(root / rel, sha)
        if c.digest(ROOT / rel) != sha:
            raise RuntimeError('Current numerical code differs from migrated source: ' + rel)
    manifest = c.read(inputs / 'manifest.json')
    if manifest != expected['inputs']:
        raise RuntimeError('Original alpha=.05 manifest changed')
    check(inputs / 'manifest.json'); check(inputs / 'counts.json', manifest['counts_sha256'])
    for rel, sha in manifest['splits'].items(): check(inputs / rel, sha)
    for seed in (2025, 2026):
        marker_file = source / 'reference' / ALPHA / str(seed) / 'COMPLETED.json'
        marker = c.read(marker_file); check(marker_file)
        # The FL-only transfer deliberately omits rounds.json and the other model-pool states.
        for rnd in (0, 30, 400, 1000): check(c.model_path(source, ALPHA, seed, rnd), marker['model_sha256'][str(rnd)])
    return expected, dict(root=str(root), case=CASE, source='outputs/dirichlet_oldserver_20261009/' + CASE + '/source24000',
        original='outputs/dirichlet_oldserver_20261009/' + CASE + '/original',
        policy='explicit root mapping; immutable old protocols and recorded content fingerprints', files_sha256=files)


@contextmanager
def migration_context(root, inputs, expected):
    """Scope the old prepare function to its immutable migrated files."""
    with ExitStack() as stack:
        for name, value in dict(ROOT=root, INPUTS=inputs, ALPHAS=(ALPHA,), CONFIGS=(common.CONFIG,),
                SELECTED={ALPHA: list(range(32))}, budget_table=lambda counts: common.allocate(counts, 10000),
                protocol=lambda: copy.deepcopy(expected)).items():
            stack.enter_context(patch.object(c, name, value))
        stack.enter_context(patch.object(common, 'protocol', lambda case: copy.deepcopy(expected)))
        yield


def initial_model(migrated_root, seed):
    if seed not in SEEDS: raise ValueError('Unsupported FL seed')
    if seed != 2025: return c.model(seed)
    _, source, _, _ = paths(migrated_root)
    net = c.reference(source, ALPHA, 2025, 0)
    if not all(torch.isfinite(v).all() for v in net.state_dict().values()):
        raise RuntimeError('Nonfinite migrated seed2025 initial state')
    return net


def route_protocol(template, route, seed, iteration, binding=None, initial_sha256=None):
    if route not in ROUTES or seed not in SEEDS or iteration not in ((0,) if route in SHARED else (8000, 32000)):
        raise ValueError('Invalid alpha=.05 route/seed/image stage')
    if template.get('case') != CASE or template.get('alpha') != ALPHA or template.get('iteration') != iteration:
        raise RuntimeError('Alpha=.05 template case/image stage mismatch')
    binding = binding if binding is not None else template['migration']
    initial_sha256 = initial_sha256 if initial_sha256 is not None else (
        template['fl_initialization']['state_sha256'] if seed == 2025 else multi.initialization_sha256(seed))
    original = copy.deepcopy(template)
    original.pop('sampler', None)  # Legacy description; canonical `sampling` follows the actual route.
    if original.get('augmentation', False) is not False: raise RuntimeError('Frozen augmentation setting changed')
    original['augmentation'] = False
    with patch.object(multi, 'initialization_sha256', return_value=initial_sha256):
        value = multi.route_protocol(original, route, seed)
    value.update(version=5, suite='dirichlet_a005_multiseed_20261012', case=CASE, alpha=ALPHA,
        version_name='shared' if route in SHARED else f'old_i{iteration}', iteration=iteration,
        budget_ratio=.2, budget_total=10000, migration=copy.deepcopy(binding))
    value['fl_initialization'] = dict(seed=seed, state_sha256=initial_sha256,
        initializer='saved migrated reference theta0' if seed == 2025 else 'dirichlet_screen_common.model(seed)',
        source_reference_seed=2025, source_reference_sha256=template['files_sha256']['source/initial_model.pt'],
        policy='seed2025 keeps original tensor state; other FL seeds freshly generated; shared across methods/stages')
    file = Path(__file__).resolve()
    value['files_sha256']['code/' + file.relative_to(ROOT).as_posix()] = c.digest(file)
    return value


def prepare(a):
    root, source, original, inputs = paths(a.migrated_root)
    expected, binding = migration_binding(root)
    mapped = SimpleNamespace(case=CASE, alpha=ALPHA, source_root=source,
        proxy_root=source if a.iteration in (0, 8000) else original, data_root=Path(a.data_root),
        output=Path(a.output).resolve(), method=a.route if a.route in ('fedavg', 'uniform_real') else
            'uniform_real' if a.route == 'history_uniform_zero' else 'residual_score',
        version='shared' if a.route in SHARED else f'old_i{a.iteration}', loss='original', iteration=a.iteration)
    if mapped.output.is_relative_to(root) or root.is_relative_to(mapped.output):
        raise RuntimeError('New FL output must not overlap the immutable migrated project')
    with migration_context(root, inputs, expected): prepared = old.prepare_fl(mapped)
    net = prepared[0] if a.seed == 2025 else initial_model(root, a.seed)
    template = prepared[-1]; template['rounds'] = a.rounds
    protocol = route_protocol(template, a.route, a.seed, a.iteration, binding, multi.state_sha256(net))
    return net, *prepared[1:-1], protocol


def verify_protocol(folder):
    protocol = c.read(Path(folder) / 'protocol.json')
    expected, binding = migration_binding(protocol['migration']['root'])
    net = initial_model(binding['root'], protocol['seed'])
    if protocol != route_protocol(protocol, protocol['route'], protocol['seed'], protocol['iteration'],
                                  binding, multi.state_sha256(net)):
        raise RuntimeError('Alpha=.05 protocol or numerical source changed')
    for name, sha in protocol['files_sha256'].items():
        if name.startswith('code/') and c.digest(ROOT / name[5:]) != sha:
            raise RuntimeError('Numerical code fingerprint changed: ' + name)
    return protocol


def verify(folder, expected_rounds=1500):
    marker = multi.base.verify(folder, expected_rounds)
    protocol = verify_protocol(folder)
    multi.replay(protocol, c.read(Path(folder) / 'rounds.json'))
    return marker


def train(a, prepared):
    # Preserve the existing update, seed, cache, checkpoint, replay and evaluation kernels.
    with patch.object(multi, 'verify', verify):
        return multi.train(a, prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrated-root', type=Path, default=DEFAULT_MIGRATED_ROOT)
    for name in ('data-root', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--iteration', type=int, choices=(0, 8000, 32000), required=True)
    parser.add_argument('--rounds', type=int, default=1500)
    parser.add_argument('--deadline', type=float); parser.add_argument('--run', action='store_true')
    a = parser.parse_args()
    if a.rounds != 1500: parser.error('Formal FL routes require 1500 rounds')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL training requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]), ensure_ascii=False)); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
