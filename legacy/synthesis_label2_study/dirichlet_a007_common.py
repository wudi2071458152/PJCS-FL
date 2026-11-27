"""Scoped, immutable alpha=.07 inputs for the frozen Dirichlet kernels."""
import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import dirichlet_screen_common as c

ROOT = c.ROOT
ALPHA = '0p07'
CONFIG = 'weighted_weak_x10'
SEEDS = (2025, 2026)
FL_SEEDS = (2025, 2027, 2028, 2029, 2030)
INPUTS = ROOT / 'inputs/dirichlet_a007_20261014'
_protocol = c.protocol
_preflight = c.preflight


@contextmanager
def context(inputs_root=INPUTS, bind_protocol=True):
    changes = dict(INPUTS=Path(inputs_root), ALPHAS=(ALPHA,), CONFIGS=(CONFIG,),
                   SELECTED={ALPHA: list(range(32))}, SEEDS=SEEDS)
    if bind_protocol: changes['protocol'] = lambda: protocol(inputs_root)
    old = {name: getattr(c, name) for name in changes}
    for name, value in changes.items(): setattr(c, name, value)
    try: yield
    finally:
        for name, value in old.items(): setattr(c, name, value)


def protocol(inputs_root=INPUTS):
    inputs_root = Path(inputs_root)
    manifest = c.read(inputs_root / 'manifest.json')
    if c.digest(inputs_root / 'counts.json') != manifest['counts_sha256']:
        raise RuntimeError('Alpha=.07 counts fingerprint changed')
    with context(inputs_root, bind_protocol=False): value = _protocol()
    if set(value['budgets']) != {ALPHA}: raise RuntimeError('Unexpected alpha input coverage')
    value.update(suite='dirichlet_a007_20261014', iterations=[8000, 32000],
        budget_total=10000, partition=dict(alpha=.07, seed=42, min_per_client=1, clients=32),
        synthesis_order='all clients to 32000; same images and momentum through 8000',
        diagnostic_clients=list(range(32)), diagnostic_contexts_per_stage=384,
        classifier_evaluation_enabled=False)
    # Retain the frozen classifier metadata only as provenance, never schedule it.
    for name in ('dirichlet_a007_common.py', 'dirichlet_dsa4000_worker.py'):
        rel = 'synthesis_label2_study/' + name
        value['code_sha256'][rel] = c.digest(ROOT / rel)
    return json.loads(json.dumps(value))


def preflight(data_root, inputs_root=INPUTS):
    expected = protocol(inputs_root)
    with context(inputs_root):
        result = _preflight(data_root)
        saved = c.load(Path(inputs_root) / 'splits/alpha0p07.pt')
        if any(saved.get(k) != v for k, v in dict(num_clients=32, alpha=.07, seed=42, min_per_client=1).items()):
            raise RuntimeError('Frozen partition definition changed')
        table = expected['budgets'][ALPHA]
        real, syn = np.asarray(table['real_counts']), np.asarray(table['synthetic_counts'])
        if real.sum() != 50000 or syn.sum() != 10000 or np.any(syn > real) or np.any((real > 0) & (syn < 1)):
            raise RuntimeError('Invalid fixed 20% image budget or class capacity')
    result.update(selected={ALPHA: list(range(32))}, synthesis_jobs=32,
                  calibration='all 32 clients; 384 contexts', fl_jobs=90,
                  vector_contexts_per_stage=384, classifier_evaluation=False)
    return result
