"""Clean HiGHS bound roundoff; retain legacy marginals, certificates and records."""
import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import joint_direction_sampling as legacy

_sample = legacy.joint_sample
_replay = legacy.replay_selection
BOUND_TOL = legacy.SOLVER_OPTIONS['primal_feasibility_tolerance']


def clean_result(result, bounds=(0., 1.)):
    if not result.success:
        return result, None
    if tuple(bounds) != (0., 1.):
        raise RuntimeError('Unexpected joint LP bounds')
    raw = np.asarray(result.x, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid joint LP variables')
    violation = np.maximum(-raw, raw - 1.)
    if float(violation.max()) > BOUND_TOL:
        raise RuntimeError(f'Joint LP bound violation exceeds solver tolerance: min={raw.min():.3e}, max={raw.max():.3e}, tolerance={BOUND_TOL:.3e}')
    clipped = np.clip(raw, 0., 1.)
    cleaned = copy.copy(result)
    cleaned.x = clipped
    changes = np.flatnonzero(raw != clipped)
    info = dict(version=1, tolerance=BOUND_TOL, raw_min=float(raw.min()),
        raw_max=float(raw.max()), clipped_count=int(len(changes)),
        clipping_l1=float(np.abs(clipped-raw).sum()),
        max_bound_violation=float(np.maximum(violation, 0.).max()),
        clipped_variables=[dict(index=int(i), raw=float(raw[i])) for i in changes],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return cleaned, info


def joint_sample(*args, **kwargs):
    solver = legacy.linprog
    repairs = []

    def solve(*a, **k):
        result, info = clean_result(solver(*a, **k), k['bounds'])
        repairs.append(info)
        return result

    with patch.object(legacy, 'linprog', solve):
        selected, pi, diagnostic = _sample(*args, **kwargs)
    if len(repairs) != 1 or repairs[0] is None:
        raise RuntimeError('Missing joint LP repair provenance')
    diagnostic['numerical_repair'] = repairs[0]
    return selected, pi, diagnostic


def replay_selection(diagnostic, weights, *args, **kwargs):
    selected, pi = _replay(diagnostic, weights, *args, **kwargs)
    if 'numerical_repair' not in diagnostic:
        return selected, pi  # Existing valid rounds retain their original provenance.
    info = diagnostic['numerical_repair']
    subsets, _ = legacy.combinations(len(weights), diagnostic['selected_count'])
    q = np.zeros(len(subsets))
    lookup = legacy._indices(len(weights), diagnostic['selected_count'])
    for item in diagnostic['lp_support']:
        q[lookup[tuple(item['clients'])]] = item['probability']
    x = q * diagnostic['raw_lp_mass'] / pi[subsets].min(axis=1)
    if np.any(x < -1e-12) or np.any(x > 1.+1e-12):
        raise RuntimeError('Saved cleaned LP variables violate bounds')
    raw = np.clip(x, 0., 1.)
    seen = set()
    for item in info.get('clipped_variables', []):
        i, value = item['index'], item['raw']
        if not isinstance(i, int) or i in seen or not 0 <= i < len(raw):
            raise RuntimeError('Invalid saved LP bound repair')
        if not np.isfinite(value) or not (value < 0. or value > 1.):
            raise RuntimeError('Invalid saved LP bound repair')
        if abs(raw[i] - np.clip(value, 0., 1.)) > 1e-12:
            raise RuntimeError('Saved LP repair differs from cleaned probabilities')
        raw[i] = value
        seen.add(i)
    result = type('Result', (), dict(success=True, x=raw))()
    _, expected = clean_result(result)
    if set(info) != set(expected):
        raise RuntimeError('Saved LP repair fields changed')
    for key, value in expected.items():
        if isinstance(value, float):
            valid = np.isclose(info[key], value, rtol=1e-10, atol=1e-12)
        else:
            valid = info[key] == value
        if not valid:
            raise RuntimeError('Saved LP repair provenance changed: '+key)
    return selected, pi


def install():
    legacy.joint_sample = joint_sample
    legacy.replay_selection = replay_selection
