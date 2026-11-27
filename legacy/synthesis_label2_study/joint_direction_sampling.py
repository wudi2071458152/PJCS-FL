"""Optimize fixed-size joint sampling while retaining the original AOCS marginals."""
import itertools
import time
import warnings
from functools import lru_cache

import numpy as np
import scipy
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import csr_matrix
import torch

from run_dirichlet_proxy_fl import fixed_aocs_sample

TARGETS = ('representative', 'diversity')
ETAS = (.25, .5, .75, 1.)
MARGINAL_RTOL = 5e-8
CERTIFICATE_TOL = 1e-7
LEGACY_NORM_RTOL = 1e-5
SOLVER_OPTIONS = dict(threads=1, primal_feasibility_tolerance=1e-9,
                      dual_feasibility_tolerance=1e-9, ipm_optimality_tolerance=1e-10)
FORMULATION = 'q_S=min(pi_i in S)*x_S; 0<=x_S<=1; (A/pi)*minpi*x=1'


@lru_cache(maxsize=None)
def combinations(n, m):
    subsets = np.asarray(list(itertools.combinations(range(n), m)), dtype=int)
    rows = subsets.ravel(); cols = np.repeat(np.arange(len(subsets)), m)
    incidence = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(subsets)))
    return subsets, incidence


@lru_cache(maxsize=None)
def _indices(n, m):
    return {tuple(row): i for i, row in enumerate(combinations(n, m)[0])}


def _parameters(weights, n, m, target, eta):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (n,) or n < m or m < 1 or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.)
            or target not in TARGETS or eta not in (0.,) + ETAS):
        raise ValueError('Invalid joint-sampling weights, target, eta or sample size')
    return weights


def _objectives(gram, weights, pi, subsets):
    n = len(weights)
    if gram.shape != (n, n) or not np.isfinite(gram).all() or np.any(np.diag(gram) < 0):
        raise RuntimeError('Invalid proxy Gram matrix')
    factors = weights / pi
    scaled = gram * factors[:, None] * factors[None, :]
    cross = factors * (gram @ weights)
    representative = np.diag(scaled)[subsets].sum(axis=1) - 2 * cross[subsets].sum(axis=1)
    representative += float(weights @ gram @ weights)
    norms = np.sqrt(np.diag(gram)); denom = norms[:, None] * norms[None, :]
    cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    cosine = np.clip(cosine, -1., 1.)
    diversity = np.zeros(len(subsets))
    for i, j in itertools.combinations(range(subsets.shape[1]), 2):
        representative += 2 * scaled[subsets[:, i], subsets[:, j]]
        diversity += np.maximum(cosine[subsets[:, i], subsets[:, j]], 0.) ** 2
    scale = max(float(np.abs(scaled).max()), float(np.abs(representative).max()), 1e-300)
    if not np.isfinite(representative).all() or np.any(representative < -1e-8 * scale):
        raise RuntimeError('Nonfinite or negative representative squared error')
    return np.maximum(representative, 0.), diversity, cosine


def _scaled_objective(cost):
    offset = float(cost.min()); scale = max(float((cost - offset).max()), 1e-300)
    # A constant objective needs no division by a tiny artificial scale.
    if scale == 1e-300: scale = 1.
    return (cost - offset) / scale, offset, scale


def _marginals(q, incidence, pi):
    if not np.isfinite(q).all() or np.any(q < 0) or abs(float(q.sum()) - 1.) > 1e-10:
        raise RuntimeError('Invalid optimized combination probabilities')
    actual = np.asarray(incidence @ q).ravel()
    relative = np.abs(actual - pi) / pi
    if np.any(actual <= 0) or float(relative.max()) > MARGINAL_RTOL:
        raise RuntimeError('Optimized combinations do not retain positive original marginals')
    return float(relative.max()), float(np.abs(actual - pi).max())


def _certificate(q, scaled_cost, scaled_incidence, variable_scale, dual):
    dual = np.asarray(dual, dtype=np.float64)
    if dual.shape != (scaled_incidence.shape[0],) or not np.isfinite(dual).all():
        raise RuntimeError('Invalid LP optimality certificate')
    reduced = scaled_cost * variable_scale - np.asarray(scaled_incidence.T @ dual).ravel()
    # x_S in [0,1]: negative reduced costs contribute their upper-bound dual term.
    primal = float(q @ scaled_cost); dual_value = float(dual.sum() + np.minimum(reduced, 0.).sum())
    if abs(primal - dual_value) > CERTIFICATE_TOL:
        raise RuntimeError('Invalid LP optimality certificate or primal-dual gap')
    return primal, dual_value, float(reduced.min())


def _draw(q, subsets, baseline, seed, eta):
    branch = 'optimized' if np.random.default_rng(seed + 600000).random() < eta else 'original'
    if branch == 'original': return baseline, branch
    index = int(np.random.default_rng(seed + 700000).choice(len(q), p=q))
    return subsets[index].tolist(), branch


def joint_sample(proxies, weights, target, eta, seed, selected_count=3):
    """Return selected clients, unchanged original pi, and a replayable LP certificate."""
    started = time.perf_counter(); n = len(proxies)
    weights = _parameters(weights, n, selected_count, target, eta)
    if (not proxies or any(not isinstance(p, torch.Tensor) or p.numel() != proxies[0].numel()
                           or not torch.isfinite(p).all() for p in proxies)):
        raise ValueError('Invalid or nonfinite proxy vectors')
    # Preserve the legacy float(P.norm()) score, separately from double Gram geometry.
    norms = np.asarray([float(p.norm()) for p in proxies]); scores = norms / (3 * .01)
    importance = weights * scores
    baseline, pi = fixed_aocs_sample(importance, selected_count, np.random.default_rng(seed + 400000))
    if not np.isfinite(pi).all() or np.any(pi <= 0): raise RuntimeError('Nonpositive original AOCS marginal')
    matrix = torch.stack([p.detach().reshape(-1).double() for p in proxies])
    gram = (matrix @ matrix.T).cpu().numpy()
    subsets, incidence = combinations(n, selected_count)
    costs, penalties, cosine = _objectives(gram, weights, pi, subsets)
    objective = costs if target == 'representative' else penalties
    scaled_cost, offset, scale = _scaled_objective(objective)
    # q_S <= min(pi_i in S) is inherent. Scaling q_S=d_S*x_S keeps x and coefficients
    # in [0,1], preserving tiny marginals without ill-conditioned 1/pi coefficients.
    variable_scale = pi[subsets].min(axis=1)
    scaled_incidence = incidence.multiply((1. / pi)[:, None]).multiply(variable_scale[None, :]).tocsr()
    geometry_seconds = time.perf_counter() - started; solve_started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads', category=OptimizeWarning)
        result = linprog(scaled_cost * variable_scale, A_eq=scaled_incidence, b_eq=np.ones(n), bounds=(0., 1.),
                         method='highs-ds', options=SOLVER_OPTIONS)
    solve_seconds = time.perf_counter() - solve_started
    if not result.success: raise RuntimeError('Joint sampling LP failed: ' + result.message)
    q = np.asarray(result.x) * variable_scale; raw_mass = float(q.sum())
    if not np.isfinite(raw_mass) or raw_mass <= 0 or abs(raw_mass - 1.) > 1e-8:
        raise RuntimeError('Invalid optimized probability mass')
    # Explicit roundoff normalization is recorded and followed by relative marginal checks.
    q = q / raw_mass
    relative_error, absolute_error = _marginals(q, incidence, pi)
    primal, dual_value, reduced_min = _certificate(q, scaled_cost, scaled_incidence, variable_scale, result.eqlin.marginals)
    selected, branch = _draw(q, subsets, baseline, seed, eta)
    lookup = _indices(n, selected_count)
    index = lookup[tuple(selected)]; baseline_index = lookup[tuple(baseline)]
    support = [dict(clients=subsets[i].tolist(), probability=float(q[i]),
                    representative_cost=float(costs[i]), direction_penalty=float(penalties[i]))
               for i in np.flatnonzero(q > 0)]
    diagnostic = dict(target=target, eta=float(eta), seed=int(seed), clients=n, selected_count=selected_count,
        selected=selected, branch=branch, original_selected=baseline, pi=pi.tolist(), scores=scores.tolist(),
        formulation=FORMULATION,
        importance=importance.tolist(), proxy_norms=norms.tolist(), gram_upper=gram[np.triu_indices(n)].tolist(),
        zero_proxy_vectors=int(np.count_nonzero(np.diag(gram) == 0)), lp_support=support,
        raw_lp_mass=raw_mass, marginal_max_relative_error=relative_error, marginal_max_absolute_error=absolute_error,
        selected_representative_cost=float(costs[index]), selected_direction_penalty=float(penalties[index]),
        original_representative_cost=float(costs[baseline_index]), original_direction_penalty=float(penalties[baseline_index]),
        selected_cosines=[float(cosine[i, j]) for i, j in itertools.combinations(selected, 2)],
        expected_optimized_representative_cost=float(q @ costs), expected_optimized_direction_penalty=float(q @ penalties),
        objective_offset=offset, objective_scale=scale, constant_objective=bool(np.ptp(objective) == 0),
        lp_dual=result.eqlin.marginals.tolist(), lp_primal_value=primal, lp_dual_value=dual_value,
        lp_min_reduced_cost=reduced_min, solver_status=int(result.status), solver_iterations=int(result.nit),
        scipy_version=scipy.__version__, geometry_seconds=geometry_seconds, solve_seconds=solve_seconds,
        selection_seconds=time.perf_counter() - started)
    return selected, pi, diagnostic


def replay_selection(diagnostic, weights, seed, target, eta, selected_count=3):
    """Verify marginals, objectives, LP dual certificate and seeded draw without re-solving."""
    d = diagnostic; n = len(weights); weights = _parameters(weights, n, selected_count, target, eta)
    if any(d[k] != v for k, v in dict(seed=seed, target=target, eta=eta, clients=n, formulation=FORMULATION,
                                     selected_count=selected_count).items()):
        raise RuntimeError('Joint selection configuration changed')
    norms = np.asarray(d['proxy_norms'], dtype=np.float64); scores = norms / (3 * .01)
    if norms.shape != (n,) or not np.isfinite(norms).all() or np.any(norms < 0):
        raise RuntimeError('Invalid saved proxy norms')
    if not np.array_equal(scores, d['scores']) or not np.array_equal(weights * scores, d['importance']):
        raise RuntimeError('Original importance or scores changed')
    baseline, pi = fixed_aocs_sample(weights * scores, selected_count, np.random.default_rng(seed + 400000))
    if not np.array_equal(pi, d['pi']) or baseline != d['original_selected']:
        raise RuntimeError('Original AOCS probabilities or sample changed')
    upper = np.asarray(d['gram_upper'], dtype=np.float64)
    if upper.shape != (n * (n + 1) // 2,): raise RuntimeError('Invalid saved Gram dimensions')
    gram = np.zeros((n, n)); gram[np.triu_indices(n)] = upper
    gram += np.triu(gram, 1).T
    if not np.isfinite(gram).all() or np.any(np.diag(gram) < 0): raise RuntimeError('Invalid saved Gram values')
    spectral_scale = max(float(np.abs(gram).max()), 1e-300)
    if float(np.linalg.eigvalsh(gram).min()) < -1e-8 * spectral_scale:
        raise RuntimeError('Saved Gram is not positive semidefinite')
    # A 320010-dimensional float32 torch.norm can differ from double geometry by
    # several ppm; float32 squared norms can also underflow for tiny vectors.
    if not np.allclose(np.sqrt(np.diag(gram)), norms, rtol=LEGACY_NORM_RTOL, atol=1e-20):
        raise RuntimeError('Saved Gram differs from original proxy norms')
    subsets, incidence = combinations(n, selected_count)
    costs, penalties, cosine = _objectives(gram, weights, pi, subsets)
    q = np.zeros(len(subsets)); lookup = _indices(n, selected_count)
    for item in d['lp_support']:
        clients = tuple(item['clients']); index = lookup.get(clients)
        if index is None or q[index] != 0 or item['probability'] <= 0:
            raise RuntimeError('Invalid or duplicate optimized support')
        q[index] = item['probability']
        for key, value in [('representative_cost', costs[index]), ('direction_penalty', penalties[index])]:
            if not np.isclose(item[key], value, rtol=1e-10, atol=1e-12):
                raise RuntimeError('Saved support objective changed')
    relative_error, absolute_error = _marginals(q, incidence, pi)
    if not np.isfinite(d['raw_lp_mass']) or abs(d['raw_lp_mass'] - 1.) > 1e-8:
        raise RuntimeError('Invalid saved pre-normalization probability mass')
    objective = costs if target == 'representative' else penalties
    scaled_cost, offset, scale = _scaled_objective(objective)
    variable_scale = pi[subsets].min(axis=1)
    primal, dual_value, reduced_min = _certificate(q, scaled_cost,
        incidence.multiply((1. / pi)[:, None]).multiply(variable_scale[None, :]).tocsr(),
        variable_scale, d['lp_dual'])
    selected, branch = _draw(q, subsets, baseline, seed, eta)
    index = lookup[tuple(selected)]; original_index = lookup[tuple(baseline)]
    checks = dict(objective_offset=offset, objective_scale=scale, lp_primal_value=primal,
        lp_dual_value=dual_value, lp_min_reduced_cost=reduced_min,
        marginal_max_relative_error=relative_error, marginal_max_absolute_error=absolute_error,
        selected_representative_cost=float(costs[index]), selected_direction_penalty=float(penalties[index]),
        original_representative_cost=float(costs[original_index]), original_direction_penalty=float(penalties[original_index]),
        expected_optimized_representative_cost=float(q @ costs), expected_optimized_direction_penalty=float(q @ penalties))
    if any(not np.isclose(d[k], v, rtol=1e-10, atol=1e-12) for k, v in checks.items()):
        raise RuntimeError('Saved joint sampling objective or certificate changed')
    if (selected != d['selected'] or branch != d['branch'] or d['solver_status'] != 0
            or d['zero_proxy_vectors'] != int(np.count_nonzero(np.diag(gram) == 0))
            or d['constant_objective'] != bool(np.ptp(objective) == 0)
            or not np.array_equal(d['selected_cosines'], [cosine[i, j] for i, j in itertools.combinations(selected, 2)])):
        raise RuntimeError('Saved joint sampling draw or geometry changed')
    return selected, pi
