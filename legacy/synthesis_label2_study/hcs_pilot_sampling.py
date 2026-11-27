"""HCS pilot geometry, capacity-limited quotas and source-style weighted sampling."""
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from cluster_sampling import round_quotas


def make_sketch(dimension, seed=602025):
    """One shared, fixed 256-bucket signed CountSketch for all clients and rounds."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=dimension), rng.choice([-1., 1.], size=dimension)


def represent(updates, representation, sketch=None):
    vectors = np.asarray(updates, dtype=np.float64)
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError('Updates must be a finite nonempty matrix')
    if representation == 'gram':
        gram = vectors @ vectors.T
        values, basis = np.linalg.eigh(gram)
        tolerance = np.finfo(float).eps * len(gram) * np.max(np.abs(values), initial=0.)
        values[values < tolerance] = 0.
        features = basis * np.sqrt(np.maximum(values, 0.))[None, :]
        # Numerical eigensolver noise must not distinguish identical updates or zeros.
        _, first, inverse = np.unique(gram, axis=0, return_index=True, return_inverse=True)
        features = features[first][inverse]
        features[np.all(gram == 0, axis=1)] = 0.
        return features
    if representation == 'countsketch':
        buckets, signs = make_sketch(vectors.shape[1]) if sketch is None else sketch
        buckets, signs = np.asarray(buckets), np.asarray(signs)
        if (buckets.shape != (vectors.shape[1],) or signs.shape != buckets.shape
                or np.any(buckets < 0) or np.any(buckets >= 256)
                or not np.isin(signs, [-1., 1.]).all()):
            raise ValueError('Invalid shared CountSketch mapping')
        return np.stack([np.bincount(buckets, weights=row * signs, minlength=256)
                         for row in vectors])
    raise ValueError('Unknown representation')


def expected_quotas(sizes, dispersion, count=3):
    """Cap proportional quotas and redistribute until the complete budget is used."""
    sizes, dispersion = np.asarray(sizes, float), np.asarray(dispersion, float)
    if (sizes.shape != dispersion.shape or not np.isfinite(sizes).all() or np.any(sizes < 0)
            or np.any(dispersion < 0) or not np.isfinite(dispersion).all()
            or not 0 <= count <= sizes.sum()):
        raise ValueError('Invalid quota inputs')
    quota = np.zeros_like(sizes)
    while count - quota.sum() > 1e-12:
        capacity = sizes - quota
        active = capacity > 1e-12
        allocation_weights = np.where(active, sizes * dispersion, 0.)
        if allocation_weights.sum() == 0:
            allocation_weights = np.where(active, capacity, 0.)
        proposed = (count - quota.sum()) * allocation_weights / allocation_weights.sum()
        quota += np.minimum(proposed, capacity)
    return quota


def geometry(updates, representation, sketch=None, seed=502025):
    features = represent(updates, representation, sketch)
    if len(features) < 3:
        raise ValueError('The pilot selects three clients')
    unique, inverse = np.unique(features, axis=0, return_inverse=True)
    if len(unique) <= 3:
        labels = inverse
    else:
        indices = np.random.default_rng(seed).choice(len(features), 3, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            labels = KMeans(n_clusters=3, init=features[indices], n_init=1,
                            max_iter=300, tol=1e-12, random_state=0).fit_predict(features)
    sizes = np.bincount(labels, minlength=3)
    dispersion = np.zeros(3)
    for h, size in enumerate(sizes):
        members = features[labels == h]
        if size > 1 and not np.all(members == members[0]):
            # Equivalent to the sum over ALL ordered pairs, including zero diagonal.
            centered = members - members.mean(axis=0)
            dispersion[h] = 2 * size * np.square(centered).sum() / (size - 1 + 1e-6)
    return dict(labels=labels.tolist(), sizes=sizes.tolist(),
                dispersion=dispersion.tolist(),
                expected_quotas=expected_quotas(sizes, dispersion).tolist(),
                scores=np.linalg.norm(features, axis=1).tolist(),
                effective_clusters=int(np.count_nonzero(sizes)))


def validate_geometry(state):
    """Check saved geometry/quota consistency without needing full cached updates."""
    labels, sizes = np.asarray(state['labels']), np.asarray(state['sizes'])
    scores = np.asarray(state['scores'], float)
    dispersion, quota = np.asarray(state['dispersion'], float), np.asarray(state['expected_quotas'], float)
    if (labels.ndim != 1 or sizes.shape != (3,) or len(labels) < 3
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= 3)
            or not np.array_equal(np.bincount(labels, minlength=3), sizes)
            or scores.shape != labels.shape or np.any(scores < 0) or not np.isfinite(scores).all()
            or state['effective_clusters'] != int(np.count_nonzero(sizes))
            or quota.shape != (3,) or not np.allclose(quota, expected_quotas(sizes, dispersion), atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid saved HCS geometry')
    return True


def sample(state, seed):
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(state['labels']), np.asarray(state['scores'], float)
    if scores.shape != labels.shape or np.any(scores < 0) or not np.isfinite(scores).all():
        raise ValueError('Invalid sampling scores')
    counts = round_quotas(np.asarray(state['expected_quotas'], float), rng)
    selected, fallback = [], []
    for h, count in enumerate(counts):
        if count == 0:
            continue
        members = np.flatnonzero(labels == h)
        if count > len(members):
            raise ValueError('Quota exceeds cluster capacity')
        positive = members[scores[members] > 0]
        if len(positive) < count:
            zero = members[scores[members] == 0]
            chosen = np.concatenate([positive, rng.choice(zero, count - len(positive), replace=False)])
            fallback.append(dict(cluster=h, kind='all_zero_uniform' if not len(positive)
                                 else 'positive_then_zero_uniform', positive_count=len(positive),
                                 zero_selected=int(count - len(positive))))
        else:
            probability = scores[members] / scores[members].sum()
            chosen = rng.choice(members, count, replace=False, p=probability)
        selected.extend(chosen.tolist())
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError('Sampling lost the three-client budget')
    return sorted(selected), dict(actual_quotas=counts.tolist(), zero_score_fallback=fallback)


def aggregation_factors(weights, selected, aggregation):
    weights, selected = np.asarray(weights, float), np.asarray(selected, int)
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError('Invalid selected clients')
    if aggregation == 'equal':
        return np.full(len(selected), 1. / len(selected))
    if aggregation == 'datasize':
        chosen = weights[selected]
        if not np.isfinite(chosen).all() or np.any(chosen <= 0):
            raise ValueError('Selected sample weights must be positive and finite')
        return chosen / chosen.sum()
    raise ValueError('Unknown aggregation')
