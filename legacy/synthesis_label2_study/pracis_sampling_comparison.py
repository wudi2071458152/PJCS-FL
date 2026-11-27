"""NumPy probability and coefficient helpers for the four Windows comparisons."""
import numpy as np


def normalize_importance(importance):
    """Normalize nonnegative scores without adding floors or sample-size factors."""
    value = np.asarray(importance, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Importance must be a finite nonnegative vector')
    maximum = float(value.max())
    if maximum == 0:
        raise ValueError('Cannot normalize all-zero importance')
    scaled = value / maximum
    return scaled / scaled.sum()


def _probabilities(value):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('Sampling probabilities must be nonnegative and sum to one')
    return p


def _selection(selected, n):
    ids = np.asarray(selected)
    if ids.ndim != 1 or (len(ids) and ids.dtype.kind not in 'iu'):
        raise ValueError('Selected clients must be integer indices')
    ids = ids.astype(int)
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n):
        raise ValueError('Selected clients must be distinct valid indices')
    return ids


def _budget(p, m):
    if not isinstance(m, (int, np.integer)) or not 0 <= m <= 3 or np.count_nonzero(p) < m:
        raise ValueError('WOR needs m in [0, 3] and at least m positive weights')


def weighted_wor_inclusion(probabilities, m):
    """Exact inclusion marginals of successive PPS draws without replacement.

    Sum the probability of each client occurring at draw 1, 2, or 3. Zero
    weights remain zero; no Monte Carlo approximation or epsilon floor is used.
    """
    p = _probabilities(probabilities)
    _budget(p, m)
    pi = np.zeros(len(p))
    if m == 0:
        return pi
    support = np.flatnonzero(p > 0)
    if len(support) == m:
        pi[support] = 1.
        return pi
    first = p / p.sum()
    pi += first
    if m >= 2:
        remaining = np.broadcast_to(p, (len(p), len(p))).copy()
        np.fill_diagonal(remaining, 0.)
        second = first[:, None] * remaining / remaining.sum(axis=1, keepdims=True)
        pi += second.sum(axis=0)
        if m == 3:
            ids = np.arange(len(p))
            eligible = ((ids[:, None, None] != ids[None, None, :]) &
                        (ids[None, :, None] != ids[None, None, :]))
            third_weights = p[None, None, :] * eligible
            third = second[:, :, None] * third_weights / third_weights.sum(axis=2, keepdims=True)
            pi += third.sum(axis=(0, 1))
    return pi


def weighted_wor_sample(probabilities, m, rng):
    """Draw successive PPS clients in draw order and return exact marginals.

    This implements the weighted WOR law. Callers preserving an author's exact
    Torch random stream may draw with torch.multinomial and use the helper above.
    """
    p = _probabilities(probabilities)
    pi = weighted_wor_inclusion(p, m)
    remaining = p.copy()
    selected = []
    for _ in range(m):
        i = int(rng.choice(len(p), p=remaining / remaining.sum()))
        selected.append(i)
        remaining[i] = 0.
    return selected, pi


def native_coefficients(batches, selected, probabilities):
    """Author coefficients B_i / (N * p_i * sum_selected B), without renormalization."""
    p = _probabilities(probabilities)
    ids = _selection(selected, len(p))
    counts = np.asarray(batches, dtype=np.float64)
    if counts.shape != p.shape or not np.isfinite(counts).all() or np.any(counts <= 0) or not len(ids) or np.any(p[ids] <= 0):
        raise ValueError('Native aggregation needs positive batches and selected probabilities')
    return counts[ids] / (len(p) * p[ids] * counts[ids].sum())


def ht_coefficients(weights, selected, inclusion):
    """Global sample fractions divided by actual selected-client inclusion marginals."""
    w = _probabilities(weights)
    ids = _selection(selected, len(w))
    pi = np.asarray(inclusion, dtype=np.float64)
    if pi.shape != w.shape or not np.isfinite(pi).all() or np.any(pi < 0) or np.any(pi > 1 + 1e-12) or not np.isclose(pi.sum(), len(ids), atol=1e-12, rtol=0) or np.any(pi[ids] <= 0) or np.any((w > 0) & (pi == 0)):
        raise ValueError('HT aggregation needs valid actual inclusion probabilities')
    return w[ids] / pi[ids]


def pracis_feedback(probabilities, selected, scores):
    """Redistribute selected old mass by scores; keep old values if all scores are zero."""
    p = _probabilities(probabilities)
    ids = _selection(selected, len(p))
    value = np.asarray(scores, dtype=np.float64)
    if value.shape != ids.shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Feedback needs finite nonnegative scores for selected clients')
    updated = p.copy()
    if len(value) and np.any(value > 0):
        updated[ids] = p[ids].sum() * normalize_importance(value)
    return updated
