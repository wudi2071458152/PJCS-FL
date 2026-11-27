"""Proxy-direction clustering and capacity-limited stratified sampling (NumPy only)."""
import numpy as np


def spherical_clusters(features, k, seed):
    norms = np.linalg.norm(features, axis=1)
    z = features / np.maximum(norms[:, None], 1e-15)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(5):
        centers = z[rng.choice(len(z), k, replace=False)].copy()
        previous = None
        for _ in range(50):
            labels = np.argmax(z @ centers.T, axis=1)
            # Repair empty clusters using a poorly represented member of a non-singleton.
            counts = np.bincount(labels, minlength=k)
            for h in np.flatnonzero(counts == 0):
                eligible = np.flatnonzero(counts[labels] > 1)
                i = eligible[np.argmin(np.sum(z[eligible] * centers[labels[eligible]], axis=1))]
                counts[labels[i]] -= 1
                labels[i] = h
                counts[h] += 1
            centers = np.stack([z[labels == h].sum(axis=0) for h in range(k)])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-15)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
        cost = float(np.sum(1 - np.sum(z * centers[labels], axis=1)))
        if best is None or cost < best[0]:
            best = cost, labels.copy(), centers.copy()
    _, labels, centers = best
    sizes = np.bincount(labels, minlength=k)
    spread = np.array([np.maximum(0, 1 - z[labels == h] @ centers[h]).mean() for h in range(k)])
    return labels, sizes, spread


def quotas(sizes, spread, m, min1):
    sizes = np.asarray(sizes, dtype=float)
    lower = np.full(len(sizes), int(min1), dtype=float)
    if not lower.sum() <= m <= sizes.sum():
        raise ValueError('Invalid quota budget')
    # Positive floor prevents permanent zero inclusion; all-zero dispersion falls back to size.
    a = sizes * np.maximum(spread, 1e-6)
    q = lower.copy()
    active = sizes > q
    remaining = m - q.sum()
    while remaining > 1e-12 and active.any():
        ids = np.flatnonzero(active)
        proposed = remaining * a[ids] / a[ids].sum()
        capacity = sizes[ids] - q[ids]
        saturated = proposed >= capacity
        if not saturated.any():
            q[ids] += proposed
            break
        full = ids[saturated]
        remaining -= (sizes[full] - q[full]).sum()
        q[full] = sizes[full]
        active[full] = False
    q[np.abs(q - np.rint(q)) < 1e-10] = np.rint(q[np.abs(q - np.rint(q)) < 1e-10])
    return q


def round_quotas(q, rng):
    floor = np.floor(q).astype(int)
    f = q - floor
    while True:
        ids = np.flatnonzero((f > 1e-12) & (f < 1 - 1e-12))
        if len(ids) < 2:
            break
        i, j = rng.choice(ids, 2, replace=False)
        a, b = min(1-f[i], f[j]), min(f[i], 1-f[j])
        if rng.random() < b / (a+b):
            f[i] += a
            f[j] -= a
        else:
            f[i] -= b
            f[j] += b
    result = floor + (f > .5)
    if result.sum() != round(q.sum()):
        raise ValueError('Quota rounding lost budget')
    return result


def stratified_sample(labels, q, scores, use_aocs, rng, aocs_sample):
    counts = round_quotas(q, rng)
    selected, pi = [], np.zeros(len(labels))
    for h, count in enumerate(counts):
        members = np.flatnonzero(labels == h)
        if use_aocs:
            lo = int(np.floor(q[h]))
            fraction = q[h] - lo
            # Deterministic helper draws are discarded; returned probabilities are exact marginals.
            _, p0 = aocs_sample(scores[members], lo, np.random.default_rng(0))
            p1 = p0
            if fraction > 0:
                _, p1 = aocs_sample(scores[members], lo+1, np.random.default_rng(0))
            pi[members] = (1-fraction)*p0 + fraction*p1
            ids, _ = aocs_sample(scores[members], int(count), rng)
        else:
            pi[members] = q[h] / len(members)
            ids = rng.choice(len(members), int(count), replace=False)
        selected.extend(members[ids].tolist())
    return sorted(selected), pi, counts
