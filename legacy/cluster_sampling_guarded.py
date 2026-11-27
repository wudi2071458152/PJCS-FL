"""Quota size-mixture ablation; existing clustering and AOCS remain unchanged."""
import numpy as np
from cluster_sampling import quotas as original_quotas


def quotas(sizes, spread, m, min1, rho=0.1):
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError('quota-rho must be in [0, 1]')
    if rho == 0:
        return original_quotas(sizes, spread, m, min1)
    sizes = np.asarray(sizes, dtype=float)
    spread = np.asarray(spread, dtype=float)
    if np.any(sizes <= 0) or not np.isfinite(spread).all() or np.any(spread < 0):
        raise ValueError('Invalid sizes or dispersion')
    weighted_mean = np.dot(sizes, spread)/sizes.sum()
    effective = np.ones_like(spread) if weighted_mean <= 1e-15 else (
        (1-rho)*spread/weighted_mean + rho)
    # original_quotas multiplies by sizes, then applies capacity-constrained allocation.
    return original_quotas(sizes, effective, m, min1)
