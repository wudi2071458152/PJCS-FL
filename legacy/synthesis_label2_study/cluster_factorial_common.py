"""Explicit quota mixing and exact full-direction representation for the factorial study."""
import numpy as np
from cluster_sampling_guarded import quotas


def mixed_quotas(sizes, spread, mixing):
    if not 0 <= mixing <= 1:
        raise ValueError('quota mixing must be in [0,1]')
    sizes=np.asarray(sizes,dtype=float)
    original=quotas(sizes,spread,3,False,.1)
    if mixing==0:
        return original
    mass=(1-mixing)*original+mixing*3/len(sizes)
    q=np.zeros(len(sizes)); active=np.ones(len(sizes),dtype=bool); budget=3.
    while active.any():
        ids=np.flatnonzero(active)
        allocation=budget*mass[ids]/mass[ids].sum()
        saturated=allocation>=sizes[ids]
        if not saturated.any():
            q[ids]=allocation; break
        full=ids[saturated]; q[full]=sizes[full]
        budget-=sizes[full].sum(); active[full]=False
    close=np.abs(q-np.rint(q))<1e-10
    q[close]=np.rint(q[close])
    return q


def full_direction_features(vectors):
    x=np.asarray(vectors,dtype=np.float64)
    values,basis=np.linalg.eigh(x@x.T)
    return basis*np.sqrt(np.maximum(values,0))[None,:]
