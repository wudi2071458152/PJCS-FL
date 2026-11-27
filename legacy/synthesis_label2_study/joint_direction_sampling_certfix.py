"""Retry the identical LP once at tighter dual precision after a certificate gap."""
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import joint_direction_sampling as legacy
import joint_direction_sampling_nonnegativefix as previous

DUAL_TOL = 1e-10
MESSAGE = 'Invalid LP optimality certificate or primal-dual gap'
_certificate = legacy._certificate
_sample = previous.joint_sample
_replay = previous.replay_selection


def failure_info(q, cost, matrix, scale, dual):
    reduced = cost*scale-np.asarray(matrix.T@dual).ravel()
    primal = float(q@cost)
    bound = float(np.asarray(dual).sum()+np.minimum(reduced,0.).sum())
    return dict(version=1, dual_feasibility_tolerance=DUAL_TOL,
        first_primal=primal, first_dual_value=bound, failure_gap=primal-bound,
        first_support=[dict(index=int(i),probability=float(q[i])) for i in np.flatnonzero(q>0)],
        first_lp_dual=np.asarray(dual).tolist(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def joint_sample(*args, **kwargs):
    solver = legacy.linprog
    inputs = []; failures = []

    def first_solve(*a, **k):
        inputs.append((a,k))
        return solver(*a,**k)

    def inspect(q,cost,matrix,scale,dual):
        try:
            return _certificate(q,cost,matrix,scale,dual)
        except RuntimeError as exc:
            if str(exc)==MESSAGE:
                failures.append(failure_info(q,cost,matrix,scale,dual))
            raise

    try:
        with patch.object(legacy,'linprog',first_solve), patch.object(legacy,'_certificate',inspect):
            return _sample(*args,**kwargs)
    except RuntimeError as exc:
        if str(exc)!=MESSAGE or len(inputs)!=1 or len(failures)!=1:
            raise

    def strict_solve(*a, **k):
        original_args, original_kwargs = inputs[0]
        same = (len(a)==len(original_args)==1 and np.array_equal(a[0],original_args[0])
            and k.keys()==original_kwargs.keys() and k['A_eq'].shape==original_kwargs['A_eq'].shape
            and (k['A_eq']!=original_kwargs['A_eq']).nnz==0
            and np.array_equal(k['b_eq'],original_kwargs['b_eq'])
            and all(k[key]==original_kwargs[key] for key in ('bounds','method','options')))
        if not same:
            raise RuntimeError('Certificate retry attempted a different LP')
        options=dict(k['options'],dual_feasibility_tolerance=DUAL_TOL)
        return solver(*a,**dict(k,options=options))

    with patch.object(legacy,'linprog',strict_solve):
        selected,pi,diagnostic = _sample(*args,**kwargs)
    diagnostic['certificate_retry']=failures[0]
    return selected,pi,diagnostic


def replay_selection(diagnostic, weights, seed, target, eta, selected_count=3):
    selected,pi = _replay(diagnostic,weights,seed,target,eta,selected_count)
    if 'certificate_retry' not in diagnostic:
        return selected,pi
    n=len(weights); upper=np.asarray(diagnostic['gram_upper'])
    gram=np.zeros((n,n)); gram[np.triu_indices(n)]=upper; gram+=np.triu(gram,1).T
    subsets,incidence=legacy.combinations(n,selected_count)
    representative,diversity,_=legacy._objectives(gram,np.asarray(weights),pi,subsets)
    cost,_,_=legacy._scaled_objective(representative if target=='representative' else diversity)
    scale=pi[subsets].min(axis=1)
    matrix=incidence.multiply((1./pi)[:,None]).multiply(scale[None,:]).tocsr()
    info=diagnostic['certificate_retry']; q=np.zeros(len(subsets))
    for item in info['first_support']:
        i,p=item['index'],item['probability']
        if not isinstance(i,int) or not 0<=i<len(q) or q[i]!=0 or not np.isfinite(p) or p<=0:
            raise RuntimeError('Invalid first-attempt LP support')
        q[i]=p
    legacy._marginals(q,incidence,pi)
    dual=np.asarray(info['first_lp_dual'])
    try:
        _certificate(q,cost,matrix,scale,dual)
    except RuntimeError as exc:
        if str(exc)!=MESSAGE:
            raise
    else:
        raise RuntimeError('Saved first-attempt certificate did not fail')
    expected=failure_info(q,cost,matrix,scale,dual)
    if set(info)!=set(expected):
        raise RuntimeError('Certificate retry provenance fields changed')
    for key,value in expected.items():
        same=np.isclose(info[key],value,rtol=1e-10,atol=1e-12) if isinstance(value,float) else info[key]==value
        if not same:
            raise RuntimeError('Certificate retry provenance changed: '+key)
    return selected,pi


def install():
    legacy.joint_sample=joint_sample
    legacy.replay_selection=replay_selection
