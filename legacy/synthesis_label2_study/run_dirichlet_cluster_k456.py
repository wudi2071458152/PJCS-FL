"""Old-loss 32000 scoring-real routes with actual K=4/5/6; frozen numerics."""
import argparse
import signal
from pathlib import Path
from unittest.mock import patch
import torch
import run_dirichlet_cluster_history as old

c = old.c
CLUSTERS = (4, 5, 6)
METHOD = 'cluster_score_real'
CODE = (*old.CODE, 'synthesis_label2_study/run_dirichlet_cluster_k456.py')
_protocol, _prepare, _round, _train, _verify = old.route_protocol, old.prepare, old.one_round, old.train, old.verify
_spherical = old.spherical_clusters


def route_protocol(original, method, clusters):
    if method != METHOD or clusters not in CLUSTERS: raise ValueError('Only scoring-real K4/5/6 is supported')
    protocol = _protocol(original, method, 3)
    protocol.update(version=3)
    protocol['clustering']['k'] = clusters
    for name in CODE: protocol['files_sha256']['code/' + name] = c.digest(c.ROOT / name)
    return protocol


def prepare(a):
    if a.method != METHOD or a.clusters not in CLUSTERS or a.gamma24 or a.iteration != 32000:
        raise ValueError('Only original-loss 32000 scoring-real K4/5/6 is supported')
    with patch.object(old, 'route_protocol', route_protocol): return _prepare(a)


def one_round(net, real, syn, weights, method, clusters, rnd, history=None, last_seen=None, update_fn=None):
    if method != METHOD or clusters not in CLUSTERS or history is not None or last_seen is not None:
        raise ValueError('Only scoring-real K4/5/6 without history is supported')
    # K3 passes the frozen function's legacy guard; the actual clustering call uses K4/5/6.
    def actual_clusters(features, legacy_k, seed): return _spherical(features, clusters, seed)
    with patch.object(old, 'spherical_clusters', actual_clusters):
        return _round(net, real, syn, weights, method, 3, rnd, update_fn=update_fn)


def verify(folder, rounds=1500):
    protocol = c.read(Path(folder) / 'protocol.json')
    if protocol['method'] != METHOD or protocol['clustering']['k'] not in CLUSTERS:
        raise RuntimeError('Unexpected K4/5/6 route')
    with patch.object(old, 'route_protocol', route_protocol): return _verify(folder, rounds)


def train(a, prepared):
    with patch.object(old, 'one_round', one_round), patch.object(old, 'verify', verify):
        return _train(a, prepared)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-root', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--proxy-root', type=Path)
    p.add_argument('--alpha', choices=c.ALPHAS, required=True)
    p.add_argument('--method', choices=(METHOD,), default=METHOD)
    p.add_argument('--clusters', type=int, choices=CLUSTERS, required=True)
    p.add_argument('--iteration', type=int, choices=(32000,), default=32000)
    p.add_argument('--rounds', type=int, default=1500); p.add_argument('--deadline', type=float)
    p.add_argument('--run', action='store_true'); p.set_defaults(gamma24=False)
    a = p.parse_args()
    if a.rounds < 1: p.error('Round target must be positive')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL requires CUDA')
    prepared = prepare(a)
    if not a.run:
        import json
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1]))); return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as scheduler
    with scheduler.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
