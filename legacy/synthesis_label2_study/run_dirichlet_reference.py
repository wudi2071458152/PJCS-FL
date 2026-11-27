"""Build two real-data FedAvg trajectories per original Dirichlet partition."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c


def train(root, data_root, alpha, seed, deadline=None):
    folder = Path(root) / 'reference' / alpha / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETED.json').exists(): return True
    x, y = c.dataset(data_root); indices = c.split(alpha); net = c.model(seed)
    checkpoint = folder / 'resume.pt'; start = 0; records = []; elapsed = 0.
    if checkpoint.exists():
        old = c.load(checkpoint); net.load_state_dict(old['model']); start = old['round']
        records = old['records']; elapsed = old['elapsed_seconds']; c.restore_rng(old['rng'])
    else: c.save(c.model_path(root, alpha, seed, 0), net.state_dict())
    tick = time.monotonic()

    def persist(round_number, state):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(round=round_number, model=net.state_dict(), records=records,
                               elapsed_seconds=duration, rng=c.rng_state()))
        c.atomic_json(folder / 'rounds.json', records)
        c.atomic_json(folder / 'status.json', dict(state=state, round=round_number, target=1500,
                                                 elapsed_seconds=duration))

    for round_number in range(start + 1, 1501):
        if c.expired(deadline): persist(round_number - 1, 'budget_exhausted'); return False
        selected = np.random.default_rng(seed + round_number * 1000).choice(32, 3, replace=False)
        sizes = np.array([len(indices[i]) for i in selected]); weights = sizes / sizes.sum()
        delta = None
        for client, weight in zip(selected, weights):
            ids = indices[client]
            update = c.update(net, x[ids], y[ids], seed + round_number * 1000 + int(client))
            delta = weight * update if delta is None else delta + weight * update
        c.add_delta(net, delta)
        records.append(dict(round=round_number, selected=selected.tolist(), weights=weights.tolist(),
                            update_norm=float(delta.norm())))
        if round_number in c.MODEL_ROUNDS:
            c.save(c.model_path(root, alpha, seed, round_number), net.state_dict())
        if round_number % 10 == 0 or round_number in c.MODEL_ROUNDS: persist(round_number, 'training')
    persist(1500, 'complete')
    c.atomic_json(folder / 'COMPLETED.json', dict(rounds=1500, model_sha256={str(r): c.digest(
        c.model_path(root, alpha, seed, r)) for r in c.MODEL_ROUNDS}))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--alpha', choices=c.ALPHAS, required=True); p.add_argument('--seed', type=int, choices=c.SEEDS, required=True)
    p.add_argument('--deadline', type=float)
    a = p.parse_args()
    c.bind(a.root / 'protocol.json', c.protocol())
    raise SystemExit(0 if train(a.root, a.data_root, a.alpha, a.seed, a.deadline) else 75)


if __name__ == '__main__': main()
