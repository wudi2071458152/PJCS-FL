"""Workers for the weighted CE, weak/strong DSA x10 follow-up.

Numerical gradients and classifier training reuse the screened implementation.
The driver supplies the selected clients, protocol and imported reference pool.
"""
import time
import numpy as np
import torch
import dirichlet_screen_common as c
from run_dirichlet_screen import client_data, update_gradient, dm_gradient, accuracy, evaluate

CONFIGS = ('weighted_weak_x10', 'weighted_strong_x10')


def synthesize(a):
    if a.config not in CONFIGS:
        raise ValueError('The follow-up only permits weighted weak/strong DSA x10')
    folder = c.route(a.root, a.alpha, a.config, a.client)
    folder.mkdir(parents=True, exist_ok=True)
    target_file = c.snapshot(a.root, a.alpha, a.config, a.client, a.iteration)
    if target_file.exists():
        c.validate_snapshot(a.root, a.alpha, a.config, a.client, a.iteration)
        c.atomic_json(folder / f'COMPLETED_{a.iteration}.json',
                      dict(iteration=a.iteration, snapshot_sha256=c.digest(target_file)))
        status = c.read(folder / 'status.json') if (folder / 'status.json').exists() else {}
        c.atomic_json(folder / 'status.json', dict(status, state='complete',
                      iteration=a.iteration, target=a.iteration))
        return True
    x, y, indices, table = client_data(a)
    ids = indices[a.client]; rx, ry = x[ids], y[ids]
    images, labels = c.initial_images(x, y, ids, table['synthetic_counts'][a.client], a.alpha, a.client)
    ce, mode, multiplier = a.config.split('_')
    coeff_file = a.root / 'calibration' / a.alpha / 'coefficients.json'
    coefficient = c.read(coeff_file)[f'{ce}_{mode}']['r'] * int(multiplier[1:])
    c.bind(folder / 'config.json', dict(alpha=a.alpha, client=a.client, config=a.config,
           lambda_dm=coefficient, calibration_sha256=c.digest(coeff_file),
           initial_images_sha256=__import__('hashlib').sha256(images.detach().cpu().numpy().tobytes()).hexdigest(),
           real_counts=table['real_counts'][a.client], synthetic_counts=table['synthetic_counts'][a.client]))
    checkpoint = folder / 'resume.pt'; start = 0; elapsed = 0.; records = []
    old = c.load(checkpoint) if checkpoint.exists() else None
    if old:
        images.copy_(old['images'].to(c.DEVICE)); start = old['iteration']
        elapsed = old['elapsed_seconds']; records = old['records']
    if start >= a.iteration:
        raise RuntimeError(f'Missing requested snapshot before saved progress: {target_file}')
    images.requires_grad_(True)
    optimizer = torch.optim.SGD([images], lr=1., momentum=.5)
    if old:
        optimizer.load_state_dict(old['optimizer'])
    else:
        c.save(c.snapshot(a.root, a.alpha, a.config, a.client, 0),
               dict(images=images.detach().cpu(), labels=labels.cpu()))
    weights = c.class_weights(table['real_counts'][a.client], table['synthetic_counts'][a.client])
    tick = time.monotonic()

    def persist(iteration, state, snap=False):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(images=images.detach().cpu(), labels=labels.cpu(), iteration=iteration,
                               optimizer=optimizer.state_dict(), elapsed_seconds=duration, records=records))
        if snap:
            c.save(c.snapshot(a.root, a.alpha, a.config, a.client, iteration),
                   dict(images=images.detach().cpu(), labels=labels.cpu(), iteration=iteration))
        c.atomic_json(folder / 'metrics.json', records)
        c.atomic_json(folder / 'status.json', dict(state=state, iteration=iteration,
                      target=a.iteration, elapsed_seconds=duration))

    for iteration in range(start + 1, a.iteration + 1):
        if c.expired(a.deadline):
            persist(iteration - 1, 'budget_exhausted'); return False
        rng = np.random.default_rng(2025 + iteration * 7919)
        model_seed = int(rng.choice(c.SEEDS)); group = c.TRAIN_POOL[int(rng.integers(3))]
        model_round = int(rng.choice(group)); net = c.reference(a.root, a.alpha, model_seed, model_round)
        seed = 2025 + iteration * 1000 + a.client
        real = c.update(net, rx, ry, seed)
        update_loss, ug = update_gradient(net, images, labels, real, seed + 500000000, weights)
        dm_loss, dg = dm_gradient(net, images, labels, rx, ry, seed + 600000000, mode)
        gradient = ug + coefficient * dg
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f'Nonfinite image gradient at {iteration}')
        optimizer.zero_grad(set_to_none=True); images.grad = gradient; optimizer.step()
        if iteration % 10 == 0 or iteration == a.iteration:
            records.append(dict(iteration=iteration, update_loss=float(update_loss), dm_loss=dm_loss,
                                model_seed=model_seed, model_round=model_round,
                                elapsed_seconds=elapsed + time.monotonic() - tick))
            c.atomic_json(folder / 'status.json', dict(state='synthesizing', iteration=iteration,
                          target=a.iteration, elapsed_seconds=elapsed + time.monotonic() - tick))
        if iteration % 100 == 0 or iteration in (250, 500, 1000) or iteration == a.iteration:
            persist(iteration, 'complete' if iteration == a.iteration else 'checkpoint_complete',
                    snap=iteration in (100, 250, 500, 1000, 2000) or iteration == a.iteration)
    c.atomic_json(folder / f'COMPLETED_{a.iteration}.json',
                  dict(iteration=a.iteration, snapshot_sha256=c.digest(target_file)))
    return True


def vectors(a):
    clients = list(getattr(a, 'clients', None) or c.SELECTED[a.alpha])
    configs = tuple(getattr(a, 'configs', None) or CONFIGS)
    if not configs or any(config not in CONFIGS for config in configs):
        raise ValueError('Invalid follow-up diagnostic configurations')
    if not clients or len(set(clients)) != len(clients) or any(client not in range(32) for client in clients):
        raise ValueError('Invalid follow-up diagnostic clients')
    dataset_root = getattr(a, 'dataset_root', None) or a.root
    folder = a.root / 'evaluation' / f'iter_{a.iteration:04d}' / a.alpha / 'vectors'
    folder.mkdir(parents=True, exist_ok=True)
    names = [f'client_{client:02d}_seed{seed}_round{model_round:04d}.json'
             for client in clients for seed in c.SEEDS for model_round in c.EVAL_ROUNDS]
    marker = folder / 'COMPLETED.json'
    if marker.exists():
        done = c.read(marker)
        expected = dict(points=len(names), repeats=30, configs=len(configs),
                        files_sha256={name: c.digest(folder / name) for name in sorted(names)})
        if done != expected:
            raise RuntimeError(f'Diagnostic completion checksum/count mismatch: {marker}')
        return True
    x, y, indices, table = client_data(a)
    datasets = {(client, config): c.load(c.snapshot(dataset_root, a.alpha, config, client, a.iteration))
                for client in clients for config in configs}
    for client in clients:
        ids = indices[client]; rx, ry = x[ids], y[ids]
        weights = c.class_weights(table['real_counts'][client], table['synthetic_counts'][client])
        for model_seed in c.SEEDS:
            for model_round in c.EVAL_ROUNDS:
                file = folder / f'client_{client:02d}_seed{model_seed}_round{model_round:04d}.json'
                if file.exists():
                    old = c.read(file)
                    if (old['client'], old['model_seed'], old['round'], old['repeats']) != (client, model_seed, model_round, 30) or [r['config'] for r in old['records']] != list(configs):
                        raise RuntimeError(f'Diagnostic point/configuration mismatch: {file}')
                    continue
                if c.expired(a.deadline): return False
                net = c.reference(a.root, a.alpha, model_seed, model_round)
                rm = c.Moments(); pm = {config: c.Moments() for config in configs}
                dm = {config: c.Moments() for config in configs}; observed = {config: [] for config in configs}
                for repeat in range(30):
                    if c.expired(a.deadline): return False
                    seed = 2000002025 + client * 100000 + model_round * 100 + (model_seed - 2025) * 10000000 + repeat
                    real = c.update(net, rx, ry, seed); rm.add(real)
                    for config in configs:
                        item = datasets[client, config]
                        proxy = c.update(net, item['images'], item['labels'], seed + 500000000, weights)
                        pm[config].add(proxy); dm[config].add(proxy - real)
                        observed[config].append(c.geometry(proxy, real))
                records = []
                for config in configs:
                    denom = float(rm.mean.square().sum()) + 1e-12
                    corrected = dm[config].corrected_squared_mean()
                    records.append(dict(config=config, mean_geometry=c.geometry(pm[config].mean, rm.mean),
                        mean_squared_error=float(dm[config].mean.square().sum()) + dm[config].m2 / dm[config].n,
                        squared_mean_error=float(dm[config].mean.square().sum()), corrected_squared_mean_error=corrected,
                        corrected_relative_squared_mean_error=corrected / denom,
                        negative_noise_corrected_estimate=corrected < 0, real_variance=rm.variance(),
                        proxy_variance=pm[config].variance(), difference_variance=dm[config].variance(),
                        individual_geometry=observed[config]))
                c.atomic_json(file, dict(client=client, model_seed=model_seed, round=model_round,
                                        repeats=30, records=records))
                completed = sum((folder / name).exists() for name in names)
                c.atomic_json(folder / 'status.json', dict(state='evaluating', completed=completed, target=len(names)))
    c.atomic_json(marker, dict(points=len(names), repeats=30, configs=len(configs),
                  files_sha256={name: c.digest(folder / name) for name in sorted(names)}))
    c.atomic_json(folder / 'status.json', dict(state='complete', completed=len(names), target=len(names)))
    return True
