"""Continue paired 24000-image branches, retaining their inherited SGD momentum."""
import time
import numpy as np
import torch
import dirichlet_screen_common as c
import dirichlet_dsa4000_worker as oldworker
import dirichlet_direction_pilot as p

BASELINE = 24000
CONFIG = 'weighted_weak_x10'
vectors = oldworker.vectors
accuracy = oldworker.accuracy


def gamma_manifest(root):
    """Verify the fixed 30% coefficients and their original calibration members."""
    result = {}
    for alpha in c.ALPHAS:
        folder = root / 'calibration' / alpha
        file = folder / 'gamma.json'; item = c.read(file)
        names = {f'client_{client:02d}.json' for client in p.CLIENTS[alpha]}
        if item.get('fixed') is not True or item.get('target_fraction') != .3 or item.get('contexts') != 36 or set(item.get('files_sha256', {})) != names:
            raise RuntimeError('Gamma calibration definition/coverage changed: ' + str(file))
        ratios = []
        for name, sha in item['files_sha256'].items():
            member = folder / name
            if c.digest(member) != sha: raise RuntimeError('Gamma calibration checksum mismatch: ' + str(member))
            rows = c.read(member)
            expected = {(seed, rnd, repeat) for seed in c.SEEDS for rnd in c.CAL_ROUNDS for repeat in range(2)}
            if len(rows) != 12 or {(r['model_seed'], r['model_round'], r['repeat']) for r in rows} != expected:
                raise RuntimeError('Gamma calibration contexts changed: ' + str(member))
            ratios.extend(row['ratio'] for row in rows)
        if not all(np.isfinite(value) and value > 0 for value in ratios) or not np.isclose(item['gamma'], .3 * np.median(ratios), rtol=1e-12, atol=0.):
            raise RuntimeError('Gamma coefficient is inconsistent with its 30% calibration')
        result[alpha] = dict(gamma=item['gamma'], gamma_sha256=c.digest(file), files_sha256=item['files_sha256'])
    return result


def sampling(client, iteration):
    rng = np.random.default_rng(2025 + iteration * 7919)
    seed = int(rng.choice(c.SEEDS))
    rnd = int(rng.choice(c.TRAIN_POOL[int(rng.integers(3))]))
    return seed, rnd, 2025 + iteration * 1000 + client


def synthesize(a):
    if a.loss == 'original': return oldworker.synthesize(a)
    if a.loss != 'low_dm_direction' or a.config != CONFIG: raise ValueError('Unexpected synthesis loss/configuration')
    folder = c.route(a.root, a.alpha, CONFIG, a.client)
    gamma = gamma_manifest(a.gamma_root)[a.alpha]
    source = c.route(a.source_root, a.alpha, CONFIG, a.client)
    source_lambda = c.read(source / 'config.json')['lambda_dm']
    settings = dict(loss=a.loss, source_iteration=BASELINE, source_lambda_dm=source_lambda,
                    effective_lambda_dm=.1 * source_lambda, gamma=gamma['gamma'], gamma_fingerprints=gamma,
                    source_config_sha256=c.digest(source / 'config.json'), source_resume_sha256=c.digest(source / 'resume.pt'),
                    source_snapshot_sha256=c.digest(c.snapshot(a.source_root, a.alpha, CONFIG, a.client, BASELINE)),
                    optimizer=dict(lr=1., momentum=.5, initial_momentum='inherit24000'))
    checkpoint = folder / 'resume.pt'; old = c.load(checkpoint)
    if old['iteration'] > BASELINE and not (folder / 'loss_settings.json').exists():
        raise RuntimeError('Advanced branch has no bound loss settings')
    c.bind(folder / 'loss_settings.json', settings)
    target_file = c.snapshot(a.root, a.alpha, CONFIG, a.client, a.iteration)
    if target_file.exists():
        c.validate_snapshot(a.root, a.alpha, CONFIG, a.client, a.iteration)
        c.atomic_json(folder / f'COMPLETED_{a.iteration}.json', dict(iteration=a.iteration, snapshot_sha256=c.digest(target_file)))
        return True
    start = old['iteration']
    if not BASELINE <= start < a.iteration: raise RuntimeError('Checkpoint is outside the requested continuation interval')
    x, y, indices, table = oldworker.client_data(a)
    ids = indices[a.client]; rx, ry = x[ids], y[ids]
    images = old['images'].to(c.DEVICE).clone().requires_grad_(True); labels = old['labels'].to(c.DEVICE)
    optimizer = torch.optim.SGD([images], lr=1., momentum=.5); optimizer.load_state_dict(old['optimizer'])
    weights = c.class_weights(table['real_counts'][a.client], table['synthetic_counts'][a.client])
    records = old['records']; elapsed = old['elapsed_seconds']; tick = time.monotonic()

    def persist(iteration, state, snap=False):
        duration = elapsed + time.monotonic() - tick
        c.save(checkpoint, dict(images=images.detach().cpu(), labels=labels.cpu(), iteration=iteration,
                               optimizer=optimizer.state_dict(), elapsed_seconds=duration, records=records))
        if snap: c.save(c.snapshot(a.root, a.alpha, CONFIG, a.client, iteration),
                        dict(images=images.detach().cpu(), labels=labels.cpu(), iteration=iteration))
        c.atomic_json(folder / 'metrics.json', records)
        c.atomic_json(folder / 'status.json', dict(state=state, iteration=iteration, target=a.iteration, elapsed_seconds=duration))

    for iteration in range(start + 1, a.iteration + 1):
        if c.expired(a.deadline): persist(iteration - 1, 'budget_exhausted'); return False
        model_seed, model_round, seed = sampling(a.client, iteration)
        net = c.reference(a.root, a.alpha, model_seed, model_round)
        real = c.update(net, rx, ry, seed)
        update_loss, direction, loss = p.objective(net, images, labels, real, seed + 500000000, weights, settings['gamma'])
        ug = torch.autograd.grad(loss, images)[0]
        dm_loss, dg = oldworker.dm_gradient(net, images, labels, rx, ry, seed + 600000000, 'weak')
        gradient = ug + settings['effective_lambda_dm'] * dg
        if not torch.isfinite(gradient).all(): raise RuntimeError(f'Nonfinite image gradient at {iteration}')
        optimizer.zero_grad(set_to_none=True); images.grad = gradient; optimizer.step()
        if iteration % 10 == 0 or iteration == a.iteration:
            records.append(dict(iteration=iteration, update_loss=float(update_loss.detach()), direction_loss=float(direction.detach()),
                                dm_loss=dm_loss, model_seed=model_seed, model_round=model_round,
                                elapsed_seconds=elapsed + time.monotonic() - tick))
            c.atomic_json(folder / 'status.json', dict(state='synthesizing', iteration=iteration, target=a.iteration,
                                                      elapsed_seconds=elapsed + time.monotonic() - tick))
        if iteration % 100 == 0 or iteration == a.iteration:
            persist(iteration, 'complete' if iteration == a.iteration else 'checkpoint_complete', snap=iteration == a.iteration)
    c.atomic_json(folder / f'COMPLETED_{a.iteration}.json', dict(iteration=a.iteration, snapshot_sha256=c.digest(target_file)))
    return True
