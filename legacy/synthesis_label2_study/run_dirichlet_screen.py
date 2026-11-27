"""Calibration, synthesis and paired evaluation workers; no automatic FL experiment."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import dirichlet_screen_common as c


def client_data(a):
    x, y = c.dataset(a.data_root); indices = c.split(a.alpha)
    return x, y, indices, c.read(a.root / 'protocol.json')['budgets'][a.alpha]


def update_gradient(net, images, labels, real, seed, weights):
    proxy = c.update(net, images, labels, seed, weights, differentiable=True)
    loss = (proxy-real).square().sum() / (real.square().sum() + 1e-12)
    return loss.detach(), torch.autograd.grad(loss, images)[0]


def dm_gradient(net, images, labels, real_x, real_y, seed, mode):
    # Release each class graph before constructing the next (same summed gradient).
    result = torch.zeros_like(images); loss = 0.
    for term in c.dm_terms(net, images, labels, real_x, real_y, seed, mode):
        loss += float(term.detach()); result += torch.autograd.grad(term, images)[0]
    return loss, result


def calibrate(a):
    folder = a.root / 'calibration' / a.alpha; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETED.json').exists(): return True
    x, y, indices, table = client_data(a)
    file = folder / 'records.json'; records = c.read(file) if file.exists() else []
    done = {(r['client'], r['model_seed'], r['round'], r['repeat']) for r in records}
    started = time.monotonic()
    for model_seed in c.SEEDS:
        for model_round in c.CAL_ROUNDS:
            net = c.reference(a.root, a.alpha, model_seed, model_round)
            for client in range(32):
                ids = indices[client]; rx, ry = x[ids], y[ids]
                images, labels = c.initial_images(x, y, ids, table['synthetic_counts'][client], a.alpha, client)
                images.requires_grad_(True)
                weights = c.class_weights(table['real_counts'][client], table['synthetic_counts'][client])
                for repeat in range(2):
                    key = (client, model_seed, model_round, repeat)
                    if key in done: continue
                    if c.expired(a.deadline): return False
                    seed = 1000002025 + client * 100000 + model_round * 100 + (model_seed-2025)*10000000 + repeat
                    real = c.update(net, rx, ry, seed)
                    unorm = {}
                    for ce, w in (('ordinary', None), ('weighted', weights)):
                        _, gradient = update_gradient(net, images, labels, real, seed + 500000000, w)
                        unorm[ce] = float(gradient.double().norm())
                    dnorm = {}
                    for mode in ('weak', 'strong'):
                        _, gradient = dm_gradient(net, images, labels, rx, ry, seed + 600000000, mode)
                        dnorm[mode] = float(gradient.double().norm())
                    if not all(np.isfinite(v) and v > 0 for v in (*unorm.values(), *dnorm.values())):
                        raise RuntimeError(f'Invalid calibration gradient {key}: {unorm}, {dnorm}')
                    records.append(dict(client=client, model_seed=model_seed, round=model_round, repeat=repeat,
                                        update_gradient_norm=unorm, dm_gradient_norm=dnorm))
                    c.atomic_json(file, records)
                    c.atomic_json(folder / 'status.json', dict(state='calibrating', completed=len(records), target=384,
                                                               session_elapsed_seconds=time.monotonic()-started))
    ratios = {}
    for ce in ('ordinary', 'weighted'):
        for mode in ('weak', 'strong'):
            values = [r['update_gradient_norm'][ce] / r['dm_gradient_norm'][mode] for r in records]
            ratios[f'{ce}_{mode}'] = dict(r=float(np.median(values)), q25=float(np.quantile(values, .25)),
                                          q75=float(np.quantile(values, .75)), n=len(values))
    c.atomic_json(folder / 'coefficients.json', ratios)
    c.atomic_json(folder / 'COMPLETED.json', dict(records=len(records), coefficients_sha256=c.digest(folder/'coefficients.json')))
    return True


def synthesize(a):
    folder = c.route(a.root, a.alpha, a.config, a.client); folder.mkdir(parents=True, exist_ok=True)
    target_file = c.snapshot(a.root, a.alpha, a.config, a.client, a.iteration)
    if target_file.exists():
        c.validate_snapshot(a.root, a.alpha, a.config, a.client, a.iteration)
        c.atomic_json(folder/f'COMPLETED_{a.iteration}.json', dict(iteration=a.iteration, snapshot_sha256=c.digest(target_file)))
        return True
    x, y, indices, table = client_data(a); ids = indices[a.client]; rx, ry = x[ids], y[ids]
    images, labels = c.initial_images(x, y, ids, table['synthetic_counts'][a.client], a.alpha, a.client)
    ce, mode, multiplier = a.config.split('_')
    coeff_file = a.root/'calibration'/a.alpha/'coefficients.json'
    coefficient = c.read(coeff_file)[f'{ce}_{mode}']['r'] * int(multiplier[1:])
    c.bind(folder/'config.json', dict(alpha=a.alpha, client=a.client, config=a.config, lambda_dm=coefficient,
           calibration_sha256=c.digest(coeff_file), initial_images_sha256=__import__('hashlib').sha256(
               images.detach().cpu().numpy().tobytes()).hexdigest(),
           real_counts=table['real_counts'][a.client], synthetic_counts=table['synthetic_counts'][a.client]))
    checkpoint = folder/'resume.pt'; start = 0; elapsed = 0.; records = []
    old = c.load(checkpoint) if checkpoint.exists() else None
    if old:
        images.copy_(old['images'].to(c.DEVICE)); start = old['iteration']; elapsed = old['elapsed_seconds']; records = old['records']
    if start >= a.iteration: raise RuntimeError(f'Missing requested snapshot before saved progress: {target_file}')
    images.requires_grad_(True); optimizer = torch.optim.SGD([images], lr=1., momentum=.5)
    if old: optimizer.load_state_dict(old['optimizer'])
    else: c.save(c.snapshot(a.root, a.alpha, a.config, a.client, 0), dict(images=images.detach().cpu(), labels=labels.cpu()))
    weights = c.class_weights(table['real_counts'][a.client], table['synthetic_counts'][a.client]) if ce=='weighted' else None
    tick = time.monotonic()

    def persist(iteration, state, snap=False):
        duration = elapsed + time.monotonic()-tick
        c.save(checkpoint, dict(images=images.detach().cpu(), labels=labels.cpu(), iteration=iteration,
                               optimizer=optimizer.state_dict(), elapsed_seconds=duration, records=records))
        if snap: c.save(c.snapshot(a.root, a.alpha, a.config, a.client, iteration),
                        dict(images=images.detach().cpu(), labels=labels.cpu(), iteration=iteration))
        c.atomic_json(folder/'metrics.json', records)
        c.atomic_json(folder/'status.json', dict(state=state, iteration=iteration, target=a.iteration,
                                               elapsed_seconds=duration))

    for iteration in range(start+1, a.iteration+1):
        if c.expired(a.deadline): persist(iteration-1, 'budget_exhausted'); return False
        rng = np.random.default_rng(2025 + iteration*7919)
        model_seed = int(rng.choice(c.SEEDS)); group = c.TRAIN_POOL[int(rng.integers(3))]
        model_round = int(rng.choice(group)); net = c.reference(a.root, a.alpha, model_seed, model_round)
        seed = 2025 + iteration*1000 + a.client
        real = c.update(net, rx, ry, seed)
        update_loss, ug = update_gradient(net, images, labels, real, seed+500000000, weights)
        dm_loss, dg = dm_gradient(net, images, labels, rx, ry, seed+600000000, mode)
        gradient = ug + coefficient*dg
        if not torch.isfinite(gradient).all(): raise RuntimeError(f'Nonfinite image gradient at {iteration}')
        optimizer.zero_grad(set_to_none=True); images.grad = gradient; optimizer.step()
        if iteration%10==0 or iteration==a.iteration:
            records.append(dict(iteration=iteration, update_loss=float(update_loss), dm_loss=dm_loss,
                                model_seed=model_seed, model_round=model_round,
                                elapsed_seconds=elapsed+time.monotonic()-tick))
            c.atomic_json(folder/'status.json', dict(state='synthesizing', iteration=iteration, target=a.iteration,
                                                   elapsed_seconds=elapsed+time.monotonic()-tick))
        if iteration%100==0 or iteration in (250, 500, 1000) or iteration==a.iteration:
            persist(iteration, 'complete' if iteration==1000 else 'checkpoint_complete', snap=iteration in (100,250,500,1000))
    c.atomic_json(folder/f'COMPLETED_{a.iteration}.json', dict(iteration=a.iteration, snapshot_sha256=c.digest(target_file)))
    return True


def vectors(a):
    folder = a.root/'evaluation'/f'iter_{a.iteration:04d}'/a.alpha/'vectors'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'COMPLETED.json').exists(): return True
    x, y, indices, table = client_data(a)
    datasets = {(client, config): c.load(c.snapshot(a.root,a.alpha,config,client,a.iteration))
                for client in c.SELECTED[a.alpha] for config in c.CONFIGS}
    # A record is one client/model point, evaluated for all configurations on identical real draws.
    for client in c.SELECTED[a.alpha]:
        ids = indices[client]; rx, ry = x[ids], y[ids]
        weights = c.class_weights(table['real_counts'][client], table['synthetic_counts'][client])
        for model_seed in c.SEEDS:
            for model_round in c.EVAL_ROUNDS:
                file = folder/f'client_{client:02d}_seed{model_seed}_round{model_round:04d}.json'
                if file.exists(): continue
                if c.expired(a.deadline): return False
                net = c.reference(a.root,a.alpha,model_seed,model_round)
                rm = c.Moments(); pm = {cfg:c.Moments() for cfg in c.CONFIGS}; dm = {cfg:c.Moments() for cfg in c.CONFIGS}
                observed = {cfg:[] for cfg in c.CONFIGS}
                for repeat in range(30):
                    if c.expired(a.deadline): return False  # retry this incomplete point unchanged on resume
                    seed = 2000002025 + client*100000 + model_round*100 + (model_seed-2025)*10000000 + repeat
                    real = c.update(net,rx,ry,seed); rm.add(real)
                    for config in c.CONFIGS:
                        item = datasets[client,config]
                        proxy = c.update(net,item['images'],item['labels'],seed+500000000,
                                         weights if config.startswith('weighted') else None)
                        pm[config].add(proxy); dm[config].add(proxy-real)
                        observed[config].append(c.geometry(proxy,real))
                records = []
                for config in c.CONFIGS:
                    denom = float(rm.mean.square().sum())+1e-12
                    corrected = dm[config].corrected_squared_mean()
                    records.append(dict(config=config, mean_geometry=c.geometry(pm[config].mean,rm.mean),
                        mean_squared_error=float(dm[config].mean.square().sum())+dm[config].m2/dm[config].n,
                        squared_mean_error=float(dm[config].mean.square().sum()), corrected_squared_mean_error=corrected,
                        corrected_relative_squared_mean_error=corrected/denom,
                        negative_noise_corrected_estimate=corrected<0, real_variance=rm.variance(),
                        proxy_variance=pm[config].variance(), difference_variance=dm[config].variance(),
                        individual_geometry=observed[config]))
                c.atomic_json(file,dict(client=client,model_seed=model_seed,round=model_round,repeats=30,records=records))
                c.atomic_json(folder/'status.json',dict(state='evaluating',completed=len(list(folder.glob('client_*.json'))),target=96))
    c.atomic_json(folder/'COMPLETED.json',dict(points=96,repeats=30,configs=8,
        files_sha256={p.name:c.digest(p) for p in sorted(folder.glob('client_*.json'))})); return True


@torch.no_grad()
def evaluate(net, x, y):
    net.eval(); correct = 0; loss = 0.
    for start in range(0,len(y),256):
        target = y[start:start+256].to(c.DEVICE); logits = net(x[start:start+256].to(c.DEVICE))
        loss += float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
        correct += int((logits.argmax(1)==target).sum())
    return dict(test_accuracy=correct/len(y),test_loss=loss/len(y))


def accuracy(a):
    folder = a.root/'evaluation'/f'iter_{a.iteration:04d}'/a.alpha/'accuracy'/a.config
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETED.json').exists(): return True
    items = [c.load(c.snapshot(a.root,a.alpha,a.config,client,a.iteration)) for client in c.SELECTED[a.alpha]]
    images = torch.cat([d['images'] for d in items]); labels = torch.cat([d['labels'] for d in items])
    tx,ty = c.dataset(a.data_root,train=False); net = c.model(2025)
    generator = torch.Generator().manual_seed(2025)
    loader = DataLoader(TensorDataset(images,labels),batch_size=256,shuffle=True,drop_last=False,
                        num_workers=0,generator=generator)
    optimizer = torch.optim.SGD(net.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
    augment = c.ParamDiffAug()  # original classifier uses default mode S, unlike DM mode M
    checkpoint = folder/'resume.pt'; start = 0; history = []; elapsed = 0.
    if checkpoint.exists():
        old = c.load(checkpoint); net.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        generator.set_state(old['generator']); c.restore_rng(old['rng']); start=old['epoch']; history=old['history']; elapsed=old['elapsed_seconds']
    else: history.append(dict(epoch=0,**evaluate(net,tx,ty)))
    tick = time.monotonic()

    def persist(epoch,state):
        duration = elapsed+time.monotonic()-tick
        c.save(checkpoint,dict(epoch=epoch,model=net.state_dict(),optimizer=optimizer.state_dict(),
                              generator=generator.get_state(),rng=c.rng_state(),history=history,elapsed_seconds=duration))
        c.atomic_json(folder/'status.json',dict(state=state,epoch=epoch,target=1000,elapsed_seconds=duration))
        c.atomic_json(folder/'history.json',history)

    for epoch in range(start+1,1001):
        if c.expired(a.deadline): persist(epoch-1,'budget_exhausted'); return False
        if epoch==501: optimizer=torch.optim.SGD(net.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
        net.train(); loss_sum=0.; correct=0
        for batch_x,batch_y in loader:
            batch_x=c.DiffAugment(batch_x.to(c.DEVICE),c.STRATEGIES['strong'],param=augment); batch_y=batch_y.to(c.DEVICE)
            optimizer.zero_grad(set_to_none=True); logits=net(batch_x); loss=c.ce_loss(logits,batch_y)
            loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())*len(batch_y); correct+=int((logits.argmax(1)==batch_y).sum())
        if epoch in (200,500,1000): history.append(dict(epoch=epoch,train_loss=loss_sum/len(labels),train_accuracy=correct/len(labels),**evaluate(net,tx,ty)))
        if epoch%10==0: persist(epoch,'complete' if epoch==1000 else 'training_classifier')
    result = dict(config=a.config,alpha=a.alpha,iteration=a.iteration,seed=2025,epochs=1000,
                  clients=c.SELECTED[a.alpha],class_counts=torch.bincount(labels,minlength=10).tolist(),history=history)
    c.atomic_json(folder/'result.json',result)
    c.atomic_json(folder/'COMPLETED.json',dict(epochs=1000,result_sha256=c.digest(folder/'result.json'))); return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('calibrate','synthesize','vectors','accuracy'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--alpha',choices=c.ALPHAS,required=True); p.add_argument('--config',choices=c.CONFIGS)
    p.add_argument('--client',type=int); p.add_argument('--iteration',type=int,choices=(500,1000),default=500)
    p.add_argument('--deadline',type=float); a=p.parse_args()
    c.bind(a.root/'protocol.json',c.protocol())
    if a.phase in ('synthesize','accuracy') and a.config is None: p.error('--config required')
    if a.phase=='synthesize' and a.client not in c.SELECTED[a.alpha]: p.error('client not in fixed screen')
    raise SystemExit(0 if globals()[a.phase](a) else 75)


if __name__=='__main__': main()
