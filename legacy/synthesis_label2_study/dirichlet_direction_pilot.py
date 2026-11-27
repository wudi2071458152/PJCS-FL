"""Paired 16k-image branches; only the synthesis loss changes."""
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c
import diagnose_dirichlet_direction as d
from run_dirichlet_screen import dm_gradient

LAYERS = ('features.4.weight', 'features.8.weight')
ARMS = {'original': (1., False), 'direction': (1., True),
        'low_dm': (.1, False), 'low_dm_direction': (.1, True)}
CLIENTS = {'0p1': (24, 17, 9), '0p2': (26, 12, 23), '0p3': (1, 26, 30)}
EVAL_ROUNDS = (20, 500, 1500)
SOURCE_ITERATION = 16000
EVAL_SEED = 3000002025


def direction_loss(net, proxy, real):
    terms, offset = [], 0
    scale = float(real.detach().double().square().sum())
    for name, parameter in net.named_parameters():
        end = offset + parameter.numel()
        if name in LAYERS:
            p, r = proxy[offset:end].double(), real[offset:end].detach().double()
            if float(r.square().sum()) > max(1e-30, scale*1e-12):
                terms.append(1 - torch.dot(p, r)/(p.norm().clamp_min(float(r.norm())*1e-8)*r.norm()))
        offset = end
    if offset != proxy.numel() or proxy.shape != real.shape:
        raise RuntimeError('Update parameter layout mismatch')
    if not all(name in dict(net.named_parameters()) for name in LAYERS):
        raise RuntimeError('Required convolution weights missing')
    return torch.stack(terms).mean() if terms else proxy.sum()*0.


def objective(net, images, labels, real, seed, weights, gamma=0.):
    proxy = c.update(net, images, labels, seed, weights, differentiable=True)
    mse = (proxy-real).square().sum()/(real.square().sum()+1e-12)
    directional = direction_loss(net, proxy, real)
    return mse, directional, mse + gamma*directional


def context(a):
    x, y = c.dataset(a.data_root); ids = c.split(a.alpha)[a.client]
    table = c.read(a.source_root/'protocol.json')['budgets'][a.alpha]
    item = c.load(c.snapshot(a.source_root, a.alpha, d.CONFIG, a.client, SOURCE_ITERATION))
    weights = c.class_weights(table['real_counts'][a.client], table['synthetic_counts'][a.client])
    return x[ids], y[ids], item, weights


def sampling(client, step):
    absolute = SOURCE_ITERATION + step
    rng = np.random.default_rng(2025 + absolute*7919)
    model_seed = int(rng.choice(c.SEEDS))
    model_round = int(rng.choice(c.TRAIN_POOL[int(rng.integers(3))]))
    return model_seed, model_round, 2025 + absolute*1000 + client


def folder(a):
    return a.output/'branches'/a.alpha/f'client_{a.client:02d}'/a.arm


def calibrate(a):
    rx, ry, item, weights = context(a)
    images = item['images'].to(c.DEVICE).clone().requires_grad_(True)
    rows = []
    for model_seed in c.SEEDS:
        for model_round in c.CAL_ROUNDS:
            net = c.reference(a.source_root, a.alpha, model_seed, model_round)
            for repeat in range(2):
                seed = 4000002025+a.client*100000+model_round*100+(model_seed-2025)*10000000+repeat
                real = c.update(net, rx, ry, seed)
                mse, direction, _ = objective(net, images, item['labels'], real, seed+500000000, weights)
                gu = torch.autograd.grad(mse, images, retain_graph=True)[0]
                gd = torch.autograd.grad(direction, images)[0]
                pair = d.gradient_pair(gu, gd)
                if not pair['valid_direction']: raise RuntimeError('Cannot calibrate zero direction/update gradient')
                rows.append(dict(model_seed=model_seed, model_round=model_round, repeat=repeat, seed=seed,
                                 update_norm=pair['first_norm'], direction_norm=pair['second_norm'],
                                 ratio=pair['first_norm']/pair['second_norm']))
    c.atomic_json(a.output/'calibration'/a.alpha/f'client_{a.client:02d}.json', rows)


def coefficients(output):
    for alpha, clients in CLIENTS.items():
        records = [r for client in clients for r in c.read(output/'calibration'/alpha/f'client_{client:02d}.json')]
        if len(records) != 36: raise RuntimeError('Missing calibration contexts')
        values = [r['ratio'] for r in records]
        c.bind(output/'calibration'/alpha/'gamma.json', dict(gamma=.3*float(np.median(values)),
            target_fraction=.3, fixed=True, contexts=36, ratio_median=float(np.median(values)),
            q25=float(np.quantile(values,.25)), q75=float(np.quantile(values,.75)),
            files_sha256={f'client_{i:02d}.json':c.digest(output/'calibration'/alpha/f'client_{i:02d}.json') for i in clients}))


def synthesize(a):
    rx, ry, item, weights = context(a); path = folder(a)
    factor, use_direction = ARMS[a.arm]
    gamma = c.read(a.output/'calibration'/a.alpha/'gamma.json')['gamma'] if use_direction else 0.
    source_lambda = c.read(c.route(a.source_root,a.alpha,d.CONFIG,a.client)/'config.json')['lambda_dm']
    images = item['images'].to(c.DEVICE).clone().requires_grad_(True)
    optimizer = torch.optim.SGD([images],lr=1.,momentum=.5)
    start, records, elapsed = 0, [], 0.
    checkpoint = path/'resume.pt'
    settings = dict(alpha=a.alpha,client=a.client,arm=a.arm,source_iteration=SOURCE_ITERATION,
        source_snapshot_sha256=c.digest(c.snapshot(a.source_root,a.alpha,d.CONFIG,a.client,SOURCE_ITERATION)),
        gamma=gamma,lambda_dm=factor*source_lambda,dm_factor=factor,
        optimizer=dict(lr=1.,momentum=.5,initial_momentum='reset'))
    c.bind(path/'config.json',settings)
    if checkpoint.exists():
        saved = c.load(checkpoint)
        if saved['settings'] != settings: raise RuntimeError('Checkpoint settings changed')
        if saved['images'].shape!=item['images'].shape or not torch.isfinite(saved['images']).all() or not torch.equal(saved['labels'],item['labels']):
            raise RuntimeError('Invalid checkpoint images/labels')
        images.data.copy_(saved['images'].to(c.DEVICE)); optimizer.load_state_dict(saved['optimizer'])
        start, records, elapsed = saved['step'], saved['records'], saved['elapsed_seconds']
    if start > a.step: raise RuntimeError('Checkpoint is beyond the requested milestone')
    tick = time.monotonic(); sequence = hashlib.sha256()
    for step in range(1,start+1): sequence.update(str(sampling(a.client,step)).encode('ascii'))
    if checkpoint.exists() and sequence.hexdigest()!=saved['sampling_sequence_sha256']:
        raise RuntimeError('Checkpoint sampling sequence changed')

    def persist(step, snapshot=False):
        duration = elapsed+time.monotonic()-tick
        c.save(checkpoint,dict(settings=settings,step=step,images=images.detach().cpu(),labels=item['labels'],
            optimizer=optimizer.state_dict(),records=records,elapsed_seconds=duration,
            sampling_sequence_sha256=sequence.hexdigest()))
        c.atomic_json(path/'metrics.json',records)
        c.atomic_json(path/'status.json',dict(step=step,target=a.step,elapsed_seconds=duration,
            session_steps=step-start,session_seconds=time.monotonic()-tick))
        if snapshot:
            target = path/'snapshots'/f'step_{step:04d}.pt'
            c.save(target,dict(images=images.detach().cpu(),labels=item['labels'],step=step))
            c.atomic_json(path/f'COMPLETED_{step}.json',dict(step=step,snapshot_sha256=c.digest(target),
                sampling_sequence_sha256=sequence.hexdigest(),metrics_sha256=c.digest(path/'metrics.json')))

    for step in range(start+1,a.step+1):
        if c.expired(a.deadline): persist(step-1); return False
        model_seed, model_round, seed = sampling(a.client,step)
        net = c.reference(a.source_root,a.alpha,model_seed,model_round)
        real = c.update(net,rx,ry,seed)
        mse, direction, loss = objective(net,images,item['labels'],real,seed+500000000,weights,gamma)
        gu = torch.autograd.grad(loss,images)[0]
        dm_loss, gd = dm_gradient(net,images,item['labels'],rx,ry,seed+600000000,'weak')
        gradient = gu+settings['lambda_dm']*gd
        if not torch.isfinite(gradient).all(): raise RuntimeError('Nonfinite synthesis gradient')
        optimizer.zero_grad(set_to_none=True); images.grad=gradient; optimizer.step()
        sequence.update(str((model_seed,model_round,seed)).encode('ascii'))
        if step%10 == 0 or step == a.step:
            records.append(dict(step=step,mse=float(mse.detach()),direction_loss=float(direction.detach()),
                dm_loss=dm_loss,model_seed=model_seed,model_round=model_round,seed=seed,
                elapsed_seconds=elapsed+time.monotonic()-tick))
            c.atomic_json(path/'status.json',dict(step=step,target=a.step,session_steps=step-start,
                session_seconds=time.monotonic()-tick,elapsed_seconds=elapsed+time.monotonic()-tick))
        if step%50 == 0 or step == a.step: persist(step,snapshot=step == a.step)
        if step%100 == 0: print(f'{a.alpha} client{a.client:02d} {a.arm}: {step}/{a.step}',flush=True)
    if start == a.step: persist(start,snapshot=True)
    return True


def evaluate(a):
    rx, ry, original, weights = context(a)
    item = original if a.step == 0 else c.load(folder(a)/'snapshots'/f'step_{a.step:04d}.pt')
    files = {}
    for model_seed in c.SEEDS:
        for model_round in EVAL_ROUNDS:
            net = c.reference(a.source_root,a.alpha,model_seed,model_round)
            pm, rm, em = ([c.Moments() for _ in range(3)] for _ in range(3))
            draws = []; real_sequence = hashlib.sha256()
            for repeat in range(30):
                seed = EVAL_SEED+a.client*100000+model_round*100+(model_seed-2025)*10000000+repeat
                real = d.trace_update(net,rx,ry,seed)
                proxy = d.trace_update(net,item['images'],item['labels'],seed+500000000,weights)
                batch_hashes = {name:hashlib.sha256(b''.join(b.numpy().tobytes() for b in c.batches(size,s))).hexdigest()
                    for name,size,s in (('real',len(ry),seed),('proxy',len(item['labels']),seed+500000000))}
                real_sequence.update(real[-1].cpu().numpy().tobytes())
                draws.append(dict(repeat=repeat,seed=seed,batch_indices_sha256=batch_hashes,
                                  steps=[d.safe_geometry(p,r) for p,r in zip(proxy,real)]))
                for i in range(3): pm[i].add(proxy[i]); rm[i].add(real[i]); em[i].add(proxy[i]-real[i])
            steps = []
            for i in range(3):
                p,r = pm[i].mean,rm[i].mean; layers, offset = {},0
                for name,parameter in net.named_parameters():
                    end=offset+parameter.numel(); layers[name]=d.safe_geometry(p[offset:end],r[offset:end]); offset=end
                correction=em[i].corrected_squared_mean()/max(float(r.square().sum()),1e-30)
                steps.append(dict(step=i+1,mean_geometry=d.safe_geometry(p,r),layers=layers,
                    corrected_relative_squared_mean_error=correction,negative_noise_corrected_estimate=correction<0))
            point=dict(alpha=a.alpha,client=a.client,arm=a.arm,additional_steps=a.step,
                       model_seed=model_seed,model_round=model_round)
            name=f'{a.alpha}_client{a.client:02d}_{a.arm}_s{a.step:04d}_seed{model_seed}_r{model_round:04d}.json'
            target=a.output/'evaluation'/name
            c.atomic_json(target,dict(point=point,repeats=30,steps=steps,individual_geometry=draws,
                real_update_sequence_sha256=real_sequence.hexdigest()))
            files[name]=c.digest(target)
            print('Complete '+name,flush=True)
    c.atomic_json(a.output/'completed'/f'evaluate_{a.alpha}_{a.client:02d}_{a.arm}_{a.step}.json',dict(files_sha256=files))
    return True


def summary(output, maximum=1000):
    rows=[]
    for alpha in c.ALPHAS:
        for arm in ARMS:
            for step in (0,500,1000):
                if step>maximum: continue
                tag='baseline' if step == 0 else arm
                files=[output/'evaluation'/f'{alpha}_client{client:02d}_{tag}_s{step:04d}_seed{seed}_r{rnd:04d}.json'
                       for client in CLIENTS[alpha] for seed in c.SEEDS for rnd in EVAL_ROUNDS]
                missing=[file.name for file in files if not file.is_file()]
                if missing: raise RuntimeError('Missing evaluation points: '+', '.join(missing))
                items=[c.read(p) for p in files]
                if len(items) != 18: raise RuntimeError('Missing evaluation points')
                rows.append(dict(alpha=alpha,arm=arm,additional_steps=step,points=len(items),
                    relative_error=float(np.mean([v['steps'][2]['mean_geometry']['relative_error'] for v in items])),
                    angle_degrees=float(np.mean([v['steps'][2]['mean_geometry']['angle_degrees'] for v in items])),
                    norm_ratio=float(np.mean([v['steps'][2]['mean_geometry']['norm_ratio'] for v in items])),
                    conv2_angle=float(np.mean([v['steps'][2]['layers'][LAYERS[0]]['angle_degrees'] for v in items])),
                    conv3_angle=float(np.mean([v['steps'][2]['layers'][LAYERS[1]]['angle_degrees'] for v in items])),
                    corrected_relative_squared_mean_error=float(np.mean([v['steps'][2]['corrected_relative_squared_mean_error'] for v in items]))))
    c.atomic_json(output/'summary.json',rows)
    c.atomic_json(output/f'summary_{maximum}.json',rows)
