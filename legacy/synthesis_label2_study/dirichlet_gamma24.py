"""Fixed Conv2/Conv3 direction coefficient calibrated at the actual 24000 split."""
import argparse
import hashlib
import json
import signal
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c
import dirichlet_direction_pilot as p

BASELINE = 24000
CONFIG = 'weighted_weak_x10'
POLICY = '24000 snapshot; all32 clients; 384 contexts; gamma=0.3*median(update_image_gradient_norm/direction_image_gradient_norm); fixed'
_MANIFEST_CACHE = {}


def contexts(client):
    return [(seed, rnd, repeat, 4000002025 + client*100000 + rnd*100 + (seed-2025)*10000000 + repeat)
            for seed in c.SEEDS for rnd in c.CAL_ROUNDS for repeat in range(2)]


def source_binding(source, alpha, client):
    source = Path(source).resolve(); protocol = c.read(source/'protocol.json')
    if alpha not in protocol['budgets'] or len(protocol['budgets'][alpha]['real_sizes']) != 32:
        raise RuntimeError('Gamma requires the original complete 32-client budget')
    if protocol['inputs'] != c.read(c.INPUTS/'manifest.json'):
        raise RuntimeError('Gamma source inputs changed')
    for rel, sha in protocol['code_sha256'].items():
        if c.digest(c.ROOT/rel) != sha: raise RuntimeError('Gamma source numerical code changed: '+rel)
    split = c.INPUTS/'splits'/f'alpha{alpha}.pt'
    if c.digest(split) != protocol['inputs']['splits'][f'splits/alpha{alpha}.pt']:
        raise RuntimeError('Gamma source split changed')
    snapshot = c.validate_snapshot(source, alpha, CONFIG, client, BASELINE)
    folder = c.route(source, alpha, CONFIG, client)
    resume = c.load(folder/'resume.pt'); saved = c.load(snapshot)
    if resume['iteration'] != BASELINE:
        raise RuntimeError('Gamma source must remain at exactly 24000')
    if not torch.equal(resume['images'],saved['images']) or not torch.equal(resume['labels'],saved['labels']):
        raise RuntimeError('Gamma source checkpoint differs from its 24000 snapshot')
    groups=resume['optimizer']['param_groups']; state=resume['optimizer']['state']
    if len(groups)!=1 or groups[0]['params']!=[0] or groups[0]['lr']!=1. or groups[0]['momentum']!=.5 or set(state)!={0}:
        raise RuntimeError('Gamma source inherited image optimizer changed')
    momentum=state[0]['momentum_buffer']
    if momentum.shape!=saved['images'].shape or not torch.isfinite(momentum).all():
        raise RuntimeError('Gamma source inherited image momentum is invalid')
    models = {}
    for seed in c.SEEDS:
        marker = c.read(source/'reference'/alpha/str(seed)/'COMPLETED.json')
        for rnd in c.CAL_ROUNDS:
            file = c.model_path(source, alpha, seed, rnd); sha = c.digest(file)
            if marker['model_sha256'].get(str(rnd)) != sha:
                raise RuntimeError('Gamma reference model changed')
            models[f'{seed}/{rnd}'] = sha
    code = (Path(__file__), Path(p.__file__), Path(p.d.__file__), Path(c.__file__), c.ROOT/'networks.py', c.ROOT/'utils.py')
    budget = protocol['budgets'][alpha]
    return dict(source_root=str(source), source_iteration=BASELINE, alpha=alpha, client=client,
        source_protocol_sha256=c.digest(source/'protocol.json'), inputs_manifest_sha256=c.digest(c.INPUTS/'manifest.json'),
        counts_sha256=c.digest(c.INPUTS/'counts.json'), split_sha256=c.digest(split),
        budget_sha256=hashlib.sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest(),
        real_counts=budget['real_counts'][client], synthetic_counts=budget['synthetic_counts'][client],
        snapshot_sha256=c.digest(snapshot), config_sha256=c.digest(folder/'config.json'),
        resume_sha256=c.digest(folder/'resume.pt'), reference_models_sha256=models,
        data_sha256=protocol['inputs']['data'],
        code_sha256={file.relative_to(c.ROOT).as_posix():c.digest(file) for file in code})


def validate_rows(item, binding, client, complete=True):
    rows = item.get('records', [])
    if item.get('source_fingerprint') != binding:
        raise RuntimeError('Gamma calibration source/fingerprint changed')
    observed = [(r['model_seed'], r['model_round'], r['repeat'], r['seed']) for r in rows]
    expected = contexts(client)
    if observed != expected[:len(rows)] or len(rows) > len(expected) or (complete and len(rows) != 12):
        raise RuntimeError('Gamma calibration contexts missing or changed')
    for row in rows:
        values = [row[k] for k in ('update_norm','direction_norm','ratio')]
        if not all(np.isfinite(v) and v > 0 for v in values) or not np.isclose(row['ratio'], row['update_norm']/row['direction_norm'], rtol=1e-12, atol=0.):
            raise RuntimeError('Gamma contains invalid gradient norms/ratio')
    return rows


def calibrate(a):
    binding = source_binding(a.source_root, a.alpha, a.client)
    for name, sha in binding['data_sha256'].items():
        if c.digest(Path(a.data_root)/name) != sha: raise RuntimeError('Gamma calibration CIFAR data changed')
    folder = Path(a.output)/'calibration'/a.alpha
    file = folder/f'client_{a.client:02d}.json'; marker = folder/f'COMPLETED_client{a.client:02d}.json'
    rows = validate_rows(c.read(file), binding, a.client, False) if file.exists() else []
    if marker.exists():
        validate_rows(c.read(file), binding, a.client)
        if c.read(marker) != dict(contexts=12, file_sha256=c.digest(file)):
            raise RuntimeError('Gamma client completion checksum changed')
        return True
    x, y = c.dataset(a.data_root); ids = c.split(a.alpha)[a.client]
    item = c.load(c.snapshot(a.source_root, a.alpha, CONFIG, a.client, BASELINE))
    images = item['images'].to(c.DEVICE).clone().requires_grad_(True); labels = item['labels'].to(c.DEVICE)
    weights = c.class_weights(binding['real_counts'], binding['synthetic_counts'])
    for seed, rnd, repeat, draw in contexts(a.client)[len(rows):]:
        if c.expired(a.deadline): return False
        net = c.reference(a.source_root, a.alpha, seed, rnd)
        real = c.update(net, x[ids], y[ids], draw)
        mse, direction, _ = p.objective(net, images, labels, real, draw+500000000, weights)
        gu = torch.autograd.grad(mse, images, retain_graph=True)[0]
        gd = torch.autograd.grad(direction, images)[0]
        pair = p.d.gradient_pair(gu, gd)
        if not pair['valid_direction']: raise RuntimeError('Cannot calibrate zero direction/update gradient')
        rows.append(dict(model_seed=seed, model_round=rnd, repeat=repeat, seed=draw,
            update_norm=pair['first_norm'], direction_norm=pair['second_norm'], ratio=pair['first_norm']/pair['second_norm']))
        c.atomic_json(file, dict(source_fingerprint=binding, records=rows))
    c.atomic_json(marker, dict(contexts=12, file_sha256=c.digest(file)))
    return True


def coefficients(output, source_root, alphas=None):
    for alpha in alphas or c.ALPHAS:
        folder = Path(output)/'calibration'/alpha; values = []; sources = {}; files = {}
        for client in range(32):
            file = folder/f'client_{client:02d}.json'; binding = source_binding(source_root, alpha, client)
            rows = validate_rows(c.read(file), binding, client)
            values.extend(r['ratio'] for r in rows); sources[str(client)] = binding; files[file.name] = c.digest(file)
        c.bind(folder/'gamma.json', dict(gamma=.3*float(np.median(values)), target_fraction=.3, fixed=True,
            source_iteration=BASELINE, calibration_clients=list(range(32)), contexts=384, policy=POLICY,
            ratio_median=float(np.median(values)), q25=float(np.quantile(values,.25)), q75=float(np.quantile(values,.75)),
            source_fingerprints=sources, files_sha256=files))
    return gamma_manifest(output, alphas, source_root)


def gamma_manifest(root, alphas=None, source_root=None):
    root = Path(root).resolve(); alphas = tuple(alphas or c.ALPHAS)
    key = (str(root), alphas, str(Path(source_root).resolve()) if source_root else None, str(c.INPUTS.resolve()))
    def stats(paths):
        return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in paths)
    cached = _MANIFEST_CACHE.get(key)
    if cached and stats(cached[0]) == cached[1]: return cached[2]
    watched = {Path(__file__)}
    result = {}
    for alpha in alphas:
        folder = Path(root)/'calibration'/alpha; file = folder/'gamma.json'; item = c.read(file)
        watched.add(file)
        names = {f'client_{i:02d}.json' for i in range(32)}
        if item.get('fixed') is not True or item.get('target_fraction') != .3 or item.get('source_iteration') != BASELINE or item.get('contexts') != 384 or item.get('calibration_clients') != list(range(32)) or item.get('policy') != POLICY or set(item.get('files_sha256',{})) != names or set(item.get('source_fingerprints',{})) != {str(i) for i in range(32)}:
            raise RuntimeError('Gamma24 calibration definition/coverage changed')
        ratios = []
        for client in range(32):
            member = folder/f'client_{client:02d}.json'
            if c.digest(member) != item['files_sha256'][member.name]: raise RuntimeError('Gamma24 member checksum changed')
            stored = item['source_fingerprints'][str(client)]
            source = Path(source_root or stored['source_root'])
            binding = source_binding(source, alpha, client)
            if binding != stored: raise RuntimeError('Gamma24 source binding changed')
            ratios.extend(r['ratio'] for r in validate_rows(c.read(member), binding, client))
            watched.update((member, source/'protocol.json', c.INPUTS/'manifest.json',c.INPUTS/'counts.json',
                c.INPUTS/'splits'/f'alpha{alpha}.pt',c.snapshot(source,alpha,CONFIG,client,BASELINE),
                c.route(source,alpha,CONFIG,client)/'config.json',c.route(source,alpha,CONFIG,client)/'resume.pt'))
            watched.update(c.ROOT/rel for rel in binding['code_sha256'])
            watched.update(c.ROOT/rel for rel in c.read(source/'protocol.json')['code_sha256'])
            for seed in c.SEEDS:
                watched.add(source/'reference'/alpha/str(seed)/'COMPLETED.json')
                watched.update(c.model_path(source,alpha,seed,rnd) for rnd in c.CAL_ROUNDS)
        if not np.isclose(item['gamma'], .3*np.median(ratios), rtol=1e-12, atol=0.):
            raise RuntimeError('Gamma24 coefficient differs from its fixed 30% rule')
        result[alpha] = dict(gamma=item['gamma'], gamma_sha256=c.digest(file), files_sha256=item['files_sha256'],
            source_iteration=BASELINE, contexts=384, source_fingerprints=item['source_fingerprints'])
    paths = sorted(watched); _MANIFEST_CACHE[key] = (paths,stats(paths),result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True); parser.add_argument('--alpha',choices=c.ALPHAS,required=True)
    parser.add_argument('--client',type=int,required=True); parser.add_argument('--deadline',type=float)
    parser.add_argument('--run',action='store_true'); a=parser.parse_args()
    if a.client not in range(32): parser.error('Client must be 0..31')
    if not a.run: source_binding(a.source_root,a.alpha,a.client); print('Preflight OK. No calibration started.'); return
    if not torch.cuda.is_available(): raise RuntimeError('Formal calibration requires CUDA')
    signal.signal(signal.SIGTERM,lambda *_:setattr(a,'deadline',0.))
    raise SystemExit(0 if calibrate(a) else 75)


if __name__=='__main__': main()
