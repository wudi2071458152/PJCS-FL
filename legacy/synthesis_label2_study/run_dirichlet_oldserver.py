"""One case worker; reuse the original numerical training and synthesis functions."""
import argparse
import copy
import signal
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c
import dirichlet_oldserver_common as common
import dirichlet_gamma24 as gamma
import run_dirichlet_reference as reference
import run_dirichlet_screen as screen
import dirichlet_dsa4000_worker as old
import dirichlet_continue32000_worker as continuation
import run_dirichlet_proxy_fl as fl


def branch_protocol(case, source, branch, loss):
    value = common.protocol(case)
    value.update(loss=loss, baseline_iteration=24000, iterations=[28000, 32000],
        source_protocol_sha256=c.digest(source/'protocol.json'),
        continuation='same24000 images and inherited SGD momentum; immutable source')
    value['loss_settings'] = dict(dm_factor=1. if loss=='original' else .1,
        direction_layers=[] if loss=='original' else ['features.4.weight','features.8.weight'],
        gamma={} if loss=='original' else gamma.gamma_manifest(branch, source_root=source),
        gamma_policy='24000 all32; 384 contexts; fixed30% gradient-norm ratio')
    return value


def prepare_fl(a):
    source = a.source_root; expected = c.read(source/'protocol.json')
    if expected != common.protocol(a.case): raise RuntimeError('FL immutable source protocol changed')
    table = expected['budgets'][a.alpha]; baseline = a.method in ('fedavg','uniform_real')
    for path in (source, a.proxy_root):
        if path is not None and (a.output.is_relative_to(path) or path.is_relative_to(a.output)):
            raise RuntimeError('FL output overlaps source/proxy images')
    files = {}
    def fingerprint(file, name):
        files[name] = c.digest(file); return files[name]
    for name, sha in expected['code_sha256'].items():
        if fingerprint(c.ROOT/name, 'code/'+name) != sha: raise RuntimeError('Bound numerical code changed: '+name)
    for name in ('manifest.json','counts.json'):
        fingerprint(c.INPUTS/name, 'inputs/'+name)
    manifest = c.read(c.INPUTS/'manifest.json')
    for name, sha in manifest['splits'].items():
        if fingerprint(c.INPUTS/name, 'inputs/'+name) != sha: raise RuntimeError('Split changed')
    for name, sha in manifest['data'].items():
        if fingerprint(a.data_root/name, 'data/'+name) != sha: raise RuntimeError('Real CIFAR changed')
    fingerprint(source/'protocol.json', 'source/protocol.json')
    x, y = c.dataset(a.data_root); indices = c.split(a.alpha)
    actual = [torch.bincount(y[row], minlength=10).tolist() for row in indices]
    if len(indices)!=32 or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(50000)) or actual!=table['real_counts'] or sum(table['synthetic_sizes'])!=common.CASES[a.case][2]:
        raise RuntimeError('Real partition/proxy budget mismatch')
    initial = c.model_path(source, a.alpha, 2025, 0)
    marker = c.read(initial.parent.parent/'COMPLETED.json')
    if fingerprint(initial, 'source/initial_model.pt') != marker['model_sha256']['0']:
        raise RuntimeError('FL initial reference model changed')
    weights = np.array([len(row) for row in indices], dtype=float)/50000
    net = c.reference(source, a.alpha, 2025, 0); syn = None; loss = None
    if not baseline:
        proxy = a.proxy_root; settings = c.read(proxy/'protocol.json')
        wanted = expected if a.version in ('old_i8000','old_i24000') else branch_protocol(a.case, source, proxy, a.loss)
        if settings != wanted: raise RuntimeError('Proxy branch protocol changed')
        fingerprint(proxy/'protocol.json', 'proxy/protocol.json'); loss = a.loss
        gamma_binding = gamma.gamma_manifest(proxy, source_root=source) if loss=='low_dm_direction' else None
        syn = []
        for client in range(32):
            file = common.verify_snapshot(proxy, a.alpha, client, a.iteration)
            fingerprint(file, f'proxy/client_{client:02d}/snapshot.pt')
            folder = c.route(proxy,a.alpha,common.CONFIG,client); src = c.route(source,a.alpha,common.CONFIG,client)
            if c.read(folder/'config.json') != c.read(src/'config.json'):
                raise RuntimeError('Inherited client config changed')
            fingerprint(folder/'config.json', f'proxy/client_{client:02d}/config.json')
            coefficient = proxy/'calibration'/a.alpha/'coefficients.json'
            if c.digest(coefficient) != c.digest(source/'calibration'/a.alpha/'coefficients.json'):
                raise RuntimeError('Inherited DM calibration changed')
            fingerprint(coefficient, 'proxy/calibration/coefficients.json')
            if gamma_binding is not None:
                gs = gamma_binding[a.alpha]
                wanted_loss = dict(loss=loss, source_iteration=24000,
                    source_lambda_dm=c.read(src/'config.json')['lambda_dm'],
                    effective_lambda_dm=.1*c.read(src/'config.json')['lambda_dm'],
                    gamma=gs['gamma'], gamma_fingerprints=gs,
                    source_config_sha256=c.digest(src/'config.json'), source_resume_sha256=c.digest(src/'resume.pt'),
                    source_snapshot_sha256=c.digest(c.snapshot(source,a.alpha,common.CONFIG,client,24000)),
                    optimizer=dict(lr=1.,momentum=.5,initial_momentum='inherit24000'))
                if c.read(folder/'loss_settings.json') != wanted_loss:
                    raise RuntimeError('New-loss settings differ from paired 24000 source')
                fingerprint(folder/'loss_settings.json', f'proxy/client_{client:02d}/loss_settings.json')
            item = c.load(file)
            syn.append((item['images'],item['labels'],c.class_weights(table['real_counts'][client],table['synthetic_counts'][client])))
    protocol = dict(version=1, case=a.case, version_name=a.version, alpha=a.alpha, method=a.method,
        proxy_loss=loss, iteration=0 if baseline else a.iteration, rounds=1500, seed=2025, clients=32, selected=3,
        local_steps=3, batch=256, shuffle=True, drop_last=False, real_lr=.01, proxy_lr=.01,
        real_ce='ordinary', proxy_ce=expected['weighted_ce'], weights=weights.tolist(),
        score='norm(P)/(3*.01)', importance='w_i*score_i', residual_alpha=1.,
        renormalize=a.method=='fedavg', aggregation={
            'fedavg':'sum_selected(n_i/sum_selected(n_j)*R_i)',
            'uniform_real':'sum_selected(w_i/pi_i*R_i)', 'score_real':'sum_selected(w_i/pi_i*R_i)',
            'residual_uniform':'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))',
            'residual_score':'sum_all(w_i*P_i)+sum_selected(w_i/pi_i*(R_i-P_i))'}[a.method],
        sampler='existing fixed_aocs_sample; exact marginals' if a.method in ('score_real','residual_score') else 'uniform without replacement',
        files_sha256=files, software=dict(torch=torch.__version__,numpy=np.__version__))
    return net, [(x[row],y[row]) for row in indices], syn, weights, c.dataset(a.data_root,train=False), protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('reference','calibrate','synthesize','vectors','accuracy','gamma','fl'))
    parser.add_argument('--case', choices=common.CASES, required=True)
    parser.add_argument('--inputs-root', type=Path, default=common.INPUTS)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--source-root', type=Path)
    parser.add_argument('--gamma-root', type=Path); parser.add_argument('--proxy-root', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--client', type=int); parser.add_argument('--seed', type=int)
    parser.add_argument('--iteration', type=int); parser.add_argument('--deadline', type=float)
    parser.add_argument('--loss', choices=('original','low_dm_direction'), default='original')
    parser.add_argument('--method', choices=fl.METHODS); parser.add_argument('--version', default='shared')
    parser.add_argument('--run',action='store_true')
    a = parser.parse_args(); common.configure(a.case,a.inputs_root)
    a.alpha = common.CASES[a.case][0]; a.config = common.CONFIG; a.clients = list(range(32)); a.configs=(common.CONFIG,)
    if a.client is not None and a.client not in range(32): parser.error('Client outside0..31')
    if not a.run: print('Preflight only. Worker needs --run to compute.'); return
    if not torch.cuda.is_available(): raise RuntimeError('Formal worker computation requires CUDA')
    def stop(_signal,_frame): a.deadline=0.
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    if a.phase=='reference': done=reference.train(a.root,a.data_root,a.alpha,a.seed,a.deadline)
    elif a.phase=='gamma':
        a.output=a.root; done=gamma.calibrate(a)
    elif a.phase=='fl':
        a.output=a.root; a.rounds=1500; done=fl.train(a,prepare_fl(a))
    else:
        if a.source_root is None: c.bind(a.root/'protocol.json',common.protocol(a.case))
        else:
            continuation.gamma_manifest=lambda root: gamma.gamma_manifest(root,source_root=a.source_root)
            c.bind(a.root/'protocol.json',branch_protocol(a.case,a.source_root,a.root,a.loss))
        function = screen.calibrate if a.phase=='calibrate' else getattr(continuation if a.source_root else old,a.phase)
        done=function(a)
    raise SystemExit(0 if done else 75)


if __name__=='__main__': main()
