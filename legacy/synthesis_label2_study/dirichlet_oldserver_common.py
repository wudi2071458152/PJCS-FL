"""Case binding and integrity checks; numerical operations stay in existing workers."""
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c

CASES = {'a0p05_b20': ('0p05', .2, 10000), 'a0p1_b30': ('0p1', .3, 15000)}
CONFIG = 'weighted_weak_x10'
INPUTS = c.ROOT/'inputs/dirichlet_oldserver_20261009'
BASE_PROTOCOL = c.protocol
SOURCE_STAGES = (4000, 8000, 16000, 24000)
BRANCH_STAGES = (28000, 32000)


def allocate(counts, total):
    counts = np.asarray(counts, dtype=int)
    sizes = counts.sum(1); present = (counts > 0).sum(1)
    budgets = c.bounded_allocate(sizes, total, present, sizes)
    classes = [c.bounded_allocate(row, int(m), (row > 0).astype(int), row).tolist()
               for row, m in zip(counts, budgets)]
    return dict(real_counts=counts.tolist(), real_sizes=sizes.tolist(),
                synthetic_sizes=budgets.tolist(), synthetic_counts=classes)


def configure(case, inputs_root=INPUTS):
    alpha, ratio, total = CASES[case]
    c.INPUTS = Path(inputs_root)/case; c.ALPHAS = (alpha,)
    c.CONFIGS = (CONFIG,); c.SELECTED = {alpha: list(range(32))}
    c.budget_table = lambda counts: allocate(counts, total)
    c.protocol = lambda: protocol(case)


def protocol(case):
    alpha, ratio, total = CASES[case]
    value = BASE_PROTOCOL()
    value.update(version=1, suite='dirichlet_oldserver_20261009', case=case,
        budget_ratio=ratio, budget_total=total, iterations=list(SOURCE_STAGES),
        diagnostic_clients='all32', configs=[CONFIG], source_iteration=24000,
        gamma_policy=dict(source_iteration=24000, clients=list(range(32)),
            contexts_per_client=12, contexts_per_case=384, fraction=.3, fixed=True),
        branch_milestones=list(BRANCH_STAGES), inherited_momentum=True)
    for name in ('dirichlet_oldserver_common.py', 'run_dirichlet_oldserver.py',
                 'schedule_dirichlet_oldserver.py', 'dirichlet_dsa4000_worker.py',
                 'dirichlet_continue32000_worker.py', 'dirichlet_direction_pilot.py',
                 'diagnose_dirichlet_direction.py', 'run_dirichlet_proxy_fl.py', 'dirichlet_gamma24.py'):
        file = Path(__file__).with_name(name)
        value['code_sha256'][file.relative_to(c.ROOT).as_posix()] = c.digest(file)
    return json.loads(json.dumps(value))


def preflight(case, data_root, inputs_root=INPUTS):
    configure(case, inputs_root)
    alpha, ratio, total = CASES[case]
    manifest = c.read(c.INPUTS/'manifest.json')
    if manifest['budget_ratio'] != ratio or manifest['budget_total'] != total:
        raise RuntimeError('Case budget manifest mismatch')
    if c.digest(c.INPUTS/'counts.json') != manifest['counts_sha256']:
        raise RuntimeError('Case counts metadata changed')
    table = c.budget_table(c.read(c.INPUTS/'counts.json')['scenarios'][0]['class_counts'])
    if table != manifest['budgets'] or sum(table['synthetic_sizes']) != total:
        raise RuntimeError('Case budget allocation mismatch')
    result = c.preflight(data_root)
    if not hasattr(torch, 'func') or not hasattr(torch.func, 'functional_call'):
        raise RuntimeError('This environment needs PyTorch 2.x with torch.func.functional_call')
    result.update(case=case, alpha=alpha, budget_ratio=ratio, budget_total=total,
        synthesis_jobs=32, calibration='all32', formal_training_started=False)
    return result


def source_files(source):
    """The paired branches receive separate copies, including the 24000 momentum."""
    patterns = ('reference/*/*/models/*.pt', 'reference/*/*/COMPLETED.json',
        'reference/*/*/rounds.json', 'calibration/*/records.json',
        'calibration/*/coefficients.json', 'calibration/*/COMPLETED.json',
        'synthesis/*/*/client_*/config.json', 'synthesis/*/*/client_*/resume.pt',
        'synthesis/*/*/client_*/metrics.json', 'synthesis/*/*/client_*/COMPLETED_*.json',
        'synthesis/*/*/client_*/snapshots/iter_*.pt')
    return sorted({file for pattern in patterns for file in source.glob(pattern)})


def import_branch(source, branch):
    files = source_files(source)
    manifest = dict(source_protocol_sha256=c.digest(source/'protocol.json') if (source/'protocol.json').exists() else None,
        source_iteration=24000, inherited_momentum=True,
        files={file.relative_to(source).as_posix(): c.digest(file) for file in files})
    path = branch/'resume_import.json'
    if path.exists():
        if c.read(path) != manifest: raise RuntimeError('Immutable 24000 source changed')
        return
    for file in files:
        target = branch/file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and c.digest(target) != c.digest(file):
            raise RuntimeError('Unbound existing branch input differs: '+str(target))
        if not target.exists(): shutil.copyfile(file, target)
    c.bind(path, manifest)


def verify_snapshot(root, alpha, client, iteration):
    file = c.validate_snapshot(root, alpha, CONFIG, client, iteration)
    marker = c.route(root, alpha, CONFIG, client)/f'COMPLETED_{iteration}.json'
    if not marker.exists(): raise RuntimeError('Missing image completion marker: '+str(marker))
    return file


def verify_resume(source, alpha, client):
    route = c.route(source, alpha, CONFIG, client)
    item = c.load(route/'resume.pt')
    if item['iteration'] != 24000 or not isinstance(item.get('optimizer'), dict):
        raise RuntimeError('Paired source needs exact 24000 checkpoint and optimizer')
    snap = c.load(verify_snapshot(source, alpha, client, 24000))
    if not torch.equal(snap['images'], item['images']) or not torch.equal(snap['labels'], item['labels']):
        raise RuntimeError('24000 checkpoint and snapshot differ')
    if not item['optimizer'].get('state') or not all('momentum_buffer' in state for state in item['optimizer']['state'].values()):
        raise RuntimeError('24000 SGD momentum is missing')
    return item


def verify_job(job):
    marker = Path(job['artifact']); root = Path(job['root']); alpha = job['alpha']
    if not marker.is_file(): return False
    item = c.read(marker); folder = marker.parent
    if job['phase'] == 'reference':
        if item['rounds'] != 1500 or set(item['model_sha256']) != {str(r) for r in c.MODEL_ROUNDS}:
            raise RuntimeError('Reference model coverage changed')
        if [row['round'] for row in c.read(folder/'rounds.json')] != list(range(1, 1501)):
            raise RuntimeError('Reference rounds incomplete')
        for rnd, sha in item['model_sha256'].items():
            if c.digest(c.model_path(root, alpha, job['seed'], int(rnd))) != sha:
                raise RuntimeError('Reference model fingerprint changed')
    elif job['phase'] == 'calibrate':
        rows = c.read(folder/'records.json')
        expected = {(i,s,r,k) for i in range(32) for s in c.SEEDS for r in c.CAL_ROUNDS for k in range(2)}
        if len(rows) != 384 or {(r['client'],r['model_seed'],r['round'],r['repeat']) for r in rows} != expected or item['records'] != 384 or c.digest(folder/'coefficients.json') != item['coefficients_sha256']:
            raise RuntimeError('DM calibration coverage/fingerprint mismatch')
    elif job['phase'] == 'synthesize':
        file=verify_snapshot(root, alpha, job['client'], job['iteration'])
        if item != dict(iteration=job['iteration'],snapshot_sha256=c.digest(file)):
            raise RuntimeError('Synthesis completion marker iteration/fingerprint mismatch')
        if job.get('loss')=='low_dm_direction':
            import dirichlet_gamma24 as gamma
            source=Path(job['source_root']); src=c.route(source,alpha,CONFIG,job['client'])
            gs=gamma.gamma_manifest(root,source_root=source)[alpha]
            source_lambda=c.read(src/'config.json')['lambda_dm']
            expected=dict(loss='low_dm_direction',source_iteration=24000,source_lambda_dm=source_lambda,
                effective_lambda_dm=.1*source_lambda,gamma=gs['gamma'],gamma_fingerprints=gs,
                source_config_sha256=c.digest(src/'config.json'),source_resume_sha256=c.digest(src/'resume.pt'),
                source_snapshot_sha256=c.digest(c.snapshot(source,alpha,CONFIG,job['client'],24000)),
                optimizer=dict(lr=1.,momentum=.5,initial_momentum='inherit24000'))
            if c.read(folder/'loss_settings.json')!=expected: raise RuntimeError('New-loss settings changed')
    elif job['phase'] == 'vectors':
        expected = {f'client_{i:02d}_seed{s}_round{r:04d}.json' for i in range(32) for s in c.SEEDS for r in c.EVAL_ROUNDS}
        if item['points'] != 384 or item['repeats'] != 30 or item['configs'] != 1 or set(item['files_sha256']) != expected:
            raise RuntimeError('All-client vector coverage mismatch')
        for name, sha in item['files_sha256'].items():
            if c.digest(folder/name) != sha: raise RuntimeError('Vector fingerprint changed')
    elif job['phase'] == 'accuracy':
        result = c.read(folder/'result.json')
        if item['epochs'] != 1000 or result['epochs'] != 1000 or result['history'][-1]['epoch'] != 1000 or c.digest(folder/'result.json') != item['result_sha256']:
            raise RuntimeError('Classifier result incomplete/changed')
    elif job['phase'] == 'gamma':
        import dirichlet_gamma24 as gamma
        file = folder/f"client_{job['client']:02d}.json"
        if item != dict(contexts=12, file_sha256=c.digest(file)):
            raise RuntimeError('Gamma client marker mismatch')
        gamma.validate_rows(c.read(file), gamma.source_binding(Path(job['source_root']), alpha, job['client']), job['client'])
    elif job['phase'] == 'fl':
        import run_dirichlet_proxy_fl as fl
        fl.verify(folder, 1500)
        settings = c.read(folder/'protocol.json')
        if settings['case'] != job['case'] or settings['method'] != job['method'] or settings['version_name'] != job['version']:
            raise RuntimeError('FL route identity changed')
    return True
