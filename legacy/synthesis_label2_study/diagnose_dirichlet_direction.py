"""Read-only layer, three-step and image-gradient diagnostics; default is preflight."""
import argparse
import hashlib
import json
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c
import schedule_dirichlet_screen as scheduler
from run_dirichlet_screen import dm_gradient

CONFIG = 'weighted_weak_x10'
SOURCE_BOUNDS = {key:(.005, 1e-6) for key in ('relative_error', 'proxy_norm', 'real_norm', 'norm_ratio')}
SOURCE_BOUNDS.update(cosine=(0., .001), angle_degrees=(0., .1))


def check_source_geometry(current, original, cross_device=False):
    strict = all(np.isclose(current[k], v, rtol=1e-5, atol=1e-7) for k, v in original.items())
    for key, value in original.items():
        rtol, atol = SOURCE_BOUNDS[key] if cross_device else (1e-5, 1e-7)
        if not np.isfinite(current[key]) or not np.isclose(current[key], value, rtol=rtol, atol=atol):
            raise RuntimeError(f'Original three-step geometry outside {"cross-device" if cross_device else "strict"} bound: '
                               f'{key}, source={value}, local={current[key]}')
    return dict(strict_match=bool(strict), absolute_difference={k:abs(current[k]-v) for k, v in original.items()})


def trace_update(net, x, y, seed, weights=None):
    # Same functional SGD, loss and iterator as c.update; only retain intermediate deltas.
    params = OrderedDict((n, p.detach().clone().requires_grad_(True)) for n, p in net.named_parameters())
    initial = list(params.values()); buffers = OrderedDict((n, b.detach().clone()) for n, b in net.named_buffers())
    result = []
    for batch in c.batches(len(y), seed):
        logits = torch.func.functional_call(net, (params, buffers), (x[batch].to(c.DEVICE),))
        loss = c.ce_loss(logits, y[batch].to(c.DEVICE), weights)
        grads = torch.autograd.grad(loss, tuple(params.values()))
        params = OrderedDict((n, p - .01 * g) for (n, p), g in zip(params.items(), grads))
        result.append(torch.cat([(p-old).reshape(-1) for p, old in zip(params.values(), initial)]).detach())
    return result


def safe_geometry(proxy, real):
    p, r = proxy.double(), real.double(); pn, rn = float(p.norm()), float(r.norm())
    result = dict(proxy_norm=pn, real_norm=rn, squared_error=float((p-r).square().sum()),
                  valid_direction=pn > 1e-15 and rn > 1e-15)
    result.update(c.geometry(p, r) if result['valid_direction'] else
                  dict(relative_error=None if rn <= 1e-15 else float((p-r).norm())/rn,
                       norm_ratio=None if rn <= 1e-15 else pn/rn, cosine=None, angle_degrees=None))
    return result


def gradient_pair(first, second):
    a, b = first.detach().double().flatten(), second.detach().double().flatten()
    an, bn = float(a.norm()), float(b.norm()); valid = an > 1e-15 and bn > 1e-15
    cosine = float(torch.dot(a, b)/(a.norm()*b.norm())) if valid else None
    return dict(first_norm=an, second_norm=bn, valid_direction=valid,
                cosine=None if cosine is None else float(np.clip(cosine, -1, 1)),
                norm_ratio=None if an <= 1e-15 else bn/an)


def image_gradients(net, images, labels, real, rx, ry, seed, weights, coefficient):
    images = images.detach().clone().to(c.DEVICE).requires_grad_(True); labels = labels.to(c.DEVICE)
    real = real.detach().to(c.DEVICE)
    if not np.isfinite(coefficient) or coefficient <= 0: raise RuntimeError('Invalid original DM coefficient')
    proxy = c.update(net, images, labels, seed+500000000, weights, differentiable=True)
    mse = (proxy-real).square().sum()/(real.square().sum()+1e-12)
    direction_valid = float(proxy.detach().norm()) > 1e-15 and float(real.norm()) > 1e-15
    ug = torch.autograd.grad(mse, images, retain_graph=direction_valid)[0]
    dg_direction = None
    if direction_valid:
        p, r = proxy.double(), real.double()
        loss = 1-torch.dot(p, r)/(p.norm()*r.norm())
        dg_direction = torch.autograd.grad(loss, images)[0]
    dm_loss, dg = dm_gradient(net, images, labels, rx, ry, seed+600000000, 'weak')
    scaled = coefficient*dg
    if not all(torch.isfinite(g).all() for g in (ug, dg, scaled)) or (
            dg_direction is not None and not torch.isfinite(dg_direction).all()):
        raise RuntimeError('Nonfinite diagnostic image gradient')
    return dict(update_loss=float(mse.detach()), dm_loss=dm_loss, lambda_dm=coefficient,
                update_vs_scaled_dm=gradient_pair(ug, scaled),
                direction_vs_scaled_dm=None if dg_direction is None else gradient_pair(dg_direction, scaled))


def point_name(point):
    return f"{point['alpha']}_client{point['client']:02d}_seed{point['model_seed']}_r{point['model_round']:04d}_i{point['iteration']}.json"


def vector_path(root, point):
    return root/'evaluation'/f"iter_{point['iteration']:04d}"/point['alpha']/'vectors'/(
        f"client_{point['client']:02d}_seed{point['model_seed']}_round{point['model_round']:04d}.json")


def validate_saved_point(path, point):
    result = c.read(path)
    cross_verified = result.get('reference_validation', {}).get('mode') == 'cross-device' and (
        result.get('reference_validation', {}).get('accepted') is True and result.get('local_update_verified') is True)
    if result.get('point') != point or not (result.get('original_geometry_reproduced') or cross_verified) or (
            result.get('repeats'), result.get('gradient_repeats')) != (30, 3): raise RuntimeError('Invalid saved point')
    if cross_verified and len(result['reference_validation'].get('checks', [])) != 31:
        raise RuntimeError('Missing cross-device comparisons')
    if [r['step'] for r in result['steps']] != [1, 2, 3] or (
            [r['repeat'] for r in result['individual_geometry']] != list(range(30)) or
            [r['repeat'] for r in result['image_gradients']] != list(range(3))):
        raise RuntimeError('Incomplete diagnostic records')
    for row in result['individual_geometry']:
        if len(row['steps']) != 3: raise RuntimeError('Missing intermediate update')
    for row in result['steps']:
        if not row['layers'] or not np.isclose(sum(r['squared_error'] for r in row['layers'].values()),
                row['mean_geometry']['squared_error'], rtol=1e-9, atol=1e-14):
            raise RuntimeError('Layer errors do not reconstruct global error')
    return result


def validate_plan(plan):
    if (plan['config'], plan['diagnostic_repeats'], plan['gradient_repeats'], plan['evaluation_seed']) != (
            CONFIG, 30, 3, 2000002025): raise RuntimeError('Diagnostic definitions changed')
    points = plan['points']
    if not points or len({point_name(p) for p in points}) != len(points): raise RuntimeError('Empty/duplicate points')
    for p in points:
        if p['alpha'] not in c.ALPHAS or p['client'] not in range(32) or p['model_seed'] not in c.SEEDS or (
                p['model_round'] not in (20, 500, 1500) or p['iteration'] not in (4000, 8000, 16000)):
            raise RuntimeError('Unexpected diagnostic point')


def preflight(source, plan):
    validate_plan(plan)
    if c.digest(source/'protocol.json') != plan['source_protocol_sha256']: raise RuntimeError('Source protocol changed')
    for name, expected in plan['source_code_sha256'].items():
        if c.digest(c.ROOT/name) != expected: raise RuntimeError('Numerical source changed: '+name)
    inputs = {'protocol.json':c.digest(source/'protocol.json')}; protocol = c.read(source/'protocol.json')
    if protocol['inputs'] != plan['inputs'] or c.read(c.INPUTS/'manifest.json') != plan['inputs']:
        raise RuntimeError('Original input manifest changed')
    for p in plan['points']:
        alpha, client = p['alpha'], p['client']
        snapshot = c.validate_snapshot(source, alpha, CONFIG, client, p['iteration'])
        marker = snapshot.parent.parent/f"COMPLETED_{p['iteration']}.json"
        if not marker.is_file(): raise RuntimeError('Missing snapshot completion marker')
        model = c.model_path(source, alpha, p['model_seed'], p['model_round'])
        model_marker = model.parent.parent/'COMPLETED.json'
        expected = c.read(model_marker)['model_sha256'][str(p['model_round'])]
        if c.digest(model) != expected: raise RuntimeError('Reference model fingerprint changed')
        record = vector_path(source, p)
        if c.digest(record) != p['source_vector_sha256']: raise RuntimeError('Paired diagnostic record changed')
        config = c.route(source, alpha, CONFIG, client)/'config.json'
        item = c.read(config); coeff_file = source/'calibration'/alpha/'coefficients.json'
        coefficient = 10*c.read(coeff_file)['weighted_weak']['r']
        table = protocol['budgets'][alpha]
        if item['lambda_dm'] != coefficient or item['calibration_sha256'] != c.digest(coeff_file) or (
                item['real_counts'] != table['real_counts'][client] or item['synthetic_counts'] != table['synthetic_counts'][client]):
            raise RuntimeError('Budget/calibrated coefficient changed')
        for file in (snapshot, marker, model, model_marker, record, config, coeff_file):
            key = file.relative_to(source).as_posix()
            if key not in inputs: inputs[key] = c.digest(file)
    return inputs


def export_inputs(source, plan, target):
    inputs = preflight(source, plan); target = target.resolve()
    if target == source or target in source.parents or source in target.parents:
        raise RuntimeError('Input archive must be outside the read-only source')
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = target.with_name(target.name+'.manifest.json')
    c.atomic_json(manifest, dict(source_protocol_sha256=plan['source_protocol_sha256'], files_sha256=inputs))
    with tarfile.open(target, 'w:gz') as tar:
        for name in sorted(inputs): tar.add(source/name, arcname=name, recursive=False)
        tar.add(manifest, arcname='input_manifest.json', recursive=False)
    with tarfile.open(target, 'r:gz') as tar:
        if set(tar.getnames()) != set(inputs)|{'input_manifest.json'}: raise RuntimeError('Input archive coverage mismatch')
        for name, expected in inputs.items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest() != expected: raise RuntimeError('Input archive checksum mismatch')
    target.with_name(target.name+'.sha256').write_bytes((c.digest(target)+'  '+target.name+'\n').encode('ascii'))
    print(f'Exported {len(inputs)} necessary input files: {target}. No diagnostic/training started.', flush=True)


def diagnose_point(source, point, plan, rx, ry, weights, cross_device=False):
    net = c.reference(source, point['alpha'], point['model_seed'], point['model_round'])
    item = c.load(c.snapshot(source, point['alpha'], CONFIG, point['client'], point['iteration']))
    old = next(r for r in c.read(vector_path(source, point))['records'] if r['config'] == CONFIG)
    coefficient = c.read(c.route(source, point['alpha'], CONFIG, point['client'])/'config.json')['lambda_dm']
    rm = [c.Moments() for _ in range(3)]; pm = [c.Moments() for _ in range(3)]
    observed = []; gradients = []; checks = []
    for repeat in range(plan['diagnostic_repeats']):
        seed = plan['evaluation_seed']+point['client']*100000+point['model_round']*100+(
            point['model_seed']-2025)*10000000+repeat
        real = trace_update(net, rx, ry, seed)
        proxy = trace_update(net, item['images'], item['labels'], seed+500000000, weights)
        row = [safe_geometry(p, r) for p, r in zip(proxy, real)]
        if repeat == 0 and cross_device:
            if not torch.equal(real[-1], c.update(net, rx, ry, seed)) or not torch.equal(proxy[-1], c.update(
                    net, item['images'], item['labels'], seed+500000000, weights)):
                raise RuntimeError('Local trace differs from the original update implementation')
        checks.append(check_source_geometry(row[-1], old['individual_geometry'][repeat], cross_device))
        batch_hashes = {name:hashlib.sha256(b''.join(b.numpy().tobytes() for b in c.batches(size, draw_seed))).hexdigest()
                       for name, size, draw_seed in (('real',len(ry),seed),('proxy',len(item['labels']),seed+500000000))}
        observed.append(dict(repeat=repeat, seed=seed, batch_indices_sha256=batch_hashes, steps=row))
        for s in range(3): rm[s].add(real[s]); pm[s].add(proxy[s])
        if repeat < plan['gradient_repeats']:
            gradients.append(dict(repeat=repeat, seed=seed, **image_gradients(
                net, item['images'], item['labels'], real[-1], rx, ry, seed, weights, coefficient)))
    steps = []
    for s in range(3):
        p, r = pm[s].mean, rm[s].mean; layers = {}; offset = 0; total = float((p-r).square().sum())
        for name, parameter in net.named_parameters():
            end = offset+parameter.numel(); metric = safe_geometry(p[offset:end], r[offset:end])
            metric['squared_error_fraction'] = None if total <= 1e-30 else metric['squared_error']/total
            layers[name] = metric; offset = end
        steps.append(dict(step=s+1, mean_geometry=safe_geometry(p, r), layers=layers))
    checks.append(check_source_geometry(steps[-1]['mean_geometry'], old['mean_geometry'], cross_device))
    validation = dict(mode='cross-device' if cross_device else 'strict', accepted=True,
                      strict_match=all(r['strict_match'] for r in checks),
                      maximum_absolute_difference={k:max(r['absolute_difference'][k] for r in checks) for k in old['mean_geometry']},
                      checks=checks, bounds=SOURCE_BOUNDS if cross_device else dict(rtol=1e-5, atol=1e-7))
    return dict(point=point, repeats=30, gradient_repeats=3, original_geometry_reproduced=validation['strict_match'],
                local_update_verified=cross_device, reference_validation=validation,
                steps=steps, individual_geometry=observed, image_gradients=gradients)


def child_command(job, a, deadline):
    cmd = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--source-root', str(a.source_root),
            '--output', str(a.output), '--plan', str(a.plan), '--data-root', str(a.data_root), '--run',
            '--worker', job['name']]
    return cmd+(['--cross-device-check'] if a.cross_device_check else [])


def worker(a, plan):
    x, y = c.dataset(a.data_root); alpha, client = a.worker.split('_client'); client = int(client)
    ids = c.split(alpha)[client]; table = c.read(a.source_root/'protocol.json')['budgets'][alpha]
    weights = c.class_weights(table['real_counts'][client], table['synthetic_counts'][client])
    files = {}
    for p in plan['points']:
        if (p['alpha'], p['client']) != (alpha, client): continue
        path = a.output/'points'/point_name(p)
        if not path.exists(): c.atomic_json(path, diagnose_point(a.source_root, p, plan, x[ids], y[ids], weights, a.cross_device_check))
        validate_saved_point(path, p)
        files[path.name] = c.digest(path); print('Complete '+path.name, flush=True)
    c.atomic_json(a.output/'completed'/(a.worker+'.json'), dict(files_sha256=files))


def bundle(output, plan):
    expected = {f"{p['alpha']}_client{p['client']:02d}": set() for p in plan['points']}
    for p in plan['points']: expected[f"{p['alpha']}_client{p['client']:02d}"].add(point_name(p))
    markers = {p.stem:p for p in (output/'completed').glob('*.json')}
    if set(markers) != set(expected): raise RuntimeError('Missing/unexpected diagnostic completion markers')
    for name, marker in markers.items():
        hashes = c.read(marker)['files_sha256']
        if set(hashes) != expected[name]: raise RuntimeError('Diagnostic marker coverage mismatch')
        for file, value in hashes.items():
            if c.digest(output/'points'/file) != value: raise RuntimeError('Diagnostic result checksum mismatch')
    for p in plan['points']: validate_saved_point(output/'points'/point_name(p), p)
    files = [output/'protocol.json', output/'status.json']+[output/'points'/point_name(p) for p in plan['points']]
    files += sorted(markers.values())
    for file in files:
        if not file.is_file(): raise RuntimeError('Missing diagnostic result: '+str(file))
    manifest = dict(complete=True, points=len(plan['points']), files=[dict(path=p.relative_to(output).as_posix(), sha256=c.digest(p)) for p in files])
    c.atomic_json(output/'packaging_manifest.json', manifest); files.append(output/'packaging_manifest.json')
    target = output/'dirichlet_direction_results.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        for file in files: tar.add(file, arcname=file.relative_to(output).as_posix(), recursive=False)
    with tarfile.open(target, 'r:gz') as tar:
        if set(tar.getnames()) != {p.relative_to(output).as_posix() for p in files}: raise RuntimeError('Archive coverage mismatch')
        for item in manifest['files']:
            if hashlib.sha256(tar.extractfile(item['path']).read()).hexdigest() != item['sha256']: raise RuntimeError('Archive hash mismatch')
    target.with_name(target.name+'.sha256').write_bytes((c.digest(target)+'  '+target.name+'\n').encode('ascii'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=c.ROOT/'outputs/dirichlet_continue24000')
    parser.add_argument('--output', type=Path, default=c.ROOT/'outputs/dirichlet_direction_diagnostic')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('dirichlet_direction_plan.json'))
    parser.add_argument('--data-root', type=Path); parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--worker')
    parser.add_argument('--cross-device-check', action='store_true', help='Explicit bounded cross-device check; retain source deviations')
    parser.add_argument('--export-inputs', type=Path, help='Pack saved inputs for another machine; compute nothing')
    parser.add_argument('--run', action='store_true'); parser.add_argument('--status', action='store_true')
    a = parser.parse_args(); a.source_root = a.source_root.resolve(); a.output = a.output.resolve(); a.plan = a.plan.resolve()
    if a.status:
        print(json.dumps(c.read(a.output/'status.json') if (a.output/'status.json').exists() else {'state':'not_started'}, indent=2)); return
    plan = c.read(a.plan); validate_plan(plan)
    if a.output == a.source_root or a.output in a.source_root.parents or a.source_root in a.output.parents:
        raise RuntimeError('Diagnostic output must be separate from synthesis results')
    if a.export_inputs:
        if a.run or a.worker: parser.error('--export-inputs cannot launch diagnostics')
        export_inputs(a.source_root, plan, a.export_inputs); return
    if a.data_root is None: a.data_root = Path(c.read(a.source_root/'run_sessions.json')[-1]['data_root'])
    a.data_root = a.data_root.resolve()
    if a.worker:
        if not a.run: parser.error('Worker requires explicit --run')
        worker(a, plan); return
    if a.workers < 1 or not a.gpus or len(set(a.gpus)) != len(a.gpus): parser.error('Invalid concurrency')
    inputs = preflight(a.source_root, plan); c.preflight(a.data_root)
    protocol = dict(plan=plan, plan_sha256=c.digest(a.plan), source_root=str(a.source_root),
                    data_root=str(a.data_root), input_sha256=inputs, code_sha256={
                        'synthesis_label2_study/'+Path(__file__).name:c.digest(__file__)},
                    read_only_inputs=True, image_optimization=False, federated_training=False)
    protocol.update(validation_mode='cross-device' if a.cross_device_check else 'strict',
                    source_runtime=plan.get('source_runtime'), local_runtime=dict(python=sys.version,
                    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_precision=torch.get_float32_matmul_precision(), dtype='float32'),
                    source_geometry_bounds=SOURCE_BOUNDS if a.cross_device_check else dict(rtol=1e-5,atol=1e-7))
    protocol = json.loads(json.dumps(protocol))
    if (a.output/'protocol.json').exists() and c.read(a.output/'protocol.json') != protocol:
        raise RuntimeError('Existing diagnostic protocol/fingerprints changed')
    for point in plan['points']:
        path = a.output/'points'/point_name(point)
        if path.exists(): validate_saved_point(path, point)
    expected_markers = {f"{p['alpha']}_client{p['client']:02d}": set() for p in plan['points']}
    for point in plan['points']:
        expected_markers[f"{point['alpha']}_client{point['client']:02d}"].add(point_name(point))
    for marker in (a.output/'completed').glob('*.json'):
        hashes = c.read(marker)['files_sha256']
        if marker.stem not in expected_markers or set(hashes) != expected_markers[marker.stem]:
            raise RuntimeError('Saved completion marker coverage changed')
        for name, expected in hashes.items():
            if c.digest(a.output/'points'/name) != expected: raise RuntimeError('Saved diagnostic checksum changed')
    print(f'Preflight OK: {len(plan["points"])} points; 30 update repeats, 3 gradient repeats. No computation started.', flush=True)
    if not a.run: return
    if not torch.cuda.is_available() or any(not g.isdigit() or int(g) >= torch.cuda.device_count() for g in a.gpus):
        raise RuntimeError('Requested GPUs unavailable')
    jobs = [dict(name=f'{alpha}_client{client:02d}', artifact=str(a.output/'completed'/f'{alpha}_client{client:02d}.json'))
            for alpha in c.ALPHAS for client in sorted({p['client'] for p in plan['points'] if p['alpha']==alpha})]
    with scheduler.lock(a.output):
        c.bind(a.output/'protocol.json', protocol)
        scheduler.child_command = child_command
        try:
            scheduler.run_queue(jobs, 'direction_diagnostic', a, None)
            c.atomic_json(a.output/'status.json', dict(state='complete', completed_points=len(plan['points']), jobs=len(jobs), failed=[]))
            bundle(a.output, plan)
        except BaseException as error:
            c.atomic_json(a.output/'status.json', dict(state='failed', error=str(error))); raise


if __name__ == '__main__': main()
