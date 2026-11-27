"""Prepare/run 39 additional FL routes; reuse completed 32000 images on amax."""
import argparse
import hashlib
import json
import signal
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import run_dirichlet_cluster_history as w
import schedule_dirichlet_screen as queue

c = w.c
OUTPUT = c.ROOT / 'outputs/dirichlet_cluster_history_39'


def jobs(a):
    routes = [('shared', 'history_uniform_zero', 0)]
    for version in ('old_i32000', 'new_i32000'):
        routes += [(version, method, k) for k in (2, 3)
                   for method in ('cluster_score_real', 'cluster_residual_score')]
        routes += [(version, method, 0) for method in ('history_score_zero', 'history_score_proxy')]
    result = []
    for version, method, k in routes:
        for alpha in c.ALPHAS:
            name = method + (f'_k{k}' if k else '')
            folder = a.output / 'fl' / alpha / version / name
            result.append(dict(name=f'fl_{alpha}_{version}_{name}', alpha=alpha, version=version,
                method=method, clusters=k, artifact=str(folder / 'COMPLETED.json')))
    return result


def worker_args(a, job):
    proxy = None if job['version'] == 'shared' else a.old_root if job['version'] == 'old_i32000' else a.new_root
    return SimpleNamespace(source_root=a.source_root, proxy_root=proxy, data_root=a.data_root,
        output=Path(job['artifact']).parent, alpha=job['alpha'], method=job['method'], clusters=job['clusters'],
        iteration=32000, gamma24=job['version'] == 'new_i32000', rounds=a.rounds)


def child_command(job, a, deadline):
    b = worker_args(a, job)
    command = [sys.executable, '-u', '-B', str(Path(w.__file__).resolve()), '--run']
    for key in ('source_root', 'data_root', 'output', 'alpha', 'method', 'clusters', 'rounds', 'iteration'):
        command += ['--' + key.replace('_', '-'), str(getattr(b, key))]
    if b.proxy_root is not None: command += ['--proxy-root', str(b.proxy_root)]
    if b.gamma24: command.append('--gamma24')
    if deadline is not None: command += ['--deadline', str(deadline)]
    return command


def preflight(a):
    templates = {}; routes = []
    for job in jobs(a):
        key = (job['version'], job['alpha'])
        if key not in templates:
            b = worker_args(a, job)
            # Nine input preparations cover all partitions, images, gamma and initial models.
            templates[key] = w.prepare(b)[-1]
            if b.proxy_root is not None:
                for client in range(32):
                    if not (c.route(b.proxy_root, b.alpha, w.base.CONFIG, client) / 'COMPLETED_32000.json').is_file():
                        raise RuntimeError('A 32000 proxy snapshot lacks its completion marker')
        protocol = w.route_protocol(templates[key], job['method'], job['clusters'])
        folder = Path(job['artifact']).parent
        if (folder / 'protocol.json').exists() and c.read(folder / 'protocol.json') != protocol:
            raise RuntimeError('Existing route inputs/settings changed: ' + job['name'])
        if Path(job['artifact']).exists(): w.verify(folder, a.rounds)
        routes.append(dict(**job, protocol=protocol))
    return dict(version=1, fl_target=39, rounds=a.rounds, source_root=str(a.source_root),
        old_root=str(a.old_root), new_root=str(a.new_root), data_root=str(a.data_root),
        routes=routes, code_sha256={name: c.digest(c.ROOT / name) for name in w.CODE},
        prior_42_routes='retained for comparison; no duplicated baseline runs')


def inventory(a):
    completed = []; missing = []; invalid = []; summary = []
    for job in jobs(a):
        if not Path(job['artifact']).exists(): missing.append(job['name']); continue
        try:
            w.verify(Path(job['artifact']).parent, a.rounds); completed.append(job['name'])
            summary.append(dict(alpha=job['alpha'], version=job['version'], method=job['method'],
                clusters=job['clusters'], **c.read(Path(job['artifact']).parent / 'summary.json')))
        except Exception as exc: invalid.append(dict(job=job['name'], error=str(exc)))
    return dict(expected_jobs=39, rounds=a.rounds, completed=completed, missing=missing, invalid=invalid,
                complete=len(completed) == 39 and not missing and not invalid), summary


def bundle(a, state, error=None):
    index, summary = inventory(a); index.update(state=state, error=error)
    c.atomic_json(a.output / 'result_index.json', index); c.atomic_json(a.output / 'summary.json', summary)
    names = [name for name in ('protocol.json', 'run_sessions.json', 'status.json', 'result_index.json', 'summary.json')
             if (a.output / name).is_file()]
    for job in jobs(a):
        if job['name'] in index['completed']:
            folder = Path(job['artifact']).parent
            names += [(folder / name).relative_to(a.output).as_posix()
                      for name in ('protocol.json', 'rounds.json', 'summary.json', 'COMPLETED.json')]
    hashes = {name: c.digest(a.output / name) for name in names}
    c.atomic_json(a.output / 'package_manifest.json', dict(complete=index['complete'], files_sha256=hashes))
    names.append('package_manifest.json')
    archive = a.output / 'dirichlet_cluster_history_39_results.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in names: tar.add(a.output / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        if set(tar.getnames()) != set(names) or len(tar.getmembers()) != len(names) or not all(m.isfile() for m in tar.getmembers()):
            raise RuntimeError('Result package member mismatch')
        for name, expected in hashes.items():
            h = hashlib.sha256()
            with tar.extractfile(name) as file:
                for block in iter(lambda: file.read(1024 * 1024), b''): h.update(block)
            if h.hexdigest() != expected: raise RuntimeError('Result package checksum mismatch: ' + name)
    # Binary LF avoids the repeated Windows-to-Linux sha256sum line-ending problem.
    archive.with_name(archive.name + '.sha256').write_bytes((c.digest(archive) + '  ' + archive.name + '\n').encode('ascii'))
    return archive, index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=c.ROOT / 'outputs/dirichlet_continue24000')
    p.add_argument('--old-root', type=Path, default=c.ROOT / 'outputs/dirichlet_continue32000/original')
    p.add_argument('--new-root', type=Path, default=c.ROOT / 'outputs/dirichlet_continue32000_gamma24/low_dm_direction')
    p.add_argument('--data-root', type=Path); p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--gpus', nargs='+', default=['0', '1']); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rounds', type=int, default=1500); p.add_argument('--hours', type=float)
    p.add_argument('--run', action='store_true'); p.add_argument('--status', action='store_true')
    a = p.parse_args()
    if a.status:
        status = c.read(a.output / 'status.json') if (a.output / 'status.json').exists() else dict(state='not_started')
        count = sum(Path(j['artifact']).exists() for j in jobs(a))
        print(json.dumps(dict(**status, fl_completed=count, fl_target=39), indent=2)); return
    if a.rounds < 1 or a.workers < 1 or (a.hours is not None and a.hours <= 0): p.error('Positive rounds/workers/hours required')
    for name in ('source_root', 'old_root', 'new_root', 'output'): setattr(a, name, getattr(a, name).resolve())
    if a.data_root is None:
        a.data_root = Path(c.read(a.source_root / 'run_sessions.json')[-1]['data_root'])
    a.data_root = a.data_root.resolve()
    protocol = preflight(a)
    if (a.output / 'protocol.json').exists() and c.read(a.output / 'protocol.json') != protocol:
        raise RuntimeError('Scheduler protocol/input fingerprints changed')
    print(json.dumps(dict(state='preflight_ok', fl_target=39, rounds=a.rounds,
        cluster_routes=24, uniform_zero_history=3, score_history=12, output=str(a.output),
        formal_training_started=False), indent=2), flush=True)
    if not a.run: print('Preflight only. Add --run to start.'); return
    settings = a.output / 'concurrency.json'
    if settings.exists():
        settings = c.read(settings); a.workers = settings.get('workers', a.workers); a.gpus = settings.get('gpus', a.gpus)
    if a.workers < 1 or not a.gpus or len(set(a.gpus)) != len(a.gpus) or not torch.cuda.is_available() or any(not str(g).isdigit() or int(g) >= torch.cuda.device_count() for g in a.gpus):
        raise RuntimeError('Requested CUDA GPU/concurrency configuration unavailable')
    a.gpus = [str(g) for g in a.gpus]
    with queue.lock(a.output):
        c.bind(a.output / 'protocol.json', protocol)
        sessions = c.read(a.output / 'run_sessions.json') if (a.output / 'run_sessions.json').exists() else []
        sessions.append(dict(argv=sys.argv, started_at=time.time(), workers=a.workers, gpus=a.gpus,
            data_root=str(a.data_root), hours=a.hours, threads=2))
        c.atomic_json(a.output / 'run_sessions.json', sessions)
        deadline = time.time() + a.hours * 3600 if a.hours is not None else None
        state = 'failed'; error = None
        def stop(*_): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            with patch.object(queue, 'child_command', child_command):
                state = 'complete' if queue.run_queue(jobs(a), 'federated_learning', a, deadline) else 'budget_exhausted'
        except KeyboardInterrupt: state = 'interrupted'
        except Exception as exc: error = str(exc)
        previous = c.read(a.output / 'status.json') if (a.output / 'status.json').exists() else {}
        previous.update(state=state, active=[], error=error); c.atomic_json(a.output / 'status.json', previous)
        archive, index = bundle(a, state, error)
        print(f'State={state}; valid={len(index["completed"])}/39; results={archive}', flush=True)
        if error or index['invalid'] or (state == 'complete' and not index['complete']):
            raise RuntimeError(error or 'Invalid/missing completed results; inspect result_index.json')
        if state != 'complete': raise SystemExit(75)


if __name__ == '__main__': main()
