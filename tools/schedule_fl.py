"""Plan the paper matrix, or explicitly run it with configurable GPU concurrency."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def plan(config, seeds=None, methods=None):
    seeds, methods = seeds or config['fl_seeds'], methods or config['routes']
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config['fl_seeds']):
        raise ValueError('Unsupported or duplicate seeds')
    if len(set(methods)) != len(methods) or not set(methods) <= set(config['routes']):
        raise ValueError('Unsupported or duplicate methods')
    return [dict(dataset=d, alpha=a, method=m, seed=s,
                 phase='first_seed' if s == seeds[0] else 'remaining_seeds')
            for s in seeds for d in config['dataset_names'] for a in config['alphas'] for m in methods]


@contextmanager
def lock(path):
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if not path.stat().st_size:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args, jobs):
    output = args.output_root.resolve()
    for protected in (args.artifacts_root, args.cifar_data_root, args.fmnist_data_root):
        protected = protected.resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Output overlaps immutable input directories')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    status = output / 'scheduler_status.json'
    completed = []
    with lock(output / 'scheduler.lock'):
        for phase in ('first_seed', 'remaining_seeds'):
            pending = [job for job in jobs if job['phase'] == phase]
            active = {}
            try:
                while pending or active:
                    for slot in range(args.workers):
                        if not pending or slot in active:
                            continue
                        job = pending.pop(0)
                        folder = output / job['dataset'] / job['alpha'] / str(job['seed']) / job['method']
                        folder.mkdir(parents=True, exist_ok=True)
                        label = '_'.join(str(job[k]) for k in ('dataset', 'alpha', 'seed', 'method'))
                        log = (logs / (label + '.log')).open('a', encoding='utf-8')
                        data = args.cifar_data_root if job['dataset'] == 'cifar10' else args.fmnist_data_root
                        command = [sys.executable, '-u', '-B', str(ROOT / 'run_fl.py'),
                                   '--dataset', job['dataset'], '--alpha', job['alpha'],
                                   '--route', job['method'], '--seed', str(job['seed']),
                                   '--artifacts-root', str(args.artifacts_root), '--data-root', str(data),
                                   '--output', str(folder), '--run']
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus[slot % len(args.gpus)])
                        env.update({name: '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                         'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                        try:
                            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                        except BaseException:
                            log.close(); raise
                        active[slot] = process, log, job
                    for slot, (process, log, job) in list(active.items()):
                        code = process.poll()
                        if code is None:
                            continue
                        log.close(); del active[slot]
                        if code:
                            raise RuntimeError('Worker failed: ' + label + '; inspect the logs directory')
                        completed.append(job)
                    snapshot = dict(state='running', phase=phase, completed=len(completed), total=len(jobs),
                                    active=[dict(job, pid=p.pid) for p, _, job in active.values()])
                    temporary = status.with_suffix('.json.partial')
                    temporary.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
                    temporary.replace(status)
                    if active:
                        time.sleep(1)
                (output / (phase + '_complete.json')).write_text(
                    json.dumps(dict(completed=[j for j in completed if j['phase'] == phase]), indent=2) + '\n',
                    encoding='utf-8')
            except BaseException as exc:
                status.write_text(json.dumps(dict(state='failed', phase=phase, completed=len(completed),
                                                  total=len(jobs), error=str(exc)), indent=2) + '\n', encoding='utf-8')
                raise
            finally:
                for process, log, _ in active.values():
                    if process.poll() is None:
                        process.terminate()
                for process, log, _ in active.values():
                    process.wait(); log.close()
        status.write_text(json.dumps(dict(state='complete', completed=len(completed), total=len(jobs)), indent=2) + '\n', encoding='utf-8')


def main():
    config = json.loads((ROOT / 'configs/paper_lr001.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--cifar-data-root', type=Path, default=ROOT / 'data/cifar10')
    parser.add_argument('--fmnist-data-root', type=Path, default=ROOT / 'data/fmnist')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/paper_lr001')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.gpus = args.gpus.split(',')
    if args.workers < 1 or any(not value.isdigit() for value in args.gpus):
        parser.error('workers must be positive; gpus must be comma-separated nonnegative IDs')
    jobs = plan(config, args.seeds, args.methods)
    print(json.dumps(dict(total=len(jobs), first_seed=sum(j['phase'] == 'first_seed' for j in jobs),
                          workers=args.workers, gpus=args.gpus, formal_training_started=args.run)))
    if args.run:
        run(args, jobs)


if __name__ == '__main__':
    main()
