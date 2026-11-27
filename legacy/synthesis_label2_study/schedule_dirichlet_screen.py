"""Run the agreed 192-route screen within a wall-clock budget; default is preflight."""
import argparse
import contextlib
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
import dirichlet_screen_common as c

HERE = Path(__file__).resolve().parent


def synthesis_jobs(root, iteration):
    # Interleave alpha and configuration so early progress covers the whole comparison.
    return [dict(name=f'{alpha}_{config}_client{client:02d}_i{iteration}', phase='synthesize',
                 alpha=alpha, config=config, client=client, iteration=iteration,
                 artifact=str(c.route(root, alpha, config, client)/f'COMPLETED_{iteration}.json'))
            for offset in range(8) for config in c.CONFIGS for alpha in c.ALPHAS
            for client in [c.SELECTED[alpha][offset]]]


def jobs(root, phase, iteration=None):
    if phase=='reference':
        return [dict(name=f'reference_{alpha}_{seed}', phase=phase, alpha=alpha, seed=seed,
                     artifact=str(root/'reference'/alpha/str(seed)/'COMPLETED.json'))
                for alpha in c.ALPHAS for seed in c.SEEDS]
    if phase=='calibrate':
        return [dict(name=f'calibrate_{alpha}', phase=phase, alpha=alpha,
                     artifact=str(root/'calibration'/alpha/'COMPLETED.json')) for alpha in c.ALPHAS]
    if phase=='synthesize': return synthesis_jobs(root, iteration)
    evaluation=root/'evaluation'/f'iter_{iteration:04d}'
    if phase=='vectors':
        return [dict(name=f'vectors_{alpha}_i{iteration}', phase=phase, alpha=alpha, iteration=iteration,
                     artifact=str(evaluation/alpha/'vectors/COMPLETED.json')) for alpha in c.ALPHAS]
    if phase=='accuracy':
        return [dict(name=f'accuracy_{alpha}_{config}_i{iteration}', phase=phase, alpha=alpha,
                     config=config, iteration=iteration,
                     artifact=str(evaluation/alpha/'accuracy'/config/'COMPLETED.json'))
                for config in c.CONFIGS for alpha in c.ALPHAS]
    raise ValueError(phase)


def complete(job): return Path(job['artifact']).is_file()


def common_iteration(root):
    for iteration in (1000,500):
        if all(complete(job) for job in synthesis_jobs(root,iteration)): return iteration
    return None


@contextlib.contextmanager
def lock(root):
    path=root/'scheduler.lock'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as file:
        try:
            if os.name=='nt':
                import msvcrt
                if path.stat().st_size==0: file.write(b'0'); file.flush()
                file.seek(0); msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,IOError) as error:
            raise RuntimeError('Another scheduler is already using this output directory') from error
        try: yield
        finally:
            if os.name=='nt': file.seek(0); msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(file,fcntl.LOCK_UN)


def child_command(job,a,deadline):
    script='run_dirichlet_reference.py' if job['phase']=='reference' else 'run_dirichlet_screen.py'
    cmd=[sys.executable,'-u','-B',str(HERE/script)]
    if job['phase']!='reference': cmd.append(job['phase'])
    cmd += ['--root',str(a.output),'--data-root',str(a.data_root),'--deadline',str(deadline)]
    for key in ('alpha','seed','config','client','iteration'):
        if key in job: cmd += ['--'+key.replace('_','-'),str(job[key])]
    return cmd


def run_queue(queue,phase,a,deadline):
    pending=[job for job in queue if not complete(job)]; active=[]; failures=[]; paused=[]
    logdir=a.output/'logs'; logdir.mkdir(parents=True,exist_ok=True)
    print(f'{phase}: {len(queue)-len(pending)}/{len(queue)} already complete',flush=True)
    try:
        while pending or active:
            for entry in active[:]:
                code=entry['process'].poll()
                if code is None: continue
                entry['log'].close(); active.remove(entry)
                if code==75: paused.append(entry['job']['name'])
                elif code!=0 or not complete(entry['job']):
                    failures.append(dict(job=entry['job']['name'],exit_code=code,log=entry['log_path']))
            while pending and len(active)<a.workers and not c.expired(deadline) and not failures:
                counts={gpu:sum(entry['gpu']==gpu for entry in active) for gpu in a.gpus}
                gpu=min(a.gpus,key=lambda item:counts[item]); job=pending.pop(0)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
                env.update({name:'2' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
                log_path=logdir/(job['name']+'.log'); log=log_path.open('a',encoding='utf-8')
                cmd=child_command(job,a,deadline)
                log.write('\nCOMMAND '+__import__('json').dumps(cmd)+'\n'); log.flush()
                try: process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                except BaseException: log.close(); raise
                active.append(dict(job=job,process=process,gpu=gpu,log=log,log_path=str(log_path)))
            state=dict(state='running',phase=phase,active=[dict(job=e['job']['name'],pid=e['process'].pid,gpu=e['gpu']) for e in active],
                       pending=[j['name'] for j in pending],failed=failures,paused=paused,
                       phase_completed=sum(complete(j) for j in queue),phase_total=len(queue),deadline=deadline)
            c.atomic_json(a.output/'status.json',state)
            if not active and (failures or c.expired(deadline)): break
            if active or pending: time.sleep(1)
        if failures: raise RuntimeError('Failed jobs: '+', '.join(f['job'] for f in failures))
        return all(complete(j) for j in queue)
    finally:
        # Only these exact child handles belong to this scheduler; never kill by broad pattern.
        for entry in active:
            entry['process'].terminate()
        for entry in active:
            try: entry['process'].wait(timeout=20)
            except subprocess.TimeoutExpired: entry['process'].kill(); entry['process'].wait()
            entry['log'].close()


def summarize(root,iteration):
    rows=[]
    for alpha in c.ALPHAS:
        for config in c.CONFIGS:
            values=[]
            if iteration:
                folder=root/'evaluation'/f'iter_{iteration:04d}'/alpha
                for file in sorted((folder/'vectors').glob('client_*.json')):
                    item=c.read(file)
                    record=next(r for r in item['records'] if r['config']==config)
                    values.append(record)
                accuracy_file=folder/'accuracy'/config/'result.json'
                accuracy=c.read(accuracy_file) if accuracy_file.exists() else None
            else: accuracy=None
            rows.append(dict(alpha=alpha,config=config,iteration=iteration,vector_points=len(values),vector_target=96,
                mean_relative_error=float(__import__('numpy').mean([r['mean_geometry']['relative_error'] for r in values])) if values else None,
                mean_angle_degrees=float(__import__('numpy').mean([r['mean_geometry']['angle_degrees'] for r in values])) if values else None,
                corrected_relative_squared_error=float(__import__('numpy').mean([r['corrected_relative_squared_mean_error'] for r in values])) if values else None,
                negative_estimates=sum(r['negative_noise_corrected_estimate'] for r in values),
                classifier_complete=accuracy is not None,
                test_accuracy=accuracy['history'][-1]['test_accuracy'] if accuracy else None))
    c.atomic_json(root/'summary.json',rows)
    import csv
    with (root/'summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def inventory(root,iteration):
    expected=jobs(root,'reference')+jobs(root,'calibrate')+jobs(root,'synthesize',500)+jobs(root,'synthesize',1000)
    if iteration: expected += jobs(root,'vectors',iteration)+jobs(root,'accuracy',iteration)
    else: expected += [dict(name='common_checkpoint_and_evaluations',artifact=str(root/'not_available'))]
    return dict(common_iteration=iteration,completed=[j['name'] for j in expected if complete(j)],
                missing=[j['name'] for j in expected if not complete(j)],expected_jobs=len(expected))


def verify_results(root):
    """Check every claimed completed result against its content and recorded fingerprints."""
    errors=[]
    queues=jobs(root,'reference')+jobs(root,'calibrate')+jobs(root,'synthesize',500)+jobs(root,'synthesize',1000)
    for iteration in (500,1000): queues+=jobs(root,'vectors',iteration)+jobs(root,'accuracy',iteration)
    for job in queues:
        if not complete(job): continue
        try:
            marker=c.read(job['artifact']); folder=Path(job['artifact']).parent
            if job['phase']=='synthesize':
                c.validate_snapshot(root,job['alpha'],job['config'],job['client'],job['iteration'])
            elif job['phase']=='reference':
                if marker['rounds']!=1500: raise ValueError('Reference round target mismatch')
                records=c.read(folder/'rounds.json')
                if [r['round'] for r in records]!=list(range(1,1501)): raise ValueError('Missing reference rounds')
                if set(marker['model_sha256'])!={str(r) for r in c.MODEL_ROUNDS}: raise ValueError('Missing model fingerprints')
                for round_number,h in marker['model_sha256'].items():
                    if c.digest(c.model_path(root,job['alpha'],job['seed'],int(round_number)))!=h:
                        raise ValueError('Model checksum mismatch')
            elif job['phase']=='calibrate':
                if marker['records']!=384 or c.digest(folder/'coefficients.json')!=marker['coefficients_sha256']:
                    raise ValueError('Calibration marker mismatch')
                keys={(r['client'],r['model_seed'],r['round'],r['repeat']) for r in c.read(folder/'records.json')}
                expected={(i,s,r,k) for i in range(32) for s in c.SEEDS for r in c.CAL_ROUNDS for k in range(2)}
                if keys!=expected: raise ValueError('Calibration coverage mismatch')
            elif job['phase']=='vectors':
                expected={f'client_{i:02d}_seed{s}_round{r:04d}.json' for i in c.SELECTED[job['alpha']]
                          for s in c.SEEDS for r in c.EVAL_ROUNDS}
                if set(marker['files_sha256'])!=expected: raise ValueError('Vector diagnostic coverage mismatch')
                for name,h in marker['files_sha256'].items():
                    if c.digest(folder/name)!=h: raise ValueError('Vector record checksum mismatch')
                    item=c.read(folder/name)
                    if item['repeats']!=30 or {r['config'] for r in item['records']}!=set(c.CONFIGS):
                        raise ValueError('Vector record protocol mismatch')
            elif job['phase']=='accuracy':
                result=c.read(folder/'result.json')
                if marker['epochs']!=1000 or result['epochs']!=1000 or result['history'][-1]['epoch']!=1000:
                    raise ValueError('Classifier epoch target mismatch')
                if c.digest(folder/'result.json')!=marker['result_sha256']: raise ValueError('Classifier checksum mismatch')
        except (Exception) as exception: errors.append(dict(job=job['name'],error=str(exception)))
    return errors


def bundle(root):
    """Explicit whitelist: all weights, images, resumes and redundant logs stay on server."""
    paths=[]
    for name in ('protocol.json','run_sessions.json','status.json','summary.json','summary.csv','result_index.json','runtime.json'):
        path=root/name
        if path.is_file(): paths.append(path)
    for pattern in ('reference/*/*/rounds.json','reference/*/*/COMPLETED.json',
                    'calibration/*/records.json','calibration/*/coefficients.json','calibration/*/COMPLETED.json',
                    'synthesis/*/*/client_*/config.json','synthesis/*/*/client_*/metrics.json',
                    'synthesis/*/*/client_*/COMPLETED_*.json','evaluation/iter_*/*/vectors/client_*.json',
                    'evaluation/iter_*/*/vectors/COMPLETED.json','evaluation/iter_*/*/accuracy/*/history.json',
                    'evaluation/iter_*/*/accuracy/*/result.json','evaluation/iter_*/*/accuracy/*/COMPLETED.json'):
        paths.extend(sorted(root.glob(pattern)))
    paths=sorted(set(paths))
    manifest=dict(complete=c.read(root/'result_index.json')['complete'],
                  missing=c.read(root/'result_index.json')['missing'],
                  files=[dict(path=p.relative_to(root).as_posix(),sha256=c.digest(p)) for p in paths])
    c.atomic_json(root/'packaging_manifest.json',manifest); paths.append(root/'packaging_manifest.json')
    archive=root/'dirichlet_screen_results.tar.gz'
    temporary=archive.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz') as tar:
        for path in paths: tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    with tarfile.open(temporary,'r:gz') as tar:
        if sorted(tar.getnames())!=sorted(p.relative_to(root).as_posix() for p in paths): raise RuntimeError('Archive member mismatch')
        for row in manifest['files']:
            stream=tar.extractfile(row['path'])
            h=__import__('hashlib').sha256()
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
            if h.hexdigest()!=row['sha256']: raise RuntimeError('Archive checksum mismatch')
    os.replace(temporary,archive)
    (root/(archive.name+'.sha256')).write_bytes((c.digest(archive)+'  '+archive.name+'\n').encode('ascii'))
    return archive


def execute(a):
    start=time.time(); deadline=start+a.hours*3600; synthesis_deadline=deadline-a.evaluation_hours*3600
    session=dict(started_at_epoch=start,deadline=deadline,synthesis_deadline=synthesis_deadline,
                 hours=a.hours,evaluation_hours=a.evaluation_hours,gpus=a.gpus,workers=a.workers,data_root=str(a.data_root))
    sessions=a.output/'run_sessions.json'; previous=c.read(sessions) if sessions.exists() else []
    c.atomic_json(sessions,previous+[session]); c.atomic_json(a.output/'runtime.json',dict(
        python=sys.version,torch=c.torch.__version__,cuda=c.torch.version.cuda,numpy=c.np.__version__,
        gpu=c.torch.cuda.get_device_name(0) if c.torch.cuda.is_available() else None))
    state='budget_exhausted'; error=None; phase='reference'
    try:
        if run_queue(jobs(a.output,'reference'),'reference',a,synthesis_deadline):
            phase='calibrate'
            if run_queue(jobs(a.output,'calibrate'),'calibrate',a,synthesis_deadline):
                phase='synthesis500'
                if run_queue(jobs(a.output,'synthesize',500),phase,a,synthesis_deadline):
                    phase='synthesis1000'
                    run_queue(jobs(a.output,'synthesize',1000),phase,a,synthesis_deadline)
        iteration=common_iteration(a.output)
        if iteration is not None:
            phase='vectors'
            vectors_done=run_queue(jobs(a.output,'vectors',iteration),phase,a,deadline)
            phase='accuracy'
            accuracy_done=run_queue(jobs(a.output,'accuracy',iteration),phase,a,deadline)
            state='complete' if iteration==1000 and vectors_done and accuracy_done else 'budget_exhausted'
    except (Exception,KeyboardInterrupt) as exception:
        state='interrupted' if isinstance(exception,KeyboardInterrupt) else 'failed'; error=str(exception)
    invalid=verify_results(a.output)
    if invalid: state='failed'; error='Result integrity verification failed'
    iteration=common_iteration(a.output); index=inventory(a.output,iteration)
    index['invalid']=invalid
    index.update(complete=state=='complete' and not index['missing'],state=state,error=error)
    c.atomic_json(a.output/'result_index.json',index)
    previous=c.read(a.output/'status.json') if (a.output/'status.json').exists() else {}
    c.atomic_json(a.output/'status.json',dict(state=state,phase=phase,common_iteration=iteration,
                   elapsed_seconds=time.time()-start,active=[],pending=index['missing'],
                   failed=previous.get('failed',[]),error=error))
    summarize(a.output,iteration); archive=bundle(a.output)
    print(f'State={state}; common_iteration={iteration}; result archive={archive}',flush=True)
    if error: raise RuntimeError(error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=c.ROOT.parent/'data')
    p.add_argument('--output',type=Path,default=c.ROOT/'outputs/dirichlet_screen')
    p.add_argument('--gpus',nargs='+',default=['0','1']); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--hours',type=float,default=12); p.add_argument('--evaluation-hours',type=float,default=2)
    p.add_argument('--run',action='store_true'); p.add_argument('--status',action='store_true')
    a=p.parse_args(); a.output=a.output.resolve(); a.data_root=a.data_root.resolve()
    if a.workers<1 or not a.gpus or len(set(a.gpus))!=len(a.gpus): p.error('Invalid GPU/worker configuration')
    if not 0<a.evaluation_hours<a.hours: p.error('Require 0 < evaluation-hours < hours')
    if a.status:
        state=c.read(a.output/'status.json') if (a.output/'status.json').exists() else dict(state='not_started')
        print(__import__('json').dumps(state,indent=2,ensure_ascii=False)); return
    print(__import__('json').dumps(c.preflight(a.data_root),indent=2,ensure_ascii=False))
    if not a.run:
        print('Preflight only. Add --run to start; target 500 then 1000, total budget 12h by default.'); return
    if not c.torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; formal CPU training is not enabled')
    if any(not gpu.isdigit() or int(gpu)>=c.torch.cuda.device_count() for gpu in a.gpus):
        raise RuntimeError('Requested GPUs are not available in this environment')
    with lock(a.output):
        c.bind(a.output/'protocol.json',c.protocol())
        invalid=verify_results(a.output)
        if invalid: raise RuntimeError('Existing results failed verification: '+str(invalid))
        execute(a)


if __name__=='__main__': main()
