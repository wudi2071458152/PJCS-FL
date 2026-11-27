"""Two cases on three GPUs: fresh synthesis, paired loss branches, then28 FL routes."""
import argparse
import hashlib
import json
import os
import signal
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
import dirichlet_screen_common as c
import dirichlet_oldserver_common as common
import dirichlet_gamma24 as gamma
import run_dirichlet_oldserver as worker
import schedule_dirichlet_screen as queue

VERSIONS = ('old_i8000','old_i24000','old_i32000','new_i32000')
METHODS = ('score_real','residual_uniform','residual_score')


def jobs(output, phase, iteration=None, branch='source24000'):
    result=[]
    for case,(alpha,ratio,total) in common.CASES.items():
        root=output/case/branch; base=dict(case=case,alpha=alpha,phase=phase,root=str(root))
        if branch!='source24000':
            base.update(source_root=str(output/case/'source24000'),loss=branch)
        if phase=='reference':
            entries=[dict(seed=seed,artifact=str(root/'reference'/alpha/str(seed)/'COMPLETED.json')) for seed in c.SEEDS]
        elif phase=='calibrate': entries=[dict(artifact=str(root/'calibration'/alpha/'COMPLETED.json'))]
        elif phase=='gamma':
            base.update(root=str(output/case/'gamma24'),source_root=str(output/case/'source24000'))
            entries=[dict(client=i,artifact=str(output/case/'gamma24/calibration'/alpha/f'COMPLETED_client{i:02d}.json')) for i in range(32)]
        elif phase=='synthesize':
            entries=[dict(client=i,iteration=iteration,artifact=str(c.route(root,alpha,common.CONFIG,i)/f'COMPLETED_{iteration}.json')) for i in range(32)]
        elif phase in ('vectors','accuracy'):
            folder=root/'evaluation'/f'iter_{iteration:04d}'/alpha/phase
            if phase=='accuracy': folder/=common.CONFIG
            entries=[dict(iteration=iteration,artifact=str(folder/'COMPLETED.json'))]
        else: raise ValueError(phase)
        for entry in entries:
            job=dict(base,**entry)
            job['name']=f"{case}_{branch}_{phase}"+''.join(f'_{k}{entry[k]}' for k in ('seed','client','iteration') if k in entry)
            result.append(job)
    # Interleave cases to use all three GPUs without finishing a whole case first.
    return sorted(result,key=lambda j:(j.get('client',j.get('seed',0)),j['case']))


def proxy_source(output,case,version):
    if version in ('old_i8000','old_i24000'):
        return output/case/'source24000',int(version.split('i')[1]),'original'
    loss='original' if version=='old_i32000' else 'low_dm_direction'
    return output/case/loss,32000,loss


def fl_jobs(output):
    result=[]
    for version,methods in [('shared',('fedavg','uniform_real'))]+[(v,METHODS) for v in VERSIONS]:
        for method in methods:
            for case,(alpha,ratio,total) in common.CASES.items():
                folder=output/case/'fl'/version/method
                job=dict(name=f'{case}_fl_{version}_{method}',phase='fl',case=case,alpha=alpha,
                    version=version,method=method,root=str(folder),source_root=str(output/case/'source24000'),
                    artifact=str(folder/'COMPLETED.json'))
                if version!='shared':
                    root,iteration,loss=proxy_source(output,case,version)
                    job.update(proxy_root=str(root),iteration=iteration,loss=loss)
                result.append(job)
    return result


def all_jobs(output):
    result=jobs(output,'reference')+jobs(output,'calibrate')
    for iteration in common.SOURCE_STAGES:
        for phase in ('synthesize','vectors','accuracy'): result+=jobs(output,phase,iteration)
    result+=jobs(output,'gamma')
    for branch in ('original','low_dm_direction'):
        for iteration in common.BRANCH_STAGES:
            for phase in ('synthesize','vectors','accuracy'): result+=jobs(output,phase,iteration,branch)
    return result+fl_jobs(output)


def child_command(job,a,deadline):
    command=[sys.executable,'-u','-B',str(Path(worker.__file__).resolve()),job['phase'],
        '--run',
        '--case',job['case'],'--inputs-root',str(a.inputs_root),'--root',job['root'],
        '--data-root',str(a.data_root)]
    if deadline is not None: command+=['--deadline',str(deadline)]
    for key in ('client','seed','iteration','source_root','proxy_root','loss','method','version'):
        if key in job: command+=['--'+key.replace('_','-'),str(job[key])]
    if job.get('loss')=='low_dm_direction': command+=['--gamma-root',job['root']]
    return command


def verify(jobs,a,require_complete=False):
    for job in jobs:
        common.configure(job['case'],a.inputs_root)
        if not common.verify_job(job) and require_complete:
            raise RuntimeError('Expected completed job is missing: '+job['name'])


def run(jobs,phase,a,deadline):
    verify(jobs,a)
    if not queue.run_queue(jobs,phase,a,deadline): return False
    verify(jobs,a,require_complete=True)
    return True


def prepare_branch(a,branch):
    for case,(alpha,ratio,total) in common.CASES.items():
        common.configure(case,a.inputs_root)
        source=a.output/case/'source24000'; target=a.output/case/branch
        if c.read(source/'protocol.json')!=common.protocol(case): raise RuntimeError('24000 source protocol changed')
        for client in range(32): common.verify_resume(source,alpha,client)
        common.import_branch(source,target)
        if branch=='low_dm_direction':
            gammas=a.output/case/'gamma24'
            gamma.gamma_manifest(gammas,source_root=source)
            for file in sorted((gammas/'calibration'/alpha).glob('*.json')):
                dest=target/file.relative_to(gammas); dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and c.digest(dest)!=c.digest(file): raise RuntimeError('Imported gamma changed')
                if not dest.exists(): dest.write_bytes(file.read_bytes())
        c.bind(target/'protocol.json',worker.branch_protocol(case,source,target,branch))


def summaries(output):
    rows=[]
    for case in common.CASES:
        for branch,stages in [('source24000',common.SOURCE_STAGES),('original',common.BRANCH_STAGES),('low_dm_direction',common.BRANCH_STAGES)]:
            for iteration in stages:
                root=output/case/branch; alpha=common.CASES[case][0]
                folder=root/'evaluation'/f'iter_{iteration:04d}'/alpha
                points=[c.read(file)['records'][0] for file in sorted((folder/'vectors').glob('client_*.json'))]
                acc=folder/'accuracy'/common.CONFIG/'result.json'
                if points or acc.exists():
                    rows.append(dict(case=case,branch=branch,iteration=iteration,vector_points=len(points),vector_target=384,
                        mean_relative_error=float(np.mean([p['mean_geometry']['relative_error'] for p in points])) if points else None,
                        mean_angle_degrees=float(np.mean([p['mean_geometry']['angle_degrees'] for p in points])) if points else None,
                        test_accuracy=c.read(acc)['history'][-1]['test_accuracy'] if acc.exists() else None))
    c.atomic_json(output/'summary.json',rows)


def bundle(a,state,error=None,name='dirichlet_oldserver_results.tar.gz'):
    planned=all_jobs(a.output); missing=[]; invalid=[]; completed=[]
    for job in planned:
        if not Path(job['artifact']).is_file(): missing.append(job['name']); continue
        try:
            common.configure(job['case'],a.inputs_root)
            if not common.verify_job(job): raise RuntimeError('Completion marker missing')
            completed.append(job['name'])
        except Exception as exception: invalid.append(dict(job=job['name'],error=str(exception)))
    summary_error=None
    try: summaries(a.output)
    except Exception as exception: summary_error=str(exception)
    complete=state=='complete' and not missing and not invalid and summary_error is None
    if state=='complete' and not complete:
        state='failed'; error=error or 'Expected results are missing or invalid'
        status=c.read(a.output/'status.json') if (a.output/'status.json').exists() else {}
        c.atomic_json(a.output/'status.json',dict(status,state=state,error=error))
    c.atomic_json(a.output/'result_index.json',dict(complete=complete,
        state=state,error=error,expected_jobs=len(planned),missing=missing,invalid=invalid,fl_target=28,
        completed=completed,summary_error=summary_error))
    whitelist={'protocol.json','status.json','run_sessions.json','runtime.json','summary.json',
        'result_index.json','resume_import.json','config.json','loss_settings.json','metrics.json',
        'rounds.json','history.json','result.json','records.json','coefficients.json','gamma.json'}
    files=sorted(file for file in a.output.rglob('*.json') if file.name in whitelist or file.name.startswith(('COMPLETED','client_')))
    manifest=dict(complete=complete,missing=missing,invalid=invalid,summary_error=summary_error,
        files=[dict(path=f.relative_to(a.output).as_posix(),sha256=c.digest(f)) for f in files])
    c.atomic_json(a.output/'packaging_manifest.json',manifest); files.append(a.output/'packaging_manifest.json')
    archive=a.output/name; temp=archive.with_name(archive.name+'.tmp')
    with tarfile.open(temp,'w:gz') as tar:
        for file in files: tar.add(file,arcname=file.relative_to(a.output).as_posix(),recursive=False)
    with tarfile.open(temp,'r:gz') as tar:
        if set(tar.getnames())!={f.relative_to(a.output).as_posix() for f in files}: raise RuntimeError('Archive members differ')
        for row in manifest['files']:
            with tar.extractfile(row['path']) as stream:
                digest=hashlib.sha256()
                for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
            if digest.hexdigest()!=row['sha256']: raise RuntimeError('Archive fingerprint differs')
    os.replace(temp,archive)
    archive.with_name(archive.name+'.sha256').write_bytes((c.digest(archive)+'  '+name+'\n').encode('ascii'))
    return archive


def execute(a):
    start=time.time(); deadline=start+a.hours*3600
    sessions=a.output/'run_sessions.json'
    previous=c.read(sessions) if sessions.exists() else []
    c.atomic_json(sessions,previous+[dict(started_at_epoch=start,deadline=deadline,gpus=a.gpus,workers=a.workers,
        data_root=str(a.data_root),inputs_root=str(a.inputs_root),hours=a.hours)])
    c.atomic_json(a.output/'runtime.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__))
    state='budget_exhausted'; phase='reference'; error=None
    def stage(queue_jobs,label):
        nonlocal phase
        phase=label
        return run(queue_jobs,label,a,deadline)
    try:
        if not stage(jobs(a.output,'reference'),'reference'): return
        if not stage(jobs(a.output,'calibrate'),'initial_dm_calibration'): return
        for iteration in common.SOURCE_STAGES:
            for kind in ('synthesize','vectors','accuracy'):
                if not stage(jobs(a.output,kind,iteration),f'{kind}{iteration}'): return
            print('Source milestone complete:',bundle(a,'milestone_complete',name=f'dirichlet_oldserver_i{iteration}_results.tar.gz'),flush=True)
        if not stage(jobs(a.output,'gamma'),'gamma24000'): return
        for case,(alpha,ratio,total) in common.CASES.items():
            common.configure(case,a.inputs_root)
            gamma.coefficients(a.output/case/'gamma24',a.output/case/'source24000')
        for branch in ('original','low_dm_direction'):
            prepare_branch(a,branch)
            for iteration in common.BRANCH_STAGES:
                for kind in ('synthesize','vectors','accuracy'):
                    if not stage(jobs(a.output,kind,iteration,branch),f'{branch}_{kind}{iteration}'): return
                print('Branch milestone complete:',bundle(a,'milestone_complete',name=f'dirichlet_oldserver_{branch}_i{iteration}_results.tar.gz'),flush=True)
        if not stage(fl_jobs(a.output),'federated_learning'): return
        state='complete'
    except (Exception,KeyboardInterrupt) as exception:
        state='interrupted' if isinstance(exception,KeyboardInterrupt) else 'failed'
        error=str(exception) or type(exception).__name__
    finally:
        previous=c.read(a.output/'status.json') if (a.output/'status.json').exists() else {}
        c.atomic_json(a.output/'status.json',dict(state=state,phase=phase,active=[],pending=[],
            failed=previous.get('failed',[]),error=error,elapsed_seconds=time.time()-start))
        print('Result archive:',bundle(a,state,error),flush=True)
        if state=='complete' and not c.read(a.output/'result_index.json')['complete']:
            error=c.read(a.output/'result_index.json')['error']
    if error: raise RuntimeError(error)


def status(a):
    state=c.read(a.output/'status.json') if (a.output/'status.json').exists() else dict(state='not_started')
    lookup={j['name']:j for j in all_jobs(a.output)}
    rates=[]; work=0.
    for active in state.get('active',[]):
        job=lookup.get(active['job'])
        if not job: continue
        path=Path(job['artifact']).parent/'status.json'
        if path.exists(): active['progress']=c.read(path)
        if job['phase']=='synthesize':
            records=path.with_name('metrics.json')
            if not records.exists(): continue
            previous={4000:0,8000:4000,16000:8000,24000:16000,28000:24000,32000:28000}[job['iteration']]
            rows=[row for row in c.read(records) if row['iteration']>previous][-21:]
            if len(rows)>=2:
                n=rows[-1]['iteration']-rows[0]['iteration']; dt=rows[-1]['elapsed_seconds']-rows[0]['elapsed_seconds']
                if n>0 and dt>0:
                    rate=dt/n; rates.append(rate)
                    work+=(job['iteration']-rows[-1]['iteration'])*rate
                    active['seconds_per_iteration']=rate
                    active['eta_seconds']=(job['iteration']-rows[-1]['iteration'])*rate
    if rates:
        mean=float(np.mean(rates))
        for name in state.get('pending',[]):
            job=lookup.get(name)
            if job and job['phase']=='synthesize':
                previous={4000:0,8000:4000,16000:8000,24000:16000,28000:24000,32000:28000}[job['iteration']]
                work+=(job['iteration']-previous)*mean
        state['phase_eta_seconds']=work/max(1,len(state.get('active',[])))
        state['eta_scope']='current synthesis phase only; recent rates; excludes later phases and evaluations'
    state['fl_completed']=sum(Path(j['artifact']).exists() for j in fl_jobs(a.output)); state['fl_target']=28
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-root',type=Path,default=common.INPUTS)
    p.add_argument('--data-root',type=Path,default=c.ROOT/'data')
    p.add_argument('--output',type=Path,default=c.ROOT/'outputs/dirichlet_oldserver_20261009')
    p.add_argument('--gpus',nargs='+',default=['0','1','2']); p.add_argument('--workers',type=int,default=3)
    p.add_argument('--hours',type=float,default=168.)
    p.add_argument('--run',action='store_true'); p.add_argument('--status',action='store_true')
    a=p.parse_args(); a.output=a.output.resolve(); a.inputs_root=a.inputs_root.resolve(); a.data_root=a.data_root.resolve()
    if a.status: print(json.dumps(status(a),indent=2)); return
    if a.workers<1 or not a.gpus or len(set(a.gpus))!=len(a.gpus) or not np.isfinite(a.hours) or a.hours<=0:
        p.error('Invalid GPU/worker/time settings')
    for protected in (a.inputs_root,a.data_root):
        if a.output.is_relative_to(protected) or protected.is_relative_to(a.output):
            raise RuntimeError('Output must not overlap input/data directories')
    settings={case:common.preflight(case,a.data_root,a.inputs_root) for case in common.CASES}
    protocols={}
    for case in common.CASES:
        common.configure(case,a.inputs_root); protocols[case]=common.protocol(case)
    binding=dict(version=1,cases=protocols,fl_target=28,gamma_policy='24000 all32 per case; fixed30%')
    if (a.output/'protocol.json').exists() and c.read(a.output/'protocol.json')!=json.loads(json.dumps(binding)):
        raise RuntimeError('Existing suite code/inputs/settings changed')
    verify(all_jobs(a.output),a)
    print(json.dumps(dict(state='preflight_ok',cases=settings,fl_target=28,gpus=a.gpus,workers=a.workers,
        formal_training_started=False),indent=2))
    if not a.run: print('Preflight only. Add --run to start.'); return
    if not torch.cuda.is_available() or any(not g.isdigit() or int(g)>=torch.cuda.device_count() for g in a.gpus):
        raise RuntimeError('Requested CUDA GPUs unavailable')
    queue.child_command=child_command
    with queue.lock(a.output):
        c.bind(a.output/'protocol.json',binding)
        for case in common.CASES:
            common.configure(case,a.inputs_root)
            c.bind(a.output/case/'source24000/protocol.json',common.protocol(case))
        def interrupt(_signal,_frame): raise KeyboardInterrupt()
        handler=signal.signal(signal.SIGTERM,interrupt)
        try: execute(a)
        finally: signal.signal(signal.SIGTERM,handler)


if __name__=='__main__': main()
