"""191 new baseline/SC-extension runs; immutable inputs, complete costs and audited resume."""
import argparse
import copy
import hashlib
import json
import signal
import time
from functools import lru_cache
from importlib.metadata import version as dependency_version
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import torch
import baseline191_algorithms as algorithms
import baseline191_costs as costs
import run_dirichlet_a005_multiseed as migrated
import run_dirichlet_a007_fl as a007
import run_dirichlet_control_variates as cv
import run_dirichlet_hcs_pilot as hcs
import run_dirichlet_multiseed as multi
import run_dirichlet_scaffold_control_refresh as refresh
import pracis_sampling_comparison as importance_helpers

c = cv.c
ALPHAS = ('0p05','0p07','0p1','0p2','0p3')
SEEDS = (2025,2027,2028,2029,2030)
ROUTES = ('poc','is_ht','delta_ht','divfl','hcs_static_gram_equal','fedvarp',
          'scaffold_weighted','scaffold_refresh_real_weighted')
SC = ROUTES[-2:]
SUITE = 'dirichlet_baseline191_20261015'


def clone(function, **overrides):
    return FunctionType(function.__code__, dict(function.__globals__, **overrides),
                        function.__name__, function.__defaults__, function.__closure__)


def input_entry(input_map, alpha):
    path = Path(input_map).resolve(); value = c.read(path)
    if value.get('version') != 1 or alpha not in value.get('alphas', {}):
        raise RuntimeError('Unsupported input-map version or alpha')
    entry = value['alphas'][alpha]
    return {k: (Path(v) if Path(v).is_absolute() else path.parent/Path(v)).resolve()
            for k,v in entry.items() if k.endswith('_root')}


def resolve_bindings(template, roots, data_root, alpha):
    """Resolve every inherited role, instead of trusting just a path label."""
    source, proxy = roots['source_root'], roots.get('proxy_root', roots['source_root'])
    inputs = roots.get('inputs_root', c.INPUTS); result = {}
    for role in template['files_sha256']:
        if role.startswith('code/'):
            path = c.ROOT/role[5:]
            if not path.exists() and Path(role).name == multi.base.EXPECTED.name: path = multi.base.EXPECTED
        elif role.startswith('data/'): path = Path(data_root)/role[5:]
        elif role.startswith('inputs/'): path = inputs/role[7:]
        elif role == 'source/initial_model.pt' or role == 'source/reference/round_0000.pt':
            path = c.model_path(source, alpha, 2025, 0)
        elif role == 'source/reference/COMPLETED.json':
            path = c.model_path(source, alpha, 2025, 0).parent.parent/'COMPLETED.json'
        elif role in ('source/calibration/coefficients.json','proxy/calibration/coefficients.json'):
            path = (source if role.startswith('source/') else proxy)/'calibration'/alpha/'coefficients.json'
        elif role.startswith(('source/client_', 'proxy/client_')):
            side, client, name = role.split('/',2)
            path = c.route(source if side == 'source' else proxy, alpha, 'weighted_weak_x10', int(client[-2:]))
            path = path/'snapshots/iter_32000.pt' if name == 'snapshot.pt' else path/('COMPLETED_32000.json' if name == 'COMPLETED.json' else name)
        elif role.startswith('source/'): path = source/role[7:]
        elif role.startswith('proxy/'): path = proxy/role[6:]
        else: raise RuntimeError('Unresolved immutable input role: '+role)
        if c.digest(path) != template['files_sha256'][role]: raise RuntimeError('Input binding changed: '+role)
        result[role] = str(path.resolve())
    return result


@lru_cache(maxsize=5)
def _prepared_base(map_path, map_sha, data_root, alpha):
    roots = input_entry(map_path, alpha)
    # Existing preparations validate saved partition, capacity, old loss and reference lineage.
    fake_output = Path(data_root).parent/'baseline191_preflight_placeholder'
    a = SimpleNamespace(alpha=alpha, seed=2025, route='residual_score', iteration=32000,
        rounds=1500, data_root=Path(data_root), output=fake_output, **roots)
    if alpha == '0p05':
        prepared = migrated.prepare(a)
        _,source,proxy,inputs = migrated.paths(a.migrated_root)
        roots.update(source_root=source, proxy_root=proxy, inputs_root=inputs)
    elif alpha == '0p07': prepared = a007.prepare(a)
    else: prepared = multi.prepare(a)
    template = copy.deepcopy(prepared[-1])
    template['files_paths'] = resolve_bindings(template, roots, data_root, alpha)
    template['input_roots'] = {k:str(v) for k,v in roots.items()}
    return (*prepared[:-1], template)


def release_prepared_cache():
    _prepared_base.cache_clear()


def route_definition(route):
    if route in algorithms.ROUTES: return algorithms.method_protocol(route)
    if route == 'hcs_static_gram_equal':
        return dict(sampling='joint expected-quota dependent rounding; norm-weighted sequential within each cluster',
            aggregation='equal mean of selected real updates; first round all32, later3',
            refresh='static',representation='gram',aggregation_mode='equal',
            clustering=dict(k=3,seed=502025,quota='N_h*dispersion; capacity capped proportional redistribution',
                geometry='exact Gram; Euclidean; no unit normalization',
                dispersion='ordered pairwise squared distances/(N_h-1+1e-6)',
                frozen='first round geometry, quotas and scores frozen'),
            warmup=dict(round=1,clients=32,counted_in_target=True),renormalize=True)
    if route == 'fedvarp':
        return dict(sampling='uniform without replacement; Pi=3/32',
            aggregation='sum_all(w_i*H_i)+sum_selected(w_i/Pi_i*(R_i-H_i))',
            history='zero initially; replace selected H AFTER aggregate; no EMA',renormalize=False)
    return dict(sampling='uniform without replacement; Pi=3/32',
        aggregation='sum_selected(w_i/Pi_i*R_i_SC)',renormalize=False,
        controls='zero initial; old c_i,c frozen within all three local steps; Option II',
        control_update='selected REAL, unselected SYNTHETIC; c=sum_all(w_i*c_i_new)' if route == SC[1] else
                       'selected REAL only; c_new=c_old+sum_selected(w_i*delta_c_i)',
        model_aggregation='real_ht',weights_mode='samples')


def prepare(a):
    if a.alpha not in ALPHAS or a.seed not in SEEDS or a.route not in ROUTES or not 1 <= a.rounds <= 1500:
        raise ValueError('Unsupported baseline route/seed/alpha/round target')
    map_path = Path(a.input_map).resolve(); map_sha = c.digest(map_path)
    original = _prepared_base(str(map_path),map_sha,str(Path(a.data_root).resolve()),a.alpha)
    source_net, real, syn, weights, test, inherited = original
    for path in (*inherited['input_roots'].values(), str(Path(a.data_root).resolve())):
        resolved, output = Path(path).resolve(), Path(a.output).resolve()
        if output.is_relative_to(resolved) or resolved.is_relative_to(output):
            raise RuntimeError('Output overlaps immutable inputs')
    net = copy.deepcopy(source_net) if a.seed == 2025 else c.model(a.seed)
    if a.alpha != '0p05' and multi.state_sha256(net) != multi.initialization_sha256(a.seed):
        raise RuntimeError('Cold initialization differs from shared same-seed model')
    proxy_required = a.route == SC[1]
    files, paths = copy.deepcopy(inherited['files_sha256']), copy.deepcopy(inherited['files_paths'])
    if 'migration' in inherited:
        for rel,sha in inherited['migration']['files_sha256'].items():
            files['migration/'+rel],paths['migration/'+rel] = sha,str(Path(inherited['migration']['root'])/rel)
    files['input-map.json'], paths['input-map.json'] = map_sha,str(map_path)
    for module in (algorithms,costs,cv,hcs,hcs.sampling,hcs.cluster_sampling,refresh,refresh.original,
                   importance_helpers,multi,migrated,a007,*multi.MODULES):
        path = Path(module.__file__).resolve(); key = 'code/'+path.relative_to(c.ROOT).as_posix()
        files[key], paths[key] = c.digest(path),str(path)
    file = Path(__file__).resolve(); key = 'code/'+file.relative_to(c.ROOT).as_posix()
    files[key],paths[key] = c.digest(file),str(file)
    protocol = dict(version=1,suite=SUITE,alpha=a.alpha,route=a.route,method=a.route,seed=a.seed,
        rounds=a.rounds,clients=32,selected=3,local_steps=3,batch=256,real_lr=.01,proxy_lr=.01 if proxy_required else None,
        real_ce='ordinary',proxy_ce=inherited['proxy_ce'] if proxy_required else None,
        proxy_loss='original' if proxy_required else None,iteration=32000 if proxy_required else 0,
        comparison_images='old-loss 32000 for proxy-dependent methods',augmentation=False,
        budget_ratio=.2,budget_total=10000,weights=np.asarray(weights).tolist(),
        real_sizes=[len(y) for _,y in real],synthetic_sizes=[len(y) for _,y,_ in syn],
        model_parameters=sum(p.numel() for p in net.parameters()),
        initial_model_state_sha256=cv.tensor_fingerprint(net.state_dict()),
        fl_initialization=dict(seed=a.seed,state_sha256=multi.state_sha256(net),
            initializer='saved migrated reference theta0' if a.alpha == '0p05' and a.seed == 2025 else
                        'dirichlet_screen_common.model(seed)',shared_across_methods=True),
        round_seeds=dict(base=f'{a.seed}+round*1000',real='+client',uniform='+17',proxy='+10000000+client'),
        files_sha256=files,files_paths=paths,input_roots=inherited['input_roots'],
        source_protocol_sha256=inherited['files_sha256']['source/protocol.json'],
        communication=costs.policy(net,syn if proxy_required else None),
        cost_policy=dict(client='real SGD and full-data queries',server='proxy, selection, geometry and aggregate',
            legacy_control='combined server_aggregation_control_diagnostics_seconds measured; unavailable separate selection/aggregation null',
            missing='null; genuine non-occurrence zero',evaluation_and_checkpoint_io='separate from algorithm timers'),
        software=dict(torch=torch.__version__,numpy=np.__version__,sklearn=hcs.sklearn.__version__,scipy=dependency_version('scipy')))
    protocol.update(route_definition(a.route))
    if 'migration' in inherited: protocol['migration'] = copy.deepcopy(inherited['migration'])
    return net,real,syn if proxy_required else None,weights,test,protocol


def initial_state(net, route, weights):
    if route in algorithms.ROUTES: return algorithms.initial_state(net,route,weights)
    if route == 'hcs_static_gram_equal': return dict(cache=None,last_seen=None,static_geometry=None)
    return cv.initial_state(net,'history_weighted' if route == 'fedvarp' else 'scaffold_weighted')


def one_round(net,real,syn,weights,route,seed,rnd,state,update_fn=None,scaffold_fn=None,proxy_fn=None,query_fn=None):
    if route in algorithms.ROUTES:
        (delta,row,state),duration = costs.timed(algorithms.one_round,net,real,weights,route,seed,rnd,state,update_fn,query_fn)
        row['costs'].update(server_proxy_calls=0,server_proxy_steps=0,server_proxy_examples=0,server_proxy_seconds=0.,
            server_total_seconds=max(0.,duration-row['costs']['client_train_seconds']-row['costs']['query_seconds']))
    else:
        measured = dict(client_train_calls=0,client_train_steps=0,client_train_examples=0,client_train_seconds=0.,
            server_proxy_calls=0,server_proxy_steps=0,server_proxy_examples=0,server_proxy_seconds=0.,
            query_calls=0,query_clients=0,query_examples=0,query_forward_examples=0,query_backward_examples=0,query_seconds=0.)
        def wrap(function,proxy=False):
            def call(net,x,y,update_seed,*args):
                value,duration = costs.timed(function,net,x,y,update_seed,*args)
                key = 'server_proxy' if proxy else 'client_train'
                measured[key+'_calls'] += 1; measured[key+'_steps'] += 3
                measured[key+'_examples'] += sum(len(batch) for batch in c.batches(len(y),update_seed))
                measured[key+'_seconds'] += duration
                return value
            return call
        update = wrap(update_fn or c.update)
        start=time.perf_counter()
        if route == 'hcs_static_gram_equal':
            kernel=clone(hcs.one_round,SEED=seed)
            delta,row,state['cache'],state['last_seen'],state['static_geometry'] = kernel(
                net,real,weights,'static','gram','equal',rnd,state['cache'],state['last_seen'],state['static_geometry'],None,update)
            measured.update(geometry_seconds=row['geometry_seconds'],selection_seconds=row['selection_seconds'],
                            aggregation_seconds=row['aggregation_seconds'],solver_seconds=0.)
        else:
            kernel=clone(refresh.one_round if route == SC[1] else cv.one_round,SEED=seed)
            sc=wrap(scaffold_fn or cv.scaffold_update)
            kwargs=dict(scaffold_fn=sc)
            if route == SC[1]: kwargs['proxy_fn']=wrap(proxy_fn or refresh.corrected_proxy_update,True)
            else: kwargs['update_fn']=update
            delta,row=kernel(net,real,syn,weights,'history_weighted' if route == 'fedvarp' else route,rnd,state,**kwargs)
            measured.update(selection_seconds=None,aggregation_seconds=None,geometry_seconds=0.,solver_seconds=0.)
        costs.synchronize(); duration=time.perf_counter()-start
        measured['server_total_seconds']=max(0.,duration-measured['client_train_seconds'])
        measured['server_aggregation_control_diagnostics_seconds']=max(0.,measured['server_total_seconds']-measured['server_proxy_seconds'])
        row['costs']=measured
    row['communication']=costs.communication(route,row,sum(p.numel() for p in net.parameters()))
    return delta,row,state


def verify_records(protocol,rows):
    route,seed=protocol['route'],protocol['seed']
    if [row['round'] for row in rows] != list(range(1,len(rows)+1)): raise RuntimeError('Round coverage mismatch')
    if route in algorithms.ROUTES: algorithms.replay_rows(rows,protocol['weights'],route,seed)
    elif route == 'hcs_static_gram_equal': clone(hcs.replay,SEED=seed)(protocol,rows)
    else:
        mapped=dict(protocol,method='history_weighted' if route == 'fedvarp' else route)
        scoped=a007._scoped(seed)
        (scoped.refresh_replay if route == SC[1] else scoped.sc_replay)(mapped,rows)
    for row in rows:
        costs.validate(row,route,protocol['model_parameters'])
        if any(not np.isfinite(row[k]) for k in ('test_accuracy','test_loss','update_norm','round_seconds','elapsed_seconds')):
            raise RuntimeError('Nonfinite training record')


def state_hash(value):
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,dict):
            for key in sorted(item): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item,(tuple,list)):
            for child in item: visit(child)
        elif isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous(); digest.update(str(tensor.dtype).encode()+str(tuple(tensor.shape)).encode()+tensor.numpy().tobytes())
        elif isinstance(item,np.ndarray): digest.update(str(item.dtype).encode()+str(item.shape).encode()+item.tobytes())
        else: digest.update(repr(item).encode())
    visit(value); return digest.hexdigest()


def validate_state(net,state,protocol,rows):
    route=protocol['route']; width=protocol['model_parameters']
    if route in algorithms.ROUTES:
        rebuilt=algorithms.replay_rows(rows,protocol['weights'],route,protocol['seed'])
        if route in ('is_ht','delta_ht') and not np.array_equal(state['u'],rebuilt['u']): raise RuntimeError('Feedback checkpoint mass changed')
        if route == 'divfl' and rows:
            if state['updated_round'] != rebuilt['updated_round'] or not np.array_equal(state['distance_matrix'],rows[-1]['distance_matrix_after']): raise RuntimeError('DivFL geometry checkpoint changed')
            if len(state['gradients']) != 32 or any(v.shape != (width,) or not torch.isfinite(v).all() for v in state['gradients']): raise RuntimeError('Invalid DivFL gradient cache')
            if not np.allclose(algorithms._distances(state['gradients']),state['distance_matrix'],rtol=1e-12,atol=1e-15): raise RuntimeError('DivFL vectors differ from recorded geometry')
        return
    if route == 'hcs_static_gram_equal':
        if not rows: return
        seen,cache_sha,geometry_sha=clone(hcs.replay,SEED=protocol['seed'])(protocol,rows)
        if hcs.cache_fingerprint(state['cache']) != cache_sha or not np.array_equal(seen,state['last_seen']) or hcs.json_fingerprint(state['static_geometry']) != geometry_sha:
            raise RuntimeError('HCS checkpoint descriptor/geometry mismatch')
        if len(state['cache']) != 32 or any(v.shape != (width,) or not torch.isfinite(v).all() for v in state['cache']): raise RuntimeError('Invalid HCS cache')
    else:
        cv._validate_state(net,state,'history_weighted' if route == 'fedvarp' else 'scaffold_weighted',rows,protocol['weights'],width)
        if rows and route == SC[1] and rows[-1]['control_sha256_after'] != cv.tensor_fingerprint({'c':state['control']}):
            raise RuntimeError('Refreshed control changed')


def summarize(rows,protocol):
    result=multi.base.summarize(rows)
    result.update(cost_totals=costs.totals(rows,'costs'),communication_totals=costs.totals(rows,'communication'))
    result['communication_totals'].update(once_proxy_upload_bytes=protocol['communication']['once_proxy_upload_bytes'],
        total_bytes_with_proxy_upload=result['communication_totals'].get('total_bytes',0)+protocol['communication']['once_proxy_upload_bytes'])
    return result


def verify_protocol(folder):
    protocol=c.read(Path(folder)/'protocol.json')
    if protocol.get('suite') != SUITE or protocol['route'] not in ROUTES or protocol['alpha'] not in ALPHAS or protocol['seed'] not in SEEDS:
        raise RuntimeError('Unexpected baseline suite/route')
    frozen=dict(clients=32,selected=3,local_steps=3,batch=256,real_lr=.01,real_ce='ordinary',augmentation=False)
    if any(protocol.get(k) != v for k,v in frozen.items()) or any(protocol.get(k) != v for k,v in route_definition(protocol['route']).items()):
        raise RuntimeError('Frozen numerical definition changed')
    for role,wanted in protocol['files_sha256'].items():
        path=Path(protocol['files_paths'][role])
        if not path.is_file() or c.digest(path) != wanted: raise RuntimeError('Bound source/input changed: '+role)
    return protocol


def verify(output,expected_rounds=1500):
    folder=Path(output); protocol=verify_protocol(folder); marker=c.read(folder/'COMPLETED.json'); rows=c.read(folder/'rounds.json')
    if protocol['rounds'] != expected_rounds or marker.get('rounds') != expected_rounds or len(rows) != expected_rounds: raise RuntimeError('Incomplete result')
    for name in ('protocol','rounds','summary'):
        if marker[name+'_sha256'] != c.digest(folder/(name+'.json')): raise RuntimeError('Result checksum mismatch: '+name)
    verify_records(protocol,rows)
    if c.read(folder/'summary.json') != summarize(rows,protocol): raise RuntimeError('Summary differs from records')
    if (folder/'resume.pt').exists():
        saved=load_checkpoint(folder,marker['protocol_sha256']); validate_state(None,saved['state'],protocol,rows)
        if saved['round'] != expected_rounds or saved['records'] != rows: raise RuntimeError('Completed checkpoint record mismatch')
    return marker


def load_checkpoint(folder,sha):
    integrity=c.read(folder/'checkpoint_integrity.json')
    if integrity['checkpoint_sha256'] != c.digest(folder/'resume.pt') or integrity['protocol_sha256'] != sha: raise RuntimeError('Checkpoint integrity mismatch')
    saved=c.load(folder/'resume.pt')
    if saved['protocol_sha256'] != sha or saved['round'] != len(saved['records']) or saved['state_sha256'] != state_hash(saved['state']): raise RuntimeError('Checkpoint state/round/protocol changed')
    if any(not torch.isfinite(v).all() for v in saved['model'].values()): raise RuntimeError('Nonfinite saved model')
    return saved


def train(a,prepared):
    net,real,syn,weights,test,protocol=prepared
    if (a.route,a.seed,a.alpha,a.rounds) != (protocol['route'],protocol['seed'],protocol['alpha'],protocol['rounds']): raise RuntimeError('Prepared route differs from arguments')
    folder=Path(a.output); folder.mkdir(parents=True,exist_ok=True)
    c.bind(folder/'protocol.json',protocol); sha=c.digest(folder/'protocol.json')
    if (folder/'COMPLETED.json').exists(): verify(folder,a.rounds); return True
    runtime=c.read(folder/'runtime.json') if (folder/'runtime.json').exists() else dict(attempts=[])
    runtime['attempts'].append(dict(started_unix=time.time(),**costs.runtime())); c.atomic_json(folder/'runtime.json',runtime)
    state=initial_state(net,a.route,weights); rows=[]; elapsed=0.; io_seconds=0.
    if (folder/'resume.pt').exists():
        saved=load_checkpoint(folder,sha); rows=saved['records']; state=saved['state']; elapsed=saved['elapsed_seconds']; io_seconds=saved['checkpoint_io_seconds']
        if (folder/'io_statistics.json').exists():
            io_seconds=c.read(folder/'io_statistics.json')['checkpoint_io_seconds']
            if not np.isfinite(io_seconds) or io_seconds < saved['checkpoint_io_seconds']: raise RuntimeError('Invalid checkpoint IO counter')
        if len(rows) > a.rounds: raise RuntimeError('Checkpoint exceeds target')
        verify_records(protocol,rows); net.load_state_dict(saved['model']); c.restore_rng(saved['rng'])
        if a.route in ('fedvarp',*SC):
            state['cache']=[v.to(c.DEVICE) for v in state['cache']]
            if state['control'] is not None: state['control']=state['control'].to(c.DEVICE)
    else:
        if (folder/'rounds.json').exists() and c.read(folder/'rounds.json'): raise RuntimeError('Partial records without checkpoint')
        if cv.tensor_fingerprint(net.state_dict()) != protocol['initial_model_state_sha256']: raise RuntimeError('Cold initial state changed')
        c.seed_all(a.seed)
    validate_state(net,state,protocol,rows); tick=time.monotonic()
    def persist(status):
        nonlocal io_seconds
        start=time.perf_counter(); duration=elapsed+time.monotonic()-tick
        c.save(folder/'resume.pt',dict(round=len(rows),model=net.state_dict(),records=rows,state=state,state_sha256=state_hash(state),
            rng=c.rng_state(),elapsed_seconds=duration,checkpoint_io_seconds=io_seconds,protocol_sha256=sha))
        c.atomic_json(folder/'checkpoint_integrity.json',dict(checkpoint_sha256=c.digest(folder/'resume.pt'),protocol_sha256=sha,round=len(rows)))
        c.atomic_json(folder/'rounds.json',rows); summary=summarize(rows,protocol)
        c.atomic_json(folder/'summary.json',summary)
        io_seconds+=time.perf_counter()-start
        c.atomic_json(folder/'io_statistics.json',dict(checkpoint_io_seconds=io_seconds,scope='all checkpoint/result writes and checksum reads; status write excluded'))
        recent=[r['round_seconds'] for r in rows[-50:]]
        c.atomic_json(folder/'status.json',dict(state=status,round=len(rows),target=a.rounds,elapsed_seconds=duration,
            eta_seconds=float(np.mean(recent))*(a.rounds-len(rows)) if recent else None))
    for rnd in range(len(rows)+1,a.rounds+1):
        if c.expired(a.deadline): persist('budget_exhausted'); return False
        start=time.monotonic(); delta,row,state=one_round(net,real,syn,weights,a.route,a.seed,rnd,state)
        _,apply_time=costs.timed(c.add_delta,net,delta)
        evaluation,eval_time=costs.timed(multi.fix.evaluate,net,*test)
        row.update(evaluation); row['costs'].update(model_apply_seconds=apply_time,evaluation_seconds=eval_time)
        row.update(round_seconds=time.monotonic()-start,elapsed_seconds=elapsed+time.monotonic()-tick)
        rows.append(row)
        if rnd%10 == 0 or rnd == a.rounds or c.expired(a.deadline): persist('training')
    persist('verifying'); verify_records(protocol,rows); validate_state(net,state,protocol,rows)
    c.atomic_json(folder/'COMPLETED.json',dict(rounds=a.rounds,protocol_sha256=sha,rounds_sha256=c.digest(folder/'rounds.json'),summary_sha256=c.digest(folder/'summary.json')))
    verify(folder,a.rounds)
    status=c.read(folder/'status.json'); status['state']='complete'; c.atomic_json(folder/'status.json',status)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-map','data-root','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--alpha',choices=ALPHAS,required=True); parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--route',choices=ROUTES,required=True); parser.add_argument('--rounds',type=int,default=1500)
    parser.add_argument('--deadline',type=float); parser.add_argument('--run',action='store_true'); a=parser.parse_args()
    if a.rounds != 1500: parser.error('Formal run requires 1500 rounds')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal training requires CUDA')
    prepared=prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok',formal_training_started=False,protocol=prepared[-1]))); return
    def stop(*_): a.deadline=0.
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a,prepared) else 75)


if __name__ == '__main__': main()
