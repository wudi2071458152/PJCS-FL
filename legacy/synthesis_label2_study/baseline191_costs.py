"""Logical FP32 communication and measured computation; never infer missing timers."""
import os
import platform
import time

import numpy as np
import torch


def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def timed(function, *args, **kwargs):
    synchronize(); start = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize()
    return result, time.perf_counter()-start


def communication(route, row, width):
    """Unicast theta/control corrections; DivFL query gradients use a second upload."""
    selected = len(row['selected']); down, up, ds, us = selected, selected, 0, 0
    if route.startswith('scaffold_'): down *= 2
    if route == 'poc': down, us = 6, 6  # candidates already hold theta; no second theta download
    elif route in ('is_ht', 'delta_ht'): us = selected  # updated feedback score
    elif route == 'divfl':
        queries = int(row['costs']['query_clients'])
        down = 32 if row['round'] == 1 else selected
        up += int(row['costs']['query_calls'])
    result = dict(downlink_vectors=down, uplink_vectors=up, downlink_scalars=ds, uplink_scalars=us)
    result.update(downlink_bytes=4*(width*down+ds), uplink_bytes=4*(width*up+us))
    result['total_bytes'] = result['downlink_bytes']+result['uplink_bytes']
    return result


def policy(net, syn):
    width = sum(p.numel() for p in net.parameters())
    upload = 0 if syn is None else sum(x.numel()*x.element_size()+y.numel()*y.element_size() for x, y, _ in syn)
    return dict(dtype='FP32', bytes_per_scalar=4, vector_parameters=width,
        transport='logical unicast payload; metadata excluded; zero vectors counted',
        scaffold='selected receives theta and c-ci; uploads R only; server computes Option II',
        divfl='initial full query theta broadcast charged as 32 unicasts; selected real update and queried gradient both uploaded',
        proxy_upload='one-time actual image/label tensor bytes, no serialization metadata; construction separate; no online proxy client contact',
        once_proxy_upload_bytes=upload, construction_in_online_bytes=False)


def runtime():
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        torch_threads=torch.get_num_threads(), timing='synchronized device wall seconds; includes observation overhead; not isolated hardware benchmark')


def totals(rows, key):
    keys = sorted(set().union(*(row.get(key, {}) for row in rows)))
    return {name: None if any(row.get(key, {}).get(name) is None for row in rows) else
            sum(row[key][name] for row in rows) for name in keys}


def logical_communication(route, rows_or_counts, model_parameters, synthetic_upload_bytes=0):
    """Rows or explicit vector/scalar counts, for deterministic legacy-result accounting."""
    if isinstance(rows_or_counts, dict):
        counts = rows_or_counts
        result = {k: int(counts.get(k, 0)) for k in ('downlink_vectors','uplink_vectors','downlink_scalars','uplink_scalars')}
        result.update(downlink_bytes=4*(result['downlink_vectors']*model_parameters+result['downlink_scalars']),
                      uplink_bytes=4*(result['uplink_vectors']*model_parameters+result['uplink_scalars']))
        result['total_bytes'] = result['downlink_bytes']+result['uplink_bytes']
    else:
        values = [communication(route, row, model_parameters) for row in rows_or_counts]
        result = {k: sum(v[k] for v in values) for k in values[0]} if values else {}
    result.update(once_proxy_upload_bytes=int(synthetic_upload_bytes),
                  total_bytes_with_proxy_upload=result.get('total_bytes',0)+int(synthetic_upload_bytes))
    return result


def validate(row, route, width):
    if row['communication'] != communication(route, row, width):
        raise RuntimeError('Logical communication differs from observed client/query counts')
    for value in row['costs'].values():
        if value is not None and (not np.isfinite(value) or value < 0):
            raise RuntimeError('Invalid measured cost')
