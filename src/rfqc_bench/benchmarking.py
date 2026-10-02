"""One synchronized inference timing round, without fitting or publishing RFs."""
import hashlib
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from .data import RFData


def timing_input(data, spec, sample_size=512, complete_views=False):
    """Select a label-blind shared pool; keep the entire pool for station FCM.

    Native views and supplied descriptors are prepared outside the timer.
    Observational arrays and sample identifiers are never included in reports.
    """
    if not isinstance(data, RFData):
        raise TypeError('Expected RFData')
    if data.sample_ids is None or len(set(data.sample_ids)) != len(data):
        raise ValueError('Timing requires unique sample_ids for label-blind selection')
    if spec.kind == 'fcm':
        if data.stations is None:
            raise ValueError('FCM requires complete same-station pools and station IDs')
        selected = data
    else:
        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError('sample_size must be a positive integer')
        eligible = np.flatnonzero(data.lengths == 7) if complete_views else np.arange(len(data))
        if len(eligible) < sample_size:
            raise ValueError(f'Need {sample_size} eligible records, received {len(eligible)}')
        order = sorted(eligible, key=lambda i: hashlib.sha256(str(data.sample_ids[i]).encode()).hexdigest())
        selected = data.subset(np.array(order[:sample_size]))
    selected = selected.select(spec.gaussian)
    return RFData(waveforms=selected.waveforms if spec.mode != 'features' else None,
                  features=selected.features if spec.mode != 'waveform' else None,
                  gaussians=selected.gaussians, lengths=selected.lengths,
                  stations=selected.stations, sample_ids=selected.sample_ids)


def benchmark_inference(predictor, data, *, sample_size=512, complete_views=False,
                        batch_size=32, warmup=10, latency_calls=100,
                        throughput_calls=32, threads=4, round_index=0):
    """Return raw durations and statistics for ONE measurement round.

    Includes prediction API preprocessing/transfers; excludes data/model loading,
    upstream RF generation and supplied-descriptor extraction. FCM instead times
    the full station pool after one warm-up. Independent methods use full batches
    only, requiring sample_size to be divisible by batch_size. Run repeated rounds
    in fresh processes via scripts/benchmark_models.py for the paper-style design.
    Timing does not modify weights, thresholds, labels or archived accuracy scores.
    """
    for name, value in [('batch_size', batch_size), ('warmup', warmup),
                        ('latency_calls', latency_calls), ('throughput_calls', throughput_calls),
                        ('threads', threads)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(round_index, int) or round_index < 0:
        raise ValueError('round_index must be nonnegative')
    prepared = timing_input(data, predictor.spec, sample_size, complete_views)
    if predictor.spec.kind != 'fcm' and (len(prepared) < batch_size or len(prepared) % batch_size):
        raise ValueError('Selected record count must be a positive multiple of batch_size')
    cuda = predictor.network is not None and predictor.device.type == 'cuda'
    mps = predictor.network is not None and predictor.device.type == 'mps'
    if predictor.network is None and predictor.device.type != 'cpu':
        raise ValueError('Statistical implementations run on CPU; specify device=cpu')
    if predictor.network is not None and any(x.dtype != torch.float32 for x in predictor.network.parameters()):
        raise ValueError('Benchmark protocol requires FP32 model parameters')
    if predictor.network is not None and any(x.training for x in predictor.network.modules()):
        raise ValueError('Benchmark requires network.eval(); training mode would change inference behavior')

    def sync():
        if cuda:
            torch.cuda.synchronize(predictor.device)
        elif mps:
            torch.mps.synchronize()

    def timed(fn):
        sync(); start = time.perf_counter_ns(); value = fn(); sync()
        return (time.perf_counter_ns() - start) / 1e9, value

    old_threads = torch.get_num_threads()
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        with threadpool_limits(limits=threads), torch.autocast(device_type=predictor.device.type, enabled=False):
            return _measure(predictor, prepared, cuda, timed, batch_size, warmup,
                            latency_calls, throughput_calls, threads, round_index, complete_views)
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old_tf32


def _measure(p, data, cuda, timed, batch, warmup, single_calls, bulk_calls, threads, repeat, complete_views):
    params = None
    if p.network is not None:
        params = sum(x.numel() for x in p.network.parameters())
    elif p.spec.kind == 'logistic':
        params = np.asarray(p.bundle['parameters']['coef']).size + np.asarray(p.bundle['parameters']['intercept']).size
    else:
        params = np.asarray(p.bundle['parameters']['centers']).size
    report = dict(schema=1, method=p.spec.name, seed=p.seed, round=repeat, kind=p.spec.kind,
                  device=str(p.device), parameters=int(params), records=len(data),
                  complete_views_selection=bool(complete_views and p.spec.kind != 'fcm'),
                  views=sorted(set(data.lengths.tolist())),
                  sample_ids_sha256=hashlib.sha256('\n'.join(data.sample_ids.tolist()).encode()).hexdigest(),
                  weights_sha256=p.bundle.get('weights_sha256'),
                  torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
                  platform=platform.system(), cpu_architecture=platform.machine(), threads=threads,
                  interop_threads=torch.get_num_interop_threads(),
                  precision='FP32 eager; AMP/TF32 off' if p.network is not None else 'Native NumPy/SciPy floating point',
                  threadpools=[{k:x.get(k) for k in ('user_api','internal_api','num_threads','version')}
                               for x in threadpool_info()],
                  timing_scope='Resident native RFData to CPU scores/decisions; includes preprocessing and transfers; excludes loading, RF generation, supplied-descriptor extraction and network transport.')
    if cuda:
        report['gpu_name'] = torch.cuda.get_device_name(p.device)
    if p.spec.kind == 'fcm':
        p.predict(data)
        seconds, result = timed(lambda: p.predict(data))
        report.update(pool_seconds=seconds, pool_records_per_second=len(data)/seconds,
                      stations=len(set(data.stations)), warmup_calls=1,
                      latency_scope='Complete station pool; no independent-record latency')
    else:
        singles = [data.subset(slice(i, i+1)) for i in range(len(data))]
        batches = [data.subset(slice(i, i+batch)) for i in range(0, len(data), batch)]
        for i in range(warmup):
            p.predict(singles[i % len(singles)], batch_size=1)
        latency = [timed(lambda i=i: p.predict(singles[(repeat*single_calls+i) % len(singles)], batch_size=1))[0]
                   for i in range(single_calls)]
        for i in range(warmup):
            p.predict(batches[i % len(batches)], batch_size=batch)
        if cuda:
            torch.cuda.reset_peak_memory_stats(p.device)
        bulk = [timed(lambda i=i: p.predict(batches[(repeat+i) % len(batches)], batch_size=batch))[0]
                for i in range(bulk_calls)]
        report.update(batch_size=batch, warmup_calls=warmup, latency_seconds=latency,
                      latency_p50_ms=1000*float(np.median(latency)),
                      latency_p95_ms=1000*float(np.percentile(latency, 95)),
                      batch_seconds=bulk, batch_records_per_second=batch*len(bulk)/sum(bulk),
                      pipeline_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(p.device) if cuda else None)
        if p.network is not None:
            args = [p.preprocessor.transform(x, p.device) for x in batches]
            with torch.inference_mode():
                for i in range(warmup):
                    p.network(*args[i % len(args)]).sigmoid()
                net = [timed(lambda i=i: p.network(*args[(repeat+i) % len(args)]).sigmoid())[0]
                       for i in range(bulk_calls)]
            report.update(forward_batch_seconds=net, forward_batch_records_per_second=batch*len(net)/sum(net))
        result = p.predict(data, batch_size=batch)
    report['finite_replay'] = bool(np.isfinite(result.p_good).all())
    return report
