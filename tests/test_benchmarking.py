"""Timing boundaries, same-pool selection, and immutable inference settings."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from rfqc_bench import benchmark_inference, synthetic_data, RFQCPredictor, create_model
from rfqc_bench.benchmarking import timing_input
from rfqc_bench.preprocessing import Preprocessor
from rfqc_bench.registry import get_spec


class Recorder:
    def __init__(self, name):
        self.spec=get_spec(name);self.network=None;self.device=torch.device('cpu');self.seed=1
        self.bundle={'parameters':{'coef':[[0]*6], 'intercept':[0], 'centers':[[0]*4,[1]*4]}}
        self.calls=[]

    def predict(self, data, batch_size=32):
        self.calls.append((len(data), data.labels, data.waveforms is None, data.features is None))
        return SimpleNamespace(p_good=np.full(len(data),.5))


def test_label_blind_order_and_native_inputs():
    data=synthetic_data(40)
    a=timing_input(data,get_spec('reference_ag3'),32,True)
    b=timing_input(data.subset(np.arange(39,-1,-1)),get_spec('reference_multifilter'),32,True)
    np.testing.assert_array_equal(a.sample_ids,b.sample_ids)
    assert a.waveforms.shape==(32,1,501) and b.waveforms.shape==(32,7,501)
    assert a.labels is b.labels is None and a.features is b.features is None
    data.labels=1-data.labels
    np.testing.assert_array_equal(a.sample_ids,timing_input(data,get_spec('logreg_ag3'),32,True).sample_ids)


def test_fcm_never_subsamples_station_context():
    p=Recorder('xiong2025_fcm')
    report=benchmark_inference(p,synthetic_data(40),sample_size=8,complete_views=True,
                               warmup=3,latency_calls=2,throughput_calls=2)
    assert [x[0] for x in p.calls]==[40,40]
    assert report['records']==40 and report['stations']==10
    assert 'latency_p50_ms' not in report and 'batch_records_per_second' not in report
    assert report['pool_records_per_second']==pytest.approx(40/report['pool_seconds'])


def test_actual_batch_denominator_and_no_data_in_report():
    p=Recorder('logreg_ag3')
    r=benchmark_inference(p,synthetic_data(32),sample_size=32,batch_size=8,warmup=2,
                          latency_calls=3,throughput_calls=5,threads=1)
    assert [x[0] for x in p.calls]==[1]*5+[8]*7+[32]
    assert r['batch_records_per_second']==pytest.approx(40/sum(r['batch_seconds']))
    assert r['latency_p50_ms']==pytest.approx(1000*np.median(r['latency_seconds']))
    assert not {'sample_ids','labels','waveforms','features','p_good','prediction'} & r.keys()
    assert all(x[1] is None and x[2] and not x[3] for x in p.calls)


def test_failed_timing_restores_global_settings():
    p=Recorder('logreg_ag3')
    def fail(*args,**kwargs):raise RuntimeError('measurement failed')
    p.predict=fail
    before=(torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
    with pytest.raises(RuntimeError,match='measurement failed'):
        benchmark_inference(p,synthetic_data(32),sample_size=32,threads=2)
    after=(torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
    assert before==after


def test_neural_weights_and_scores_unchanged():
    data=synthetic_data(4).select(3)
    network=create_model('gong_cnn').eval()
    p=RFQCPredictor(dict(schema=1,method='gong_cnn',seed=1,threshold=.7,
                        preprocessing=Preprocessor.fit(data).state),network)
    expected=p.predict(data).p_good.copy();weights={k:v.clone() for k,v in network.state_dict().items()}
    r=benchmark_inference(p,data,sample_size=4,batch_size=2,warmup=1,latency_calls=2,throughput_calls=2,threads=1)
    assert r['finite_replay'] and r['forward_batch_records_per_second']>0
    np.testing.assert_allclose(expected,p.predict(data).p_good,atol=1e-7)
    assert all(torch.equal(v,network.state_dict()[k]) for k,v in weights.items())
    assert p.threshold==.7
    network.train()
    with pytest.raises(ValueError,match='network.eval'):
        benchmark_inference(p,data,sample_size=4,batch_size=2)


@pytest.mark.parametrize('kwargs',[{'sample_size':31},{'batch_size':0},{'warmup':0},{'round_index':-1}])
def test_invalid_measurement_budget(kwargs):
    with pytest.raises(ValueError):
        benchmark_inference(Recorder('logreg_ag3'),synthetic_data(32),**dict({'sample_size':32},**kwargs))
