import json
import signal
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score
from rfqc_bench import RFData,RFQCPredictor,create_model,list_models,synthetic_data,fit,evaluate,station_bootstrap
from rfqc_bench.api import create_app
from rfqc_bench.preprocessing import Preprocessor
from rfqc_bench.registry import get_spec
from rfqc_bench.training import save_bundle
from rfqc_bench.diagnostics import stack_comparison,cross_filter_association


@pytest.mark.parametrize('name',[m['name'] for m in list_models() if m['kind']=='neural'])
def test_every_neural_architecture(name):
    d=synthetic_data(2).select(get_spec(name).gaussian)
    model=create_model(name).eval()
    with torch.inference_mode():out=model(*Preprocessor.fit(d).transform(d))
    assert out.shape==(2,) and torch.isfinite(out).all()
    if name.startswith('reference'):assert sum(p.numel() for p in model.parameters())==395393


def predictor(tmp_path,name='gong_cnn'):
    d=synthetic_data(4).select(get_spec(name).gaussian)
    net=create_model(name).eval()
    save_bundle(tmp_path,name,123,.7,Preprocessor.fit(d).state,net)
    return RFQCPredictor.from_directory(tmp_path)


def test_safe_serialization_and_http(tmp_path):
    p=predictor(tmp_path);d=synthetic_data(3)
    expected=p.predict(d).to_dict()
    client=TestClient(create_app(p))
    body=dict(waveforms=d.waveforms.tolist(),gaussians=d.gaussians.tolist(),lengths=d.lengths.tolist())
    response=client.post('/predict',json=body)
    assert response.status_code==200 and response.json()==expected
    assert client.get('/health').json()['status']=='ok'
    assert '/predict' in client.get('/openapi.json').json()['paths']
    assert client.post('/predict',json=dict(body,labels=[0,1,1])).status_code==422
    assert client.post('/predict',json=dict(body,start_time=0.)).status_code==422
    assert client.post('/predict',json={'waveforms':[[1,2,3]]}).status_code==422
    with (tmp_path/'weights.safetensors').open('ab') as f:f.write(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):RFQCPredictor.from_directory(tmp_path)


def test_missing_filter_mask_and_batches(tmp_path):
    p=predictor(tmp_path,'reference_multifilter');d=synthetic_data(3)
    d.lengths[0]=6;d.waveforms[0,-1,:]=1e20
    pred=p.predict(d,batch_size=1).p_good
    np.testing.assert_allclose(pred,p.predict(d,batch_size=3).p_good,atol=2e-6)
    truncated=RFData(d.waveforms[:1,:6],d.gaussians[:1,:6])
    np.testing.assert_allclose(pred[:1],p.predict(truncated).p_good,atol=2e-6)
    with pytest.raises(ValueError,match='AG5'):d.select(5.)


def test_input_contract():
    with pytest.raises(ValueError):RFData(np.zeros((2,7,501)))
    with pytest.raises(ValueError):RFData(np.zeros((2,2,501)),[3.,1.])
    with pytest.raises(ValueError):RFData(np.zeros((2,1,501)),lengths=[.5,1])
    with pytest.raises(ValueError):RFData(np.full((2,501),np.inf))
    with pytest.raises(ValueError):RFData(np.zeros((2,501)),features=np.zeros((3,6)))


def splits():
    a=synthetic_data(20,seed=17);b=synthetic_data(8,seed=23)
    b.stations=np.array(['val_'+s for s in b.stations]);b.sample_ids=np.array(['val_'+s for s in b.sample_ids])
    return a,b


@pytest.mark.parametrize('name',['logreg_ag3','logreg_multifilter','xiong2025_fcm'])
def test_statistical_fit_and_reload(tmp_path,name):
    a,b=splits();p=fit(name,a,b,tmp_path/name)
    before=p.predict(b).p_good
    after=RFQCPredictor.from_directory(tmp_path/name).predict(b).p_good
    np.testing.assert_array_equal(before,after)
    assert np.isfinite(before).all()


def test_neural_manual_resume_at_batch_boundary(tmp_path,monkeypatch):
    a,b=splits();options=dict(seed=42,epochs=2,batch_size=4,patience=10,device='cpu')
    full=fit('descriptors_multifilter',a,b,tmp_path/'full',**options)
    original=torch.optim.AdamW.step;calls=[0]
    def stop_after_second(self,*args,**kwargs):
        result=original(self,*args,**kwargs);calls[0]+=1
        if calls[0]==2:signal.raise_signal(signal.SIGINT)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW,'step',stop_after_second)
        with pytest.raises(KeyboardInterrupt):fit('descriptors_multifilter',a,b,tmp_path/'resumed',**options)
    state=json.loads((tmp_path/'resumed/status.json').read_text())
    assert state['state']=='paused' and state['next_record']==8
    restored=fit('descriptors_multifilter',a,b,tmp_path/'resumed',resume=True,**options)
    np.testing.assert_array_equal(full.predict(b).p_good,restored.predict(b).p_good)
    assert full.threshold==restored.threshold
    with pytest.raises(ValueError,match='settings differ'):
        fit('descriptors_multifilter',a,b,tmp_path/'resumed',resume=True,**dict(options,seed=99))


def test_metrics_and_station_pairing():
    y=np.array([0,1,0,1,0,1]);p=np.array([.1,.7,.6,.8,.4,.9])
    assert evaluate(y,p)['macro_f1']==f1_score(y,p>=.5,average='macro')
    result=station_bootstrap(y,p,p,np.array(['a','a','b','b','c','c']),draws=100)
    assert result['lower_pp']==result['upper_pp']==0
    d=synthetic_data(6)
    comparison=stack_comparison(d.select(3).waveforms[:,0],d.labels==1,d.labels==1)
    assert comparison['raw']['nrmse_whole']==0
    assert comparison['shape']['corr_post_p']==pytest.approx(1.)
    assert len(cross_filter_association(d))==6


def test_no_train_validation_station_overlap(tmp_path):
    a,b=splits();b.stations[:]='synthetic_0'
    with pytest.raises(ValueError,match='stations overlap'):fit('logreg_ag3',a,b,tmp_path/'bad')
