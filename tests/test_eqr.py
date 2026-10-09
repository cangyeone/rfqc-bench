import csv
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from rfqc_bench import RFQCPredictor, create_model, screen_eqr, synthetic_data
from rfqc_bench.api import create_app
from rfqc_bench.cli import main
from rfqc_bench.eqr import read_eqr
from rfqc_bench.predictor import Prediction
from rfqc_bench.preprocessing import Preprocessor
from rfqc_bench.registry import get_spec
from rfqc_bench.training import save_bundle


def sac(path, *, value=1., endian='<', version=6, dt=.1, begin=-15., n=601,
        station='EW27', baz=40., nan=False, uneven=False):
    """Synthetic binary SAC fixture; no observational data in the test suite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = np.full(70, -12345., dtype=endian+'f4')
    f[[0,5,6,44,52,53]] = [dt,begin,begin+(n-1)*dt,.065,baz,70.]
    i = np.full(40, -12345, dtype=endian+'i4')
    i[[6,9,15,35]] = [version,n,1,0 if uneven else 1]
    chars = bytearray(b' '*192)
    chars[:8] = station.encode().ljust(8)[:8]
    chars[160:168] = b'EQR     '
    chars[168:176] = b'DB      '
    wave = np.full(n,value,dtype=endian+'f4')
    if nan:wave[100] = np.nan
    footer = np.full(22, -12345., dtype=endian+'f8')
    footer[:3] = [dt,begin,begin+(n-1)*dt]
    path.write_bytes(f.tobytes()+i.tobytes()+chars+wave.tobytes()+(footer.tobytes() if version==7 else b''))
    return path


class Dummy:
    def __init__(self, model='reference_multifilter'):
        self.spec=get_spec(model);self.seed=12;self.threshold=.6;self.device='cpu'
        self.bundle={'input_grid':{},'method':model}
        self.calls=[]

    def predict(self, data, batch_size=32):
        self.calls.append(data)
        p=np.where(data.waveforms[:,0,100]>0,.9,.1)
        return Prediction(p,(p>=self.threshold).astype(int),self.threshold,self.spec.name,self.seed)

    def screen_eqr(self,*args,**kwargs):
        return screen_eqr(*args,predictor=self,**kwargs)


@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('version',[6,7])
def test_sac_matches_obspy_grid(tmp_path,endian,version):
    path=sac(tmp_path/'test.eqr',endian=endian,version=version)
    wave,metadata=read_eqr(path)
    assert wave.shape==(501,) and wave.dtype==np.float32
    np.testing.assert_array_equal(wave,np.ones(501))
    assert metadata['station']=='EW27' and metadata['baz']==40.
    assert len(metadata['sha256'])==64
    # Independent optional oracle for the SAC header and exact waveform slice.
    obspy=pytest.importorskip('obspy')
    # ObsPy's size check predates the v7 double footer; use it as a waveform
    # oracle only for v7. The reader's own exact file-size checks remain on.
    trace=obspy.read(str(path),format='SAC',fsize=(version==6))[0]
    np.testing.assert_array_equal(wave,trace.data[50:551])


def test_v7_uses_double_footer_time_grid(tmp_path):
    path=sac(tmp_path/'v7.eqr',version=7)
    raw=bytearray(path.read_bytes())
    raw[:4]=np.array([.2],dtype='<f4').tobytes()
    path.write_bytes(raw)
    assert read_eqr(path)[0].shape==(501,)


@pytest.mark.parametrize('options,match',[
    ({'dt':.2},'delta'),({'begin':0},'cover'),({'begin':-15.01},'cover'),
    ({'n':500},'cover'),({'nan':True},'Nonfinite'),({'value':0},'All-zero'),
    ({'uneven':True},'evenly sampled')])
def test_invalid_sac(tmp_path,options,match):
    with pytest.raises(ValueError,match=match):read_eqr(sac(tmp_path/'x.eqr',**options))


def test_truncated_and_extra_bytes(tmp_path):
    path=sac(tmp_path/'x.eqr')
    raw=path.read_bytes()
    for bad in [raw[:200],raw[:-4],raw+b'junk']:
        path.write_bytes(bad)
        with pytest.raises(ValueError):read_eqr(path)


def test_grouping_missing_view_and_no_folder_label_leakage(tmp_path):
    root=tmp_path/'input'
    # A good prediction must still be possible under a folder named bad.
    sac(root/'A/AG1/bad/same.eqr')
    sac(root/'A/AG3/same.eqr')
    sac(root/'A/AG3/other.eqr',value=-1)
    sac(root/'B/AG3/same.eqr',value=-1,station='B')
    before={p:p.read_bytes() for p in root.rglob('*.eqr')}
    predictor=Dummy()
    report=screen_eqr(root,predictor=predictor)
    assert (root/'record').read_text().splitlines()==['A/AG1/bad/same.eqr','A/AG3/same.eqr']
    assert report['events_screened']==3 and report['good_events']==1 and report['retained_files']==2
    assert [len(d) for d in predictor.calls]==[2,1]
    assert sorted(predictor.calls[0].lengths)==[1,2]
    assert all(p.read_bytes()==b for p,b in before.items())
    rows=list(csv.DictReader((root/'record.predictions.csv').open()))
    assert len({r['sample_id'] for r in rows})==3
    assert report==json.loads((root/'record.json').read_text())
    with pytest.raises(FileExistsError):screen_eqr(root,predictor=predictor)
    screen_eqr(root,predictor=predictor,overwrite=True)


def test_single_filter_and_root_itself_ag(tmp_path):
    root=tmp_path/'STA'
    sac(root/'AG1/event.eqr');sac(root/'AG3/event.eqr')
    report=screen_eqr(root,predictor=Dummy('reference_ag3'))
    assert (root/'record').read_text()=='AG3/event.eqr\n'
    assert report['rejected_or_unused_entries']==1
    report=screen_eqr(root/'AG3',predictor=Dummy())
    assert (root/'AG3/record').read_text()=='event.eqr\n'
    assert report['good_events']==1


def test_duplicates_and_header_conflicts_are_not_chosen(tmp_path):
    root=tmp_path/'input'
    sac(root/'STA/AG3/duplicate.eqr')
    sac(root/'STA/AG3/bad/duplicate.eqr')
    sac(root/'STA/AG1/conflict.eqr')
    sac(root/'STA/AG3/conflict.eqr',baz=70)
    report=screen_eqr(root,predictor=Dummy())
    assert report['events_screened']==0 and report['status']=='no_valid_events'
    assert (root/'record').read_bytes()==b''
    reasons=(root/'record.rejected.csv').read_text()
    assert 'Ambiguous duplicate' in reasons and 'Conflicting' in reasons


def test_flat_gaussian_required_and_fcm_station_pool(tmp_path):
    root=tmp_path/'flat'
    for i in range(5):sac(root/f'{i}.eqr',station='A' if i<3 else 'B')
    report=screen_eqr(root,tmp_path/'missing',predictor=Dummy())
    assert report['status']=='no_valid_events'
    predictor=Dummy('xiong2025_fcm')
    report=screen_eqr(root,predictor=predictor,gaussian=3.,batch_size=1)
    assert sorted(len(d) for d in predictor.calls)==[2,3]
    assert report['retained_files']==5


def test_rejections_limits_and_failed_inference_preserve_outputs(tmp_path):
    root=tmp_path/'input'
    sac(root/'AG3/valid.eqr');sac(root/'AG3/invalid.eqr',nan=True)
    predictor=Dummy()
    report=screen_eqr(root,predictor=predictor)
    assert report['retained_files']==1 and report['rejected_or_unused_entries']==1
    before={p:p.read_bytes() for p in root.glob('record*')}
    def fail(*args,**kwargs):raise RuntimeError('inference failed')
    predictor.predict=fail
    with pytest.raises(RuntimeError):screen_eqr(root,predictor=predictor,overwrite=True)
    assert all(p.read_bytes()==data for p,data in before.items())
    with pytest.raises(ValueError,match='limit'):
        screen_eqr(root,tmp_path/'limited',predictor=Dummy(),max_files=1)
    assert not (tmp_path/'limited').exists()
    with pytest.raises(ValueError,match='waveform models'):
        screen_eqr(root,tmp_path/'feature',predictor=Dummy('combined_ag3'))
    with pytest.raises(ValueError,match='EQR'):
        screen_eqr(root,root/'AG3/valid.eqr',predictor=Dummy())
    (root/'.busy.rfqc-lock').mkdir()
    with pytest.raises(FileExistsError,match='Publication lock'):
        screen_eqr(root,root/'busy',predictor=Dummy())
    assert not (root/'busy').exists()


def test_http_and_symlinks(tmp_path):
    root=tmp_path/'root';sac(root/'input/AG3/ok.eqr')
    outside=tmp_path/'outside';sac(outside/'secret.eqr')
    (root/'outside').symlink_to(outside,target_is_directory=True)
    (root/'input/AG3/link.eqr').symlink_to(outside/'secret.eqr')
    disabled=TestClient(create_app(Dummy()))
    assert disabled.post('/screen-eqr',json={'directory':str(root)}).status_code==403
    client=TestClient(create_app(Dummy(),eqr_root=root))
    for directory in ['../outside',str(outside),'outside']:
        assert client.post('/screen-eqr',json={'directory':directory}).status_code==422
    assert client.post('/screen-eqr',json={'directory':'input','output':'../oops'}).status_code==422
    response=client.post('/screen-eqr',json={'directory':'input'})
    assert response.status_code==200 and response.json()['retained_files']==1
    assert response.json()['rejected_or_unused_entries']==1
    assert client.post('/screen-eqr',json={'directory':'input'}).status_code==409
    assert not (outside/'record').exists()
    assert '/screen-eqr' in client.get('/openapi.json').json()['paths']


def test_actual_model_python_and_cli(tmp_path,capsys):
    training=synthetic_data(4)
    bundle=tmp_path/'model'
    save_bundle(bundle,'reference_multifilter',123,.5,Preprocessor.fit(training).state,
                create_model('reference_multifilter').eval())
    root=tmp_path/'input'
    for gaussian in [1,3,5]:sac(root/f'STA/AG{gaussian}/test.eqr')
    predictor=RFQCPredictor.from_directory(bundle)
    expected=predictor.predict(waveforms=np.ones((1,3,501)),gaussians=[1,3,5])
    result=predictor.screen_eqr(root)
    rows=list(csv.DictReader((root/'record.predictions.csv').open()))
    assert float(rows[0]['p_good'])==expected.p_good[0]
    main(['screen-eqr',str(root),'--model-dir',str(bundle),'--output',str(tmp_path/'cli-record')])
    assert json.loads(capsys.readouterr().out)['events_screened']==1
    assert (tmp_path/'cli-record').read_bytes()==(root/'record').read_bytes()
    assert result['model_weights_sha256']==predictor.bundle['weights_sha256']
