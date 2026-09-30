"""Start rfqc-bench serve first; no dataset files are needed for this demo."""
import json
from urllib.request import Request, urlopen
from rfqc_bench import synthetic_data

data=synthetic_data(n=2)
payload=dict(waveforms=data.waveforms.tolist(),features=data.features.tolist(),
             gaussians=data.gaussians.tolist(),lengths=data.lengths.tolist(),stations=data.stations.tolist())
request=Request('http://127.0.0.1:8000/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urlopen(request) as response:print(json.load(response))
