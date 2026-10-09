# HTTP API and deployment

Install the `api` extra and start one model per process:

```bash
python -m pip install "rfqc-bench[api] @ git+https://github.com/cangyeone/rfqc-bench.git@v0.1.2"
rfqc-bench serve --model gong_cnn --seed 20260928 --device cpu --host 127.0.0.1 --port 8000
```

Routes: `GET /health`, `GET /model`, `GET /models`, `POST /predict`, `POST /screen-eqr`.
Interactive docs: `/docs`; machine-readable schema: `/openapi.json`.
This uses [FastAPI's documented OpenAPI interface](https://fastapi.tiangolo.com/tutorial/first-steps/).

Create a **synthetic** request file, then call with curl:

```python
import json
from rfqc_bench import synthetic_data
d = synthetic_data(2)
request = dict(waveforms=d.waveforms.tolist(), gaussians=d.gaussians.tolist(),
               features=d.features.tolist(), lengths=d.lengths.tolist(),
               stations=d.stations.tolist(), start_time=-10., sampling_interval=.1)
with open("request.json", "w") as f:
    json.dump(request, f)
```

```bash
curl -H 'Content-Type: application/json' --data-binary @request.json http://127.0.0.1:8000/predict
```

Response schema:

```json
{"p_good": [0.7, 0.2], "prediction": [1, 0], "threshold": 0.5,
 "method": "gong_cnn", "seed": 20260928}
```

Numbers above illustrate the schema, not an experimental output. The actual
threshold comes from the selected bundle. Invalid shape, time grid, Gaussian
order, missing required features or unexpected fields return HTTP 422. Labels
are intentionally not part of the prediction request. Default maximum is 4096
records per request, configurable with `--max-records`; neural compute uses
`--batch-size`. Very large datasets should use the CLI/Python API.

For Xiong-FCM, one request must contain each station's complete pool: dividing it
between requests changes the correlation feature. For other methods request
batching does not change their input context. For offline deployment use
`--model-dir /path/to/downloaded/bundle`; no runtime downloads are then needed.

The default address is localhost. To expose a service on a managed network,
use `--host 0.0.0.0` and supply your organization's authentication/TLS/reverse
proxy and body-size limits. The package does not provision a public server or
store request datasets. Select one GPU per service with `--device cuda:0` or
`cuda:1`; use separate ports. Model execution is serialized inside a process;
multiple workers independently allocate weights and can multiply GPU memory.

## Optional server-side EQR directories

```bash
rfqc-bench serve --eqr-root /srv/rfqc --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/screen-eqr \
  -H 'Content-Type: application/json' \
  -d '{"directory":"incoming/all_eqr","output":"results/record"}'
```

`directory` and optional `output` are relative to the configured **server-side**
root, not the client's filesystem. Flat inputs can specify `"gaussian":3`.
Files must already exist on the server; there is no upload endpoint. The response
is a JSON summary including paths to the plain record, prediction/rejection CSVs,
counts, model identity and hashes. The endpoint returns 403 unless `--eqr-root`
was configured. Path traversal and symlinks are refused. Existing output returns
409; the HTTP endpoint never overwrites results. `--max-eqr-files` defaults to
100000 and caps the scan before inference/output. Large jobs should use the CLI;
HTTP execution is synchronous. This service assumes a trusted server filesystem
and does not replace authentication or filesystem permissions.

The Python factory is `create_app(predictor, eqr_root='/srv/rfqc',
max_eqr_files=100000)`; existing array clients are unchanged.
