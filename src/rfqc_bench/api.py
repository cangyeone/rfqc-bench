"""Optional FastAPI service; configure the model at startup, not per request."""
from threading import Lock
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from . import __version__
from .data import RFData
from .registry import list_models


class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    waveforms: list[Any] | None = None
    features: list[Any] | None = None
    gaussians: list[Any] | None = None
    lengths: list[int] | None = None
    stations: list[str] | None = None
    start_time: float = -10.
    sampling_interval: float = .1


class PredictResponse(BaseModel):
    p_good: list[float]
    prediction: list[int]
    threshold: float
    method: str
    seed: int


def create_app(predictor,max_records=4096,batch_size=32):
    """Serve a preloaded predictor. Swagger docs: /docs; OpenAPI: /openapi.json."""
    app=FastAPI(title='RFQC Bench API',version=__version__,description='RF quality agreement scores; good=1, bad=0. No structural inversion.')
    lock=Lock()

    @app.get('/health')
    def health():
        return dict(status='ok',version=__version__,model=predictor.spec.name,seed=predictor.seed)

    @app.get('/models')
    def models():return list_models()

    @app.get('/model')
    def model():
        return dict(name=predictor.spec.name,seed=predictor.seed,threshold=predictor.threshold,
                    device=str(predictor.device),max_records=max_records,input_grid=predictor.bundle['input_grid'])

    @app.post('/predict',response_model=PredictResponse)
    def predict(request:PredictRequest):
        try:
            raw=request.model_dump();n=len(raw.get('waveforms') or raw.get('features') or [])
            if not 1<=n<=max_records:raise ValueError(f'Request must contain 1..{max_records} RF records')
            data=RFData(**raw)
            with lock:result=predictor.predict(data,batch_size=batch_size)
            return result.to_dict()
        except (ValueError,TypeError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    return app
