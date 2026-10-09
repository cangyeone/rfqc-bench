"""Optional FastAPI service; configure the model at startup, not per request."""
from threading import Lock
from typing import Any
from pathlib import Path
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


class ScreenEQRRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    directory: str = Field(description='Directory relative to the configured server eqr_root')
    output: str | None = Field(default=None,description='Record path relative to eqr_root; default: directory/record')
    gaussian: float | None = None


def create_app(predictor,max_records=4096,batch_size=32,*,eqr_root=None,max_eqr_files=100000):
    """Serve a preloaded predictor. Swagger docs: /docs; OpenAPI: /openapi.json."""
    app=FastAPI(title='RFQC Bench API',version=__version__,description='RF quality agreement scores; good=1, bad=0. No structural inversion.')
    lock=Lock()
    root=Path(eqr_root).expanduser().resolve(strict=True) if eqr_root is not None else None
    if root is not None and not root.is_dir():raise ValueError('eqr_root must be a directory')
    if max_eqr_files<1:raise ValueError('max_eqr_files must be positive')

    def confined(relative):
        part=Path(relative)
        if part.is_absolute() or '..' in part.parts:
            raise ValueError('Use a relative path within eqr_root; traversal is not allowed')
        target=root/part
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError('Server directory paths must not contain symlinks')
        if not target.resolve().is_relative_to(root):
            raise ValueError('Path escapes eqr_root')
        return target

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

    @app.post('/screen-eqr')
    def screen_eqr(request:ScreenEQRRequest):
        if root is None:
            raise HTTPException(status_code=403,detail='Directory screening is disabled; configure --eqr-root at server startup')
        try:
            directory=confined(request.directory)
            output=confined(request.output) if request.output is not None else directory/'record'
            with lock:
                return predictor.screen_eqr(directory,output,gaussian=request.gaussian,
                                            batch_size=batch_size,max_files=max_eqr_files)
        except FileExistsError as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc
        except (ValueError,OSError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    return app
