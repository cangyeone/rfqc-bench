"""Command line for listing, downloading, prediction, training and serving."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__, RFData, RFQCPredictor, evaluate, fit, list_models, synthetic_data, download_model


def _predictor(a):
    if getattr(a,'model_dir',None):return RFQCPredictor.from_directory(a.model_dir,a.device)
    return RFQCPredictor.from_pretrained(a.model,a.seed,a.device,getattr(a,'cache_dir',None))


def main(argv=None):
    p=argparse.ArgumentParser(description='RFQC benchmark algorithms; datasets are never downloaded')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='List every model configuration')
    sub.add_parser('doctor',help='Show installed runtime and available compute devices')
    for cmd in ['download','predict','demo','serve']:
        c=sub.add_parser(cmd)
        group=c.add_mutually_exclusive_group()
        group.add_argument('--model',default='reference_multifilter')
        group.add_argument('--model-dir',type=Path)
        c.add_argument('--seed',type=int);c.add_argument('--cache-dir',type=Path)
        c.add_argument('--device',default='cpu');c.add_argument('--batch-size',type=int,default=32)
        if cmd=='predict':
            c.add_argument('--input',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
            c.add_argument('--evaluate',action='store_true',help='Also score supplied labels; never tunes the threshold')
        elif cmd=='demo':c.add_argument('--output',type=Path,default=Path('synthetic_predictions.csv'))
        elif cmd=='serve':
            c.add_argument('--host',default='127.0.0.1');c.add_argument('--port',type=int,default=8000)
            c.add_argument('--max-records',type=int,default=4096)
    t=sub.add_parser('train');t.add_argument('--model',required=True)
    t.add_argument('--train',type=Path,required=True);t.add_argument('--validation',type=Path,required=True)
    t.add_argument('--output',type=Path,required=True);t.add_argument('--device',default='cpu')
    t.add_argument('--seed',type=int,default=20260928);t.add_argument('--epochs',type=int,default=50)
    t.add_argument('--patience',type=int,default=10);t.add_argument('--batch-size',type=int,default=32)
    t.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.command=='list':print(json.dumps(list_models(),indent=2));return
        if a.command=='doctor':
            import torch,sklearn,numpy
            print(json.dumps(dict(rfqc_bench=__version__,python=sys.version.split()[0],torch=str(torch.__version__),
                                  sklearn=sklearn.__version__,numpy=numpy.__version__,cuda=torch.cuda.is_available(),
                                  mps=torch.backends.mps.is_available()),indent=2));return
        if a.command=='download':print(download_model(a.model,a.seed,a.cache_dir));return
        if a.command=='train':
            trained=fit(a.model,RFData.load(a.train),RFData.load(a.validation),a.output,seed=a.seed,epochs=a.epochs,
                        patience=a.patience,batch_size=a.batch_size,device=a.device,resume=a.resume)
            print(json.dumps(dict(output=str(a.output),model=trained.spec.name,threshold=trained.threshold)));return
        predictor=_predictor(a)
        if a.command=='serve':
            try:
                import uvicorn
                from .api import create_app
            except ImportError:raise ValueError('Install the [api] extra to run the HTTP service') from None
            uvicorn.run(create_app(predictor,a.max_records,a.batch_size),host=a.host,port=a.port);return
        data=synthetic_data() if a.command=='demo' else RFData.load(a.input)
        result=predictor.predict(data,batch_size=a.batch_size);result.to_csv(a.output,data.sample_ids)
        report=dict(output=str(a.output),records=len(data),method=result.method,seed=result.seed,threshold=result.threshold)
        if a.command=='demo':report['scope']='Synthetic signals only; this tests installation, not scientific accuracy.'
        elif a.evaluate:
            if data.labels is None:raise ValueError('--evaluate requires labels in the input')
            report['metrics']=evaluate(data.labels,result.p_good,result.threshold)
        print(json.dumps(report,indent=2))
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')
    except KeyboardInterrupt:p.exit(130,'Training interrupted; repeat the same train command with --resume.\n')


if __name__=='__main__':main()
