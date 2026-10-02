"""RFQC benchmark: models, trained inference, fitting and evaluation."""
__version__ = "0.1.1"
from .data import RFData, synthetic_data
from .metrics import evaluate, label_agreement, station_bootstrap
from .models import create_model
from .predictor import Prediction, RFQCPredictor
from .registry import list_models
from .zoo import available_weights, download_model
from .benchmarking import benchmark_inference


def fit(*args, **kwargs):
    from .training import fit as train
    return train(*args, **kwargs)


__all__ = ['RFData','Prediction','RFQCPredictor','create_model','list_models','available_weights',
           'download_model','fit','evaluate','label_agreement','station_bootstrap','synthetic_data','benchmark_inference']
