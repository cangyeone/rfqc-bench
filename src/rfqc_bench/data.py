"""Explicit RF inputs; no hidden download, alignment, resampling or labeling."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

GAUSSIANS = np.array([1., 1.5, 2., 2.5, 3., 4., 5.])


@dataclass
class RFData:
    waveforms: np.ndarray | None = None
    gaussians: np.ndarray | None = None
    lengths: np.ndarray | None = None
    features: np.ndarray | None = None
    stations: np.ndarray | None = None
    labels: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    start_time: float = -10.0
    sampling_interval: float = 0.1

    def __post_init__(self):
        if not np.isclose(self.start_time, -10., atol=1e-9, rtol=0) or not np.isclose(self.sampling_interval, .1, atol=1e-9, rtol=0):
            raise ValueError("Expected 501 observed samples at dt=0.1 s, t0=-10 s; explicitly adapt other grids before use")
        shape = None
        for name, width in [('waveforms', 501), ('features', 6)]:
            v = getattr(self, name)
            if v is None: continue
            v = np.asarray(v)
            if v.ndim == 2: v = v[:, None, :]
            if v.ndim != 3 or v.shape[-1] != width or v.dtype.kind not in 'fiu':
                raise ValueError(f"{name} must be a numeric (N, K, {width}) array")
            if shape is not None and v.shape[:2] != shape: raise ValueError("Waveform/feature shapes differ")
            shape = v.shape[:2]
            setattr(self, name, v)
        if shape is None or shape[0] < 1 or not 1 <= shape[1] <= 7:
            raise ValueError("Provide nonempty waveforms or features with 1..7 filtered views")
        n,k = shape
        if self.gaussians is None:
            if k != 1: raise ValueError("Multi-filter data require explicit Gaussian coefficients")
            self.gaussians = np.full((n,1), 3.)
        else:
            g = np.asarray(self.gaussians, dtype=float)
            self.gaussians = np.broadcast_to(g, (n,k)).copy()
        raw = np.full(n,k) if self.lengths is None else np.asarray(self.lengths)
        if raw.shape != (n,) or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all() or np.any((raw<1)|(raw>k)):
            raise ValueError("lengths must contain N integers in 1..K")
        self.lengths = raw.astype(np.int64)
        valid = np.arange(k)[None,:] < self.lengths[:,None]
        if not np.isin(self.gaussians[valid], GAUSSIANS).all(): raise ValueError("Unsupported Gaussian coefficient (not Hz)")
        for i in range(n):
            if np.any(np.diff(self.gaussians[i,:self.lengths[i]]) <= 0):
                raise ValueError("Valid Gaussian coefficients must be unique and ascending")
        for name in ['waveforms','features']:
            v = getattr(self,name)
            if v is not None and not np.isfinite(v[valid]).all(): raise ValueError(f"Nonfinite valid {name}")
        for name in ['labels','stations','sample_ids']:
            v = getattr(self,name)
            if v is None: continue
            v = np.asarray(v)
            if v.shape != (n,): raise ValueError(f"{name} must have shape (N,)")
            if name=='labels' and not np.isin(v,[0,1]).all(): raise ValueError("Labels are bad=0, good=1")
            if name!='labels' and (v.dtype.kind not in 'US' or np.any(v=='')): raise ValueError(f"{name} must contain nonempty strings")
            setattr(self,name,v)

    def __len__(self):
        return len(self.lengths)

    def subset(self, indices):
        return RFData(**{name: (None if getattr(self,name) is None else getattr(self,name)[indices]) for name in
                        ['waveforms','features','gaussians','lengths','stations','labels','sample_ids']})

    def select(self, gaussian):
        if gaussian is None: return self
        mask = (self.gaussians==gaussian)&(np.arange(self.gaussians.shape[1])[None,:]<self.lengths[:,None])
        if not mask.any(1).all(): raise ValueError(f"Every record must contain AG{gaussian:g} for this model")
        idx = mask.argmax(1); rows = np.arange(len(self))
        return RFData(waveforms=None if self.waveforms is None else self.waveforms[rows,idx,None,:],
                      features=None if self.features is None else self.features[rows,idx,None,:],
                      gaussians=[gaussian], stations=self.stations, labels=self.labels, sample_ids=self.sample_ids)

    @classmethod
    def load(cls, path):
        """Load a user-owned NPZ without pickle, or an existing NPY cache directory."""
        path = Path(path)
        names = ['waveforms','features','gaussians','lengths','stations','labels','sample_ids']
        if path.is_dir():
            return cls(**{n:np.load(path/f'{n}.npy',allow_pickle=False,mmap_mode='r') for n in names if (path/f'{n}.npy').exists()})
        with np.load(path,allow_pickle=False) as z:
            keys = names+['start_time','sampling_interval']
            return cls(**{n:z[n].item() if n in keys[-2:] else z[n] for n in keys if n in z})


def synthetic_data(n=8, multiview=True, seed=0):
    """Runtime-generated demonstration signals, never observational RF data."""
    rng=np.random.default_rng(seed);t=np.linspace(-10,40,501)
    g=GAUSSIANS if multiview else np.array([3.])
    wave=np.stack([np.exp(-.5*(t*a)**2)+.2*np.exp(-.5*((t-4)*a/2)**2) for a in g])
    wave=wave[None,:,:]+rng.normal(0,.08,(n,len(g),501))
    return RFData(wave, g, features=rng.normal(size=(n,len(g),6)),
                  stations=np.array([f'synthetic_{i//4}' for i in range(n)]),
                  labels=np.arange(n)%2, sample_ids=np.array([f'synthetic_{i}' for i in range(n)]))
