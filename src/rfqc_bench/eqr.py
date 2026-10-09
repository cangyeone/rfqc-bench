"""Screen user-owned SAC EQR directories without changing the source files."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .data import GAUSSIANS, RFData
from .io import digest


def read_eqr(path):
    """Read a regular, binary SAC v6/v7 RF; crop observed [-10, 40] s at 0.1 s.

    Time zero must already be the direct P arrival. No resampling, padding,
    deconvolution, moveout correction or amplitude scaling is performed here.
    Returns (waveform, metadata). Big- and little-endian SAC are supported.
    """
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(632)
        if len(header) != 632:
            raise ValueError('Truncated SAC header (expected 632 bytes)')
        orders = [o for o in ('<', '>') if np.frombuffer(header, o+'i4', 40, 280)[6] in (6, 7)]
        if len(orders) != 1:
            raise ValueError('Expected binary SAC version 6 or 7')
        order = orders[0]
        floats = np.frombuffer(header, order+'f4', 70)
        ints = np.frombuffer(header, order+'i4', 40, 280)
        n, version = int(ints[9]), int(ints[6])
        if ints[15] != 1 or ints[35] != 1:
            raise ValueError('Expected an evenly sampled SAC time series (IFTYPE=ITIME, LEVEN=1)')
        size = 632 + 4*n + (176 if version == 7 else 0)
        if n < 1 or n > 10_000_000 or os.fstat(stream.fileno()).st_size != size:
            raise ValueError('Invalid SAC sample count or file length')
        raw = stream.read(4*n)
        wave = np.frombuffer(raw, order+'f4').astype(np.float32)
        dt, begin, end = map(float, floats[[0, 5, 6]])
        footer = stream.read(176) if version == 7 else b''
        if version == 7:
            dt, begin, end = map(float, np.frombuffer(footer, order+'f8')[:3])
    if not all(np.isfinite([dt, begin, end])) or begin == -12345 or end == -12345:
        raise ValueError('Missing or nonfinite SAC time grid')
    if not np.isclose(dt, .1, atol=1e-7, rtol=0):
        raise ValueError('Expected delta=0.1 s; adapt other sampling intervals explicitly')
    if not np.isclose(begin+(n-1)*dt, end, atol=max(1e-4, abs(end)*2e-7), rtol=0):
        raise ValueError('SAC b/e/npts/delta are inconsistent')
    start = round((-10.-begin)/dt)
    if start < 0 or start+501 > n or not np.isclose(begin+start*dt, -10., atol=1e-5, rtol=0):
        raise ValueError('RF must cover the observed [-10, 40] s grid relative to P=0')
    if not np.isfinite(wave).all():
        raise ValueError('Nonfinite SAC waveform')
    wave = wave[start:start+501].copy()
    if not np.any(wave):
        raise ValueError('All-zero RF in the model window')

    def number(index):
        value = float(floats[index])
        return value if np.isfinite(value) and value != -12345 else None

    def string(index):
        value = header[440+8*index:448+8*index].decode('ascii', errors='replace').strip(' \x00')
        return '' if value == '-12345' else value

    return wave, dict(station=string(0), network=string(21), component=string(20),
                      baz=number(52), gcarc=number(53), user4=number(44),
                      sha256=hashlib.sha256(header+raw+footer).hexdigest())


def _location(path, root, gaussian):
    # Include ancestors when the supplied root is itself e.g. DB_EW27/AG3.
    for folder in path.parents:
        match = re.fullmatch(r'AG(\d+(?:\.\d+)?)', folder.name, flags=re.IGNORECASE)
        if match:
            value = float(match[1])
            if value not in GAUSSIANS:
                raise ValueError(f'Unsupported Gaussian coefficient AG{value:g}')
            if gaussian is not None and gaussian != value:
                raise ValueError('--gaussian conflicts with the AG directory name')
            return folder.parent, value, True
    if gaussian is None:
        raise ValueError('No AG directory: specify the known Gaussian coefficient with --gaussian')
    # A flat station directory may contain arbitrary good/bad subdirectories.
    # SAC station headers distinguish stations in a shared flat input directory.
    return root, float(gaussian), False


def screen_eqr(directory, output=None, *, predictor=None, model='reference_multifilter',
               seed=None, model_dir=None, device='cpu', cache_dir=None, gaussian=None,
               batch_size=32, overwrite=False, max_files=None):
    """Write a UTF-8 ``record`` list of retained relative EQR paths.

    AG folders identify filter views; identical basenames within one station
    anchor identify an event. Missing views are allowed for multi-filter models.
    A single-filter model lists only its required view. Folder names such as
    ``bad`` are never treated as labels. Invalid/unused files are recorded in a
    sidecar, not silently accepted. Original files are never moved or rewritten.

    ``predictor`` accepts a preloaded RFQCPredictor. Otherwise use a registered
    model or a local model_dir. The returned JSON-compatible summary also lives
    at <output>.json. Existing outputs are protected unless overwrite=True.
    FCM is evaluated with the complete valid input pool for each station.
    """
    from .predictor import RFQCPredictor

    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('directory must be an existing directory')
    if gaussian is not None and gaussian not in GAUSSIANS:
        raise ValueError('Unsupported Gaussian coefficient (not Hz)')
    if batch_size < 1 or (max_files is not None and max_files < 1):
        raise ValueError('batch_size and max_files must be positive')
    target = Path(output).expanduser().absolute() if output is not None else root/'record'
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('Output paths must not contain symlinks')
    if target.suffix.lower() == '.eqr':
        raise ValueError('The record output must not overwrite an EQR file')
    destinations = dict(record=target, predictions=Path(str(target)+'.predictions.csv'),
                        rejected=Path(str(target)+'.rejected.csv'), summary=Path(str(target)+'.json'))
    for path in destinations.values():
        if path.is_symlink() or (path.exists() and (not overwrite or not path.is_file())):
            raise FileExistsError(f'Output exists or is not a regular file: {path}; choose a new output or use --overwrite')
    if predictor is None:
        predictor = (RFQCPredictor.from_directory(model_dir, device) if model_dir is not None else
                     RFQCPredictor.from_pretrained(model, seed, device, cache_dir))
    elif model_dir is not None:
        raise ValueError('Pass predictor or model_dir, not both')
    if predictor.spec.mode != 'waveform':
        raise ValueError('Directory screening supports waveform models only; descriptor/combined models need supplied descriptors')

    # Store only the index in RAM. Load and evaluate one station at a time.
    pools = defaultdict(lambda: defaultdict(list))
    rejected = []
    count = 0

    def reject(path, reason):
        rejected.append((path.relative_to(root).as_posix(), reason))

    def walk_error(error):
        raise error

    for folder, subdirs, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        subdirs.sort()
        for subdir in list(subdirs):
            path = Path(folder)/subdir
            if path.is_symlink():
                reject(path, 'Skipped symlink directory; contents not scanned')
                subdirs.remove(subdir)
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() != '.eqr':
                continue
            path = Path(folder)/filename
            count += 1
            if max_files is not None and count > max_files:
                raise ValueError(f'Directory exceeds the configured limit of {max_files} EQR files')
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Symlink or non-regular EQR file is not read')
                if '\n' in str(path) or '\r' in str(path):
                    raise ValueError('A record path cannot contain a newline')
                anchor, view, has_ag = _location(path, root, gaussian)
                # Header-only station discovery for layouts without AG folders.
                # Full parsing below still verifies the header and the waveform.
                station = anchor.name
                if not has_ag:
                    with path.open('rb') as stream:
                        stream.seek(440)
                        name = stream.read(8).decode('ascii', errors='replace').strip(' \x00')
                        stream.seek(608)
                        network = stream.read(8).decode('ascii', errors='replace').strip(' \x00')
                    if name and name != '-12345':
                        station = (network+'.' if network and network != '-12345' else '')+name
                    elif predictor.spec.kind == 'fcm':
                        raise ValueError('Flat FCM input requires SAC station names, or use a station/AG directory')
                pools[(str(anchor), station)][filename].append((view, path))
            except (ValueError, OSError) as exc:
                reject(path, str(exc))
    if not count:
        raise ValueError(f'No .eqr files found in {root}')

    target.parent.mkdir(parents=True, exist_ok=True)
    totals = dict(files_found=count, events_screened=0, good_events=0, bad_events=0,
                  retained_files=0, stations_screened=0)
    source_hash = hashlib.sha256()
    started = datetime.now(timezone.utc).isoformat()
    # Inference failures leave previous outputs intact. Publish the plain record
    # last, so it is never mistaken for a successful partial run.
    with tempfile.TemporaryDirectory(prefix='.rfqc-screen-', dir=target.parent) as temporary:
        staging = {key: Path(temporary)/key for key in destinations}
        with staging['record'].open('w', encoding='utf-8', newline='\n') as records, \
             staging['predictions'].open('w', encoding='utf-8', newline='') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['sample_id','station','event','gaussians','p_good','prediction',
                             'threshold','method','seed','files','source_sha256'])
            for (anchor, station), events in sorted(pools.items()):
                samples = []
                for event, files in sorted(events.items()):
                    views = [g for g, _ in files]
                    if len(set(views)) != len(views):
                        for _, path in files:
                            reject(path, 'Ambiguous duplicate event/filter view; entire event skipped')
                        continue
                    good = []
                    for view, path in sorted(files):
                        if predictor.spec.gaussian is not None and view != predictor.spec.gaussian:
                            reject(path, f'Unused by single-filter model requiring AG{predictor.spec.gaussian:g}')
                            continue
                        try:
                            wave, metadata = read_eqr(path)
                            good.append((view, path, wave, metadata))
                        except (ValueError, OSError) as exc:
                            reject(path, str(exc))
                    if not good:
                        continue
                    # Known station/event geometry must agree across views.
                    conflict = False
                    for key, tolerance in [('baz', .01), ('gcarc', .001), ('user4', 1e-6)]:
                        values = [entry[3][key] for entry in good if entry[3][key] is not None]
                        if values and max(values)-min(values) > tolerance:
                            conflict = True
                    names = {entry[3]['station'] for entry in good if entry[3]['station']}
                    if conflict or len(names) > 1:
                        for _, path, _, _ in good:
                            reject(path, 'Conflicting station/event headers across filter views; event skipped')
                        continue
                    samples.append((event, good))
                if not samples:
                    continue
                n, k = len(samples), max(len(views) for _, views in samples)
                waveforms = np.zeros((n, k, 501), dtype=np.float32)
                gaussians = np.zeros((n, k), dtype=np.float32)
                lengths = np.array([len(views) for _, views in samples])
                for i, (_, views) in enumerate(samples):
                    for j, (view, _, wave, _) in enumerate(views):
                        waveforms[i,j] = wave
                        gaussians[i,j] = view
                data = RFData(waveforms, gaussians, lengths, stations=np.repeat(station,n))
                prediction = predictor.predict(data, batch_size=batch_size)
                totals['stations_screened'] += 1
                for i, (event, views) in enumerate(samples):
                    paths = [path.relative_to(root).as_posix() for _, path, _, _ in views]
                    hashes = {path.relative_to(root).as_posix(): metadata['sha256'] for _, path, _, metadata in views}
                    sample_id = json.dumps([os.path.relpath(anchor, root), station, event], ensure_ascii=False)
                    decision = int(prediction.prediction[i])
                    writer.writerow([sample_id, station, event, ';'.join(f'{g:g}' for g, *_ in views),
                                     float(prediction.p_good[i]), decision, predictor.threshold,
                                     predictor.spec.name, predictor.seed, json.dumps(paths, ensure_ascii=False),
                                     json.dumps(hashes, ensure_ascii=False, sort_keys=True)])
                    for path in paths:
                        source_hash.update(json.dumps([path, hashes[path]], ensure_ascii=False).encode('utf-8'))
                    totals['events_screened'] += 1
                    totals['good_events' if decision else 'bad_events'] += 1
                    if decision:
                        for path in paths:
                            records.write(path+'\n')
                        totals['retained_files'] += len(paths)
        with staging['rejected'].open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['path','reason'])
            writer.writerows(rejected)
        bundle = getattr(predictor, 'bundle', {})
        summary = dict(schema=1, status='complete' if totals['events_screened'] else 'no_valid_events',
                       started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                       directory=str(root), outputs={key:str(path) for key,path in destinations.items()},
                       model=predictor.spec.name, seed=predictor.seed, threshold=predictor.threshold,
                       device=str(predictor.device), model_weights_sha256=bundle.get('weights_sha256'),
                       model_bundle_sha256=hashlib.sha256(json.dumps(bundle, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                       **totals, rejected_or_unused_entries=len(rejected),
                       source_manifest_sha256=source_hash.hexdigest(),
                       output_sha256={key:digest(staging[key]) for key in ['record','predictions','rejected']},
                       record_format='UTF-8; one retained EQR path per line, relative to directory; no header',
                       input_contract='P-referenced radial RF; observed -10..40 s at 0.1 s; Gaussian coefficients are not Hz',
                       selection='Joint event decision lists only the valid views used by this model; folder names are not labels')
        staging['summary'].write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        # mkdir is exclusive on Windows/POSIX, including filesystems without
        # hard links (e.g. external exFAT disks). Recheck all targets while
        # holding the publication lock; then atomically rename each file.
        lock = target.parent/('.'+target.name+'.rfqc-lock')
        try:
            lock.mkdir()
        except FileExistsError:
            raise FileExistsError(f'Publication lock exists: {lock}; another call may be publishing. '
                                  'If a previous process crashed, inspect outputs before removing this empty lock directory.') from None
        try:
            for path in destinations.values():
                if path.is_symlink() or (path.exists() and (not overwrite or not path.is_file())):
                    raise FileExistsError(f'Output appeared during inference: {path}')
            for key in ['predictions', 'rejected', 'summary', 'record']:
                os.replace(staging[key], destinations[key])
        finally:
            lock.rmdir()
    return summary
