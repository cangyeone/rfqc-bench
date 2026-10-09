"""Check user-owned SAC fixtures against archived predictions; ships no RF data.

Example:
python scripts/check_eqr_parity.py --directory /path/input --model-dir /path/bundle \
  --expected /path/test_predictions.csv --output /path/check

Expected CSV uses station, sample_id (<station>:eqr:<filename>), prediction and
p_good columns. Only events in the fixture are compared; the expected CSV may
contain the full test set. Numerical tolerance permits CPU/CUDA roundoff, but
the retained-file set and decisions must match exactly.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from rfqc_bench import RFQCPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--expected',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tolerance',type=float,default=1e-4)
    args=parser.parse_args()
    if args.tolerance<0:raise ValueError('tolerance must be nonnegative')
    root=args.directory.resolve()
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.eqr')}
    model=RFQCPredictor.from_directory(args.model_dir)
    summary=model.screen_eqr(root,args.output)
    with args.expected.open(newline='') as stream:
        expected={(r['station'],r['sample_id'].split(':eqr:')[-1]):r for r in csv.DictReader(stream)}
    with Path(str(args.output)+'.predictions.csv').open(newline='') as stream:
        predicted=list(csv.DictReader(stream))
    if not predicted:raise ValueError('No valid events were predicted')
    deltas=[];wanted=set()
    for row in predicted:
        original=expected[row['station'],row['event']]
        if int(row['prediction'])!=int(original['prediction']):raise ValueError('Archived decision mismatch')
        deltas.append(abs(float(row['p_good'])-float(original['p_good'])))
        if int(row['prediction']):wanted.update(json.loads(row['files']))
    if max(deltas)>args.tolerance:raise ValueError('Score difference exceeds tolerance')
    retained=args.output.read_text(encoding='utf-8').splitlines()
    if set(retained)!=wanted or len(retained)!=len(wanted):raise ValueError('Record list mismatch')
    if not all((root/p).is_file() for p in retained):raise ValueError('Retained path missing')
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in before.items()):
        raise ValueError('Input file changed')
    report=dict(scope='Interface parity, not a new accuracy experiment',events=len(predicted),files=len(before),
                good_events=summary['good_events'],retained_files=len(retained),
                max_probability_difference=max(deltas),same_decisions=True,source_files_unchanged=True,
                record_matches_predictions=True,weights_sha256=summary['model_weights_sha256'])
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
