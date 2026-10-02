"""Maintainer: export aggregate paper results, never observational records.

Every source is explicitly allowlisted. The public manifest distinguishes
unaltered aggregate files from field-filtered timing records and source scripts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'original_runs.csv': 'SourceData/benchmark_v1/run_metrics.csv',
    'original_summary.csv': 'SourceData/benchmark_v1/summary.csv',
    'logistic_runs.csv': 'SourceData/revision_20260930/logistic_metrics.csv',
    'label_agreement.csv': 'SourceData/revision_20260930/label_agreement_summary.csv',
    'station_bootstrap.csv': 'SourceData/revision_20260930/station_bootstrap.csv',
    'eight_seed_metrics.csv': 'SourceData/resolution_20260930/eight_seed_metrics.csv',
    'eight_seed_differences.csv': 'SourceData/resolution_20260930/eight_seed_differences.csv',
    'eight_seed_summary.csv': 'SourceData/resolution_20260930/eight_seed_summary.csv',
    'cross_protocol_metrics.csv': 'SourceData/resolution_20260930/cross_protocol_metrics.csv',
    'cross_protocol_counts.csv': 'SourceData/resolution_20260930/cross_protocol_contingencies.csv',
    'input_perturbation.csv': 'SourceData/resolution_20260930/input_perturbation.csv',
    'label_perturbation.csv': 'SourceData/resolution_20260930/label_perturbation.csv',
    'physical_summary.csv': 'SourceData/physical_validation_v1/results/summary.csv',
    'subsample_summary.csv': 'SourceData/physical_validation_v1/results/subsample_summary.csv',
    'frequency_control.csv': 'SourceData/physical_validation_v1/results/frequency_control/classification.csv',
    'inference_summary.csv': 'SourceData/inference_20261002/summary.csv',
    'accuracy_cost.csv': 'SourceData/inference_20261002/accuracy_cost.csv',
}
PAPER_SCRIPTS = [
    'analyze_benchmark.py', 'revision_analysis_20260930.py', 'analyze_resolution_20260930.py',
    'render_resolution_results_20260930.py', 'verify_completed_resolution_20261002.py',
    'summarize_completed_controls_20261002.py', 'analyze_inference_20261002.py', 'analyze_results.py',
]
TIMING_FIELDS = {'method','round','seed','kind','device','parameters','records','views',
    'sample_ids_sha256','bundle_sha256','source_model_sha256','protocol_sha256','script_sha256',
    'torch','numpy','cpu_model','latency_seconds','latency_p50_ms','latency_p95_ms',
    'batch32_seconds','batch32_records_per_second','pipeline_peak_gpu_allocated_bytes',
    'forward_batch32_seconds','forward_batch32_records_per_second','finite_replay',
    'archive_max_abs_score_difference','archive_decisions_changed','max_rss_kib',
    'pool_seconds','pool_records_per_second','mean_ms_per_record','warmup_calls',
    'predicted_good','latency_scope'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'benchmarks/2026-10-02')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def copy(name, source):
        target = a.output/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.paper/source, target)
        sources[name] = dict(source=source, source_sha256=digest(a.paper/source), public_sha256=digest(target), projection='unchanged')

    for target, source in FILES.items():
        copy('results/'+target, source)
    for script in PAPER_SCRIPTS:
        copy('reproduction/paper_scripts/'+script, 'scripts/'+script)
    for script in ('analyze_physical.py','classification_metrics.py','plot_physical.py','summarize_physical.py','verify_physical.py',
                   'analyze_frequency_controls.py','prepare_frequency_controls.py','run_frequency_controls.py'):
        copy('reproduction/physical_scripts/'+script, 'SourceData/physical_validation_v1/scripts/'+script)
    copy('reproduction/original_timing_runner.py', 'SourceData/inference_20261002/benchmark_inference.py')
    completion = json.loads((a.paper/'SourceData/inference_20261002/completion.json').read_text())
    for source in sorted((a.paper/'SourceData/inference_20261002/results').glob('*.json')):
        assert digest(source) == completion['sha256']['results/'+source.name]
        original = json.loads(source.read_text())
        target = a.output/'timings'/source.name; target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({k:v for k,v in original.items() if k in TIMING_FIELDS}, indent=2)+'\n')
        name = str(target.relative_to(a.output))
        sources[name] = dict(source=str(source.relative_to(a.paper)), source_sha256=digest(source),
                            public_sha256=digest(target), projection='timing fields only; host paths, GPU UUIDs and raw IDs omitted')
    protocol = json.loads((a.paper/'SourceData/inference_20261002/protocol.json').read_text())
    protocol.pop('cache'); protocol.pop('gpu_uuid')
    for item in protocol['methods']:
        item.pop('source_directory')
    protocol['source_protocol_sha256'] = completion['protocol_sha256']
    protocol['source_script_sha256'] = completion['script_sha256']
    protocol['publication_note'] = 'Public projection; sample IDs, filesystem paths and GPU UUIDs omitted. Not the original byte-identical protocol.'
    path = a.output/'protocol.json'; path.write_text(json.dumps(protocol, indent=2)+'\n')
    sources['protocol.json'] = dict(source='SourceData/inference_20261002/protocol.json', source_sha256=completion['protocol_sha256'],
                                   public_sha256=digest(path), projection='deployment paths and GPU UUID omitted')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=a.paper, text=True).strip()
    manifest = dict(schema=1, paper_commit=commit, title='Receiver-function quality control on a common task: a benchmark with published baselines and waveform diagnostics',
                    package_used_for_recorded_timing='rfqc-bench 0.1.0',
                    policy='No observational arrays, manual label vectors, sample/station IDs or per-record prediction vectors; aggregates and raw call durations only.',
                    files=sources)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(exported_files=len(sources), paper_commit=commit)))


if __name__ == '__main__':main()
