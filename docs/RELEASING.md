# Maintainer release procedure

The initial distribution is a pip-installable GitHub wheel and tagged source.
Weight assets are separate from both git history and the wheel. No training data
or source-paper PDFs are published. Do not add observational arrays or prediction
CSVs; the synthetic tests generate their inputs at runtime.

1. Run the tests and the optional local archive parity check.
2. Export verified weight bundles with `scripts/export_model_zoo.py --workspace
   /path/to/original/workspace --output /path/outside/repository`. This needs the
   original completed experiments; a normal installation does not.
3. Update the version, static model catalog and notices; `python -m build` and
   `python -m twine check dist/*`.
4. Commit source and verification receipts. Tag the tested commit; upload wheel,
   sdist and model ZIPs to the corresponding GitHub release. Do not silently
   replace a model asset; change the version and catalog hashes.

## Comparison-only release v0.1.1

The new wheel and source distribution add inference timing and a public
comparison snapshot. Existing model catalog entries and all 53 weight bundles
continue to reference the immutable v0.1.0 assets. Do not duplicate, replace or
retrain these weights when releasing v0.1.1. Aggregate comparison CSVs and timing
durations are permitted; observational arrays and individual predictions remain
excluded. Run `scripts/reproduce_comparisons.py` as well as the synthetic tests,
and preserve `benchmarks/2026-10-02/manifest.json` when packaging the snapshot.

## Optional PyPI publication

The repository includes a **manual-only** `.github/workflows/publish-pypi.yml`.
It is not invoked automatically on push/tag. Before running it, the maintainer
must establish ownership of the `rfqc-bench` project on PyPI (or a pending trusted
publisher) and configure GitHub trusted publishing for owner `cangyeone`, repository
`rfqc-bench`, workflow `publish-pypi.yml`, environment `pypi`. Protect the GitHub
environment as appropriate. No token belongs in this repository.

After a successful PyPI upload, verify installation from PyPI in a clean virtual
environment before documenting `pip install rfqc-bench` as available. Until then,
use the tested GitHub wheel or `pip install` from the versioned Git URL. The
software license does not imply that RF data have a public DOI or data license.
