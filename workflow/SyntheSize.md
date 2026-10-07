# SyntheSize fitting and reruns

Stage 06 evaluates classifiers with the existing seed (42), candidate-size grid,
class proportions and draw settings in `config/synthesize.tsv`. It passes the
per-draw metrics to **SyNG-BTS 3.6.1** `fit_sample_sizes` for each source and metric.
The package averages rows by method and candidate subset size, sorts sizes, and
fits `1 - a - b*n**c` with weights `i/m`. Approximate pointwise 95% confidence
intervals use the fitted-parameter covariance and parameter derivatives. These
are intervals for the fitted mean, not prediction intervals or simultaneous bands.

The archived prediction grid is the observed subset sizes, matching the package
plotter. This replaces stage 06's former 100-point grid. A fitted curve can exist
without an interval. Observations remain available when fitting fails. Singleton
draw SDs are unavailable, not zero. Neither fitting nor plotting clips numerical
results to the score range; displayed y-limits come from the plotting config.

## Numerical files

Each completed job writes these files below
`data/processed/curves/<dataset>/<norm>/<offaug_tag>/<model>/`:

- `metrics_generated.csv`: per-draw classifier evaluations.
- `fits.json`: public API `to_dict()` results plus dataset, normalization, model,
  offline-augmentation and source identity. Includes observations, weights,
  predictions, parameters, covariance, `fit_status`, `interval_status`, and the
  package diagnostic message. Non-finite values are JSON `null`.
  The API's equation field `model` is stored as `curve_model` to distinguish it
  from the study's generative-model identifier.
- `points.csv`: observed and fitted rows with `n`, `value`, `observed_std`,
  `ci_low`, `ci_high`, `n_draws`, and `weight`. Unavailable values are blank.
- `fits.csv`: one row per source/method/metric, containing `fit_ok`, both status
  fields, `message`, `a,b,c`, and the nine `cov_aa` through `cov_cc` entries.
- `complete.json`: producing package/API, configuration identity, seed, input
  identities, evaluation status and hashes of the completed output files.

Real metrics are shared at `<dataset>/<norm>/metrics_real.csv`, with their own
completion record `metrics_real.json`. `results/curves.csv` and
`results/curve_fits.csv` concatenate only this run's completed jobs in plan order.
`results/learning_curve_fits.json` contains the same numerical results, producing
provenance, job completion records, job counts and hashes of the two CSV tables.
It is written last. The append-only `results/synthesize_log.csv` is diagnostic;
it never controls cache reuse.

Stages 07 and 08 require this completed archive and matching CSV hashes and
configuration. Stage 07 renders the stored observations/predictions. Stage 08
converts them to the existing Plotly array fields without rounding or refitting,
retaining parameters, covariance and status. Failed fits and missing bands are
labelled; a missing real or generated source does not hide the available source.

## Cache handling

`skip_existing=true` reuses a job only when its completion record matches the
current package, export format, scientific configuration, seed and source inputs,
and every recorded output has the expected hash. Input identity uses the resolved
path, size and nanosecond modification time, not a content hash. Replacing an
input while preserving all those properties requires explicit cache removal.
Changing plotting limits, the default display metric or `skip_existing` does not
invalidate numerical fits. Configuration changes can invalidate multiple jobs.

Incomplete or incompatible job outputs are removed before retrying. Missing
inputs are reported; available sources are retained. Jobs without generated
evaluations are retried on the next invocation. A failed invocation leaves no
completed aggregate archive; consumers stop instead of using old results. With
no available jobs, stage 06 writes empty tables and an empty completed archive.
Run only one stage 06 invocation against a cache at a time; `--workers` manages
parallelism within that invocation.

Stage 08 builds in a temporary directory and replaces all `website/plots/`, then
rebuilds `website/manifest.json`. Old bootstrap configurations are never retained.
`--manifest-only` rejects incompatible plot JSON using the standard library and
can create an empty awaiting-rerun manifest when there are no plot files.

## Manual rerun, after invalidating historical results

Run from the supplement repository. Installing the lockfile uses the published
PyPI package; no sibling source override is needed:

```bash
uv sync --locked
uv run --locked python -c 'import syng_bts; print(syng_bts.__version__)'
```

For a completely fresh SyntheSize run, remove only these derived outputs. Preserve
`data/generated/`, empirical inputs and all fidelity results:

```bash
rm -rf data/processed/curves figures/learning_curves website/plots
rm -f results/curves.csv results/curves.csv.tmp \
  results/curve_fits.csv results/curve_fits.csv.tmp \
  results/learning_curve_fits.json results/learning_curve_fits.json.tmp \
  results/synthesize_log.csv results/synthesize_log.csv.tmp \
  website/manifest.json
uv run --locked workflow/08_website/run.py --manifest-only
```

Then run, in order:

```bash
# Expensive: repeated classifier cross-validation across the configured study.
# Adjust workers to available RAM/CPU; 1 is the default.
uv run --locked workflow/06_synthesize/run.py --workers 1

# Rendering only: no classifier evaluation or fitting.
uv run --locked workflow/07_figures/figure_learning_curves.py
uv run --locked workflow/07_figures/figure_learning_curves.py --metric auc
uv run --locked workflow/07_figures/run.py --figures learning-curve-agreement
uv run --locked workflow/08_website/run.py

# Same dependency-free validation performed by Pages CI.
uv run --locked python -S workflow/08_website/run.py --manifest-only
uv run --locked python -m http.server --directory website 8000
```

The agreement command builds the manuscript overview, selected comparisons and
shared-range gap table from the same archive. Figure selections are recorded in
`config/learning_curve_figures.tsv`; see the [figure guide](07_figures/README.md).
The AUC command renders all configured AUC combinations, including HNSC when its
inputs are available. These commands do not regenerate data or run fidelity
analyses. Stage 06 reports missing generated files; inspect the counts and log
before interpreting coverage. If it fails, fix the reported job and rerun; valid
completed jobs can be reused. Static figures and the website must be regenerated
after a successful stage 06 run. Historical manuscript figures and findings are
not validated by correcting the code. No command above deploys the website.
