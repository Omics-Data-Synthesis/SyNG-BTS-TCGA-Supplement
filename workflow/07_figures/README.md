# Manuscript figures

## Learning-curve agreement

Build the manuscript learning-curve overview, selected comparisons and supporting
tables from the completed stage 06 archive:

```bash
uv run --locked workflow/07_figures/run.py --figures learning-curve-agreement
```

The standalone entry point is
`uv run --locked workflow/07_figures/figure_learning_curve_agreement.py`.
It accepts `--output-dir PATH` for figures and `--results-dir PATH` for derived
tables and provenance. Defaults are `figures/` and `results/`.
The renderer reads only this repository's archive and configuration; it requires
neither patient-level inputs nor another repository. Archive validation checks
scientific configuration and the hashes of `curves.csv` and `curve_fits.csv`.
It does not evaluate classifiers or refit learning curves.

`config/learning_curve_figures.tsv` selects the comparison columns in file order.
Each row specifies dataset, normalization, generative model, offline augmentation
and a display label. The overview includes every archived binary task and
configuration. The selected comparisons use HNSC raw/cVAE-1:50, HNSC
TC/cVAE-1:50 and COAD RNA TC/cVAE-1:100, all without an AE head.

| Output under `figures/` | Content |
|---|---|
| `learning_curve_agreement.pdf`, `.png` | Task-by-classifier signed and absolute AUC gaps, medians across configurations |
| `learning_curve_comparison_auc.pdf`, `.png` | Selected AUC curves with archived fitted-mean intervals |
| `learning_curve_comparison_f1_score.pdf`, `.png` | Selected macro-F1 curves with draw SDs and fitted-mean intervals |
| `learning_curve_comparison_accuracy.pdf`, `.png` | Selected accuracy curves with draw SDs and fitted-mean intervals |
| `learning_curve_agreement_configurations.pdf` | One page per task with all configuration-level AUC gaps |
| `learning_curve_agreement_<dataset_id>.png` | Configuration-page previews for the selected tasks |

PNGs are 300 dpi; PDFs contain vector plots. Blue denotes empirical data and
orange generated data. Dotted lines mark the empirical subset-size limit;
the region to the right contains generated-only evaluations. Lines and bands
come from the archived predictions. Bands are approximate pointwise 95%
intervals for the fitted mean; error bars in macro-F1 and accuracy panels show
±1 SD across draw means. Score axes include all displayed values without clipping.

`results/tables/learning_curve_agreement.csv` retains every configuration,
classifier and metric comparison. The signed and absolute gaps integrate
piecewise-linear observed mean scores over the shared subset-size range,
divided by its width. Sign crossings are accounted for in the absolute integral.
No source is extrapolated, and these descriptive gaps are not equivalence tests
or sample-size recommendation thresholds. Positive signed gaps mean higher
within-generated performance. Empirical series reused across generator
configurations are not independent empirical experiments.

`results/learning_curve_agreement_summary.json` records aggregate summaries,
archive provenance, source CSV hashes and the figure-selection configuration
hash. These derived results follow the repository's existing ignored-results
policy. The original per-configuration renderer remains available through
`--figures learning-curves`.

## Fidelity figures

Build all three fidelity figure families from the repository root:

```bash
uv run workflow/07_figures/run.py --figures source-ari fidelity-mirna fidelity-rnaseq
```

Stage 07 uses Python only. The selected fidelity figures need stage 05 results;
they do not need the stage 06 learning-curve archive. Running stage 07 without
`--figures` builds every registered figure, including learning curves, and requires
all of their inputs.

Run the source-ARI figure independently from the repository root:

```bash
uv run workflow/07_figures/figure_source_ari.py
```

This writes `figures/data_source_separation_miRNA_RNA.pdf` and `.png` (300 dpi).
Use `--output-dir PATH` to render candidates for review elsewhere. Rendering needs
only the Python environment and stage 05 results, without a learning-curve archive.

Render the miRNA fidelity figures for all configured cohorts, or only STAD:

```bash
uv run workflow/07_figures/figure_fidelity_mirna.py
uv run workflow/07_figures/figure_fidelity_mirna.py --dataset STAD_prospective_collection
```

Each cohort produces `<dataset_id>_miRNA_fidelity.pdf` and `.png` in `figures/`.
The miRNA renderer also accepts `--output-dir PATH`. It reads the source tables
once for the cohort batch and draws the saved UMAP coordinates without refitting.

Render the selected RNA-seq fidelity figure:

```bash
uv run workflow/07_figures/figure_fidelity_rnaseq.py
```

This writes `figures/RNA_selected_fidelity_UMAP.pdf` and `.png` and also accepts
`--output-dir PATH`. The summary and UMAP panels use the same `rnaseq` selection
as the source-ARI overview. One-group cohorts appear first, followed by two-group
cohorts; cohorts are alphabetical within each group. Each metric's scale is shared
across the selected cohorts. UMAP coordinates are fitted separately by cohort and
are not comparable across panels. Circles represent one-group samples; circles
and triangles distinguish groups 0 and 1 for two-group cohorts.

## Inputs and selections

Stage 05 writes `results/fidelity.csv` (one row per configuration and subsampling
draw) and `results/umap.csv` (saved two-dimensional coordinates). These files are
local, ignored outputs; the renderer does not download them. Recompute them with
`uv run workflow/05_fidelity/run.py` after preparing the processed and generated
datasets through stages 01–04, as described in [the workflow guide](../README.md).
Stage 05 needs the Python and R environments. An existing complete fidelity
configuration is skipped according to `config/fidelity.tsv`.

`config/fidelity_figures.tsv` selects the normalization, model, and augmentation
for each panel and modality/grouping combination. Cohorts and grouping definitions
come from `config/datasets.tsv`. The `mirna_ari` selection covers all configured
miRNA cohorts; `mirna_umap` specifies the representative miRNA embedding setting;
`rnaseq` specifies the settings shared by the RNA overview and detailed figure.
Zero in `AE_head_num` means no AE head. Full configuration keys are used to match
results, and missing selected configurations are errors.

The miRNA comparison grid takes normalization and model order from
`config/generate_norms.tsv` and `config/generate_models.tsv`. The figure selection
specifies augmentation and the representative UMAP setting. Embeddings must have
the empirical/generated sample counts recorded in the fidelity table and matching
binary-group counts between sources. The coordinate table does not identify AE
heads, so these UMAP figures require configurations without AE-head augmentation.

ARI is signed and retained at its stored precision. Points show arithmetic means
across the configured subsampling draws; rounding is confined to display labels.
Zero is the chance-adjusted agreement baseline. Negative values indicate
below-chance agreement and do not by themselves establish better mixing.
The fidelity summary helper calculates observed minimum–maximum ranges, not
confidence intervals, and rejects missing values in requested metrics.

In the miRNA figure, CCC-PCC compares within-cluster partial correlations;
CCC −log10(p) compares differential-expression significance; CCC mean compares
marker-wise means. Higher CCC indicates closer agreement. MedAE is the median
absolute difference in marker-wise means, SDs, or CVs; lower values indicate
closer agreement. Each metric has one scale shared across the cohort's rows.
CCC scales include negative values when present; MedAE scales depend on the
cohort's observed range. Green segments show the full minimum–maximum range
across subsampling draws.
