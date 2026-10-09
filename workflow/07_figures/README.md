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
tables and summaries. Defaults are `figures/` and `results/`.
The renderer reads this repository's stage 06 archive, stage 05 fidelity table
and configurations; it requires
neither patient-level inputs nor another repository. It reads
`results/learning_curve_fits.json` directly and checks selected curve coverage
and candidate-size/draw completeness.
It does not evaluate classifiers or refit learning curves.

`config/learning_curve_figures.tsv` selects the comparison columns in file order.
Each row specifies dataset, modality, normalization, generative model, offline
augmentation, AE-head iteration count (0 for none), and a display label. The
overview includes every archived binary task and configuration. Selected columns
are BRCA miRNA TC/cVAE-1:5, HNSC miRNA raw/cVAE-1:50, HNSC miRNA TC/cVAE-1:50
(all without an AE head), and COAD RNA TC/cVAE-1:50 with two AE-head iterations.
Each column reports mean source ARI from its exact configuration in
`results/fidelity.csv`, averaging the five unrounded subsampling draws. Missing
configurations, incorrect AE-head counts, or incomplete draws are errors.

| Output under `figures/` | Content |
|---|---|
| `learning_curve_agreement.pdf`, `.png` | Task-by-classifier signed and absolute AUC gaps, medians across configurations |
| `learning_curve_comparison_auc.pdf`, `.png` | Selected AUC curves over shared support, with draw SDs, fitted-mean intervals, B/D and source ARI |
| `learning_curve_comparison_f1_score.pdf`, `.png` | Matching shared-support macro-F1 curves |
| `learning_curve_comparison_accuracy.pdf`, `.png` | Matching shared-support accuracy curves |
| `learning_curve_comparison_full_support.pdf` | Three pages: AUC, macro-F1 and accuracy over full source ranges |
| `learning_curve_comparison_<metric>_full_support.png` | PNG counterparts of the full-support PDF pages |
| `learning_curve_agreement_configurations.pdf` | One page per task with all configuration-level AUC gaps |
| `learning_curve_agreement_<dataset_id>.png` | Configuration-page previews for the selected tasks |

PNGs are 300 dpi; PDFs contain vector plots. Blue denotes empirical data and
orange generated data. Shared-support plots use each task's intersection of
observed source ranges: BRCA 31–1,000, HNSC 19–569 and COAD 14–467. Full-support
plots retain all observations; pale blue and pale orange mark empirical-only and
generated-only support, respectively, and a dotted boundary marks the upper
shared limit. Lines and bands connect archived predictions without refitting or new
curve evaluations. Bands are approximate pointwise 95% intervals for the fitted
mean; error bars in every metric show +-1 SD across draw means. Score axes preserve
displayed values and intervals outside [0,1]. Generator fits use their full
archived range even when their display is restricted to shared support.

`results/tables/learning_curve_agreement.csv` retains every configuration,
classifier and metric comparison. The signed and absolute gaps integrate
piecewise-linear observed mean scores over the shared subset-size range,
divided by its width. Sign crossings are accounted for in the absolute integral.
No source is extrapolated, and these descriptive gaps are not equivalence tests
or sample-size recommendation thresholds. Positive signed gaps mean higher
within-generated performance. Empirical series reused across generator
configurations are not independent empirical experiments.

`results/learning_curve_agreement_summary.json` records aggregate summaries,
the source archive path, completed-job counts and selected fidelity means.
`results/tables/selected_learning_curve_fidelity.csv` records the exact
selected configurations, draw counts, mean source ARI and mean DE-p-value CCC.
These derived results follow the repository's existing ignored-results
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
