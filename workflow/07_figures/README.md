# Manuscript figures and tables

## Manuscript table CSVs

Export six manuscript tables and the runtime source-data CSV from existing results:

```bash
uv run --locked workflow/07_figures/run.py --tables
```

This table-only command writes to the top-level `tables/` directory beside
`figures/`. It does not render plots, load fidelity results, refit curves or rerun
upstream stages. The standalone command is
`uv run --locked workflow/07_figures/table_manuscript.py`; it accepts
`--output-dir PATH` for exports elsewhere. Running stage 07 without selection
flags builds all registered figures and then all manuscript tables. Use
`--figures NAME --tables` to build named figures and the tables together.

| CSV under `tables/` | Manuscript item | Inputs |
|---|---|---|
| `benchmark_label_recoding.csv` | S1: label recoding and group counts | `results/group_selection.csv`, `config/datasets.tsv` |
| `marker_filtering.csv` | S2: marker filtering and sample counts | `results/marker_filtering.csv` |
| `training_runtime_mirna.csv` | Figure S1 source data: miRNA training epochs and runtimes | `results/training_time.csv` |
| `learning_curve_agreement_summary.csv` | S3: aggregate agreement summaries | `results/learning_curve_fits.json` |
| `learning_curve_selected_configurations.csv` | S4: configurations selected for Figure 6 | Same archive, `config/datasets.tsv`, `config/generate_off_aug.tsv` |
| `learning_curve_selected_agreement.csv` | S5: comparisons at those configurations | Same archive |
| `learning_curve_agreement.csv` | S6: complete comparison grid | Same archive |

`table_manuscript.py` selects columns and reshapes the existing results. Runtime
export uses successful miRNA records without offline augmentation and pivots the
three loss ratios into paired epoch/time columns. Runtime logging is unchanged.
Numeric values retain the precision available in the source results; round only for typesetting
(three decimals for filtering cutoffs and agreement gaps in the current manuscript).
One-group datasets have blank binary-group counts in the recoding CSV.

Both exporters check candidate-size and draw counts against `config/synthesize.tsv`
before writing outputs. Table export also requires exactly one empirical and one
generated series for every classifier and metric in each archived configuration;
incomplete or duplicate series are errors.

## Training epochs and runtime

Render a two-panel heatmap of the complete miRNA runtime table:

```bash
uv run --locked workflow/07_figures/run.py --figures training-runtime
```

`figure_training_runtime.py` reads original `results/training_time.csv` records and writes
`figures/training_runtime_mirna.pdf` and `.png`. Its standalone entry point accepts
`--source PATH` and `--output-dir PATH`. It does not regenerate the CSV or rerun
training. It shares successful-miRNA/no-offline-augmentation selection and reshaping
with `table_manuscript.training_runtime`. Duplicate successful records are rejected.
The exported `tables/training_runtime_mirna.csv` remains available as unnumbered
source data for exact numerical lookup; it is not a prerequisite for rendering.

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
overview selects one configuration per archived binary task by the lowest
mean absolute AUC gap across all five classifiers. Both signed and absolute
panels use that same configuration. Selection uses unrounded values; exact ties
resolve lexicographically by the full configuration key. Incomplete classifier
coverage is an error. This is retrospective selection within the evaluated grid,
not a prospective configuration-selection validation. The median across all
configurations is retained as Supplementary Figure S2. Selected comparison columns
are BRCA miRNA TC/cVAE-1:5, HNSC miRNA raw/cVAE-1:50, HNSC miRNA TC/cVAE-1:50
(all without an AE head), and COAD RNA TC/cVAE-1:50 with two AE-head iterations.
Each column reports mean source ARI from its exact configuration in
`results/fidelity.csv`, averaging the five unrounded subsampling draws. Missing
configurations, incorrect AE-head counts, or incomplete draws are errors.

Both agreement overviews and the configuration-level PDF pages use the same task
order: miRNA first, then RNA, with increasing empirical cohort size within each
modality and dataset identifier breaking ties. Compact `n=…` labels in overview
rows and configuration-page titles give the total empirical cohort size after
grouping and exclusions, before cross-validation splitting. Sizes come from the
maximum observed subset size of each task's archived empirical series; all such
series must agree. They are not capped at the shared-support upper limit or at
the generated sample count. Configuration selection and numerical summaries
are independent of this display order.
Agreement row labels use fixed text columns so their middle-dot separators align
vertically. Figure 6 omits AE-head status from the labels: all selected miRNA
configurations omit the head, while both selected RNA configurations use two
AE-head iterations, as stated in the caption. Configuration-level pages use a
dagger (†) after the loss ratio to distinguish AE-head configurations; unmarked
rows omit the head.

| Output under `figures/` | Content |
|---|---|
| `learning_curve_agreement.pdf`, `.png` | Figure 6: task-by-classifier signed and absolute AUC gaps at one selected configuration per task |
| `learning_curve_agreement_median.pdf`, `.png` | Supplementary Figure S2: medians across all configurations |
| `learning_curve_comparison_auc.pdf`, `.png` | Selected AUC curves over shared support, with draw SDs, fitted-mean intervals, B/D and source ARI |
| `learning_curve_comparison_f1_score.pdf`, `.png` | Matching shared-support macro-F1 curves |
| `learning_curve_comparison_accuracy.pdf`, `.png` | Matching shared-support accuracy curves |
| `learning_curve_comparison_full_support.pdf` | Three pages: AUC, macro-F1 and accuracy over full source ranges |
| `learning_curve_comparison_<metric>_full_support.png` | PNG counterparts of the full-support PDF pages |
| `learning_curve_agreement_configurations.pdf` | One page per task with all configuration-level AUC gaps |
| `learning_curve_agreement_<dataset_id>.png` | Configuration-page previews for the selected tasks |

PNGs are 300 dpi; PDFs contain vector plots. Blue denotes empirical data and
orange generated data. Each comparison subplot has separate colored B and D
labels: B uses the overview's blue–white–red signed-gap scale, and D uses its
yellow–orange–red absolute-gap scale. A compact horizontal legend band places
the B colorbar on the left, empirical/generated source labels in the center,
and the D colorbar on the right, with sparse ticks and metric units appended to
each colorbar title on the same line. Column subtitles combine normalization,
model and AE-head status on one line, with mean source ARI below. The two-iteration
AE-head setting is specified in the captions and configuration table.
Both AUC overviews, all configuration-level heatmaps,
and all comparison metrics and support views use the same numerical color limits.
Limits use the full configuration–classifier grid across AUC, macro-F1 and
accuracy, rounded upward to the next 0.05: B is symmetric around zero and D
starts at zero. For the current archive these limits are −0.45 to +0.45 for B
and 0 to 0.45 for D. Backgrounds use opaque colormap colors so equal values have
equal colors. Figure titles are descriptive; manuscript numbering stays in the
captions and figure manifest.
Typography is sized for placement at 180 mm width: essential labels and values
are approximately 6–6.5 pt after scaling. Heatmap colorbar ticks explicitly include
the shared limits. Comparison footers retain a short visual key; detailed
definitions are given in the manuscript captions.
Shared-support plots use each task's intersection of
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
the source archive path, completed-job counts, selected fidelity means, and
the Figure 6 selection criterion, configurations and selected-configuration
metric summaries. `results/tables/learning_curve_selected_configurations.csv`
records all 26 Figure 6 choices and their mean and maximum absolute AUC gaps.
`results/tables/learning_curve_selected_agreement.csv` retains all three metrics
for those same AUC-selected configurations (390 rows); other metrics are not
used to reselect configurations. The all-grid summaries remain unchanged.
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
`--figures` or `--tables` builds every registered figure, including learning curves,
and all manuscript tables, and requires all of their inputs.

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
