# Fidelity figures

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
