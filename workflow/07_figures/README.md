# Fidelity figures

Run the source-ARI figure independently from the repository root:

```bash
uv run workflow/07_figures/figure_source_ari.py
```

This writes `figures/data_source_separation_miRNA_RNA.pdf` and `.png` (300 dpi).
Use `--output-dir PATH` to render candidates for review elsewhere. Rendering needs
only the Python environment and stage 05 results, without a learning-curve archive.

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

ARI is signed and retained at its stored precision. Points show arithmetic means
across the configured subsampling draws; rounding is confined to display labels.
Zero is the chance-adjusted agreement baseline. Negative values indicate
below-chance agreement and do not by themselves establish better mixing.
The fidelity summary helper calculates observed minimum–maximum ranges, not
confidence intervals, and rejects missing values in requested metrics.
