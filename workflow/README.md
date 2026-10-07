# Running the workflow

Stages should be run in order by hand. There is no workflow engine. Ideally, each stage directory has exactly one entry point, `run.R` or `run.py`, and that is the only file documented here. 


## Stages

Stages should be run in this order. **Define groups before preprocess:** marker
thresholds (mean / SD on `log2(x+1)`) are computed after samples with missing
group labels have been dropped, so the sample set has to be fixed first.
`grouping=none` rows skip stage 02 and keep every sample.

| # | Stage | Command | Reads | Writes |
|---|---|---|---|---|
| 01 | Download TCGA | `Rscript workflow/01_download_tcga/run.R` | cohort list from `config/` | `data/raw/` (expression + clinical) |
| 02 | Define groups | `uv run workflow/02_define_groups/run.py` | `data/raw/`, `config/datasets.tsv`, `config/group_maps.tsv` | `data/processed/grouped/`, `results/group_selection.csv` |
| 03 | Preprocess | `uv run workflow/03_preprocess/run.py` | `data/raw/`, `data/processed/grouped/`, `config/datasets.tsv`, `config/preprocess.tsv` | `data/processed/filtered/`, `results/marker_filtering.csv` |
| 04 | Generate | `uv run workflow/04_generate/run.py` | `data/processed/`, generation config | `data/generated/` |
| 05 | Fidelity | `uv run workflow/05_fidelity/run.py` | `data/processed/`, `data/generated/`, `config/mirna_coordinates.csv` | `results/fidelity.csv`, `results/umap.csv` |
| 06 | SyntheSize | `uv run workflow/06_synthesize/run.py [--workers N]` | `data/processed/`, `data/generated/` | `results/learning_curve_fits.json`, `results/curves.csv`, `results/curve_fits.csv`, `data/processed/curves/` |
| 07 | Figures and tables | `uv run workflow/07_figures/run.py` | `results/` | `figures/`, `results/tables/` |
| 08 | Website | `uv run workflow/08_website/run.py` | `results/learning_curve_fits.json` and companion CSVs, `config/` | `website/plots/`, `website/manifest.json` |

**Stage 05 also writes the UMAP coordinates.** Signed source ARI, MAD, CCC of means, and UMAP run in Python. DANA `cccpcc` and voom+limma `ccc_log10_pvalue` / `ccc_log2_FC` are computed by `dana_dea.R` (needs `Rscript` on PATH) for binary grouped jobs. `grouping=none` RNA-seq skips DANA/DEA and SyntheSize (stage 06). A job whose `generated.csv` is missing is skipped; rerun after stage 04. RNA-seq `cccpcc` is NA until a gene-coordinate table is added — miRNA uses `config/mirna_coordinates.csv`.

`results/fidelity.csv` stores full-precision signed `source_ari` per draw and the actual
`draw_seed` (currently 101–105); `seed` remains the shared workflow/UMAP seed, 42.
Configuration keys are dataset, modality, normalization, model, off_aug and AE_head_num,
with `draw` identifying each row. Figure means and minimum–maximum ranges are calculated
from unrounded draws; rounding is only for labels. Zero is the chance-adjusted agreement
baseline, and negative values indicate below-chance agreement, not automatically better
mixing. This measure alone does not establish distributional equivalence.

Stage 07 keeps one `figure_*.py` module per figure, each exposing `main()`, registered in the
`FIGURES` mapping in `run.py`. All rendering uses Python. `figure_cvae_architecture.py`
is a schematic of `syng_bts.helper_models.CVAE`, so it reads nothing from `results/`.
The source-ARI, miRNA, and RNA-seq fidelity renderers consume stage 05 tables and
`config/fidelity_figures.tsv`. Build these independently of stage 06 with
`uv run workflow/07_figures/run.py --figures source-ari fidelity-mirna fidelity-rnaseq`.
The [figure guide](07_figures/README.md) documents inputs and individual commands.

The `learning-curve-agreement` figure family builds the manuscript overview,
selected source overlays, configuration heatmaps and comparison table from stage
06. Run `uv run --locked workflow/07_figures/run.py --figures learning-curve-agreement`.
Selections live in `config/learning_curve_figures.tsv`; outputs use descriptive
names under `figures/`, with the derived comparison CSV in `results/tables/` and
summary provenance in `results/learning_curve_agreement_summary.json`. Shared-range
gap summaries use interpolated observed means. Curve displays use the archived
predictions without refitting.

Figure files are named for what they show (`cvae_architecture.pdf`), never for a manuscript
number. The numbering is still changing, and a stale `fig1.pdf` is worse than no name at all.

The stage 07 learning-curve renderer and stage 08 read stage 06's completed
numerical archive. They do not fit curves. Stage 08 replaces all generated plot
JSON, then validates and rebuilds the manifest. See [SyntheSize.md](SyntheSize.md) for the fitting exports, cache
invalidation and the commands to rerun only the learning-curve analysis.


## Entry points

Every stage directory should have a `run.R` or `run.py`, which:

- takes no required arguments; optional arguments control runtime behavior only,
- reads its parameters from `config/`. So no constants in its own body, except the seed;
- declares a fixed `SEED` for every stochastic call,
- takes its cohort list and per-cohort settings from `config/`,
- writes only into `data/`, `results/`, `figures/`, or `website/`.

Each stage can have additional helper scripts in the same directory.
