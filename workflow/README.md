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
| 06 | SyntheSize | `uv run workflow/06_synthesize/run.py [--workers N]` | `data/processed/`, `data/generated/` | `results/curves.csv`, `results/curve_fits.csv`, `data/processed/curves/` |
| 07 | Figures and tables | `uv run workflow/07_figures/run.py` | `results/` | `figures/`, `results/tables/` |
| 08 | Website | `uv run workflow/08_website/run.py` | `data/processed/curves/`, `config/datasets.tsv` | `website/plots/`, `website/manifest.json` |

**Stage 05 should also write the UMAP coordinates.** cARI, MAD, CCC of means, and UMAP run in Python. DANA `cccpcc` and voom+limma `ccc_log10_pvalue` / `ccc_log2_FC` are computed by `dana_dea.R` (needs `Rscript` on PATH) for binary grouped jobs. `grouping=none` RNA-seq skips DANA/DEA and SyntheSize (stage 06). A job whose `generated.csv` is missing is skipped; rerun after stage 04. RNA-seq `cccpcc` is NA until a gene-coordinate table is added — miRNA uses `config/mirna_coordinates.csv`.

Stage 07 keeps one `figure_*.py` module per figure, each exposing `main()`, registered in the
`FIGURES` list in `run.py`. `figure_cvae_architecture.py` is the worked example: it is a
schematic of `syng_bts.helper_models.CVAE`, so it reads nothing from `results/`.

Figure files are named for what they show (`cvae_architecture.pdf`), never for a manuscript
number. The numbering is still changing, and a stale `fig1.pdf` is worse than no name at all.

Stage 08 rebuilds the interactive site from the per-draw `metrics_*.csv` cache that
stage 06 writes under `data/processed/curves/`. It does **not** use `results/curves.csv`
or the per-job `points.csv` / `fits.csv` (those are an unweighted dense-grid fit).
The default run is incremental so bootstrap JSON for cohorts that have not been
re-run is kept. Preview with `python3 -m http.server --directory website 8000`.


## Entry points

Every stage directory should have a `run.R` or `run.py`, which:

- takes no required arguments; optional arguments control runtime behavior only,
- reads its parameters from `config/`. So no constants in its own body, except the seed;
- declares a fixed `SEED` for every stochastic call,
- takes its cohort list and per-cohort settings from `config/`,
- writes only into `data/`, `results/`, `figures/`, or `website/`.

Each stage can have additional helper scripts in the same directory.
