# Running the workflow

Stages should be run in order by hand. There is no workflow engine. Ideally, each stage directory has exactly one entry point, `run.R` or `run.py`, and that is the only file documented here. 


## Stages

An suggested example workflow structure could be:

| # | Stage | Command | Reads | Writes |
|---|---|---|---|---|
| 01 | Download TCGA | `Rscript workflow/01_download_tcga/run.R` | cohort list from `config/` | `data/raw/` |
| 02 | Preprocess | `Rscript workflow/02_preprocess/run.R` | `data/raw/`, preprocessing and filtering config | `data/processed/`, `results/marker_filtering.csv` |
| 03 | Define groups | `Rscript workflow/03_define_groups/run.R` | `data/processed/`, group-label config | `data/processed/`, `results/group_selection.csv` |
| 04 | Generate | `uv run workflow/04_generate/run.py` | `data/processed/`, generation config | `data/generated/` |
| 05 | Fidelity | `uv run workflow/05_fidelity/run.py` | `data/processed/`, `data/generated/` | `results/fidelity_*.csv`, `results/umap_*.csv` |
| 06 | SyntheSize | `uv run workflow/06_synthesize/run.py` | `data/processed/`, `data/generated/` | `results/curves_*.csv` |
| 07 | Figures and tables | `uv run workflow/07_figures/run.py` | `results/` | `figures/`, `results/tables/` |

**Stage 05 should also write the UMAP coordinates.**

Stage 07 keeps one `figure_*.py` module per figure, each exposing `main()`, registered in the
`FIGURES` list in `run.py`. `figure_cvae_architecture.py` is the worked example: it is a
schematic of `syng_bts.helper_models.CVAE`, so it reads nothing from `results/`.

Figure files are named for what they show (`cvae_architecture.pdf`), never for a manuscript
number. The numbering is still changing, and a stale `fig1.pdf` is worse than no name at all.


## Entry points

Every stage directory should have a `run.R` or `run.py`, which:

- takes no additional arguments,
- reads its parameters from `config/`. So no constants in its own body, except the seed;
- declares a fixed `SEED` for every stochastic call,
- takes its cohort list and per-cohort settings from `config/`,
- writes only into `data/`, `results/`, or `figures/`.

Each stage can have additional helper scripts in the same directory.