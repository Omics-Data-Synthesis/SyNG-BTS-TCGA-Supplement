"""Stage 07: build the manuscript figures and tables.

Each figure is its own ``figure_*.py`` module in this directory, exposing ``main()``
and writing descriptively named files into ``figures/``. Register a new module in
FIGURES below and it gets built with the rest.

Figure files are named for what they show, not for their manuscript number, because
the numbering is still changing.

Run from the repository root:

    uv run workflow/07_figures/run.py
"""

import figure_cvae_architecture
import figure_learning_curves

# (label, module) pairs, built in this order.
FIGURES = [
    ("cVAE architecture", figure_cvae_architecture),
    ("SyntheSize learning curves", figure_learning_curves),
    # TODO: add the remaining data-driven figures here
]

# TODO: add the remaining data-driven figures as their source data lands in results/:
#   cARI across the 24 miRNA cohorts        <- results/fidelity.csv
#   quality evaluation for STAD             <- results/fidelity.csv, results/umap.csv
#   HNSC learning curves                   <- results/curves.csv (empty until HNSC is generated)
#   cARI across the 5 RNA-seq cohorts       <- results/fidelity.csv
#   RNA-seq UMAP and MAD panels             <- results/fidelity.csv, results/umap.csv


def main() -> None:
    for label, module in FIGURES:
        print(f"building {label}")
        module.main()


if __name__ == "__main__":
    main()
