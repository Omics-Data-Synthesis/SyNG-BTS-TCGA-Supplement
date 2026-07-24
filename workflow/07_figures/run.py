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

# (label, module) pairs, built in this order.
FIGURES = [
    ("cVAE architecture", figure_cvae_architecture),
    # TODO: add the data-driven figures here
]

# TODO: add the data-driven figures as their source data lands in results/:
#   cARI across the 24 miRNA cohorts        <- results/cari_mirna.csv
#   quality evaluation for STAD             <- results/fidelity_stad.csv, results/umap_stad.csv
#   HNSC learning curves                   <- results/curves_hnsc.csv
#   cARI across the 5 RNA-seq cohorts       <- results/cari_rna.csv
#   RNA-seq UMAP and MAD panels             <- results/fidelity_rna.csv, results/umap_rna.csv


def main() -> None:
    for label, module in FIGURES:
        print(f"building {label}")
        module.main()


if __name__ == "__main__":
    main()
