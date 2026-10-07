"""Stage 07: build the manuscript figures and tables.

Each figure is its own ``figure_*.py`` module in this directory, exposing ``main()``
and writing descriptively named files into ``figures/``. Register a new module in
FIGURES below and it gets built with the rest.

Figure files are named for what they show, not for their manuscript number, because
the numbering is still changing.

Run from the repository root:

    uv run workflow/07_figures/run.py
    uv run workflow/07_figures/run.py --figures source-ari
"""

import argparse

import figure_cvae_architecture
import figure_fidelity_mirna
import figure_fidelity_rnaseq
import figure_learning_curve_agreement
import figure_learning_curves
import figure_source_ari

# Command names and modules, built in this order by default.
FIGURES = {
    "cvae-architecture": figure_cvae_architecture,
    "learning-curves": figure_learning_curves,
    "learning-curve-agreement": figure_learning_curve_agreement,
    "source-ari": figure_source_ari,
    "fidelity-mirna": figure_fidelity_mirna,
    "fidelity-rnaseq": figure_fidelity_rnaseq,
}


def main(figures: list[str] | None = None) -> None:
    for name in FIGURES if figures is None else figures:
        print(f"building {name}", flush=True)
        FIGURES[name].main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", nargs="+", choices=FIGURES, help="Build only the named figures."
    )
    main(parser.parse_args().figures)
