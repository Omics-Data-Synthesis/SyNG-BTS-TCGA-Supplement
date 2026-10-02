"""Stage 07: build the manuscript figures and tables.

Each figure is its own ``figure_*.py`` module in this directory, exposing ``main()``
and writing descriptively named files into ``figures/``. Register a new module in
FIGURES below and it gets built with the rest.

Figure files are named for what they show, not for their manuscript number, because
the numbering is still changing.

The fidelity figures are the R scripts in this directory. ``figure_fidelity.py``
runs them, and needs ``Rscript`` plus ``renv::restore()``.

Run from the repository root:

    uv run workflow/07_figures/run.py
"""

import figure_cvae_architecture
import figure_fidelity
import figure_learning_curves

# (label, module) pairs, built in this order.
FIGURES = [
    ("cVAE architecture", figure_cvae_architecture),
    ("SyntheSize learning curves", figure_learning_curves),
    ("fidelity figures", figure_fidelity),
]


def main() -> None:
    for label, module in FIGURES:
        print(f"building {label}")
        module.main()


if __name__ == "__main__":
    main()
