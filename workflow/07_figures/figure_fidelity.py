"""Run the fidelity R scripts.

``ari_figure_code.R`` and ``fidelity_RNA_code.R`` are each called once.
``fidelity_miRNA_code.R`` is called once per miRNA cohort in ``config/datasets.tsv``.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _run(rscript: str, script: str, *args: str) -> None:
    label = " ".join((script, *args))
    print(f"    {label}", flush=True)
    subprocess.run(
        [rscript, str(HERE / script), *args],
        cwd=ROOT,
        env={**os.environ, "RENV_PROJECT": str(ROOT)},
        check=True,
    )


def main() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise SystemExit("Rscript not found on PATH; install R, then run renv::restore()")

    _run(rscript, "ari_figure_code.R")
    _run(rscript, "fidelity_RNA_code.R")
    with (ROOT / "config" / "datasets.tsv").open(newline="") as handle:
        dataset_ids = [
            row["dataset_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("modality") == "mirna"
        ]
    for dataset_id in dataset_ids:
        _run(rscript, "fidelity_miRNA_code.R", dataset_id)


if __name__ == "__main__":
    main()
