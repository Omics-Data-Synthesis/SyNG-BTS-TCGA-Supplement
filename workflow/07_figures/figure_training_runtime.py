"""Plot original successful miRNA runtime records without rerunning generation.

Run: uv run --locked workflow/07_figures/figure_training_runtime.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from table_manuscript import read_csv, training_runtime

ROOT = Path(__file__).resolve().parents[2]
STEM = "training_runtime_mirna"
NORMS = ("raw", "TC", "DESeq")
RATIOS = (5, 10, 50)


def load_values(source: Path):
    """Return cohort labels and two 24-by-9 arrays in explicit config order."""
    # Share selection and reshaping with the CSV exporter, not its output file.
    rows = training_runtime(read_csv(source))
    indexed = {}
    for row in rows:
        key = (row["Dataset (cancer_variable)"], row["Norm"])
        if key in indexed:
            raise ValueError(f"Duplicate dataset/normalization: {key}")
        indexed[key] = row
    datasets = sorted({key[0] for key in indexed})
    if len(datasets) != 24 or set(indexed) != {
        (dataset, norm) for dataset in datasets for norm in NORMS
    }:
        raise ValueError("Expected 24 datasets, each with raw, TC and DESeq")
    cohorts = [dataset.split("_")[0] for dataset in datasets]
    if len(set(cohorts)) != len(cohorts):
        raise ValueError("Short cohort labels must identify unique tasks")
    arrays = []
    for column in ("Epochs", "Time (s)"):
        values = np.array([
            [float(indexed[(dataset, norm)][f"CVAE1-{ratio} {column}"])
             for norm in NORMS for ratio in RATIOS]
            for dataset in datasets
        ])
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Missing, nonfinite or nonpositive {column}")
        if column == "Epochs" and not np.equal(values, np.floor(values)).all():
            raise ValueError("Epoch counts must be integers")
        arrays.append(values)
    return cohorts, *arrays


def contrast_text(rgba):
    """Choose black or white by the actual background's relative luminance."""
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ [0.2126, 0.7152, 0.0722])
    return "black" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "white"


def draw_figure(cohorts, epochs, seconds):
    """Render at 180 by 145 mm; text sizes are final placement sizes."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(180 / 25.4, 145 / 25.4))
        specs = [
            (0.07, epochs, "A  Completed training epochs", "Blues", 50, ".0f", "Epochs"),
            (0.58, seconds, "B  Training and generation time", "PuBuGn", 5, ".1f", "Time (s)"),
        ]
        for left, values, title, cmap_name, step, fmt, unit in specs:
            ax = fig.add_axes([left, 0.18, 0.405, 0.69])
            upper = np.ceil(values.max() / step) * step
            norm = Normalize(0, upper)
            cmap = plt.get_cmap(cmap_name)
            im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(cohorts)), cohorts, fontsize=6.8)
            ax.set_xticks(range(9), [f"1:{ratio}" for _ in NORMS for ratio in RATIOS], fontsize=6.7)
            ax.xaxis.tick_top()
            ax.tick_params(axis="both", length=0, pad=3)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(cohorts), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.35)
            ax.tick_params(which="minor", length=0)
            for boundary in (2.5, 5.5):
                ax.axvline(boundary, color="white", linewidth=2)
            for i in range(len(cohorts)):
                for j in range(9):
                    value = values[i, j]
                    ax.text(j, i, format(value, fmt), ha="center", va="center",
                            fontsize=6.4, color=contrast_text(cmap(norm(value))))
            for group, label in enumerate(NORMS):
                ax.text((group + 0.5) / 3, 1.045, label, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7.2, weight="bold")
            fig.text(left, 0.957, title, fontsize=8.5, weight="bold", va="top")
            cax = fig.add_axes([left + 0.035, 0.137, 0.335, 0.018])
            ticks = np.arange(0, upper + 0.1, 150 if unit == "Epochs" else 5)
            if ticks[-1] != upper:
                ticks = np.r_[ticks, upper]
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
            cb.ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=2)
            cb.outline.set_linewidth(0.4)
            cb.set_label(unit, fontsize=7, labelpad=3)
        fig.text(0.5, 0.063, "Columns: normalization and cVAE reconstruction-to-KL loss ratio",
                 ha="center", fontsize=6.7)
    return fig


def main(output_dir=ROOT / "figures", source=ROOT / "results" / "training_time.csv"):
    cohorts, epochs, seconds = load_values(Path(source))
    fig = draw_figure(cohorts, epochs, seconds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove 5.5 mm of unused bottom canvas without rescaling text or panels.
    export_bounds = Bbox.from_bounds(0, 5.5 / 25.4, 180 / 25.4, 139.5 / 25.4)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{STEM}.{ext}", dpi=300, facecolor="white",
                    bbox_inches=export_bounds, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--source", type=Path, default=ROOT / "results" / "training_time.csv")
    args = parser.parse_args()
    main(args.output_dir, args.source)
