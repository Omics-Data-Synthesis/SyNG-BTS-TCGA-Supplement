"""Render selected RNA-seq fidelity summaries and saved UMAP embeddings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fidelity_data import (
    KEYS,
    ROOT,
    read_fidelity,
    read_umap,
    select_panel,
    select_umap,
    summarize,
)
from fidelity_plot import draw_metric, draw_umap, legend_handles, model_label
from matplotlib.figure import Figure

STEM = "RNA_selected_fidelity_UMAP"
METRIC_LABELS = {
    "source_ari": "Signed source ARI",
    "medae_mean": "MedAE mean",
    "medae_sd": "MedAE SD",
    "medae_cv": "MedAE CV",
}


def metric_limits(values: pd.DataFrame, metric: str) -> tuple[float, float]:
    """One scale per metric, shared across every selected cohort."""
    low, high = values.low.min(), values.high.max()
    if metric == "source_ari":
        return min(-0.01, np.floor(low * 100) / 100), max(
            0.01, np.ceil(high * 100) / 100
        )
    if low < 0:
        raise ValueError(f"Negative absolute error: {metric}")
    step = 0.5 if metric == "medae_sd" else 0.1
    return 0.0, max(step, np.ceil(high / step) * step)


def draw_figure(fidelity: pd.DataFrame, umap: pd.DataFrame) -> Figure:
    """Draw a cohort summary above one-group and two-group embedding rows."""
    keys = KEYS + ["cohort", "grouping"]
    rows = fidelity[keys].drop_duplicates()
    rows = (
        rows.assign(order=rows.grouping.map({"none": 0, "binary": 1}))
        .sort_values(["order", "cohort"])
        .drop(columns="order")
    )
    if (
        rows.dataset_id.duplicated().any()
        or not rows.grouping.isin(["none", "binary"]).all()
    ):
        raise ValueError(
            "RNA figure requires one selected setting per one/two-group cohort"
        )
    values = summarize(fidelity, list(METRIC_LABELS))
    y = np.arange(len(rows))[::-1]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig = plt.figure(figsize=(11.5, 12.5))
        layout = fig.add_gridspec(
            11,
            1,
            left=0.045,
            right=0.985,
            top=0.98,
            bottom=0.025,
            hspace=0.12,
            height_ratios=[0.3, 0.25, 2, 0.73, 0.4, 0.58, 2.55, 0.58, 2.55, 0.5, 0.3],
        )
        heading = layout[0].get_position(fig)
        fig.text(
            heading.x0,
            heading.y1,
            "A  RNA-seq configurations and fidelity",
            fontsize=15,
            weight="bold",
            va="top",
        )
        grid = layout[2].subgridspec(
            1, 5, width_ratios=[2.9, 1.1, 1, 1, 1], wspace=0.25
        )
        labels = fig.add_subplot(grid[0, 0])
        labels.set(xlim=(0, 1), ylim=(-0.65, len(rows) - 0.35))
        labels.set_axis_off()
        for position, row in zip(y, rows.itertuples()):
            setup = "One-group VAE" if row.grouping == "none" else "Two-group cVAE"
            for x, text, align in [
                (0.01, row.cohort, "left"),
                (0.46, setup, "center"),
                (0.85, row.norm, "center"),
            ]:
                labels.text(x, position, text, ha=align, va="center", fontsize=10.5)
        for x, text, align in [
            (0.01, "Cohort", "left"),
            (0.46, "Setup", "center"),
            (0.85, "Normalization", "center"),
        ]:
            labels.annotate(
                text,
                (x, 1),
                xycoords=labels.transAxes,
                xytext=(0, 13),
                textcoords="offset points",
                ha=align,
                va="baseline",
                fontsize=10.5,
                weight="bold",
            )
        for column, (metric, label) in enumerate(METRIC_LABELS.items(), 1):
            ax = fig.add_subplot(grid[0, column], sharey=labels)
            d = rows.merge(
                values[values.metric.eq(metric)], on=keys, validate="one_to_one"
            )
            lower, upper = metric_limits(d, metric)
            ticks = (
                [lower, 0, upper] if metric == "source_ari" else [0, upper / 2, upper]
            )
            draw_metric(ax, d, y, (lower, upper), ticks, label)

        note = layout[3].get_position(fig)
        fig.text(
            note.x0,
            note.y0 + 0.006,
            "Points: means; green lines: minimum–maximum across "
            f"{fidelity.draw.nunique()} subsampling draws.\n"
            "ARI: zero = chance baseline. "
            "Lower MedAE is better. No AE-head augmentation.",
            fontsize=9,
            color="#53636D",
            linespacing=1.5,
            va="bottom",
        )
        heading = layout[4].get_position(fig)
        fig.text(
            heading.x0,
            heading.y1 - 14 / (72 * fig.get_figheight()),
            "B  UMAP embeddings of empirical and generated RNA-seq samples",
            fontsize=15,
            weight="bold",
            va="top",
        )
        max_cohorts = rows.groupby("grouping").size().max()
        for grouping, plot_row, title in [
            ("none", 6, "One-group VAE cohorts"),
            ("binary", 8, "Two-group cVAE cohorts"),
        ]:
            cohorts = rows[rows.grouping.eq(grouping)]
            grid = layout[plot_row].subgridspec(1, 2 * max_cohorts, wspace=0.65)
            start = max_cohorts - len(cohorts)
            for i, row in enumerate(cohorts.itertuples()):
                ax = fig.add_subplot(grid[0, start + 2 * i : start + 2 * i + 2])
                draw_umap(
                    ax,
                    umap[umap.dataset_id.eq(row.dataset_id)],
                    point_size=14,
                    aspect=0.72,
                    boxed=True,
                )
                ax.set_title(
                    row.cohort, loc="left", fontsize=11.5, weight="bold", pad=24
                )
                if i == 0:
                    # Anchor each group heading to the row it describes.
                    bounds = ax.get_position()
                    fig.text(
                        bounds.x0,
                        bounds.y1 + 41 / (72 * fig.get_figheight()),
                        title,
                        fontsize=14,
                        color="#36454F",
                        weight="bold",
                        va="baseline",
                    )
                ax.text(
                    0,
                    1.025,
                    f"{model_label(row.model)} · {row.norm}",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#53636D",
                    va="bottom",
                )
        legend = layout[9].get_position(fig)
        fig.legend(
            handles=legend_handles(),
            loc="center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, (legend.y0 + legend.y1) / 2),
            fontsize=10.5,
        )
        fig.text(
            legend.x0,
            legend.y0 - 8 / (72 * fig.get_figheight()),
            "One-group samples are circles. Embeddings are fitted separately "
            "and are not comparable across cohorts.",
            fontsize=9,
            color="#53636D",
            va="top",
        )
    return fig


def main(output_dir: Path = ROOT / "figures") -> None:
    selected = select_panel(read_fidelity(), "rnaseq")
    embeddings = select_umap(read_umap(), selected)
    fig = draw_figure(selected, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output_dir / f"{STEM}.{extension}"
        with plt.rc_context({"pdf.fonttype": 42}):
            fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    main(parser.parse_args().output_dir)
