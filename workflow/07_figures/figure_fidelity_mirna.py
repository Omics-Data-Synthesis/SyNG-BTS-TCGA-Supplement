"""Render miRNA UMAP and fidelity panels from the current stage 05 tables."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fidelity_data import (
    ROOT,
    read_fidelity,
    read_umap,
    select_mirna_grid,
    select_panel,
    select_umap,
    summarize,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

COLORS = {"Real": "#2878B5", "Generated": "#E6863B"}
GREEN = "#007A4D"
METRIC_LABELS = {
    "cccpcc": "CCC-PCC",
    "ccc_log10_pvalue": "CCC −log₁₀(p)",
    "ccc_meandiff": "CCC mean",
    "medae_mean": "MedAE mean",
    "medae_sd": "MedAE SD",
    "medae_cv": "MedAE CV",
}


def metric_limits(values: pd.DataFrame, metric: str) -> tuple[float, float]:
    """One unclipped scale per metric across the cohort's configurations."""
    low, high = values.low.min(), values.high.max()
    if metric.startswith("ccc"):
        if low < -1 or high > 1:
            raise ValueError(f"CCC outside [-1, 1]: {metric}")
        return min(0, np.floor(low * 2) / 2), 1.0
    if low < 0:
        raise ValueError(f"Negative absolute error: {metric}")
    return 0.0, max(0.05, np.ceil(high * 20) / 20)


def draw_figure(
    fidelity: pd.DataFrame, umap: pd.DataFrame, rows: pd.DataFrame
) -> Figure:
    """Draw one cohort: saved UMAP above a configuration-by-metric comparison."""
    values = summarize(fidelity, list(METRIC_LABELS))
    cohort = fidelity.cohort.unique().item()
    norm, model = umap[["norm", "model"]].drop_duplicates().iloc[0]
    model_label = model.replace("CVAE", "cVAE").replace("1-", "-1:")
    n_rows = len(rows)
    y = np.arange(n_rows)[::-1]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig = plt.figure(figsize=(12.5, 10.2))
        layout = fig.add_gridspec(
            7,
            1,
            left=0.035,
            right=0.985,
            bottom=0.025,
            top=0.98,
            height_ratios=[0.55, 4, 0.48, 0.65, 0.36, 3.55, 0.3],
            hspace=0.06,
        )
        heading = layout[0].get_position(fig)
        fig.text(
            heading.x0,
            heading.y1,
            f"A  {cohort} miRNA samples in UMAP space",
            fontsize=15,
            weight="bold",
            va="top",
        )
        fig.text(
            heading.x0,
            heading.y0 + 0.005,
            f"Empirical and generated data · {model_label}, {norm}",
            fontsize=11,
            color="#53636D",
            va="bottom",
        )
        top = layout[1].subgridspec(1, 3, width_ratios=[1, 6, 1])
        ax = fig.add_subplot(top[0, 1])
        # Plot saved coordinates directly; groups control marker shape.
        for group, marker in [(0, "o"), (1, "^")]:
            points = umap[umap.groups.eq(group)]
            ax.scatter(
                points.UMAP1,
                points.UMAP2,
                c=points.datatype.map(COLORS),
                marker=marker,
                s=22 if group == 0 else 29,
                alpha=0.9,
                linewidths=0,
            )
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.set_axisbelow(True)
        ax.grid(color="#D5D5D5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_box_aspect(0.5)
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                color=COLORS[source],
                linestyle="none",
                label=label,
                markersize=6,
            )
            for source, label in [("Real", "Empirical"), ("Generated", "Generated")]
        ] + [
            Line2D(
                [],
                [],
                marker=marker,
                color="#222222",
                linestyle="none",
                label=f"Group {group}",
                markersize=6,
            )
            for group, marker in [(0, "o"), (1, "^")]
        ]
        legend = layout[2].get_position(fig)
        fig.legend(
            handles=handles,
            loc="center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, (legend.y0 + legend.y1) / 2),
            fontsize=10,
        )

        grid = layout[5].subgridspec(
            1, 7, width_ratios=[2.05, 1, 1, 1, 1, 1, 1], wspace=0.25
        )
        labels = fig.add_subplot(grid[0, 0])
        labels.set(xlim=(0, 1), ylim=(-0.65, n_rows - 0.35))
        labels.set_axis_off()
        for normalization in rows.norm.drop_duplicates():
            positions = y[rows.norm.eq(normalization)]
            labels.text(0.01, positions.mean(), normalization, va="center", fontsize=11)
        for position, row in zip(y, rows.itertuples()):
            labels.text(
                0.79,
                position,
                row.model.removeprefix("CVAE").replace("-", ":"),
                ha="center",
                va="center",
                fontsize=11,
            )
        labels.text(
            0.01,
            1.035,
            "Normalization",
            transform=labels.transAxes,
            weight="bold",
            fontsize=10.5,
        )
        labels.text(
            0.79,
            1.035,
            "Loss ratio",
            transform=labels.transAxes,
            ha="center",
            weight="bold",
            fontsize=10.5,
        )

        metric_axes = []
        for column, (metric, label) in enumerate(METRIC_LABELS.items(), 1):
            ax = fig.add_subplot(grid[0, column], sharey=labels)
            metric_axes.append(ax)
            d = rows.merge(
                values[values.metric.eq(metric)],
                on=["norm", "model"],
                validate="one_to_one",
            )
            lower, upper = metric_limits(d, metric)
            ax.hlines(y, lower, upper, color="#A8A8A8", linewidth=1)
            ax.hlines(y, d.low, d.high, color=GREEN, linewidth=2)
            ax.scatter(d["mean"], y, color=GREEN, s=26, zorder=3)
            if lower < 0:
                ax.axvline(0, color="#A8A8A8", linestyle="--", linewidth=0.7)
            padding = 0.08 * (upper - lower)
            ticks = (
                np.arange(lower, upper + 0.01, 0.5)
                if lower < 0
                else [0, upper / 2, upper]
            )
            ax.set(xlim=(lower - padding, upper + padding), xticks=ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_title(label, fontsize=10.5, weight="bold", pad=13)

        header = layout[3].get_position(fig)
        fig.text(
            header.x0,
            header.y1,
            "B  Fidelity across cVAE configurations",
            fontsize=15,
            weight="bold",
            va="top",
        )
        for first, last, text, direction in [
            (0, 2, "Higher is better", 1),
            (3, 5, "Lower is better", -1),
        ]:
            left = metric_axes[first].get_position().x0
            right = metric_axes[last].get_position().x1
            fig.text(
                (left + right) / 2, header.y0 + 0.018, text, ha="center", fontsize=11
            )
            start, end = (left, right) if direction == 1 else (right, left)
            labels.annotate(
                "",
                xy=(end, header.y0 + 0.009),
                xytext=(start, header.y0 + 0.009),
                xycoords=fig.transFigure,
                textcoords=fig.transFigure,
                arrowprops={"arrowstyle": "->", "lw": 1.1},
            )
        divider = (
            metric_axes[2].get_position().x1 + metric_axes[3].get_position().x0
        ) / 2
        fig.add_artist(
            Line2D(
                [divider, divider],
                [metric_axes[0].get_position().y0, header.y0 + 0.05],
                transform=fig.transFigure,
                color="#808080",
                linewidth=0.8,
            )
        )
        footer = layout[6].get_position(fig)
        fig.text(
            footer.x0,
            footer.y0,
            "Points: means; green lines: minimum–maximum across "
            f"{fidelity.draw.nunique()} generated-sample subsampling draws.",
            fontsize=9,
            color="#53636D",
        )
    return fig


def main(output_dir: Path = ROOT / "figures", dataset_id: str | None = None) -> None:
    fidelity = read_fidelity()
    selected = select_mirna_grid(fidelity)
    embedding_settings = select_panel(fidelity, "mirna_umap")
    if dataset_id is not None:
        if dataset_id not in selected.dataset_id.values:
            raise ValueError(f"Unknown configured miRNA dataset: {dataset_id}")
        selected = selected[selected.dataset_id.eq(dataset_id)]
        embedding_settings = embedding_settings[
            embedding_settings.dataset_id.eq(dataset_id)
        ]
    embeddings = select_umap(read_umap(), embedding_settings)
    rows = selected[["norm", "model"]].drop_duplicates().reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, data in selected.groupby("dataset_id", sort=False):
        fig = draw_figure(data, embeddings[embeddings.dataset_id.eq(dataset)], rows)
        for extension in ("pdf", "png"):
            path = output_dir / f"{dataset}_miRNA_fidelity.{extension}"
            with plt.rc_context({"pdf.fonttype": 42}):
                fig.savefig(path, dpi=300, facecolor="white")
            print(path, flush=True)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", help="Render only this configured miRNA dataset ID"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    main(args.output_dir, args.dataset)
