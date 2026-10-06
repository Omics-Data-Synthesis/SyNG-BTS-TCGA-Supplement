"""Plot elements shared by the miRNA and RNA fidelity figures."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

COLORS = {"Real": "#2878B5", "Generated": "#E6863B"}
GREEN = "#007A4D"


def model_label(model: str) -> str:
    return model.replace("CVAE", "cVAE").replace("1-", "-1:")


def draw_umap(
    ax: Axes,
    data: pd.DataFrame,
    *,
    point_size: float = 22,
    aspect: float = 0.5,
    boxed: bool = False,
) -> None:
    """Plot saved coordinates, coloring source and using shape for binary groups."""
    groups = [(None, "o")] if data.grouping.eq("none").all() else [(0, "o"), (1, "^")]
    for group, marker in groups:
        points = data if group is None else data[data.groups.eq(group)]
        ax.scatter(
            points.UMAP1,
            points.UMAP2,
            c=points.datatype.map(COLORS),
            marker=marker,
            s=point_size * 29 / 22 if group == 1 else point_size,
            alpha=0.9,
            linewidths=0,
        )
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_axisbelow(True)
    ax.grid(color="#D5D5D5", linewidth=0.6)
    if boxed:
        for spine in ax.spines.values():
            spine.set(color="#909090", linewidth=0.5)
    else:
        ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(aspect)


def legend_handles() -> list[Line2D]:
    """Use the same filled source/group markers as the scatterplots."""
    return [
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


def draw_metric(
    ax: Axes,
    values: pd.DataFrame,
    y: np.ndarray,
    bounds: tuple[float, float],
    ticks: list[float] | np.ndarray,
    label: str,
) -> None:
    """Draw means and complete observed ranges on the supplied metric scale."""
    lower, upper = bounds
    ax.hlines(y, lower, upper, color="#A8A8A8", linewidth=1)
    ax.hlines(y, values.low, values.high, color=GREEN, linewidth=2)
    ax.scatter(values["mean"], y, color=GREEN, s=26, zorder=3)
    if lower < 0:
        ax.axvline(0, color="#A8A8A8", linestyle="--", linewidth=0.7)
    padding = 0.08 * (upper - lower)
    ax.set(xlim=(lower - padding, upper + padding), xticks=ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(label, fontsize=10.5, weight="bold", pad=13)
