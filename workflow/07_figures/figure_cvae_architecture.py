"""Draw the standard SyNG-BTS cVAE architecture used in the miRNA benchmark.

The figure is a schematic of ``syng_bts.helper_models.CVAE``, not a plot of data, so
it reads nothing from ``results/``. It writes editable SVG and PDF outputs plus a
600-dpi PNG for the Word workflow.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
STEM = "cvae_architecture"


COLORS = {
    "ink": "#1F2933",
    "muted": "#5F6B76",
    "line": "#52606D",
    "neutral_fill": "#F3F5F6",
    "neutral_edge": "#7A8793",
    "encoder_fill": "#D7EAF4",
    "encoder_edge": "#397A9C",
    "decoder_fill": "#F9DDD0",
    "decoder_edge": "#B75D36",
    "latent_fill": "#E8E0F2",
    "latent_edge": "#785A9B",
    "condition_fill": "#EEE8F5",
    "condition_edge": "#785A9B",
    "group_fill": "#F7F9FA",
}

HIDDEN_BLOCK_WIDTH = 8.0


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.15,
    radius: float = 0.9,
    zorder: int = 3,
) -> FancyBboxPatch:
    """Add a centered rounded rectangle and return it."""
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    linewidth: float = 1.15,
    connectionstyle: str = "arc3",
    zorder: int = 2,
) -> FancyArrowPatch:
    """Add a clean directional connector."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=linewidth,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def line(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.0,
    zorder: int = 1,
) -> None:
    """Add an undirected connector segment."""
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=zorder,
    )


def section_label(
    ax: plt.Axes,
    x: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    """Add a two-line group label."""
    ax.text(
        x,
        52.1,
        title,
        ha="center",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x,
        49.2,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=color,
    )


def draw_hidden_block(
    ax: plt.Axes,
    x: float,
    y: float,
    height: float,
    units: int,
    *,
    kind: str,
) -> None:
    """Draw one hidden block; the shared operations appear in the figure key."""
    if kind == "encoder":
        facecolor = COLORS["encoder_fill"]
        edgecolor = COLORS["encoder_edge"]
    else:
        facecolor = COLORS["decoder_fill"]
        edgecolor = COLORS["decoder_edge"]

    rounded_box(
        ax,
        x,
        y,
        HIDDEN_BLOCK_WIDTH,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        radius=0.75,
    )
    ax.text(
        x,
        y,
        f"{units}",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_figure() -> plt.Figure:
    """Construct the architecture figure."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0,
        }
    )

    # The canvas is the manuscript's final 6.5-inch text width. Font sizes
    # therefore correspond directly to their printed point sizes.
    fig = plt.figure(figsize=(6.5, 2.35), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 158)
    ax.set_ylim(0, 57.108)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    center_y = 35.0

    # Subtle group fields organize the architecture without enclosing the figure.
    rounded_box(
        ax,
        41.25,
        31.75,
        54.5,
        33.5,
        facecolor="#F6FAFC",
        edgecolor="#D4E4EC",
        linewidth=0.8,
        radius=1.3,
        zorder=0,
    )
    rounded_box(
        ax,
        79.0,
        31.75,
        19.0,
        33.5,
        facecolor="#FAF8FC",
        edgecolor="#E2D9EC",
        linewidth=0.8,
        radius=1.3,
        zorder=0,
    )
    rounded_box(
        ax,
        115.75,
        31.75,
        52.5,
        33.5,
        facecolor="#FEF8F5",
        edgecolor="#F1D8CC",
        linewidth=0.8,
        radius=1.3,
        zorder=0,
    )

    section_label(
        ax,
        41.25,
        "Encoder",
        r"$q_\phi(z\mid x,y)$",
        COLORS["encoder_edge"],
    )
    section_label(
        ax,
        79.0,
        "Latent space",
        "32 dimensions",
        COLORS["latent_edge"],
    )
    section_label(
        ax,
        115.75,
        "Decoder",
        r"$p_\theta(x\mid z,y)$",
        COLORS["decoder_edge"],
    )

    # Inputs and encoder concatenation.
    rounded_box(
        ax,
        6.0,
        center_y,
        10.0,
        7.6,
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["neutral_edge"],
        radius=1.0,
    )
    ax.text(
        6.0,
        center_y,
        r"$x$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
    )
    concat_left = (16.5, center_y)
    concat_right = (91.4, center_y)
    for cx, cy in (concat_left, concat_right):
        circle = Circle(
            (cx, cy),
            1.9,
            facecolor="white",
            edgecolor=COLORS["condition_edge"],
            linewidth=1.15,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            cx,
            cy,
            "∥",
            ha="center",
            va="center",
            fontsize=10.5,
            color=COLORS["condition_edge"],
            fontweight="bold",
            zorder=5,
        )

    arrow(ax, (11.0, center_y), (14.6, center_y))

    encoder_x = (25.5, 37.0, 48.5)
    encoder_heights = (22.0, 15.0, 8.0)
    encoder_units = (256, 128, 64)
    arrow(ax, (18.4, center_y), (21.5, center_y))
    for x, height, units in zip(
        encoder_x,
        encoder_heights,
        encoder_units,
        strict=True,
    ):
        draw_hidden_block(
            ax,
            x,
            center_y,
            height,
            units,
            kind="encoder",
        )
    arrow(ax, (29.5, center_y), (33.0, center_y))
    arrow(ax, (41.0, center_y), (44.5, center_y))

    # Latent parameter heads.
    head_x = 62.0
    head_height = 6.2
    for y, symbol in (
        (39.5, r"$\mu$"),
        (30.5, r"$\log\,\sigma^2$"),
    ):
        rounded_box(
            ax,
            head_x,
            y,
            12.5,
            head_height,
            facecolor=COLORS["latent_fill"],
            edgecolor=COLORS["latent_edge"],
            radius=0.8,
        )
        ax.text(
            head_x,
            y,
            symbol,
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
        )

    arrow(
        ax,
        (52.5, center_y + 0.5),
        (55.75, 38.8),
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        ax,
        (52.5, center_y - 0.5),
        (55.75, 31.2),
        connectionstyle="arc3,rad=0.12",
    )

    # Reparameterization node and noise input.
    z_center = (78.75, center_y)
    z_circle = Circle(
        z_center,
        2.5,
        facecolor=COLORS["latent_fill"],
        edgecolor=COLORS["latent_edge"],
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(z_circle)
    ax.text(
        z_center[0],
        z_center[1],
        r"$z$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        zorder=5,
    )
    arrow(
        ax,
        (68.25, 39.1),
        (76.55, 36.2),
        connectionstyle="arc3,rad=-0.08",
    )
    arrow(
        ax,
        (68.25, 30.9),
        (76.55, 33.8),
        connectionstyle="arc3,rad=0.08",
    )
    ax.text(
        z_center[0],
        43.0,
        r"$z=\mu+\sigma\odot\varepsilon$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    rounded_box(
        ax,
        79.0,
        18.5,
        17.0,
        5.6,
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["neutral_edge"],
        linewidth=0.9,
        radius=0.75,
    )
    ax.text(
        79.0,
        18.5,
        r"$\varepsilon\sim\mathcal{N}(0,\mathrm{I})$",
        ha="center",
        va="center",
        fontsize=8.0,
        math_fontfamily="dejavuserif",
        color=COLORS["ink"],
    )
    arrow(
        ax,
        (79.0, 21.3),
        (78.75, 32.5),
        linewidth=0.95,
        connectionstyle="arc3",
    )

    # Decoder blocks, output projection, and reconstructed output.
    arrow(ax, (81.25, center_y), (89.5, center_y))
    decoder_x = (100.0, 112.0, 124.0)
    decoder_heights = (8.0, 15.0, 22.0)
    decoder_units = (64, 128, 256)
    arrow(ax, (93.3, center_y), (96.0, center_y))
    for x, height, units in zip(
        decoder_x,
        decoder_heights,
        decoder_units,
        strict=True,
    ):
        draw_hidden_block(
            ax,
            x,
            center_y,
            height,
            units,
            kind="decoder",
        )
    arrow(ax, (104.0, center_y), (108.0, center_y))
    arrow(ax, (116.0, center_y), (120.0, center_y))

    rounded_box(
        ax,
        136.0,
        center_y,
        9.0,
        8.0,
        facecolor=COLORS["decoder_fill"],
        edgecolor=COLORS["decoder_edge"],
        radius=0.75,
    )
    ax.text(
        136.0,
        center_y,
        r"$p$",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        136.0,
        43.0,
        "features",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    arrow(ax, (128.0, center_y), (131.5, center_y))

    rounded_box(
        ax,
        152.0,
        center_y,
        10.0,
        7.6,
        facecolor=COLORS["neutral_fill"],
        edgecolor=COLORS["neutral_edge"],
        radius=1.0,
    )
    ax.text(
        152.0,
        center_y,
        r"$\hat{x}$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
    )
    arrow(ax, (140.5, center_y), (147.0, center_y))

    # Shared conditioning label, branched to encoder and decoder concatenation.
    condition_y = 6.5
    rail_y = 12.5
    rounded_box(
        ax,
        55.0,
        condition_y,
        32.0,
        6.5,
        facecolor=COLORS["condition_fill"],
        edgecolor=COLORS["condition_edge"],
        linewidth=1.0,
        radius=0.9,
    )
    ax.text(
        55.0,
        condition_y,
        r"group label $y$ (scalar)",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["ink"],
    )
    line(
        ax,
        (55.0, condition_y + 3.25),
        (55.0, rail_y),
        color=COLORS["condition_edge"],
        linewidth=1.0,
    )
    line(
        ax,
        (16.5, rail_y),
        (91.4, rail_y),
        color=COLORS["condition_edge"],
        linewidth=1.0,
    )
    arrow(
        ax,
        (16.5, rail_y),
        (16.5, center_y - 1.9),
        color=COLORS["condition_edge"],
        linewidth=1.0,
    )
    arrow(
        ax,
        (91.4, rail_y),
        (91.4, center_y - 1.9),
        color=COLORS["condition_edge"],
        linewidth=1.0,
    )

    ax.text(
        18.0,
        6.5,
        "∥  concatenate",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    ax.text(
        121.0,
        8.0,
        "Hidden blocks: Dense → BN → ReLU",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    ax.text(
        124.0,
        4.9,
        r"Output layer: Dense($p$) → ReLU",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    return fig


def main() -> None:
    """Write all publication and Word-friendly outputs."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = draw_figure()
    title = "Conditional variational autoencoder architecture used in SyNG-BTS"
    common = {"facecolor": "white"}
    svg_path = FIGURES / f"{STEM}.svg"
    fig.savefig(
        svg_path,
        metadata={"Title": title, "Description": title},
        **common,
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        FIGURES / f"{STEM}.pdf",
        metadata={"Title": title, "Subject": title},
        **common,
    )
    fig.savefig(
        FIGURES / f"{STEM}.png",
        dpi=600,
        metadata={"Title": title, "Description": title},
        **common,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
