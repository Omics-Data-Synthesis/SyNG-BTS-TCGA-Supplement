"""Draw signed source ARI from stage 05 results, without recomputing metrics.

Run: uv run workflow/07_figures/figure_source_ari.py [--output-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fidelity_data import ROOT, read_fidelity, select_panel, summarize
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

STEM = "data_source_separation_miRNA_RNA"


def ari_label(value: float) -> str:
    """Keep small nonzero values distinguishable from zero, including their sign."""
    return f"{value:.1g}" if 0 < abs(value) < 0.01 else f"{value:.2f}"


def setting_label(frame: pd.DataFrame) -> str:
    """Describe the single normalization/model setting in a cohort group."""
    settings = frame[["norm", "model"]].drop_duplicates()
    if len(settings) != 1:
        raise ValueError("Expected one normalization/model setting per cohort group")
    norm, model = settings.iloc[0]
    return f"{norm} · {model.replace('CVAE', 'cVAE').replace('1-', '-1:')}"


def draw_figure(mirna: pd.DataFrame, rnaseq: pd.DataFrame, n_draws: int) -> Figure:
    """Render two cohort panels with the same signed ARI scale."""
    panels = [frame.sort_values(["mean", "cohort"]) for frame in (mirna, rnaseq)]
    means = pd.concat(panels)["mean"]
    lower = min(-0.01, np.floor(means.min() * 100) / 100 - 0.01)
    upper = max(0.12, np.ceil(means.max() * 100) / 100 + 0.01)
    ticks = np.unique(np.r_[lower, np.arange(0, upper + 0.001, 0.05)])
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.8, 6.0),
            sharey=True,
            gridspec_kw={"width_ratios": [1.9, 1]},
        )
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.26, top=0.81, wspace=0.06)
        titles = [
            f"A  miRNA-seq · {setting_label(mirna)}",
            "B  RNA-seq · selected configurations",
        ]
        for ax, frame, color, title in zip(
            axes, panels, ["#180F80", "#F28E00"], titles
        ):
            x = np.arange(len(frame))
            y = frame["mean"].to_numpy()
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#C9C9C9", linewidth=0.6, linestyle=(0, (1, 4)))
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.vlines(x, 0, y, color="#9B9B9B", linewidth=1.2)
            ax.scatter(x, y, color=color, s=48, zorder=3)
            for position, value in zip(x, y):
                ax.annotate(
                    ari_label(value),
                    (position, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=color,
                    fontsize=8,
                    rotation=90 if len(frame) > 12 else 0,
                )
            ax.set_xticks(x, frame.cohort, rotation=58, ha="right", fontsize=10)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, len(frame) - 0.4)
            ax.set_ylim(lower, upper + 0.01)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=16)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Signed source ARI", fontsize=11)
        axes[1].spines["left"].set_visible(False)
        axes[1].tick_params(axis="y", left=False)
        fig.suptitle(
            "Data-source separation across TCGA cohorts",
            x=0.025,
            ha="left",
            y=0.97,
            fontsize=17,
            weight="bold",
        )
        rna_settings = "; ".join(
            f"{setting_label(frame)} for "
            f"{'one-group' if grouping == 'none' else 'two-group'} cohorts"
            for grouping, frame in rnaseq.groupby("grouping", sort=False)
        )
        augmentation = (
            "AE-head augmentation is disabled."
            if rnaseq.off_aug.eq("none").all()
            else ""
        )
        fig.text(
            0.025,
            0.065,
            f"Points and labels: means across {n_draws} subsampling draws. "
            "Zero is the chance-adjusted agreement baseline.\n"
            f"RNA: {rna_settings}. {augmentation}",
            fontsize=9,
            color="#555555",
            linespacing=1.6,
        )
    return fig


def main(output_dir: Path = ROOT / "figures") -> None:
    data = read_fidelity()
    mirna = select_panel(data, "mirna_ari")
    rnaseq = select_panel(data, "rnaseq")
    fig = draw_figure(
        summarize(mirna, ["source_ari"]),
        summarize(rnaseq, ["source_ari"]),
        mirna.draw.nunique(),
    )
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
