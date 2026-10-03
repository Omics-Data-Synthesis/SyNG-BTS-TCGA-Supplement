"""Draw SyntheSize learning-curve fits stored by stage 06.

Stage 06 already ran ``evaluate_sample_sizes`` and fitted the IPLF
``(1 - a) - b * n^c``. This module only renders ``results/learning_curve_fits.json``:
observed means (with draw SD) and the stored fitted curve plus 95% CI.
Layout matches ``helper_source/augmentation_learning_curves.py``: one
figure per generate() combo, rows = classifiers, columns = real | generated.

Metric, classifier order, and y-limits come from ``config/synthesize.tsv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))
from synthesize_results import POINT_COLUMNS, point_rows, read_archive
SYNTHESIZE_TSV = ROOT / "config" / "synthesize.tsv"
CURVES_CSV = ROOT / "results" / "curves.csv"
FIGURES = ROOT / "figures" / "learning_curves"

SOURCES = ("real", "generated")
COLORS = {"real": "#d62728", "generated": "#1f77b4"}


def read_kv_tsv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {r["param"]: r["value"] for r in csv.DictReader(f, delimiter="\t")}


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def combo_stem(dataset_id: str, norm: str, model: str, off_aug: str) -> Path:
    return FIGURES / dataset_id / norm / model / off_aug


def draw_combo(
    points: pd.DataFrame,
    *,
    fits: list[dict],
    methods: list[str],
    metric: str,
    ylim: tuple[float, float],
    title: str,
) -> plt.Figure:
    present = [m for m in methods if not points[points["method"] == m].empty]
    if not present:
        raise ValueError("no classifier rows to plot")
    n_rows = len(present)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(9.6, 2.35 * n_rows + 0.7),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_i, method in enumerate(present):
        for col_i, source in enumerate(SOURCES):
            ax = axes[row_i, col_i]
            sel = points[(points["method"] == method) & (points["source"] == source)]
            obs = sel[sel["kind"] == "observed"].sort_values("n")
            fit = sel[sel["kind"] == "fitted"].sort_values("n")
            color = COLORS[source]

            result = next((f for f in fits if f["method"] == method and f["source"] == source), None)
            if result and result["interval_status"] == "ok":
                ax.fill_between(
                    fit["n"],
                    fit["ci_low"],
                    fit["ci_high"],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )
            if result and result["fit_status"] == "ok":
                ax.plot(
                    fit["n"],
                    fit["value"],
                    color=color,
                    linestyle="--",
                    linewidth=1.3,
                    zorder=2,
                    label="Fitted",
                )
            if not obs.empty:
                ax.plot(obs["n"], obs["value"], "o", color=color, markersize=4.5, zorder=3, label="Observed")
                with_sd = obs[obs["observed_std"].notna()]
                ax.errorbar(
                    with_sd["n"],
                    with_sd["value"],
                    yerr=with_sd["observed_std"],
                    fmt="none",
                    color=color,
                    markersize=4.5,
                    capsize=2,
                    elinewidth=0.9,
                    zorder=3,
                    label="Observed",
                )

            notice = "No observations" if result is None else (
                "Fit unavailable" if result["fit_status"] != "ok" else
                "Confidence band unavailable" if result["interval_status"] != "ok" else ""
            )
            if notice:
                ax.text(0.02, 0.04, notice, transform=ax.transAxes, fontsize=8)
            ax.set_ylim(*ylim)
            ax.set_ylabel(metric if col_i == 0 else "")
            if row_i == n_rows - 1:
                ax.set_xlabel("Candidate subset size")
            if row_i == 0:
                ax.set_title(source.capitalize())
            if col_i == 0:
                ax.annotate(
                    method,
                    xy=(-0.22, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=10,
                )
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, dest_dir: Path, stem: str, title: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    common = {"facecolor": "white"}
    fig.savefig(dest_dir / f"{stem}.pdf", metadata={"Title": title, "Subject": title}, **common)
    fig.savefig(dest_dir / f"{stem}.png", dpi=200, metadata={"Title": title}, **common)
    plt.close(fig)


def main(metric: str | None = None) -> None:
    archive = read_archive(ROOT)
    cfg = read_kv_tsv(SYNTHESIZE_TSV)
    methods = split_csv(cfg["methods"])
    metric = metric or cfg["static_metric"]
    if metric not in split_csv(cfg["metrics"]):
        raise ValueError(f"Metric {metric} is not configured in synthesize.tsv")
    ylim = (float(cfg["plot_ylim_lo"]), float(cfg["plot_ylim_hi"]))
    curves = pd.DataFrame([r for f in archive["fits"] for r in point_rows(f)], columns=POINT_COLUMNS)
    # Replace this metric's generated figures, including obsolete configurations.
    for extension in ("pdf", "png"):
        for path in FIGURES.rglob(f"learning_curve_{metric}.{extension}"):
            path.unlink()
    if curves.empty:
        print("No archived SyntheSize fits; no learning-curve figures generated.")
        return

    sub = curves[curves["metric"] == metric].copy()
    if sub.empty:
        raise SystemExit(f"no rows in results/curves.csv for metric={metric}")

    keys = ["dataset_id", "norm", "model", "off_aug"]
    combos = sub[keys].drop_duplicates().sort_values(keys)
    stem = f"learning_curve_{metric}"
    n_ok = 0
    for rec in combos.itertuples(index=False):
        points = sub[
            (sub["dataset_id"] == rec.dataset_id)
            & (sub["norm"] == rec.norm)
            & (sub["model"] == rec.model)
            & (sub["off_aug"] == rec.off_aug)
        ]
        title = (
            f"{rec.dataset_id}  {rec.norm}  {rec.model}  {rec.off_aug}  — {metric}"
        )
        fits = [f for f in archive["fits"] if f["metric_name"] == metric and
                all(f[key] == getattr(rec, key) for key in keys)]
        fig = draw_combo(points, fits=fits, methods=methods, metric=metric, ylim=ylim, title=title)
        dest = combo_stem(rec.dataset_id, rec.norm, rec.model, rec.off_aug)
        save_figure(fig, dest, stem, title)
        n_ok += 1
        print(f"    {dest.relative_to(ROOT)}/{stem}.pdf")

    print(f"wrote {n_ok} learning-curve figure(s) under {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", help="configured metric (default: static_metric)")
    main(parser.parse_args().metric)
