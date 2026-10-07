"""Render manuscript learning-curve agreement figures from the stage 06 archive.

Uses observed means for shared-support gap summaries and archived predictions for
curve displays. No classifiers are rerun and no learning curves are refitted.
Run from the supplement root:

    uv run workflow/07_figures/figure_learning_curve_agreement.py
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))
import itertools

from synthesize_results import file_hash, read_archive

SELECTIONS = ROOT / "config" / "learning_curve_figures.tsv"
KEYS = ("dataset_id", "modality", "norm", "model", "off_aug", "method", "metric_name")
METHODS = ("LOGIS", "SVM", "KNN", "RF", "XGB")


def interpolate(observed, n):
    assert observed[0]["n"] <= n <= observed[-1]["n"]
    for left, right in itertools.pairwise(observed):
        if left["n"] <= n <= right["n"]:
            t = (n - left["n"]) / (right["n"] - left["n"])
            return left["observed_mean"] * (1 - t) + right["observed_mean"] * t
    raise ValueError(n)


def compare(real, generated):
    r, g = real["observed"], generated["observed"]
    lo, hi = max(r[0]["n"], g[0]["n"]), min(r[-1]["n"], g[-1]["n"])
    knots = sorted({lo, hi} | {o["n"] for o in r + g if lo <= o["n"] <= hi})
    delta = [interpolate(g, n) - interpolate(r, n) for n in knots]
    signed = absolute = 0
    for x, y, d, e in zip(knots, knots[1:], delta, delta[1:]):
        width = y - x
        signed += width * (d + e) / 2
        # Exact integral of the absolute piecewise-linear difference, including crossings.
        absolute += width * (
            (d * d + e * e) / (2 * (abs(d) + abs(e)))
            if d * e < 0
            else (abs(d) + abs(e)) / 2
        )
    return {
        "n_low": lo,
        "n_high": hi,
        "signed_gap": signed / (hi - lo),
        "absolute_gap": absolute / (hi - lo),
        "endpoint_gap": delta[-1],
    }


def main(output_dir=None, results_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "figures"
    results_dir = Path(results_dir) if results_dir is not None else ROOT / "results"
    archive = read_archive(ROOT)
    with SELECTIONS.open(newline="") as stream:
        selections = list(csv.DictReader(stream, delimiter="\t"))
    if not selections or len({tuple(row.items()) for row in selections}) != len(
        selections
    ):
        raise ValueError("Missing or duplicate learning-curve figure selections")
    with (ROOT / "config" / "synthesize.tsv").open(newline="") as stream:
        config = {
            row["param"]: row["value"] for row in csv.DictReader(stream, delimiter="\t")
        }

    fits = archive["fits"]
    if not fits:
        raise ValueError("No archived learning curves available")
    if not all(
        len(f["observed"]) == int(config["n_sizes"])
        and all(o["n_draws"] == int(config["n_draws"]) for o in f["observed"])
        for f in fits
    ):
        raise ValueError("Incomplete candidate-size or draw coverage")
    for selection in selections:
        for method in METHODS:
            for metric in ("auc", "f1_score", "accuracy"):
                for source in ("real", "generated"):
                    matches = [
                        f
                        for f in fits
                        if all(
                            f[k] == selection[k]
                            for k in ("dataset_id", "norm", "model", "off_aug")
                        )
                        and f["method"] == method
                        and f["metric_name"] == metric
                        and f["source"] == source
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Missing or duplicate selected curve: {selection}, {method}, {metric}, {source}"
                        )
    index = {tuple(f[k] for k in KEYS) + (f["source"],): f for f in fits}
    rows = []
    for f in fits:
        if f["source"] == "generated":
            key = tuple(f[k] for k in KEYS)
            rows.append({**dict(zip(KEYS, key)), **compare(index[key + ("real",)], f)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    with (results_dir / "tables" / "learning_curve_agreement.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "provenance": archive["provenance"],
        "source_hashes": archive["tables"],
        "completed_jobs": archive["counts"],
        "comparisons": len(rows),
        "figure_selection_sha256": file_hash(SELECTIONS),
    }
    unique_real = {}
    for f in fits:
        if f["source"] == "real":
            key = tuple(f[k] for k in ("dataset_id", "norm", "method", "metric_name"))
            value = {
                k: f[k]
                for k in (
                    "observed",
                    "predictions",
                    "parameters",
                    "covariance",
                    "fit_status",
                    "interval_status",
                )
            }
            assert key not in unique_real or unique_real[key] == value
            unique_real[key] = value
    summary["unique_empirical_series"] = len(unique_real)
    summary["generated_series"] = sum(f["source"] == "generated" for f in fits)
    summary["unavailable_intervals"] = [
        {k: f[k] for k in KEYS + ("source", "message")}
        for f in fits
        if f["interval_status"] != "ok"
    ]
    for metric in ("auc", "f1_score", "accuracy"):
        selected = [r for r in rows if r["metric_name"] == metric]
        summary[metric] = {
            "comparisons": len(selected),
            "positive_signed_gap": sum(r["signed_gap"] > 0 for r in selected),
            "median_signed_gap": statistics.median(r["signed_gap"] for r in selected),
            "median_absolute_gap": statistics.median(
                r["absolute_gap"] for r in selected
            ),
        }
        summary[metric]["by_modality"] = {}
        for modality in ("mirna", "rnaseq"):
            subset = [r for r in selected if r["modality"] == modality]
            summary[metric]["by_modality"][modality] = {
                "comparisons": len(subset),
                "positive_signed_gap": sum(r["signed_gap"] > 0 for r in subset),
                "median_signed_gap": statistics.median(r["signed_gap"] for r in subset),
                "median_absolute_gap": statistics.median(
                    r["absolute_gap"] for r in subset
                ),
            }
    (results_dir / "learning_curve_agreement_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    plot(fits, rows, selections, output_dir)
    print(f"Wrote learning-curve agreement figures to {output_dir}")
    print(f"Wrote {len(rows)} comparisons and provenance to {results_dir}")


def plot(fits, rows, selections, output_dir):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    cohorts = sorted(
        {r["dataset_id"] for r in rows}, key=lambda ds: (ds.startswith("RNA_"), ds)
    )
    matrices = []
    for measure in ("signed_gap", "absolute_gap"):
        matrices.append(
            np.array(
                [
                    [
                        statistics.median(
                            r[measure]
                            for r in rows
                            if r["dataset_id"] == ds
                            and r["method"] == method
                            and r["metric_name"] == "auc"
                        )
                        for method in METHODS
                    ]
                    for ds in cohorts
                ]
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 9), layout="constrained")
    labels = [
        ("RNA" if ds.startswith("RNA_") else "miRNA")
        + " · "
        + ds.removeprefix("RNA_").split("_")[0]
        for ds in cohorts
    ]
    limit = max(abs(matrices[0].min()), abs(matrices[0].max()))
    for ax, data, title, cmap, low, high in zip(
        axes,
        matrices,
        ("A  Signed AUC gap", "B  Absolute AUC gap"),
        ("RdBu_r", "YlOrRd"),
        (-limit, 0),
        (limit, matrices[1].max()),
    ):
        im = ax.imshow(data, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        ax.set_xticks(range(5), ["LR", "SVM", "KNN", "RF", "XGB"])
        ax.set_yticks(range(len(cohorts)), labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(
            sum(not ds.startswith("RNA_") for ds in cohorts) - 0.5,
            color="white",
            linewidth=2,
        )
        fig.colorbar(
            im, ax=ax, shrink=0.65, label="Median across configurations (AUC units)"
        )
    fig.suptitle(
        "Empirical–generated learning-curve agreement\nObserved means over the shared subset-size range",
        fontsize=13,
    )
    destination = output_dir / "learning_curve_agreement"
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)

    for metric in ("auc", "f1_score", "accuracy"):
        plot_cases(
            fits,
            metric,
            output_dir / f"learning_curve_comparison_{metric}",
            selections,
            draw_sd=metric != "auc",
        )
    plot_configurations(rows, cohorts, selections, output_dir)


def plot_cases(fits, metric, destination, selections, draw_sd=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        5,
        len(selections),
        figsize=(3.5 * len(selections), 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    metric_label = {"auc": "AUC", "f1_score": "Macro-F1", "accuracy": "Accuracy"}[
        metric
    ]
    plotted_values = [0.0, 1.0]
    for col, selection in enumerate(selections):
        ds, norm, model, off_aug = (
            selection[k] for k in ("dataset_id", "norm", "model", "off_aug")
        )
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            for source, color in (("real", "#2878B5"), ("generated", "#E58332")):
                f = next(
                    f
                    for f in fits
                    if f["dataset_id"] == ds
                    and f["model"] == model
                    and f["norm"] == norm
                    and f["off_aug"] == off_aug
                    and f["method"] == method
                    and f["metric_name"] == metric
                    and f["source"] == source
                )
                obs, pred = f["observed"], f["predictions"]
                plotted_values.extend(o["observed_mean"] for o in obs)
                plotted_values.extend(
                    p[k]
                    for p in pred
                    for k in ("predicted", "ci_low", "ci_high")
                    if p[k] is not None
                )
                ax.plot(
                    [o["n"] for o in obs],
                    [o["observed_mean"] for o in obs],
                    "o",
                    color=color,
                    ms=2.6,
                )
                if draw_sd:
                    ax.errorbar(
                        [o["n"] for o in obs],
                        [o["observed_mean"] for o in obs],
                        yerr=[o["observed_std"] for o in obs],
                        fmt="none",
                        color=color,
                        linewidth=0.6,
                        capsize=1.5,
                        alpha=0.7,
                    )
                    plotted_values.extend(
                        o["observed_mean"] + sign * o["observed_std"]
                        for o in obs
                        for sign in (-1, 1)
                    )
                ax.plot(
                    [p["n"] for p in pred],
                    [p["predicted"] for p in pred],
                    color=color,
                    lw=1.2,
                )
                if f["interval_status"] == "ok":
                    ax.fill_between(
                        [p["n"] for p in pred],
                        [p["ci_low"] for p in pred],
                        [p["ci_high"] for p in pred],
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )
                if source == "real":
                    boundary = obs[-1]["n"]
                    ax.axvline(boundary, color=".55", linestyle=":", lw=0.8)
                    ax.axvspan(boundary, 1000, color=".8", alpha=0.13)
            ax.set_xlim(0, 1020)
            ax.grid(alpha=0.16)
            ax.set_xticks([0, 500, 1000])
            if col == 0:
                ax.set_ylabel(
                    ("LR" if method == "LOGIS" else method) + "\n" + metric_label
                )
            if row == 0:
                ax.set_title(
                    chr(65 + col)
                    + "  "
                    + selection["label"]
                    + " · "
                    + norm
                    + "\n"
                    + model.replace("CVAE1-", "cVAE 1:"),
                    fontsize=11,
                )
            if row == 4:
                ax.set_xlabel("Candidate subset size")
    # Preserve all displayed values, including intervals or SD bars outside [0,1].
    axes[0, 0].set_ylim(
        min(0.0, min(plotted_values) - 0.015), max(1.03, max(plotted_values) + 0.015)
    )
    fig.suptitle(
        "Agreement depends on the configuration\nSelected examples; "
        + ("no AE head; " if all(s["off_aug"] == "none" for s in selections) else "")
        + "common "
        + metric_label
        + " scale",
        fontsize=14,
        y=0.995,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=c, marker="o", label=l)
            for c, l in (("#2878B5", "Empirical"), ("#E58332", "Generated"))
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.023),
        frameon=False,
    )
    fig.text(
        0.5,
        0.008,
        "Points: observed means."
        + (" Error bars: ±1 draw SD." if draw_sd else "")
        + " Lines/bands: archived fit and pointwise 95% fitted-mean interval.\nDotted line: empirical limit. Raw: marker-filtered and log-transformed, without depth normalization.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        top=0.91, bottom=0.11, left=0.07, right=0.985, hspace=0.2, wspace=0.12
    )
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_configurations(rows, cohorts, selections, output_dir):
    """One AUC page per task: expose all configurations behind Figure 6."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    selected = [r for r in rows if r["metric_name"] == "auc"]
    signed_limit = math.ceil(max(abs(r["signed_gap"]) for r in selected) * 20) / 20
    absolute_limit = math.ceil(max(r["absolute_gap"] for r in selected) * 20) / 20
    with PdfPages(output_dir / "learning_curve_agreement_configurations.pdf") as pdf:
        for ds in cohorts:
            task = [r for r in selected if r["dataset_id"] == ds]
            configs = sorted(
                {(r["norm"], r["model"], r["off_aug"]) for r in task},
                key=lambda key: (
                    ("raw", "TC", "DESeq").index(key[0]),
                    int(key[1].split("-")[-1]),
                    key[2],
                ),
            )
            lookup = {
                (r["norm"], r["model"], r["off_aug"], r["method"]): r for r in task
            }
            labels = [
                f"{n} · {m.replace('CVAE1-', '1:')} · {'no head' if a == 'none' else 'AE head'}"
                for n, m, a in configs
            ]
            fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
            for ax, measure, title, cmap, low, high in zip(
                axes,
                ("signed_gap", "absolute_gap"),
                ("A  Signed AUC gap", "B  Absolute AUC gap"),
                ("RdBu_r", "YlOrRd"),
                (-signed_limit, 0),
                (signed_limit, absolute_limit),
            ):
                values = np.array(
                    [
                        [lookup[config + (method,)][measure] for method in METHODS]
                        for config in configs
                    ]
                )
                im = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
                ax.set_yticks(range(len(configs)), labels)
                ax.set_xticks(range(5), ["LR", "SVM", "KNN", "RF", "XGB"])
                ax.set_title(title, loc="left", fontweight="bold")
                for i in range(1, len(configs)):
                    if configs[i][0] != configs[i - 1][0]:
                        ax.axhline(i - 0.5, color="white", linewidth=2)
                for i in range(len(configs)):
                    for j in range(5):
                        value = values[i, j]
                        dark = (
                            abs(value) > signed_limit * 0.55
                            if measure == "signed_gap"
                            else value > absolute_limit * 0.6
                        )
                        ax.text(
                            j,
                            i,
                            f"{value:+.3f}"
                            if measure == "signed_gap"
                            else f"{value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if dark else "black",
                        )
                fig.colorbar(im, ax=ax, shrink=0.65, label="Average gap (AUC units)")
            cohort = ds.removeprefix("RNA_").split("_")[0]
            modality = "RNA-seq" if ds.startswith("RNA_") else "miRNA-seq"
            fig.suptitle(
                f"Supplementary Figure S1 · {cohort} {modality}\n{ds}", fontsize=13
            )
            pdf.savefig(fig)
            if ds in {s["dataset_id"] for s in selections}:
                fig.savefig(output_dir / f"learning_curve_agreement_{ds}.png", dpi=300)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Figure destination (default: figures/)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Table and summary destination (default: results/)",
    )
    args = parser.parse_args()
    main(args.output_dir, args.results_dir)
