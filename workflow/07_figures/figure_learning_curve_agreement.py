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

SELECTIONS = ROOT / "config" / "learning_curve_figures.tsv"
from agreement_data import (
    KEYS, METHODS, compare, comparison_rows, select_agreement_configurations, validate_archive,
)

def empirical_sample_sizes(fits):
    """Read full empirical cohort sizes from the archived observed size grids."""
    sizes = {}
    for fit in fits:
        if fit["source"] != "real":
            continue
        ds = fit["dataset_id"]
        n = max(point["n"] for point in fit["observed"])
        if not math.isfinite(n) or n <= 0 or int(n) != n:
            raise ValueError(f"Invalid empirical sample size for {ds}: {n}")
        if ds in sizes and sizes[ds] != n:
            raise ValueError(f"Inconsistent empirical sample sizes for {ds}")
        sizes[ds] = int(n)
    return sizes


def agreement_overviews(rows, sample_sizes):
    """Align both overviews by modality, empirical size, then dataset identifier."""
    import numpy as np

    selections = select_agreement_configurations(rows)
    missing = {s["dataset_id"] for s in selections} - sample_sizes.keys()
    if missing:
        raise ValueError(f"Missing empirical sample sizes: {sorted(missing)}")
    selections.sort(key=lambda s: (
        ("mirna", "rnaseq").index(s["modality"]),
        sample_sizes[s["dataset_id"]], s["dataset_id"],
    ))
    cohorts = [s["dataset_id"] for s in selections]
    chosen = {tuple(s[k] for k in KEYS[:5]) for s in selections}
    auc = [r for r in rows if r["metric_name"] == "auc"]
    lookup = {(r["dataset_id"], r["method"]): r for r in auc
              if tuple(r[k] for k in KEYS[:5]) in chosen}
    selected = [np.array([[lookup[ds, method][measure] for method in METHODS]
                          for ds in cohorts])
                for measure in ("signed_gap", "absolute_gap")]
    median = [np.array([[statistics.median(r[measure] for r in auc
                         if r["dataset_id"] == ds and r["method"] == method)
                         for method in METHODS] for ds in cohorts])
              for measure in ("signed_gap", "absolute_gap")]
    return cohorts, selections, selected, median


def agreement_color_maps(rows):
    """Full-grid limits across all metrics, shared by every agreement display."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    signed_limit = math.ceil(max(abs(r["signed_gap"]) for r in rows) * 20) / 20 or 0.05
    absolute_limit = math.ceil(max(r["absolute_gap"] for r in rows) * 20) / 20 or 0.05
    return [ScalarMappable(norm=Normalize(-signed_limit, signed_limit), cmap="RdBu_r"),
            ScalarMappable(norm=Normalize(0, absolute_limit), cmap="YlOrRd")]


def selected_fidelity(selections, root=ROOT):
    """Match full configuration keys and average unrounded stage 05 draws."""
    import pandas as pd
    from fidelity_data import KEYS as FIDELITY_KEYS
    from fidelity_data import read_fidelity, select_configurations

    settings = pd.DataFrame(selections)
    settings["AE_head_num"] = settings["AE_head_num"].astype(int)
    if settings.duplicated(FIDELITY_KEYS).any():
        raise ValueError("Duplicate selected fidelity configurations")
    selected = select_configurations(
        read_fidelity(root), settings, "learning-curve fidelity"
    )
    metrics = ["source_ari", "ccc_log10_pvalue"]
    if not all(math.isfinite(v) for v in selected[metrics].to_numpy().flat):
        raise ValueError("Missing or nonfinite selected fidelity metric")
    result = []
    for selection in settings.to_dict("records"):
        rows = selected[
            (selected[FIDELITY_KEYS] == pd.Series(selection)[FIDELITY_KEYS]).all(axis=1)
        ]
        result.append({
            **selection,
            "n_fidelity_draws": len(rows),
            "mean_source_ari": statistics.mean(rows.source_ari),
            "mean_ccc_log10_pvalue": statistics.mean(rows.ccc_log10_pvalue),
        })
    return result


def case_support(real, generated, full_support=False):
    """Shared or union support, derived from each source's observed endpoints."""
    r, g = real["observed"], generated["observed"]
    if full_support:
        return min(r[0]["n"], g[0]["n"]), max(r[-1]["n"], g[-1]["n"])
    return max(r[0]["n"], g[0]["n"]), min(r[-1]["n"], g[-1]["n"])


def main(output_dir=None, results_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "figures"
    results_dir = Path(results_dir) if results_dir is not None else ROOT / "results"
    archive = json.loads((ROOT / "results/learning_curve_fits.json").read_text())
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
    validate_archive(fits, config["n_sizes"], config["n_draws"], selections)
    fidelity = selected_fidelity(selections)
    rows = comparison_rows(fits)
    output_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    with (results_dir / "tables" / "learning_curve_agreement.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (results_dir / "tables" / "selected_learning_curve_fidelity.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fidelity[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fidelity)
    summary = {
        "source_archive": "results/learning_curve_fits.json",
        "completed_jobs": archive["counts"],
        "comparisons": len(rows),
        "selected_fidelity": fidelity,
    }
    overview_selections = select_agreement_configurations(rows)
    chosen_keys = {tuple(s[k] for k in KEYS[:5]) for s in overview_selections}
    chosen_rows = [r for r in rows if tuple(r[k] for k in KEYS[:5]) in chosen_keys]
    for filename, records in (
        ("learning_curve_selected_configurations.csv", overview_selections),
        ("learning_curve_selected_agreement.csv", chosen_rows),
    ):
        with (results_dir / "tables" / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    summary["overview_selection"] = {
        "criterion": "Minimum mean absolute AUC gap across all five classifiers, one configuration per task",
        "tie_break": "Lexicographic full configuration key; unrounded values",
        "interpretation": "Retrospective descriptive selection within the evaluated grid",
        "configurations": overview_selections,
        "metrics_at_auc_selected_configurations": {
            metric: {"comparisons": sum(r["metric_name"] == metric for r in chosen_rows),
                     "median_absolute_gap": statistics.median(
                         r["absolute_gap"] for r in chosen_rows if r["metric_name"] == metric)}
            for metric in ("auc", "f1_score", "accuracy")
        },
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
    plot(fits, rows, selections, fidelity, output_dir)
    print(f"Wrote learning-curve agreement figures to {output_dir}")
    print(f"Wrote {len(rows)} comparisons and summaries to {results_dir}")


def plot(fits, rows, selections, fidelity, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # About 6.2 pt when the widest figure is placed at 180 mm.
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    sample_sizes = empirical_sample_sizes(fits)
    cohorts, overview_selections, selected, median = agreement_overviews(rows, sample_sizes)
    gap_colors = agreement_color_maps(rows)
    for matrices, is_selected in ((selected, True), (median, False)):
        fig = draw_overview(cohorts, overview_selections, matrices, gap_colors, is_selected, sample_sizes)
        stem = "learning_curve_agreement" if is_selected else "learning_curve_agreement_median"
        fig.savefig(output_dir / f"{stem}.png", dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)

    from matplotlib.backends.backend_pdf import PdfPages

    # Full-support counterparts are one PDF, ordered AUC / macro-F1 / accuracy.
    with PdfPages(output_dir / "learning_curve_comparison_full_support.pdf") as pdf:
        for metric in ("auc", "f1_score", "accuracy"):
            fig = draw_cases(fits, metric, selections, fidelity, gap_colors=gap_colors)
            destination = output_dir / f"learning_curve_comparison_{metric}"
            fig.savefig(destination.with_suffix(".png"), dpi=300)
            fig.savefig(destination.with_suffix(".pdf"))
            plt.close(fig)
            fig = draw_cases(fits, metric, selections, fidelity, full_support=True, gap_colors=gap_colors)
            pdf.savefig(fig)
            fig.savefig(output_dir / f"learning_curve_comparison_{metric}_full_support.png", dpi=300)
            plt.close(fig)
    plot_configurations(rows, cohorts, selections, output_dir, sample_sizes)


def aligned_row_labels(ax, rows, fontsize):
    """Set proportional-font row labels in fixed columns with aligned separators."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.text import Text

    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(size=fontsize)
    def width(text):
        return renderer.get_text_width_height_descent(text, font, ismath=False)[0] * 72 / ax.figure.dpi

    widths = [max(width(row[col]) for row in rows) for col in range(len(rows[0]))]
    separator_width = width(" · ")
    total = sum(widths) + separator_width * (len(widths) - 1)
    height = fontsize * 1.25
    ax.set_yticks(range(len(rows)), [""] * len(rows))
    for y, row in enumerate(rows):
        area = DrawingArea(total, height, 0, 0)
        x = 0
        for col, value in enumerate(row):
            area.add_artist(Text(x, height / 2, value, fontsize=fontsize, va="center"))
            x += widths[col]
            if col < len(widths) - 1:
                area.add_artist(Text(x, height / 2, " · ", fontsize=fontsize, va="center"))
                x += separator_width
        ax.add_artist(AnnotationBbox(
            area, (0, y), xycoords=ax.get_yaxis_transform(),
            xybox=(-7, 0), boxcoords="offset points", box_alignment=(1, .5),
            frameon=False, pad=0, annotation_clip=False,
        ))


def draw_overview(cohorts, selections, matrices, gap_colors, is_selected, sample_sizes):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5 if is_selected else 11, 9), layout="constrained")
    labels = [
        ["RNA" if ds.startswith("RNA_") else "miRNA",
         ds.removeprefix("RNA_").split("_")[0], f"n={sample_sizes[ds]}"]
        for ds in cohorts
    ]
    if is_selected:
        labels = [label + [s["norm"], s["model"].replace("CVAE1-", "1:")]
                  for label, s in zip(labels, selections)]
    for i, (ax, data, title, mapping) in enumerate(zip(
        axes,
        matrices,
        ("A  Signed AUC gap", "B  Absolute AUC gap"),
        gap_colors,
    )):
        im = ax.imshow(data, cmap=mapping.cmap, norm=mapping.norm, aspect="auto")
        ax.set_xticks(range(5), ["LR", "SVM", "KNN", "RF", "XGB"])
        if i == 0 or not is_selected:
            aligned_row_labels(ax, labels, fontsize=11 if is_selected else 10)
        else:
            ax.set_yticks([])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        if not is_selected:
            ax.tick_params(labelsize=10)
        ax.axhline(
            sum(not ds.startswith("RNA_") for ds in cohorts) - 0.5,
            color="white",
            linewidth=2,
        )
        bar = fig.colorbar(
            im, ax=ax, shrink=0.65, label="Average gap (AUC units)" if is_selected
            else "Median across configurations (AUC units)"
        )
        set_heatmap_gap_ticks(bar, mapping)
        if not is_selected:
            bar.ax.tick_params(labelsize=10)
            bar.ax.yaxis.label.set_size(10)
    fig.suptitle(
        ("Empirical–generated AUC agreement · selected configuration per task\n"
         "Selection: lowest mean absolute AUC gap across all five classifiers"
         if is_selected else
         "Empirical–generated AUC agreement\nMedian across all evaluated configurations"),
        fontsize=12,
    )
    fig.supxlabel("Observed means over shared subset-size support.\n" + (
        "Retrospective selection; the same configuration supplies both panels."
        if is_selected else "Nine configurations per miRNA task; 18 per RNA-seq task."),
        fontsize=11 if is_selected else 10)
    return fig


def set_heatmap_gap_ticks(bar, mapping):
    """Include the true endpoints, matching the comparison color scales."""
    limit = mapping.norm.vmax
    fractions = range(-3, 4) if mapping.norm.vmin < 0 else range(4)
    ticks = [limit * fraction / 3 for fraction in fractions]
    bar.set_ticks(ticks, labels=["0" if value == 0 else f"{value:.2f}" for value in ticks])


def draw_cases(fits, metric, selections, fidelity, full_support=False, gap_colors=None):
    """Draw archived observations/predictions; no dense re-evaluation or refitting."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from matplotlib.ticker import MaxNLocator

    if gap_colors is None:
        # Standalone callers use the same full-archive scale calculation as plot().
        index = {tuple(f[k] for k in KEYS) + (f["source"],): f for f in fits}
        rows = [{**dict(zip(KEYS, key[:-1])),
                 **compare(index[key[:-1] + ("real",)], generated)}
                for key, generated in index.items() if key[-1] == "generated"]
        gap_colors = agreement_color_maps(rows)

    fig, axes = plt.subplots(
        5, len(selections), figsize=(3.15 * len(selections), 10.65),
        sharex="col", sharey=True, squeeze=False,
    )
    metric_label = {"auc": "AUC", "f1_score": "Macro-F1", "accuracy": "Accuracy"}[metric]
    plotted_values = [0.0, 1.0]
    for col, selection in enumerate(selections):
        selected = [f for f in fits if all(f[k] == selection[k] for k in KEYS[:5])
                    and f["metric_name"] == metric]
        fid = next(f for f in fidelity if all(f[k] == selection[k] for k in KEYS[:5])
                   and int(f["AE_head_num"]) == int(selection["AE_head_num"]))
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            real = next(f for f in selected if f["method"] == method and f["source"] == "real")
            generated = next(f for f in selected if f["method"] == method and f["source"] == "generated")
            shared_lo, shared_hi = case_support(real, generated)
            lo, hi = case_support(real, generated, full_support)
            if full_support:
                # Either source can have the larger observed range (BRCA: empirical).
                for f, color in ((real, "#2878B5"), (generated, "#E58332")):
                    first, last = f["observed"][0]["n"], f["observed"][-1]["n"]
                    if first < shared_lo:
                        ax.axvspan(first, shared_lo, color=color, alpha=0.07, linewidth=0)
                    if last > shared_hi:
                        ax.axvspan(shared_hi, last, color=color, alpha=0.07, linewidth=0)
                ax.axvline(shared_hi, color=".5", linestyle=":", lw=0.8)
            for f, color in ((real, "#2878B5"), (generated, "#E58332")):
                obs, pred = f["observed"], f["predictions"]
                # Keep bracketing archived points for line segments crossing the window.
                # No new observations or inverse-power predictions are computed.
                for o in obs:
                    if lo <= o["n"] <= hi:
                        plotted_values.extend(o["observed_mean"] + sign * o["observed_std"]
                                              for sign in (-1, 1))
                plotted_values.extend(p[k] for p in pred
                                      for k in ("predicted", "ci_low", "ci_high")
                                      if p[k] is not None)
                ax.errorbar(
                    [o["n"] for o in obs], [o["observed_mean"] for o in obs],
                    yerr=[o["observed_std"] for o in obs], fmt="o", color=color,
                    ms=2.5, linewidth=0.6, capsize=1.3, zorder=3,
                )
                ax.plot([p["n"] for p in pred], [p["predicted"] for p in pred],
                        color=color, lw=1.15, zorder=2)
                if f["interval_status"] == "ok":
                    ax.fill_between([p["n"] for p in pred], [p["ci_low"] for p in pred],
                                    [p["ci_high"] for p in pred], color=color,
                                    alpha=0.09, linewidth=0, zorder=1)
            # Exact bounds distinguish the shared range from any source-only support.
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(alpha=0.15)
            if metric == "auc":
                ax.axhline(0.5, color=".65", linestyle=":", linewidth=0.6)
            gap = compare(real, generated)
            badges = []
            for label, measure, mapping in zip(
                ("B", "D"), ("signed_gap", "absolute_gap"), gap_colors
            ):
                value = gap[measure]
                color = mapping.to_rgba(value)
                luminance = sum(c * w for c, w in zip(color[:3], (0.2126, 0.7152, 0.0722)))
                number = f"{value:+.3f}" if label == "B" else f"{value:.3f}"
                badges.append(TextArea(f"{label} {number}", textprops=dict(
                    fontsize=10.7, color="white" if luminance < 0.45 else "black",
                    bbox=dict(facecolor=color, edgecolor="none", pad=2),
                )))
            ax.add_artist(AnchoredOffsetbox(
                loc="lower right", child=HPacker(children=badges, align="baseline", pad=0, sep=8),
                bbox_to_anchor=(0.97, 0.04), bbox_transform=ax.transAxes,
                frameon=False, pad=0, borderpad=0,
            ))
            if col == 0:
                ax.set_ylabel(("LR" if method == "LOGIS" else method) + "\n" + metric_label)
            if row == 0:
                aug = "AE head" if selection["off_aug"] == "AE_head" else "No AE head"
                ari = fid["mean_source_ari"]
                ari_text = f"{ari:.4f}" if abs(ari) >= 0.001 else f"{ari:.2g}"
                ax.set_title(
                    f"{chr(65 + col)}  {selection['label']}",
                    fontsize=11, fontweight="bold", pad=40,
                )
                ax.annotate(
                    f"{selection['norm']} · {selection['model'].replace('CVAE1-', 'cVAE 1:')} · {aug}\n"
                    f"Mean source ARI: {ari_text}",
                    xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11,
                )
            if row == 4:
                ax.set_xlabel("Candidate subset size")
    axes[0, 0].set_ylim(min(0.0, min(plotted_values) - 0.015),
                        max(1.03, max(plotted_values) + 0.015))
    fig.suptitle("Empirical–generated learning-curve comparisons\n"
                 + ("Full observed source ranges" if full_support else "Shared observed subset-size ranges"),
                 fontsize=13, y=0.995)
    fig.legend(handles=[Line2D([0], [0], color=c, marker="o", label=l)
                        for c, l in (("#2878B5", "Empirical"), ("#E58332", "Generated"))],
               loc="center", ncol=2, bbox_to_anchor=(0.5, 0.103), frameon=False)
    for i, (left, mapping, label) in enumerate(zip(
        (0.085, 0.665), gap_colors, ("B: signed average gap", "D: absolute average gap")
    )):
        cax = fig.add_axes([left, 0.097, 0.25, 0.012])
        bar = fig.colorbar(mapping, cax=cax, orientation="horizontal")
        if i == 0:
            bar.set_ticks([mapping.norm.vmin, 0, mapping.norm.vmax],
                          labels=[f"{mapping.norm.vmin:.2f}", "0", f"+{mapping.norm.vmax:.2f}"])
        else:
            ticks = [mapping.norm.vmax * fraction / 3 for fraction in range(4)]
            bar.set_ticks(ticks, labels=[f"{value:.2f}" for value in ticks])
        bar.set_label(f"{label} ({metric_label} units)", fontsize=10.7)
        bar.ax.tick_params(labelsize=10.7, length=2)
    support_note = ("Pale blue/orange: empirical-only/generated-only support. Dotted boundary: upper shared limit."
                    if full_support else "Each column uses its shared observed range; x-axis ranges differ between tasks.")
    # Detailed definitions are in the manuscript captions; keep the in-figure key brief.
    fig.text(0.5, 0.010,
             "Points: mean ±1 draw SD. Bands: pointwise 95% fitted-mean intervals.\n" + support_note,
             ha="center", fontsize=10.7)
    # Allow the larger scale labels and footer to breathe within the same page size.
    fig.subplots_adjust(top=0.866, bottom=0.170, left=0.065, right=0.975,
                        hspace=0.20, wspace=0.13)
    return fig


def plot_configurations(rows, cohorts, selections, output_dir, sample_sizes):
    """One AUC page per task: expose all configurations behind Figure 6."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    selected = [r for r in rows if r["metric_name"] == "auc"]
    gap_colors = agreement_color_maps(rows)
    signed_limit = gap_colors[0].norm.vmax
    absolute_limit = gap_colors[1].norm.vmax
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
                [n, m.replace("CVAE1-", "1:") + ("†" if a == "AE_head" else "")]
                for n, m, a in configs
            ]
            fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
            for ax, measure, title, mapping in zip(
                axes,
                ("signed_gap", "absolute_gap"),
                ("A  Signed AUC gap", "B  Absolute AUC gap"),
                gap_colors,
            ):
                values = np.array(
                    [
                        [lookup[config + (method,)][measure] for method in METHODS]
                        for config in configs
                    ]
                )
                im = ax.imshow(values, cmap=mapping.cmap, norm=mapping.norm, aspect="auto")
                aligned_row_labels(ax, labels, fontsize=11)
                ax.set_xticks(range(5), ["LR", "SVM", "KNN", "RF", "XGB"])
                ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
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
                            fontsize=10.2,
                            color="white" if dark else "black",
                        )
                bar = fig.colorbar(im, ax=ax, shrink=0.65, label="Average gap (AUC units)")
                set_heatmap_gap_ticks(bar, mapping)
            cohort = ds.removeprefix("RNA_").split("_")[0]
            modality = "RNA-seq" if ds.startswith("RNA_") else "miRNA-seq"
            fig.suptitle(
                f"{cohort} {modality} · n={sample_sizes[ds]}\n{ds}", fontsize=13
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
