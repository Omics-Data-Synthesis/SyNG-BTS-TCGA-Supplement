"""Shared observed-mean gap calculations and AUC-based configuration selection."""

import itertools
import math
import statistics
from collections import Counter

KEYS = ("dataset_id", "modality", "norm", "model", "off_aug", "method", "metric_name")
METHODS = ("LOGIS", "SVM", "KNN", "RF", "XGB")


def validate_archive(fits, n_sizes, n_draws, selections=None):
    """Check observed coverage and unique source series before publication export.

    Tables require every archived configuration; figures may specify their cases.
    """
    if not fits:
        raise ValueError("No archived learning curves available")
    if not all(
        len(f["observed"]) == int(n_sizes)
        and all(o["n_draws"] == int(n_draws) for o in f["observed"])
        for f in fits
    ):
        raise ValueError("Incomplete candidate-size or draw coverage")
    counts = Counter(tuple(f[k] for k in KEYS) + (f["source"],) for f in fits)
    configurations = (
        {tuple(f[k] for k in KEYS[:5]) for f in fits}
        if selections is None
        else {tuple(s[k] for k in KEYS[:5]) for s in selections}
    )
    for key in sorted(configurations):
        for method in METHODS:
            for metric in ("auc", "f1_score", "accuracy"):
                for source in ("real", "generated"):
                    if counts[key + (method, metric, source)] != 1:
                        raise ValueError(
                            f"Missing or duplicate curve: {key}, {method}, {metric}, {source}"
                        )


def select_agreement_configurations(rows):
    """Choose one complete configuration per task by mean AUC D, never |B|.

    Exact ties resolve lexicographically by the full configuration key.
    Selection uses unrounded values and is reused for both gaps and all metrics.
    """
    grouped = {}
    for row in rows:
        if row["metric_name"] == "auc":
            grouped.setdefault(tuple(row[k] for k in KEYS[:5]), []).append(row)
    tasks = {}
    for key, group in sorted(grouped.items()):
        if len(group) != len(METHODS) or {r["method"] for r in group} != set(METHODS):
            raise ValueError(f"Expected all five classifiers exactly once: {key}")
        if not all(math.isfinite(r[m]) for r in group for m in ("signed_gap", "absolute_gap")):
            raise ValueError(f"Nonfinite agreement values: {key}")
        tasks.setdefault(key[0], []).append({
            **dict(zip(KEYS[:5], key)),
            "mean_absolute_auc_gap": statistics.mean(r["absolute_gap"] for r in group),
            "max_absolute_auc_gap": max(r["absolute_gap"] for r in group),
        })
    if not tasks:
        raise ValueError("No AUC configurations available for selection")
    return [{**min(tasks[ds], key=lambda r: r["mean_absolute_auc_gap"]),
             "n_configurations_evaluated": len(tasks[ds]), "n_classifiers": len(METHODS)}
            for ds in sorted(tasks, key=lambda ds: (ds.startswith("RNA_"), ds))]


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


def comparison_rows(fits):
    """Compare each generated series with its matching archived empirical series."""
    index = {tuple(f[k] for k in KEYS) + (f["source"],): f for f in fits}
    rows = []
    for f in fits:
        if f["source"] == "generated":
            key = tuple(f[k] for k in KEYS)
            rows.append({**dict(zip(KEYS, key)), **compare(index[key + ("real",)], f)})
    return rows
