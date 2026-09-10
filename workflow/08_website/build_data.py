"""Build ``website/plots/**/data.json`` from stage 06 SyntheSize metrics.

Reads the per-draw tables written by ``workflow/06_synthesize/run.py``:

    data/processed/curves/<dataset_id>/<norm>/metrics_real.csv
    data/processed/curves/<dataset_id>/<norm>/<offaug_tag>/<model>/metrics_generated.csv

and writes one compact JSON per configuration:

    website/plots/<data_type>/<subtype>/<group_label>/<norm>/<offaug>/<param>/data.json

Fitted curves and 95% confidence bands are recomputed here with a weighted
inverse power-law fit (mirroring ``syng_bts.synthesize._fit_curve``). Stage 06
``points.csv`` / ``fits.csv`` / ``results/curves.csv`` are not used: those come
from an unweighted dense-grid fit and do not match the reference figures.

Default is incremental: only configurations present in the cache are
overwritten. Existing JSON (including bootstrap miRNA files) is left in place.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import norm

from dataset_keys import grouping_of, split_dataset_id

REPO = Path(__file__).resolve().parents[2]
DATASETS_TSV = REPO / "config" / "datasets.tsv"
CACHE_ROOT = REPO / "data" / "processed" / "curves"
DEST_ROOT = REPO / "website" / "plots"

REAL_FILE = "metrics_real.csv"
GEN_FILE = "metrics_generated.csv"
OUTPUT_FILE = "data.json"

CLASSIFIERS = ["LOGIS", "SVM", "KNN", "RF", "XGB"]
METRICS = ["f1_score", "accuracy", "auc"]
SOURCES = ["real", "generated"]

NORM_LABELS = {"raw": "none"}
OFFAUG_LABELS = {"offaug_none": "none", "offaug_AE_head_2": "AE"}

VALUE_DIGITS = 5


def _power_law(x, a, b, c):
    """Inverse power-law function: ``(1 - a) - b * x^c``."""
    return (1 - a) - (b * (x**c))


def _power_law_gradient(x, a, b, c):
    x_power_c = x**c
    return np.array([-1.0, -x_power_c, -b * x_power_c * np.log(x)])


def _prediction_variance(x, params, covariance):
    gradient = _power_law_gradient(x, *params)
    return float(gradient @ covariance @ gradient.T)


def fit_learning_curve(ns: np.ndarray, ys: np.ndarray) -> dict:
    """Weighted IPLF fit + delta-method 95% CI at the observed ``ns``.

    Returns dict with keys fit_ok, ci_ok, and (when applicable) predicted,
    ci_low, ci_high arrays aligned with ``ns``. Mirrors
    ``syng_bts.synthesize._fit_curve`` including its CI-omission rules.
    """
    result = {"fit_ok": False, "ci_ok": False}
    if len(np.unique(ns)) < 3:
        return result

    weights = np.arange(1, len(ns) + 1) / len(ns)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            popt, pcov = curve_fit(
                _power_law,
                ns,
                ys,
                p0=[0, 1, -0.5],
                sigma=1 / np.sqrt(weights),
                maxfev=50000,
            )
    except (RuntimeError, ValueError):
        return result

    if not np.isfinite(popt).all():
        return result
    predicted = _power_law(ns, *popt)
    if not np.isfinite(predicted).all():
        return result

    result["fit_ok"] = True
    result["predicted"] = predicted

    optimizer_warnings = [w for w in caught if issubclass(w.category, OptimizeWarning)]
    if optimizer_warnings or not np.isfinite(pcov).all():
        return result

    pred_var = np.array([_prediction_variance(float(x), popt, pcov) for x in ns])
    if not np.isfinite(pred_var).all() or (pred_var < 0).any():
        return result

    t = norm.ppf(0.975)
    result["ci_ok"] = True
    result["ci_low"] = predicted - t * np.sqrt(pred_var)
    result["ci_high"] = predicted + t * np.sqrt(pred_var)
    return result


def _mean_table(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-(n, method) mean and draw std — mirrors plot_sample_sizes' aggregation."""
    return (
        metrics.groupby(["total_size", "method"])
        .agg(
            observed_mean=(metric, "mean"),
            observed_std=(metric, "std"),
            n_draws=(metric, "size"),
        )
        .reset_index()
        .rename(columns={"total_size": "n"})
        .sort_values(["method", "n"])
    )


def load_metrics(metrics_csv: Path) -> dict:
    """Aggregate per-draw metrics and fit curves.

    Returns series[method][metric] = {obs_n, obs_y, obs_sd[, fit_n, fit_y
    [, fit_lo, fit_hi]]}.
    """
    metrics = pd.read_csv(metrics_csv)
    series: dict[str, dict[str, dict]] = {c: {} for c in CLASSIFIERS}
    for metric in METRICS:
        if metric not in metrics.columns:
            continue
        means = _mean_table(metrics, metric)
        for clf in CLASSIFIERS:
            sub = means[means["method"] == clf]
            if sub.empty:
                continue
            ns = sub["n"].to_numpy()
            obs_y = sub["observed_mean"].to_numpy()
            obs_sd = sub["observed_std"].fillna(0.0).to_numpy()
            entry = {
                "obs_n": [int(n) for n in ns],
                "obs_y": [round(float(v), VALUE_DIGITS) for v in obs_y],
                "obs_sd": [round(float(v), VALUE_DIGITS) for v in obs_sd],
            }
            fit = fit_learning_curve(ns.astype(float), obs_y)
            if fit["fit_ok"]:
                entry["fit_n"] = [int(n) for n in ns]
                entry["fit_y"] = [round(float(v), VALUE_DIGITS) for v in fit["predicted"]]
                if fit["ci_ok"]:
                    entry["fit_lo"] = [round(float(v), VALUE_DIGITS) for v in fit["ci_low"]]
                    entry["fit_hi"] = [round(float(v), VALUE_DIGITS) for v in fit["ci_high"]]
            series[clf][metric] = entry
    return series


def completeness_warnings(config_id: str, series: dict) -> list[str]:
    warnings_out = []
    for clf in CLASSIFIERS:
        for metric in METRICS:
            for source in SOURCES:
                s = series[clf][metric][source]
                if s is None or "obs_n" not in s:
                    warnings_out.append(f"{config_id}: missing observed {clf}/{metric}/{source}")
                elif "fit_n" not in s:
                    warnings_out.append(f"{config_id}: fit failed {clf}/{metric}/{source}")
                elif "fit_lo" not in s:
                    warnings_out.append(f"{config_id}: CI omitted {clf}/{metric}/{source}")
    return warnings_out


def write_config(dest_dir: Path, series: dict) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {"classifiers": CLASSIFIERS, "metrics": METRICS, "series": series}
    out = dest_dir / OUTPUT_FILE
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return out.stat().st_size


def load_dataset_index() -> dict[str, dict[str, str]]:
    if not DATASETS_TSV.exists():
        raise SystemExit("config/datasets.tsv not found; run from the repository root")
    with DATASETS_TSV.open(newline="") as f:
        return {row["dataset_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def build_from_cache() -> tuple[int, int, list[str]]:
    """Overwrite JSON for every metrics pair found in the stage 06 cache."""
    datasets = load_dataset_index()
    configs = 0
    total_bytes = 0
    warnings_out: list[str] = []

    if not CACHE_ROOT.is_dir():
        warnings_out.append(f"cache not found, skipped: {CACHE_ROOT.relative_to(REPO)}")
        return configs, total_bytes, warnings_out

    real_cache: dict[Path, dict] = {}
    for gen_csv in sorted(CACHE_ROOT.rglob(GEN_FILE)):
        rel = gen_csv.relative_to(CACHE_ROOT).parts
        # <dataset_id>/<norm>/<offaug_tag>/<model>/metrics_generated.csv
        if len(rel) != 5:
            warnings_out.append(f"unexpected depth, skipped: {gen_csv.relative_to(REPO)}")
            continue
        dataset_id, norm, offaug_dir, param, _ = rel
        ds = datasets.get(dataset_id)
        if ds is None:
            warnings_out.append(f"unknown dataset_id, skipped: {dataset_id}")
            continue
        if grouping_of(ds) == "none":
            warnings_out.append(f"grouping=none, skipped: {dataset_id}")
            continue

        real_csv = CACHE_ROOT / dataset_id / norm / REAL_FILE
        if not real_csv.is_file():
            warnings_out.append(f"{REAL_FILE} missing for {gen_csv.relative_to(REPO)}")
            continue

        if real_csv not in real_cache:
            real_cache[real_csv] = load_metrics(real_csv)
        real_series = real_cache[real_csv]
        gen_series = load_metrics(gen_csv)

        data_type, subtype, group_label = split_dataset_id(dataset_id, ds["modality"])
        norm_label = NORM_LABELS.get(norm, norm)
        offaug_label = OFFAUG_LABELS.get(offaug_dir, offaug_dir)

        series = {
            clf: {
                metric: {
                    "real": real_series.get(clf, {}).get(metric),
                    "generated": gen_series.get(clf, {}).get(metric),
                }
                for metric in METRICS
            }
            for clf in CLASSIFIERS
        }

        config_id = f"{data_type}/{dataset_id}/{norm_label}/{offaug_label}/{param}"
        warnings_out.extend(completeness_warnings(config_id, series))

        dest_dir = DEST_ROOT / data_type / subtype / group_label / norm_label / offaug_label / param
        total_bytes += write_config(dest_dir, series)
        configs += 1

    return configs, total_bytes, warnings_out


def main() -> None:
    configs, nbytes, all_warnings = build_from_cache()
    dest = DEST_ROOT.relative_to(REPO)
    print(
        f"Wrote {configs} {OUTPUT_FILE} files under {dest}/ "
        f"({nbytes / 1e6:.1f} MB)."
    )
    if all_warnings:
        ci_omitted = [w for w in all_warnings if "CI omitted" in w]
        others = [w for w in all_warnings if "CI omitted" not in w]
        print(f"{len(ci_omitted)} curves with CI omitted (covariance unusable).")
        if others:
            print(f"{len(others)} other warnings:")
            for w in others:
                print("  ", w)


if __name__ == "__main__":
    main()
