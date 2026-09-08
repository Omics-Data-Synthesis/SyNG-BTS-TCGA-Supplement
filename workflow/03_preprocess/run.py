# Stage 03: drop NA groups (binary only), log2(x+1), filter markers, reverse-log,
# TC / DESeq.
#
# miRNA: auto-detect mean threshold x* (Xinyi half-max-after-peak; std is
# recorded but not applied). RNA-seq COAD / SKCM: fixed mean and SD cutoffs
# from config/datasets.tsv. grouping=none RNA-seq (LAML / READ / PAAD) uses
# the same fixed mean>5, sd>2 filter but keeps every sample and writes no
# groups column.
#
# Reads group labels from stage 02 (binary rows) and the matching raw
# expression matrix from data/raw/. Writes filtered and normalised tables
# under data/processed/filtered/, diagnostic plots under data/processed/plots/,
# and results/marker_filtering.csv.
#
# This stage is deterministic and declares no seed.
#
# Run from the repository root:
#
#   uv run workflow/03_preprocess/run.py

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
DATASETS_TSV = ROOT / "config" / "datasets.tsv"
PREPROCESS_TSV = ROOT / "config" / "preprocess.tsv"
RESULTS_CSV = ROOT / "results" / "marker_filtering.csv"
META_COLS = {"Sample", "groups"}


def read_kv_tsv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {r["param"]: r["value"] for r in csv.DictReader(f, delimiter="\t")}


def read_datasets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def grouping_of(row: dict[str, str]) -> str:
    text = (row.get("grouping") or "binary").strip()
    return text or "binary"


def find_halfmax_threshold(
    col_mean,
    col_std,
    n_bins=80,
    max_peak_frac=0.70,
    max_search_frac=0.20,
):
    """Auto-detect x* (mean filter). Port of Xinyi_version/preprocess.py."""
    sort_idx = np.argsort(col_mean)
    xs, ys = col_mean[sort_idx], col_std[sort_idx]

    bins = np.array_split(np.arange(len(xs)), n_bins)
    bx = np.array([xs[b].mean() for b in bins])
    by_p75 = np.array([np.percentile(ys[b], 75) for b in bins])
    by_med = np.array([np.median(ys[b]) for b in bins])

    win = max(7, n_bins // 10)
    win = win + 1 if win % 2 == 0 else win
    poly = min(3, win - 1)
    sm_p75 = savgol_filter(by_p75, window_length=win, polyorder=poly)

    d1 = np.gradient(sm_p75)
    peak_search_end = max(1, int(max_peak_frac * len(d1)))
    peak_i = int(np.nanargmax(d1[:peak_search_end]))

    search_window = max(5, int(max_search_frac * n_bins))
    search_end = min(peak_i + search_window, len(d1))
    half_max = 0.5 * d1[peak_i]
    flat_after = np.where(d1[peak_i:search_end] <= half_max)[0]
    if len(flat_after):
        thresh_i = int(flat_after[0]) + peak_i
    else:
        thresh_i = peak_i + search_window // 2
    thresh_i = min(thresh_i, len(bx) - 1)
    x_star = float(bx[thresh_i])
    y_star = float(sm_p75[thresh_i])
    return x_star, y_star, bx, by_p75, by_med, sm_p75, d1, peak_i, thresh_i


def normalize_groups(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(lambda x: str(int(round(x))) if pd.notna(x) else "NA")
    out = series.astype("string")
    return out.where(out.notna() & (out != "") & (out != "<NA>"), "NA")


def tc_normalise(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=1)
    return counts.div(lib_sizes, axis=0) * 1e6


def deseq_normalise(counts: pd.DataFrame) -> pd.DataFrame:
    log_counts = np.log(counts.replace(0, np.nan))
    log_geo_mean = log_counts.mean(axis=0)
    valid = log_geo_mean.notna()
    if valid.sum() == 0:
        raise ValueError(
            "No valid features for DESeq size factor estimation "
            "(every feature has at least one zero)."
        )
    log_ratios = log_counts.loc[:, valid] - log_geo_mean[valid]
    size_factors = np.exp(log_ratios.median(axis=1))
    return counts.div(size_factors, axis=0)


def save_scatter(path, dataset, col_mean, col_std, keep_mask, x_star):
    fig, ax = plt.subplots(figsize=(6, 5))
    rm = ~keep_mask
    ax.scatter(
        col_mean[rm], col_std[rm],
        s=5, alpha=0.30, color="lightcoral", label="Removed", rasterized=True,
    )
    ax.scatter(
        col_mean[keep_mask], col_std[keep_mask],
        s=5, alpha=0.45, color="steelblue", label="Kept", rasterized=True,
    )
    ax.axvline(x_star, color="crimson", lw=1.5, linestyle="--", label=f"x* = {x_star:.2f}")
    ax.set_xlabel("Mean  (log2(x+1))", fontsize=11)
    ax.set_ylabel("Std   (log2(x+1))", fontsize=11)
    ax.set_title(
        f"{dataset}\nkept {int(keep_mask.sum())}/{len(keep_mask)}"
        f"  ({100 * keep_mask.mean():.1f}%)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_envelope(path, dataset, x_star, bx, by_p75, by_med, sm_p75, d1, peak_i, thresh_i):
    fig, ax2 = plt.subplots(figsize=(6, 5))
    ax2_r = ax2.twinx()
    ax2.plot(bx, by_med, color="steelblue", lw=1.2, alpha=0.5, label="Median std")
    ax2.plot(bx, by_p75, color="seagreen", lw=1.2, alpha=0.5, label="75th pct std")
    ax2.plot(bx, sm_p75, color="navy", lw=2.2, label="Smooth envelope")
    ax2.scatter(
        [bx[thresh_i]], [sm_p75[thresh_i]],
        color="crimson", zorder=6, s=70, label=f"x* = {x_star:.2f}",
    )
    ax2.axvline(x_star, color="crimson", lw=1.5, linestyle="--")
    ax2_r.plot(bx, d1, color="goldenrod", lw=1.2, linestyle=":", alpha=0.8, label="Slope")
    ax2_r.axhline(0.5 * d1[peak_i], color="goldenrod", lw=1, linestyle="--", alpha=0.6)
    ax2_r.scatter([bx[peak_i]], [d1[peak_i]], color="goldenrod", s=50, zorder=5)
    ax2_r.set_ylabel("d(smooth)/d(bin)", fontsize=9, color="goldenrod")
    ax2_r.tick_params(axis="y", labelcolor="goldenrod")
    ax2.set_xlabel("Mean  (log2(x+1))", fontsize=11)
    ax2.set_ylabel("Std (75th pct per bin)", fontsize=11)
    ax2.set_title(f"{dataset}\nEnvelope + slope  ->  x* = half-max-after-peak", fontsize=10)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def as_int(text: str) -> int | None:
    text = (text or "").strip()
    return int(text) if text else None


def as_float(text: str) -> float | None:
    text = (text or "").strip()
    return float(text) if text else None


def process_one(row: dict[str, str], cfg: dict[str, str]) -> dict | None:
    dataset_id = row["dataset_id"]
    method = row["preprocess"]
    if method == "skip":
        print(f"[{dataset_id}] skip", flush=True)
        return None

    expr_path = ROOT / cfg["raw_dir"] / row["modality"] / f"TCGA-{row['cohort']}.csv"
    if not expr_path.exists():
        raise FileNotFoundError(f"missing expression matrix: {expr_path}")

    print(f"[{dataset_id}] filtering ({method})", flush=True)
    expr = pd.read_csv(expr_path, index_col=0)
    ungrouped = grouping_of(row) == "none"
    df = expr.T
    df.insert(0, "Sample", df.index.astype(str))
    if ungrouped:
        meta_cols = ["Sample"]
    else:
        groups_path = ROOT / cfg["grouped_dir"] / f"{dataset_id}.csv"
        if not groups_path.exists():
            raise FileNotFoundError(f"missing group table (run stage 02 first): {groups_path}")
        groups = pd.read_csv(groups_path, dtype={"Sample": str, "groups": "string"})
        gmap = dict(zip(groups["Sample"], groups["groups"]))
        missing = [s for s in expr.columns if s not in gmap]
        if missing:
            raise ValueError(
                f"{dataset_id}: {len(missing)} expression barcodes have no group row "
                f"(e.g. {missing[0]})"
            )
        df["groups"] = [gmap[s] for s in expr.columns]
        meta_cols = [c for c in df.columns if c in META_COLS]
    numeric_cols = [c for c in df.columns if c not in META_COLS]

    n_samples_raw = len(df)
    n_features_raw = len(numeric_cols)

    if not ungrouped and "groups" in df.columns:
        g = df["groups"]
        keep = g.notna() & ~g.astype(str).isin(["", "<NA>", "nan", "None", "NA"])
        df = df.loc[keep].copy()
    n_samples_filt = len(df)

    df[numeric_cols] = np.log2(df[numeric_cols].astype(float) + 1)
    nz_cols = [c for c in numeric_cols if (df[c] != 0).any()]
    df = df[nz_cols + meta_cols].copy()

    col_mean = df[nz_cols].mean(axis=0).to_numpy()
    col_std = df[nz_cols].std(axis=0).to_numpy()

    n_bins = int(cfg["n_bins"])
    max_peak_frac = float(cfg["max_peak_frac"])
    max_search_frac = float(cfg["max_search_frac"])
    env = None
    if method == "auto_mean":
        env = find_halfmax_threshold(
            col_mean, col_std,
            n_bins=n_bins,
            max_peak_frac=max_peak_frac,
            max_search_frac=max_search_frac,
        )
        x_star, y_star = env[0], env[1]
        keep_mask = col_mean >= x_star
    elif method == "fixed_mean_sd":
        x_star = float(row["mean_threshold"])
        y_star = float(row["sd_threshold"])
        keep_mask = (col_mean >= x_star) & (col_std >= y_star)
        # Envelope plot still uses the auto curve as a diagnostic overlay.
        env = find_halfmax_threshold(
            col_mean, col_std,
            n_bins=n_bins,
            max_peak_frac=max_peak_frac,
            max_search_frac=max_search_frac,
        )
    else:
        raise ValueError(f"{dataset_id}: unknown preprocess method {method!r}")

    cols_final = [c for c, k in zip(nz_cols, keep_mask) if k]
    pct_rm = 100.0 * (n_features_raw - len(cols_final)) / n_features_raw if n_features_raw else 0.0

    df_filtered = df[cols_final + meta_cols].copy()
    if "groups" in df_filtered.columns:
        df_filtered["groups"] = normalize_groups(df_filtered["groups"])

    filtered_dir = ROOT / cfg["filtered_dir"]
    plot_dir = ROOT / cfg["plot_dir"]
    plot_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(df_filtered, filtered_dir / f"{dataset_id}_filtered.csv")

    save_scatter(
        plot_dir / f"{dataset_id}_scatter.png",
        dataset_id, col_mean, col_std, keep_mask, x_star,
    )
    if env is not None:
        _, _, bx, by_p75, by_med, sm_p75, d1, peak_i, thresh_i = env
        save_envelope(
            plot_dir / f"{dataset_id}_envelope.png",
            dataset_id, x_star, bx, by_p75, by_med, sm_p75, d1, peak_i, thresh_i,
        )

    meta_df = df_filtered[meta_cols].copy()
    log_data = df_filtered[cols_final]
    raw_counts = np.exp2(log_data) - 1
    raw_counts = raw_counts.clip(lower=0)

    tc_df = pd.concat([np.log2(tc_normalise(raw_counts) + 1), meta_df], axis=1)
    atomic_to_csv(tc_df, filtered_dir / f"{dataset_id}_filtered_TC.csv")
    deseq_status = "ok"
    try:
        deseq_df = pd.concat(
            [np.log2(deseq_normalise(raw_counts) + 1), meta_df], axis=1
        )
        atomic_to_csv(deseq_df, filtered_dir / f"{dataset_id}_filtered_DESeq.csv")
    except ValueError as e:
        deseq_status = f"SKIPPED - {e}"

    n_after_ref = as_int(row.get("n_after_na_ref", ""))
    n_nz_ref = as_int(row.get("markers_nonzero_ref", ""))
    n_fin_ref = as_int(row.get("markers_final_ref", ""))
    x_ref = as_float(row.get("x_star_ref", ""))
    y_ref = as_float(row.get("y_star_ref", ""))

    match_n = n_after_ref is None or n_samples_filt == n_after_ref
    match_nz = n_nz_ref is None or len(nz_cols) == n_nz_ref
    match_fin = n_fin_ref is None or len(cols_final) == n_fin_ref
    flags = []
    if not match_n:
        flags.append(f"N(after NA) {n_samples_filt} vs {n_after_ref}")
    if not match_nz:
        flags.append(f"nonzero {len(nz_cols)} vs {n_nz_ref}")
    if not match_fin:
        flags.append(f"final {len(cols_final)} vs {n_fin_ref}")
    status = "OK" if not flags else "MISMATCH: " + "; ".join(flags)
    print(
        f"  samples {n_samples_raw} -> {n_samples_filt}; "
        f"markers {n_features_raw} -> {len(nz_cols)} nonzero -> {len(cols_final)} final; "
        f"x*={x_star:.4f} y*={y_star:.4f}  {status}",
        flush=True,
    )
    print(f"    TC ok; DESeq {deseq_status}", flush=True)

    return {
        "dataset_id": dataset_id,
        "modality": row["modality"],
        "preprocess": method,
        "n_samples_raw": n_samples_raw,
        "n_samples_after_na": n_samples_filt,
        "n_samples_after_na_ref": n_after_ref,
        "n_markers_raw": n_features_raw,
        "n_markers_nonzero": len(nz_cols),
        "n_markers_nonzero_ref": n_nz_ref,
        "x_star": round(x_star, 4),
        "x_star_ref": x_ref,
        "y_star": round(y_star, 4),
        "y_star_ref": y_ref,
        "n_markers_final": len(cols_final),
        "n_markers_final_ref": n_fin_ref,
        "removed_pct": round(pct_rm, 1),
        "match_n_after_na": match_n,
        "match_markers_nonzero": match_nz,
        "match_markers_final": match_fin,
        "deseq": deseq_status,
    }


def main() -> None:
    if not DATASETS_TSV.exists():
        raise SystemExit("config/datasets.tsv not found; run from the repository root")
    cfg = read_kv_tsv(PREPROCESS_TSV)
    datasets = read_datasets(DATASETS_TSV)
    rows = []
    for ds in datasets:
        rec = process_one(ds, cfg)
        if rec is not None:
            rows.append(rec)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_bad = sum(
        not (r["match_n_after_na"] and r["match_markers_nonzero"] and r["match_markers_final"])
        for r in rows
    )
    print(
        f"wrote {RESULTS_CSV.relative_to(ROOT)}  "
        f"({len(rows)} datasets processed, {n_bad} count mismatches vs config refs)"
    )


if __name__ == "__main__":
    main()
