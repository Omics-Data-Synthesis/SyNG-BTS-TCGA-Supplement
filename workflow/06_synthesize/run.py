# Stage 06: SyntheSize sample-size learning curves.
#
# For every generate() combo that has a generated.csv, runs
# syng_bts.evaluate_sample_sizes on real (stage 03) and generated (stage 04)
# tables. Real metrics are computed once per (dataset, norm) and reused.
# IPLF fits are stored so stage 07 can draw the curves without refitting.
#
# Processed data is already log2(x+1), so apply_log comes from
# config/synthesize.tsv (false). Skip-if-missing / skip-if-exists match stage 04.
# grouping=none datasets are skipped: classifiers need a binary label.
#
# Run from the repository root:
#
#   uv run workflow/06_synthesize/run.py
#   uv run workflow/06_synthesize/run.py --workers 4

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import shutil
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "data" / ".numba_cache"))
# XGBoost 3 + libomp SIGSEGVs on macOS when OpenMP oversubscribes inside
# evaluate_sample_sizes (XGB after sklearn). Must be set before native imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings(
    "ignore",
    message="l1_ratios parameter is only used when penalty is 'elasticnet'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime, curve_fit
from scipy.stats import norm
from syng_bts import evaluate_sample_sizes

SEED = 42

DATASETS_TSV = ROOT / "config" / "datasets.tsv"
MODELS_TSV = ROOT / "config" / "generate_models.tsv"
NORMS_TSV = ROOT / "config" / "generate_norms.tsv"
OFF_AUG_TSV = ROOT / "config" / "generate_off_aug.tsv"
SYNTHESIZE_TSV = ROOT / "config" / "synthesize.tsv"
META_COLS = ("groups", "samples", "Sample", "sample", "Unnamed: 0")

CURVES_CSV = ROOT / "results" / "curves.csv"
FITS_CSV = ROOT / "results" / "curve_fits.csv"
LOG_CSV = ROOT / "results" / "synthesize_log.csv"


def read_kv_tsv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {r["param"]: r["value"] for r in csv.DictReader(f, delimiter="\t")}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def grouping_of(row: dict[str, str]) -> str:
    text = (row.get("grouping") or "binary").strip()
    return text or "binary"


def as_bool(text: str) -> bool:
    return (text or "").strip().lower() in {"1", "true", "yes"}


def as_optional_str(text: str) -> str | None:
    text = (text or "").strip()
    return text or None


def off_aug_tag(off_aug: str | None, ae_head_num: int | None) -> str:
    if not off_aug:
        return "offaug_none"
    if off_aug == "AE_head":
        return f"offaug_AE_head_{ae_head_num}"
    return f"offaug_{off_aug}"


def load_off_aug(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_tsv(path):
        by_key.setdefault((row["modality"], grouping_of(row)), []).append(row)
    return by_key


def planned_jobs(datasets, models_by_key, norms, off_aug_by_key):
    jobs = []
    for ds in datasets:
        if (ds.get("preprocess") or "").strip() == "skip":
            continue
        if grouping_of(ds) == "none":
            continue
        key = (ds["modality"], grouping_of(ds))
        models = models_by_key.get(key, [])
        aug_rows = off_aug_by_key.get(key, [{"off_aug": "", "AE_head_num": ""}])
        for norm in norms:
            for model in models:
                for aug in aug_rows:
                    jobs.append((ds, norm, model, aug))
    return jobs


def load_features_groups(path: Path):
    df = pd.read_csv(path)
    if "groups" not in df.columns:
        return None, None
    raw = df["groups"]
    if len(raw.dropna().unique()) != 2:
        return None, None
    if pd.api.types.is_numeric_dtype(raw):
        groups = raw.apply(lambda x: str(int(round(x))) if pd.notna(x) else "NA")
    else:
        groups = raw.astype(str)
    drop = [c for c in df.columns if c in META_COLS or not pd.api.types.is_numeric_dtype(df[c])]
    data = df.drop(columns=drop)
    return data.reset_index(drop=True), groups.reset_index(drop=True)


def allocate_stratified(total_size: int, group_counts: dict[str, int]) -> dict[str, int]:
    total_available = sum(group_counts.values())
    groups = list(group_counts.keys())
    raw = {g: total_size * group_counts[g] / total_available for g in groups}
    alloc = {g: min(int(np.floor(raw[g])), group_counts[g]) for g in groups}
    remaining = total_size - sum(alloc.values())
    if remaining > 0:
        order = sorted(groups, key=lambda g: raw[g] - alloc[g], reverse=True)
        while remaining > 0:
            progressed = False
            for g in order:
                if alloc[g] < group_counts[g]:
                    alloc[g] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
    return alloc


def make_sample_sizes(groups: pd.Series, n_sizes: int, min_class: int) -> list[int]:
    group_counts = {str(g): int(c) for g, c in groups.value_counts().items()}
    total = sum(group_counts.values())
    n_classes = len(group_counts)

    def feasible(s: int) -> bool:
        if s < min_class * n_classes or s > total:
            return False
        alloc = allocate_stratified(s, group_counts)
        return all(c >= min_class for c in alloc.values())

    candidates = sorted({int(round(x)) for x in np.linspace(min_class * 2, total, 400)})
    feasible_sizes = [s for s in candidates if feasible(s)]
    if not feasible_sizes:
        return []
    if feasible_sizes[-1] != total and feasible(total):
        feasible_sizes = sorted(set(feasible_sizes + [total]))
    idx = np.linspace(0, len(feasible_sizes) - 1, min(n_sizes, len(feasible_sizes)))
    return sorted({feasible_sizes[int(round(i))] for i in idx})


def power_law(x, a, b, c):
    return (1 - a) - b * (x ** c)


def fit_iplf(ns: np.ndarray, ys: np.ndarray):
    try:
        popt, pcov = curve_fit(power_law, ns, ys, p0=[0, 1, -0.5], maxfev=50000)
        return popt, pcov, True
    except (RuntimeError, ValueError):
        return None, None, False


def predict_with_ci(popt, pcov, xs: np.ndarray):
    pred = power_law(xs, *popt)
    eps = np.sqrt(np.finfo(float).eps)
    jac = np.empty((len(xs), len(popt)))
    for i, x in enumerate(xs):
        jac[i] = approx_fprime([x], lambda z: power_law(z[0], *popt), eps)
    var = np.sum((jac @ pcov) * jac, axis=1)
    sd = np.sqrt(np.clip(var, 0, None))
    t = norm.ppf(0.975)
    return pred, pred - t * sd, pred + t * sd


def mean_table(metrics: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    sub = metrics[metrics["method"] == method]
    agg = (
        sub.groupby("total_size")
        .agg(observed_mean=(metric, "mean"), observed_std=(metric, "std"), n_draws=(metric, "size"))
        .reset_index()
        .rename(columns={"total_size": "n"})
        .sort_values("n")
    )
    agg["observed_std"] = agg["observed_std"].fillna(0.0)
    return agg


def build_points_and_fits(metrics_real, metrics_gen, keys, methods, metrics):
    point_rows, fit_rows = [], []
    sources = [("real", metrics_real), ("generated", metrics_gen)]
    for source, table in sources:
        if table is None or table.empty:
            continue
        for method in methods:
            if method not in set(table["method"]):
                continue
            for metric in metrics:
                mt = mean_table(table, method, metric)
                if mt.empty:
                    continue
                ns = mt["n"].to_numpy(dtype=float)
                ys = mt["observed_mean"].to_numpy(dtype=float)
                for _, r in mt.iterrows():
                    point_rows.append(
                        {
                            **keys,
                            "source": source,
                            "method": method,
                            "metric": metric,
                            "kind": "observed",
                            "n": int(r["n"]),
                            "value": float(r["observed_mean"]),
                            "observed_std": float(r["observed_std"]),
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "n_draws": int(r["n_draws"]),
                        }
                    )
                popt, pcov, ok = fit_iplf(ns, ys) if len(ns) >= 3 else (None, None, False)
                fit_rows.append(
                    {
                        **keys,
                        "source": source,
                        "method": method,
                        "metric": metric,
                        "fit_ok": ok,
                        "a": float(popt[0]) if ok else np.nan,
                        "b": float(popt[1]) if ok else np.nan,
                        "c": float(popt[2]) if ok else np.nan,
                    }
                )
                if ok:
                    grid = np.linspace(ns.min(), ns.max(), 100)
                    pred, lo, hi = predict_with_ci(popt, pcov, grid)
                    for xg, pg, lg, hg in zip(grid, pred, lo, hi):
                        point_rows.append(
                            {
                                **keys,
                                "source": source,
                                "method": method,
                                "metric": metric,
                                "kind": "fitted",
                                "n": float(xg),
                                "value": float(pg),
                                "observed_std": np.nan,
                                "ci_low": float(lg),
                                "ci_high": float(hg),
                                "n_draws": np.nan,
                            }
                        )
    return pd.DataFrame(point_rows), pd.DataFrame(fit_rows)


def write_df(path: Path, df: pd.DataFrame) -> None:
    """Write a dataframe atomically so interrupted jobs are safe to retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def append_log(row: dict) -> None:
    fields = [
        "dataset_id",
        "modality",
        "norm",
        "model",
        "off_aug",
        "status",
        "real_status",
        "seconds",
        "seed",
    ]
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def group_jobs(jobs):
    """Keep jobs sharing real-data metrics in the same worker process."""
    grouped = {}
    for index, job in enumerate(jobs, start=1):
        ds, norm, _, _ = job
        key = (ds["dataset_id"], norm["label"])
        grouped.setdefault(key, []).append((index, *job))
    return list(grouped.values())


class JobResult(NamedTuple):
    """Small, picklable worker result; the parent owns logging and progress."""

    index: int
    category: str
    label: str
    status: str = ""
    real_status: str = ""
    seconds: float = 0
    traceback_text: str | None = None
    keys: dict | None = None


def run_group(group, worker_cfg):
    """Run one (dataset, norm) group, reusing its real-data evaluation."""
    methods = worker_cfg["methods"]
    metric_names = worker_cfg["metric_names"]
    n_draws = worker_cfg["n_draws"]
    n_sizes = worker_cfg["n_sizes"]
    min_class = worker_cfg["min_class"]
    apply_log = worker_cfg["apply_log"]
    skip_existing = worker_cfg["skip_existing"]
    filtered_dir = Path(worker_cfg["filtered_dir"])
    generated_dir = Path(worker_cfg["generated_dir"])
    cache_dir = Path(worker_cfg["cache_dir"])

    real_cache = {}
    real_status = {}
    records = []

    for index, ds, norm, model, aug in group:
        off_aug = as_optional_str(aug.get("off_aug", ""))
        ae_head_raw = (aug.get("AE_head_num") or "").strip()
        ae_head_num = int(ae_head_raw) if ae_head_raw else None
        tag = off_aug_tag(off_aug, ae_head_num)
        dataset_id = ds["dataset_id"]
        keys = dict(
            dataset_id=dataset_id,
            modality=ds["modality"],
            norm=norm["label"],
            model=model,
            off_aug=off_aug or "none",
        )
        label = f"{dataset_id}  {norm['label']}  {model}  {tag}"
        model_dir = cache_dir / dataset_id / norm["label"] / tag / model
        points_path = model_dir / "points.csv"

        if skip_existing and points_path.exists():
            records.append(
                JobResult(index, "skip", label, "SKIPPED (exists)", "cached", keys=keys)
            )
            continue

        gen_path = generated_dir / dataset_id / norm["label"] / model / tag / "generated.csv"
        if not gen_path.exists():
            records.append(JobResult(index, "missing", label))
            continue

        real_key = (dataset_id, norm["label"])
        t0 = time.time()
        try:
            # Only this worker owns real_key, so metrics_real.csv cannot race.
            if real_key not in real_cache:
                real_csv = cache_dir / dataset_id / norm["label"] / "metrics_real.csv"
                real_path = filtered_dir / f"{dataset_id}{norm['suffix']}"
                if skip_existing and real_csv.exists():
                    real_cache[real_key] = pd.read_csv(real_csv)
                    real_status[real_key] = "cached"
                elif not real_path.exists():
                    real_cache[real_key] = None
                    real_status[real_key] = "real_file_missing"
                else:
                    data_r, groups_r = load_features_groups(real_path)
                    if data_r is None:
                        real_cache[real_key] = None
                        real_status[real_key] = "real_no_groups"
                    else:
                        sizes_r = make_sample_sizes(groups_r, n_sizes, min_class)
                        if not sizes_r:
                            real_cache[real_key] = None
                            real_status[real_key] = "real_no_feasible_sizes"
                        else:
                            metrics_real = evaluate_sample_sizes(
                                data=data_r,
                                sample_sizes=sizes_r,
                                groups=groups_r,
                                n_draws=n_draws,
                                methods=methods,
                                apply_log=apply_log,
                                random_seed=SEED,
                                verbose="silent",
                            )
                            write_df(real_csv, metrics_real)
                            real_cache[real_key] = metrics_real
                            real_status[real_key] = "ok"

            data_g, groups_g = load_features_groups(gen_path)
            if data_g is None:
                status = "generated_no_groups"
                metrics_gen = None
            else:
                sizes_g = make_sample_sizes(groups_g, n_sizes, min_class)
                if not sizes_g:
                    status = "generated_no_feasible_sizes"
                    metrics_gen = None
                else:
                    metrics_gen = evaluate_sample_sizes(
                        data=data_g,
                        sample_sizes=sizes_g,
                        groups=groups_g,
                        n_draws=n_draws,
                        methods=methods,
                        apply_log=apply_log,
                        random_seed=SEED,
                        verbose="silent",
                    )
                    write_df(model_dir / "metrics_generated.csv", metrics_gen)
                    status = "ok"

            points, fits = build_points_and_fits(
                real_cache.get(real_key), metrics_gen, keys, methods, metric_names
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            if not points.empty:
                write_df(points_path, points)
            if not fits.empty:
                write_df(model_dir / "fits.csv", fits)
            if status == "ok" and real_cache.get(real_key) is None:
                status = "ok_generated_only"
            secs = round(time.time() - t0, 1)
            records.append(
                JobResult(
                    index, "ok", label, status, real_status.get(real_key, ""), secs,
                    keys=keys,
                )
            )
        except Exception as exc:
            secs = round(time.time() - t0, 1)
            records.append(
                JobResult(
                    index, "fail", label, f"ERROR: {exc}",
                    real_status.get(real_key, ""), secs, traceback.format_exc(),
                    keys=keys,
                )
            )
    return records


def completed_groups(groups, worker_cfg, workers):
    """Yield completed groups; serial mode follows the same worker path."""

    if workers == 1:
        for group in groups:
            yield run_group(group, worker_cfg)
        return

    # Spawn avoids inheriting initialized sklearn/XGBoost native runtimes.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(run_group, group, worker_cfg) for group in groups]
        for future in as_completed(futures):
            yield future.result()


def aggregate_job_files(jobs, cache_dir, filename, destination):
    """Rebuild an aggregate in plan order without loading it all into memory."""
    sources = []
    for ds, norm, model, aug in jobs:
        off_aug = as_optional_str(aug.get("off_aug", ""))
        ae_head_raw = (aug.get("AE_head_num") or "").strip()
        ae_head_num = int(ae_head_raw) if ae_head_raw else None
        source = (
            cache_dir
            / ds["dataset_id"]
            / norm["label"]
            / off_aug_tag(off_aug, ae_head_num)
            / model
            / filename
        )
        if source.exists():
            sources.append(source)

    if not sources:
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    # Workers never append to shared files; the parent publishes one atomic rebuild.
    with tmp.open("w", newline="") as out:
        for source_index, source in enumerate(sources):
            with source.open(newline="") as src:
                if source_index:
                    next(src, None)
                shutil.copyfileobj(src, out)
    tmp.replace(destination)
    return len(sources)


def parse_args():
    parser = argparse.ArgumentParser(description="Run stage 06 SyntheSize evaluations.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of independent (dataset, norm) worker processes (default: 1)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if not DATASETS_TSV.exists():
        raise SystemExit("config/datasets.tsv not found; run from the repository root")

    cfg = read_kv_tsv(SYNTHESIZE_TSV)
    datasets = read_tsv(DATASETS_TSV)
    norms = read_tsv(NORMS_TSV)
    off_aug_by_key = load_off_aug(OFF_AUG_TSV)
    models_by_key: dict[tuple[str, str], list[str]] = {}
    for row in read_tsv(MODELS_TSV):
        models_by_key.setdefault((row["modality"], grouping_of(row)), []).append(row["model"])

    methods = [m.strip() for m in cfg["methods"].split(",") if m.strip()]
    metric_names = [m.strip() for m in cfg["metrics"].split(",") if m.strip()]
    n_draws = int(cfg["n_draws"])
    n_sizes = int(cfg["n_sizes"])
    min_class = int(cfg["min_class_subset"])
    apply_log = as_bool(cfg["apply_log"])
    skip_existing = as_bool(cfg.get("skip_existing", "true"))
    filtered_dir = ROOT / cfg["filtered_dir"]
    generated_dir = ROOT / cfg["generated_dir"]
    cache_dir = ROOT / "data" / "processed" / "curves"

    jobs = planned_jobs(datasets, models_by_key, norms, off_aug_by_key)
    np.random.seed(SEED)
    print(f"SEED             : {SEED}")
    print(f"Jobs             : {len(jobs)}")
    print(f"methods          : {methods}")
    print(f"n_draws / n_sizes: {n_draws} / {n_sizes}")
    print(f"apply_log        : {apply_log}")
    print(f"workers          : {args.workers}\n", flush=True)

    groups = group_jobs(jobs)
    worker_cfg = {
        "methods": methods,
        "metric_names": metric_names,
        "n_draws": n_draws,
        "n_sizes": n_sizes,
        "min_class": min_class,
        "apply_log": apply_log,
        "skip_existing": skip_existing,
        "filtered_dir": str(filtered_dir),
        "generated_dir": str(generated_dir),
        "cache_dir": str(cache_dir),
    }
    # Results arrive in completion order for prompt progress reporting. They are
    # sorted back into plan order before logs and aggregates are finalized.
    records = []
    completed = 0
    for batch in completed_groups(groups, worker_cfg, args.workers):
        records.extend(batch)
        for record in sorted(batch, key=lambda r: r.index):
            completed += 1
            category = record.category
            if category == "ok":
                detail = f"{record.status}  ({record.seconds}s)"
            else:
                detail = {
                    "skip": "skipped (exists)",
                    "missing": "skipped (generated missing)",
                }.get(category, record.status)
            print(
                f"[completed {completed}/{len(jobs)}] {record.label}: {detail}",
                flush=True,
            )

    records.sort(key=lambda r: r.index)

    counts = Counter(record.category for record in records)
    for record in records:
        if record.traceback_text:
            print(record.traceback_text, end="")
        if record.keys is not None:
            append_log(
                {
                    **record.keys,
                    "status": record.status,
                    "real_status": record.real_status,
                    "seconds": record.seconds,
                    "seed": SEED,
                }
            )

    aggregate_job_files(jobs, cache_dir, "points.csv", CURVES_CSV)
    aggregate_job_files(jobs, cache_dir, "fits.csv", FITS_CSV)

    print(
        f"\ndone: {counts['ok']} ok, {counts['skip']} skipped exists, "
        f"{counts['missing']} missing, {counts['fail']} failed; "
        f"curves → {CURVES_CSV.relative_to(ROOT)}"
    )
    if counts["fail"]:
        raise SystemExit(f"{counts['fail']} combo(s) failed")


if __name__ == "__main__":
    main()
