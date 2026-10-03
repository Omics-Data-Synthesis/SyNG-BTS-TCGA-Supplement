# Stage 06: SyntheSize sample-size learning curves.
#
# For every generate() combo that has a generated.csv, runs
# syng_bts.evaluate_sample_sizes on real (stage 03) and generated (stage 04)
# tables. Real metrics are computed once per (dataset, norm) and reused.
# Public package fits are archived for stages 07 and 08 without refitting.
#
# Processed data is already log2(x+1), so apply_log comes from
# config/synthesize.tsv (false). Only completed, compatible caches are reused.
# grouping=none datasets are skipped: classifiers need a binary label.
#
# Run from the repository root:
#
#   uv run workflow/06_synthesize/run.py
#   uv run workflow/06_synthesize/run.py --workers 4

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import sys
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
from syng_bts import __version__, evaluate_sample_sizes, fit_sample_sizes

sys.path.insert(0, str(ROOT / "workflow"))
from synthesize_results import (
    ARCHIVE_NAME, PACKAGE_VERSION, POINT_COLUMNS, FIT_COLUMNS,
    cache_valid, file_hash, fit_row, input_identity, point_rows, provenance,
    write_completion, write_json,
)

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


def build_fits(metrics_real, metrics_gen, keys, methods, metrics):
    """Fit once in the package; archive its default (observed-size) grid."""
    records = []
    for source, table in (("real", metrics_real), ("generated", metrics_gen)):
        if table is None or table.empty:
            continue
        table = table[table["method"].isin(methods)]
        if table.empty:
            continue
        for metric in metrics:
            for fit in fit_sample_sizes(table, metric_name=metric).values():
                payload = fit.to_dict()
                # The study's "model" is the generator, not the learning-curve equation.
                payload["curve_model"] = payload.pop("model")
                records.append({**keys, "source": source, **payload})
    return records


def build_points_and_fits(metrics_real, metrics_gen, keys, methods, metrics):
    """Tabular views of the same package results (also useful interactively)."""
    records = build_fits(metrics_real, metrics_gen, keys, methods, metrics)
    return tables_from_fits(records)


def tables_from_fits(records):
    return (
        pd.DataFrame([row for fit in records for row in point_rows(fit)], columns=POINT_COLUMNS),
        pd.DataFrame([fit_row(fit) for fit in records], columns=FIT_COLUMNS),
    )


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

        completion_path = model_dir / "complete.json"
        gen_path = generated_dir / dataset_id / norm["label"] / model / tag / "generated.csv"
        real_path = filtered_dir / f"{dataset_id}{norm['suffix']}"
        identity = dict(worker_cfg["provenance"], keys=keys,
                        real_input=input_identity(real_path), generated_input=input_identity(gen_path))
        required = {"points.csv", "fits.csv", "fits.json", "metrics_generated.csv"}
        if skip_existing and cache_valid(completion_path, identity, required):
            records.append(JobResult(index, "skip", label, "SKIPPED (validated)", "cached", keys=keys))
            continue

        # Remove only this job's obsolete derived artifacts before a retry.
        for name in ("complete.json", "points.csv", "fits.csv", "fits.json", "metrics_generated.csv"):
            (model_dir / name).unlink(missing_ok=True)
        if not gen_path.exists():
            records.append(JobResult(index, "missing", label, keys=keys))
            continue

        real_key = (dataset_id, norm["label"])
        t0 = time.time()
        try:
            # Only this worker owns real_key, so metrics_real.csv cannot race.
            if real_key not in real_cache:
                real_csv = cache_dir / dataset_id / norm["label"] / "metrics_real.csv"
                real_path = filtered_dir / f"{dataset_id}{norm['suffix']}"
                real_completion = real_csv.with_suffix(".json")
                real_identity = dict(worker_cfg["provenance"], real_input=input_identity(real_path))
                reuse_real = skip_existing and cache_valid(real_completion, real_identity, {real_csv.name})
                if not reuse_real:
                    real_csv.unlink(missing_ok=True)
                    real_completion.unlink(missing_ok=True)
                if reuse_real:
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
                            write_completion(real_completion, real_identity, [real_csv])
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

            numerical = build_fits(real_cache.get(real_key), metrics_gen, keys, methods, metric_names)
            points, fits = tables_from_fits(numerical)
            write_df(points_path, points)
            write_df(model_dir / "fits.csv", fits)
            write_json(model_dir / "fits.json", numerical)
            if status == "ok" and real_cache.get(real_key) is None:
                status = "ok_generated_only"
            files = [points_path, model_dir / "fits.csv", model_dir / "fits.json"]
            if metrics_gen is not None:
                files.append(model_dir / "metrics_generated.csv")
            write_completion(completion_path, identity, files, status=status,
                             real_status=real_status.get(real_key, ""))
            secs = round(time.time() - t0, 1)
            records.append(
                JobResult(
                    index, "ok", label, status, real_status.get(real_key, ""), secs,
                    keys=keys,
                )
            )
        except Exception as exc:
            for name in ("complete.json", "points.csv", "fits.csv", "fits.json", "metrics_generated.csv"):
                (model_dir / name).unlink(missing_ok=True)
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


def job_directory(job, cache_dir):
    ds, norm, model, aug = job
    head = (aug.get("AE_head_num") or "").strip()
    tag = off_aug_tag(as_optional_str(aug.get("off_aug", "")), int(head) if head else None)
    return cache_dir / ds["dataset_id"] / norm["label"] / tag / model


def publish_results(jobs, records, cache_dir, producing):
    """Publish only this invocation's completed jobs; the JSON archive is last."""
    valid = {r.index for r in records if r.category in {"ok", "skip"}}
    directories = [job_directory(job, cache_dir) for i, job in enumerate(jobs, 1) if i in valid]
    numerical, completions = [], []
    for directory in directories:
        completion = json.loads((directory / "complete.json").read_text())
        if not cache_valid(directory / "complete.json", completion["identity"], {"points.csv", "fits.csv", "fits.json"}):
            raise ValueError(f"Incomplete job: {directory}")
        numerical.extend(json.loads((directory / "fits.json").read_text()))
        completions.append(completion)
    for filename, destination, columns in (("points.csv", CURVES_CSV, POINT_COLUMNS),
                                           ("fits.csv", FITS_CSV, FIT_COLUMNS)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as out:
            csv.writer(out).writerow(columns)
            for directory in directories:
                with (directory / filename).open(newline="") as src:
                    next(src, None)
                    shutil.copyfileobj(src, out)
        tmp.replace(destination)
    write_json(CURVES_CSV.parent / ARCHIVE_NAME, dict(
        provenance=producing, jobs=completions, counts=dict(Counter(r.category for r in records)),
        tables={p.name: file_hash(p) for p in (CURVES_CSV, FITS_CSV)}, fits=numerical,
    ))


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
    if __version__ != PACKAGE_VERSION:
        raise SystemExit(f"Expected syng-bts=={PACKAGE_VERSION}, installed {__version__}; run uv sync --locked")
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

    producing = provenance(ROOT, SEED)
    # An interrupted/failed run cannot leave an apparently completed old aggregate.
    for path in (CURVES_CSV, FITS_CSV, CURVES_CSV.parent / ARCHIVE_NAME):
        path.unlink(missing_ok=True)
    groups = group_jobs(jobs)
    worker_cfg = {
        "provenance": producing,
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
                    "skip": "skipped (validated)",
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

    if not counts["fail"]:
        publish_results(jobs, records, cache_dir, producing)

    print(
        f"\ndone: {counts['ok']} ok, {counts['skip']} skipped validated, "
        f"{counts['missing']} missing, {counts['fail']} failed; "
        f"curves → {CURVES_CSV.relative_to(ROOT)}"
    )
    if counts["fail"]:
        raise SystemExit(f"{counts['fail']} combo(s) failed")


if __name__ == "__main__":
    main()
