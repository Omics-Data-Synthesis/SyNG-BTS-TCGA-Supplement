# Stage 04: train SyNG-BTS CVAE models and write synthetic samples.
#
# Reads filtered tables from stage 03 (already log2(x+1)). miRNA and RNA-seq
# share apply_log=False, new_size, and the generate() call. Models come from
# config/generate_models.tsv keyed by (modality, grouping); learning_rate /
# epoch / patience from config/generate_train.tsv. grouping=none RNA uses VAE
# with groups=None. The seed is SEED below. Binary RNA also sweeps off_aug
# in config/generate_off_aug.tsv (none vs AE_head, AE_head_num=2).
#
# Skip-if-exists: a run whose generated.csv is already on disk is not retrained.
# A failing run is recorded in results/training_time.csv and does not stop
# the rest; rerun the script to retry it.
#
# Run from the repository root:
#
#   uv run workflow/04_generate/run.py

from __future__ import annotations

import csv
import os
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Must be set before importing matplotlib / syng_bts (umap → numba cache).
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "data" / ".numba_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from syng_bts.core import generate

SEED = 42

DATASETS_TSV = ROOT / "config" / "datasets.tsv"
GENERATE_TSV = ROOT / "config" / "generate.tsv"
MODELS_TSV = ROOT / "config" / "generate_models.tsv"
NORMS_TSV = ROOT / "config" / "generate_norms.tsv"
TRAIN_TSV = ROOT / "config" / "generate_train.tsv"
OFF_AUG_TSV = ROOT / "config" / "generate_off_aug.tsv"
RESULTS_CSV = ROOT / "results" / "training_time.csv"
META_COLS = ("groups", "samples", "Sample", "sample", "Unnamed: 0")

SUMMARY_FIELDS = [
    "dataset_id",
    "modality",
    "norm",
    "model",
    "n_features",
    "n_real_samples",
    "n_generated",
    "epochs_trained",
    "elapsed_s",
    "learning_rate",
    "epoch",
    "early_stop_patience",
    "off_aug",
    "AE_head_num",
    "seed",
    "status",
    "output",
]


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


def load_features_and_groups(csv_path: Path) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(csv_path)
    groups: pd.Series | None = None
    if "groups" in df.columns:
        raw = df["groups"]
        unique_vals = raw.dropna().unique()
        if len(unique_vals) != 2:
            raise ValueError(
                f"{csv_path.name}: groups has {len(unique_vals)} unique value(s); "
                "conditional models need exactly 2 (ungrouped tables must omit groups)"
            )
        if pd.api.types.is_numeric_dtype(raw):
            groups = raw.apply(lambda x: str(int(round(x))) if pd.notna(x) else "NA")
        else:
            groups = raw.astype(str)
        groups.name = "groups"

    feature_df = df.select_dtypes(include=np.number)
    for col in META_COLS:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])
    id_col = next((c for c in ("Sample", "samples") if c in df.columns), None)
    if id_col is not None:
        feature_df.index = df[id_col]
        if not feature_df.index.is_unique:
            feature_df = feature_df.reset_index(drop=True)
    return feature_df, groups


def attach_groups(gen_df: pd.DataFrame, result) -> pd.DataFrame:
    out = gen_df.copy()
    gen_groups = getattr(result, "generated_groups", None)
    if gen_groups is not None:
        out["groups"] = gen_groups.values if hasattr(gen_groups, "values") else gen_groups
    return out


def save_heatmaps(result, save_dir: Path) -> None:
    for which in ("generated", "original"):
        try:
            fig = result.plot_heatmap(which=which, log_scale=True)
            fig.savefig(save_dir / f"heatmap_{which}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"      heatmap_{which} skipped: {exc}")


def append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def already_done(save_dir: Path) -> bool:
    return (save_dir / "generated.csv").exists()


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


def planned_jobs(
    datasets: list[dict[str, str]],
    models_by_key: dict[tuple[str, str], list[str]],
    norms: list[dict[str, str]],
    off_aug_by_key: dict[tuple[str, str], list[dict[str, str]]],
) -> list[tuple[dict[str, str], dict[str, str], str, dict[str, str]]]:
    jobs = []
    for ds in datasets:
        if (ds.get("preprocess") or "").strip() == "skip":
            continue
        key = (ds["modality"], grouping_of(ds))
        models = models_by_key.get(key, [])
        aug_rows = off_aug_by_key.get(key, [{"off_aug": "", "AE_head_num": ""}])
        for norm in norms:
            for model in models:
                for aug in aug_rows:
                    jobs.append((ds, norm, model, aug))
    return jobs


def load_train_cfg(path: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(path)
    by_mod = {r["modality"]: r for r in rows}
    for mod in ("mirna", "rnaseq"):
        if mod not in by_mod:
            raise SystemExit(f"{path} missing training row for {mod}")
    return by_mod


def run_one(
    ds: dict[str, str],
    norm: dict[str, str],
    model: str,
    aug: dict[str, str],
    cfg: dict[str, str],
    train: dict[str, str],
    filtered_dir: Path,
    output_dir: Path,
) -> dict:
    dataset_id = ds["dataset_id"]
    norm_label = norm["label"]
    off_aug = as_optional_str(aug.get("off_aug", ""))
    ae_head_raw = (aug.get("AE_head_num") or "").strip()
    ae_head_num = int(ae_head_raw) if ae_head_raw else None
    tag = off_aug_tag(off_aug, ae_head_num)
    src = filtered_dir / f"{dataset_id}{norm['suffix']}"
    save_dir = output_dir / dataset_id / norm_label / model / tag
    rel_out = str(save_dir.relative_to(ROOT))

    base = {
        "dataset_id": dataset_id,
        "modality": ds["modality"],
        "norm": norm_label,
        "model": model,
        "n_features": "",
        "n_real_samples": "",
        "n_generated": 0,
        "epochs_trained": "",
        "elapsed_s": 0,
        "learning_rate": train["learning_rate"],
        "epoch": train["epoch"],
        "early_stop_patience": train["early_stop_patience"],
        "off_aug": off_aug or "none",
        "AE_head_num": ae_head_num if ae_head_num is not None else "",
        "seed": SEED,
        "output": rel_out,
    }

    if not src.exists():
        return {**base, "status": "SKIPPED (file missing)"}
    if as_bool(cfg.get("skip_existing", "true")) and already_done(save_dir):
        return {**base, "status": "SKIPPED (exists)"}

    data, groups = load_features_and_groups(src)
    base["n_features"] = data.shape[1]
    base["n_real_samples"] = data.shape[0]
    save_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = dict(
        data=data,
        groups=groups,
        name=f"{dataset_id}_{norm_label}_{model}_{tag}",
        model=model,
        new_size=int(cfg["new_size"]),
        apply_log=as_bool(cfg["apply_log"]),
        batch_frac=float(cfg["batch_frac"]),
        learning_rate=float(train["learning_rate"]),
        epoch=int(train["epoch"]),
        early_stop_patience=int(train["early_stop_patience"]),
        val_ratio=float(cfg.get("val_ratio", "0.2")),
        off_aug=off_aug,
        CVAE_wide_network=as_bool(cfg.get("CVAE_wide_network", "false")),
        random_seed=SEED,
        output_dir=str(save_dir),
        verbose=cfg.get("verbose", "minimal") or "minimal",
    )
    if off_aug == "AE_head" and ae_head_num is not None:
        gen_kwargs["AE_head_num"] = ae_head_num

    t0 = time.time()
    try:
        result = generate(**gen_kwargs)
        elapsed = time.time() - t0
        gen_df = attach_groups(result.generated_data, result)
        tmp = save_dir / "generated.csv.tmp"
        gen_df.to_csv(tmp, index=False)
        tmp.replace(save_dir / "generated.csv")
        if as_bool(cfg.get("save_heatmaps", "true")):
            save_heatmaps(result, save_dir)
        epochs = result.metadata.get("epochs_trained", "")
        print(
            f"    generated={tuple(result.generated_data.shape)}  "
            f"epochs={epochs}  time={elapsed:.1f}s",
            flush=True,
        )
        return {
            **base,
            "n_generated": int(result.generated_data.shape[0]),
            "epochs_trained": epochs,
            "elapsed_s": round(elapsed, 1),
            "status": "ok",
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    ERROR after {elapsed:.1f}s: {exc}", flush=True)
        traceback.print_exc()
        return {
            **base,
            "elapsed_s": round(elapsed, 1),
            "status": f"ERROR: {exc}",
        }


def main() -> None:
    if not DATASETS_TSV.exists():
        raise SystemExit("config/datasets.tsv not found; run from the repository root")

    cfg = read_kv_tsv(GENERATE_TSV)
    datasets = read_tsv(DATASETS_TSV)
    norms = read_tsv(NORMS_TSV)
    train_by_mod = load_train_cfg(TRAIN_TSV)
    off_aug_by_key = load_off_aug(OFF_AUG_TSV)
    models_by_key: dict[tuple[str, str], list[str]] = {}
    for row in read_tsv(MODELS_TSV):
        models_by_key.setdefault((row["modality"], grouping_of(row)), []).append(row["model"])

    filtered_dir = ROOT / cfg["filtered_dir"]
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = planned_jobs(datasets, models_by_key, norms, off_aug_by_key)
    print(f"SEED             : {SEED}")
    print(f"Jobs             : {len(jobs)}")
    for key in sorted(models_by_key):
        mod, grouping = key
        t = train_by_mod[mod]
        augs = [
            off_aug_tag(
                as_optional_str(r.get("off_aug", "")),
                int(r["AE_head_num"]) if (r.get("AE_head_num") or "").strip() else None,
            )
            for r in off_aug_by_key.get(key, [])
        ]
        print(
            f"{mod}/{grouping:8s} : models={models_by_key[key]}  "
            f"lr={t['learning_rate']}  epoch={t['epoch']}  "
            f"patience={t['early_stop_patience']}  off_aug={augs}"
        )
    print(f"Norms            : {[n['label'] for n in norms]}")
    print(f"new_size         : {cfg['new_size']}")
    print(f"apply_log        : {cfg['apply_log']}")
    print(f"Output           : {output_dir}\n", flush=True)

    n_ok = n_skip = n_fail = 0
    for i, (ds, norm, model, aug) in enumerate(jobs, start=1):
        tag = off_aug_tag(
            as_optional_str(aug.get("off_aug", "")),
            int(aug["AE_head_num"]) if (aug.get("AE_head_num") or "").strip() else None,
        )
        print(
            f"({i}/{len(jobs)}) {ds['dataset_id']}  {norm['label']}  {model}  {tag}",
            flush=True,
        )
        rec = run_one(
            ds, norm, model, aug, cfg,
            train_by_mod[ds["modality"]],
            filtered_dir, output_dir,
        )
        append_summary(RESULTS_CSV, rec)
        status = rec["status"]
        if status == "ok":
            n_ok += 1
        elif status.startswith("SKIPPED"):
            n_skip += 1
        else:
            n_fail += 1

    print(
        f"\ndone: {n_ok} ok, {n_skip} skipped, {n_fail} failed; "
        f"log → {RESULTS_CSV.relative_to(ROOT)}"
    )
    if n_fail:
        raise SystemExit(f"{n_fail} run(s) failed; rerun to retry (skip_existing keeps finished runs)")


if __name__ == "__main__":
    main()
