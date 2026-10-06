# Stage 05: fidelity metrics and UMAP coordinates for every generated table.
#
# Python computes signed source ARI, CCC of feature means, MAD, MedAE, and UMAP.
# DANA cccpcc and voom+limma CCC of -log10(p) and log2 FC run in dana_dea.R
# (same maths as helper_source/Xinyi_version/evaluate_synthetic.R) for binary
# grouped jobs only. grouping=none RNA skips DANA/DEA (no two-class contrast)
# but still computes source ARI, MAD, MedAE, and UMAP. miRNA coordinates are
# config/mirna_coordinates.csv; dots in `name` are turned into hyphens
# to match TCGA feature names. RNA-seq has no gene-coordinate table yet, so
# cccpcc is NA for those jobs; the other metrics still run.
#
# Skip-if-missing: a job without generated.csv is recorded and skipped.
# Skip-if-exists: combos already in results/fidelity.csv are not recomputed.
#
# Run from the repository root:
#
#   uv run workflow/05_fidelity/run.py

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "data" / ".numba_cache"))

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from metrics import (
    clip_nonfinite,
    source_ari,
    load_table,
    mad_metrics,
    sample_to_n,
    stratified_sample,
    umap_coordinates,
)

SEED = 42

DATASETS_TSV = ROOT / "config" / "datasets.tsv"
MODELS_TSV = ROOT / "config" / "generate_models.tsv"
NORMS_TSV = ROOT / "config" / "generate_norms.tsv"
OFF_AUG_TSV = ROOT / "config" / "generate_off_aug.tsv"
FIDELITY_TSV = ROOT / "config" / "fidelity.tsv"
DANA_DEA_R = Path(__file__).resolve().parent / "dana_dea.R"

FIDELITY_CSV = ROOT / "results" / "fidelity.csv"
UMAP_CSV = ROOT / "results" / "umap.csv"
LOG_CSV = ROOT / "results" / "fidelity_log.csv"

FIDELITY_FIELDS = [
    "dataset_id",
    "modality",
    "norm",
    "model",
    "off_aug",
    "AE_head_num",
    "draw",
    "n_real",
    "n_generated",
    "n_features",
    "source_ari",
    "cccpcc",
    "ccc_log10_pvalue",
    "ccc_log2_FC",
    "ccc_meandiff",
    "mad_mean",
    "mad_sd",
    "mad_cv",
    "medae_mean",
    "medae_sd",
    "medae_cv",
    "seed",
    "draw_seed",
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
        key = (ds["modality"], grouping_of(ds))
        models = models_by_key.get(key, [])
        aug_rows = off_aug_by_key.get(key, [{"off_aug": "", "AE_head_num": ""}])
        for norm in norms:
            for model in models:
                for aug in aug_rows:
                    jobs.append((ds, norm, model, aug))
    return jobs


def ensure_csv_fields(path: Path, fields: list[str]) -> None:
    """Require the expected result columns before appending."""
    if not path.exists():
        return
    if list(pd.read_csv(path, nrows=0).columns) != fields:
        raise ValueError(f"Incompatible fidelity schema in {path}; recompute signed source ARI.")


def append_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def align_matrices(real_feat: pd.DataFrame, gen_feat: pd.DataFrame):
    cols = [c for c in real_feat.columns if c in gen_feat.columns]
    real = clip_nonfinite(real_feat[cols].to_numpy())
    gen = clip_nonfinite(gen_feat[cols].to_numpy())
    return real, gen, cols


def run_dana_dea(jobs_csv: Path, out_csv: Path) -> pd.DataFrame:
    cmd = ["Rscript", str(DANA_DEA_R), str(jobs_csv), str(out_csv)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"dana_dea.R failed ({proc.returncode})")
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if not out_csv.exists():
        return pd.DataFrame(columns=["draw", "cccpcc", "ccc_log10_pvalue", "ccc_log2_FC"])
    return pd.read_csv(out_csv)


def already_done(done, ds, norm, model, off_aug, ae_head_num) -> bool:
    return (ds["dataset_id"], ds["modality"], norm["label"], model,
            off_aug, ae_head_num or 0) in done


def load_done_keys(path: Path, n_draws: int, draw_seed_base: int) -> set[tuple]:
    if not path.exists():
        return set()
    ensure_csv_fields(path, FIDELITY_FIELDS)
    df = pd.read_csv(path)
    if df.empty:
        return set()
    df["AE_head_num"] = df["AE_head_num"].fillna(0)
    keys = ["dataset_id", "modality", "norm", "model", "off_aug", "AE_head_num"]
    if df[keys].isna().any().any():
        raise ValueError("Missing fidelity configuration keys.")
    done = set()
    for key, rows in df.groupby(keys, dropna=False):
        if (len(rows) != n_draws or set(rows["draw"]) != set(range(1, n_draws + 1))
                or not rows["source_ari"].between(-0.5, 1).all()
                or not (rows["draw_seed"] == draw_seed_base + rows["draw"]).all()):
            raise ValueError(f"Invalid or incomplete fidelity results for {key}; recompute this configuration.")
        done.add(key)
    return done


def main() -> None:
    if not DATASETS_TSV.exists():
        raise SystemExit("config/datasets.tsv not found; run from the repository root")

    cfg = read_kv_tsv(FIDELITY_TSV)
    datasets = read_tsv(DATASETS_TSV)
    norms = read_tsv(NORMS_TSV)
    off_aug_by_key = load_off_aug(OFF_AUG_TSV)
    models_by_key: dict[tuple[str, str], list[str]] = {}
    for row in read_tsv(MODELS_TSV):
        models_by_key.setdefault((row["modality"], grouping_of(row)), []).append(row["model"])

    filtered_dir = ROOT / cfg["filtered_dir"]
    generated_dir = ROOT / cfg["generated_dir"]
    n_draws = int(cfg["n_draws"])
    draw_seed_base = int(cfg["draw_seed_base"])
    skip_existing = as_bool(cfg.get("skip_existing", "true"))
    coords_mirna = ROOT / cfg["coords_mirna"]
    if not coords_mirna.exists():
        raise SystemExit(f"missing miRNA coordinates: {coords_mirna}")

    jobs = planned_jobs(datasets, models_by_key, norms, off_aug_by_key)
    existing = load_done_keys(FIDELITY_CSV, n_draws, draw_seed_base)
    if existing and not skip_existing:
        raise ValueError("Remove existing fidelity results before recomputing; appending would duplicate draws.")
    done = existing
    print(f"SEED             : {SEED}")
    print(f"Jobs             : {len(jobs)}")
    print(f"n_draws          : {n_draws}")
    print(f"miRNA coords     : {coords_mirna.relative_to(ROOT)}")
    print(f"Already done     : {len(done)} combos\n", flush=True)

    n_ok = n_skip = n_miss = n_fail = 0
    for i, (ds, norm, model, aug) in enumerate(jobs, start=1):
        off_aug = as_optional_str(aug.get("off_aug", ""))
        ae_head_raw = (aug.get("AE_head_num") or "").strip()
        ae_head_num = int(ae_head_raw) if ae_head_raw else None
        tag = off_aug_tag(off_aug, ae_head_num)
        dataset_id = ds["dataset_id"]
        print(f"({i}/{len(jobs)}) {dataset_id}  {norm['label']}  {model}  {tag}", flush=True)

        log_base = {
            "dataset_id": dataset_id,
            "modality": ds["modality"],
            "norm": norm["label"],
            "model": model,
            "off_aug": off_aug or "none",
            "AE_head_num": ae_head_num if ae_head_num is not None else "",
            "tag": tag,
            "seed": SEED,
        }

        if skip_existing and already_done(done, ds, norm, model, off_aug or "none", ae_head_num):
            print("    skipped (exists)", flush=True)
            n_skip += 1
            append_rows(LOG_CSV, [{**log_base, "status": "SKIPPED (exists)"}], list(log_base) + ["status"])
            continue

        real_path = filtered_dir / f"{dataset_id}{norm['suffix']}"
        gen_path = generated_dir / dataset_id / norm["label"] / model / tag / "generated.csv"
        if not real_path.exists():
            print("    skipped (real missing)", flush=True)
            n_miss += 1
            continue
        if not gen_path.exists():
            print("    skipped (generated missing)", flush=True)
            n_miss += 1
            continue

        try:
            ungrouped = grouping_of(ds) == "none"
            real_feat, real_groups = load_table(real_path)
            gen_feat, gen_groups = load_table(gen_path)
            if not ungrouped and (real_groups is None or gen_groups is None):
                raise ValueError("both real and generated tables need a groups column")

            if ungrouped:
                umap_feat, umap_groups = sample_to_n(
                    gen_feat, gen_groups, len(real_feat), SEED
                )
            else:
                umap_feat, umap_groups = stratified_sample(
                    gen_feat, gen_groups, real_groups, SEED
                )
            real_m, gen_m, _ = align_matrices(real_feat, umap_feat)
            umap_df = umap_coordinates(real_m, gen_m, real_groups, umap_groups, SEED)
            umap_df.insert(0, "off_aug", off_aug or "none")
            umap_df.insert(0, "model", model)
            umap_df.insert(0, "norm", norm["label"])
            umap_df.insert(0, "modality", ds["modality"])
            umap_df.insert(0, "dataset_id", dataset_id)

            py_rows = []
            need_dana = not ungrouped
            tmpdir = None
            if need_dana:
                (ROOT / "data").mkdir(parents=True, exist_ok=True)
                tmpdir = Path(tempfile.mkdtemp(prefix="fidelity_", dir=str(ROOT / "data")))
            try:
                r_jobs = []
                for draw in range(1, n_draws + 1):
                    if ungrouped:
                        samp_feat, samp_groups = sample_to_n(
                            gen_feat, gen_groups, len(real_feat), draw_seed_base + draw
                        )
                    else:
                        samp_feat, samp_groups = stratified_sample(
                            gen_feat, gen_groups, real_groups, draw_seed_base + draw
                        )
                    real_a, gen_a, _ = align_matrices(real_feat, samp_feat)
                    ari = source_ari(real_a, gen_a)
                    mads = mad_metrics(real_a, gen_a)
                    if need_dana:
                        gen_csv = tmpdir / f"gen_draw{draw}.csv"
                        real_csv = tmpdir / "real.csv"
                        if draw == 1:
                            real_out = real_feat.copy()
                            real_out["groups"] = real_groups.to_numpy()
                            real_out.to_csv(real_csv, index=False)
                        gen_out = samp_feat.copy()
                        gen_out["groups"] = samp_groups.to_numpy()
                        gen_out.to_csv(gen_csv, index=False)
                        r_jobs.append(
                            {
                                "draw": draw,
                                "real_csv": str(real_csv),
                                "gen_csv": str(gen_csv),
                                "coords_csv": str(coords_mirna) if ds["modality"] == "mirna" else "",
                                "reverse_log": cfg["reverse_log"],
                                "tzero": cfg["dana_tzero"],
                                "tpoor": cfg["dana_tpoor"],
                                "twell": cfg["dana_twell"],
                                "failure": cfg["failure"],
                            }
                        )
                    py_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "modality": ds["modality"],
                            "norm": norm["label"],
                            "model": model,
                            "off_aug": off_aug or "none",
                            "AE_head_num": ae_head_num if ae_head_num is not None else "",
                            "draw": draw,
                            "n_real": int(real_a.shape[0]),
                            "n_generated": int(gen_a.shape[0]),
                            "n_features": int(real_a.shape[1]),
                            "source_ari": ari,
                            "cccpcc": "",
                            "ccc_log10_pvalue": "",
                            "ccc_log2_FC": "",
                            **mads,
                            "seed": SEED,
                            "draw_seed": draw_seed_base + draw,
                        }
                    )

                r_map = {}
                if need_dana:
                    jobs_csv = tmpdir / "jobs.csv"
                    r_out = tmpdir / "dana_dea.csv"
                    pd.DataFrame(r_jobs).to_csv(jobs_csv, index=False)
                    try:
                        r_df = run_dana_dea(jobs_csv, r_out)
                        r_map = {int(r["draw"]): r for r in r_df.to_dict("records")} if not r_df.empty else {}
                    except Exception as exc:
                        print(f"    DANA/DEA failed: {exc}", flush=True)
                        r_map = {}
                for row in py_rows:
                    rec = r_map.get(int(row["draw"]), {})
                    row["cccpcc"] = rec.get("cccpcc", "")
                    row["ccc_log10_pvalue"] = rec.get("ccc_log10_pvalue", "")
                    row["ccc_log2_FC"] = rec.get("ccc_log2_FC", "")
            finally:
                if tmpdir is not None:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            append_rows(FIDELITY_CSV, py_rows, FIDELITY_FIELDS)
            umap_header = not UMAP_CSV.exists()
            UMAP_CSV.parent.mkdir(parents=True, exist_ok=True)
            umap_df.to_csv(UMAP_CSV, mode="a", index=False, header=umap_header)
            append_rows(LOG_CSV, [{**log_base, "status": "ok"}], list(log_base) + ["status"])
            n_ok += 1
            print(f"    ok  draws={n_draws}  umap={len(umap_df)}", flush=True)
        except Exception as exc:
            n_fail += 1
            print(f"    ERROR: {exc}", flush=True)
            append_rows(LOG_CSV, [{**log_base, "status": f"ERROR: {exc}"}], list(log_base) + ["status"])

    print(
        f"\ndone: {n_ok} ok, {n_skip} skipped exists, {n_miss} missing, {n_fail} failed; "
        f"metrics → {FIDELITY_CSV.relative_to(ROOT)}"
    )
    if n_fail:
        raise SystemExit(f"{n_fail} combo(s) failed")


if __name__ == "__main__":
    main()
