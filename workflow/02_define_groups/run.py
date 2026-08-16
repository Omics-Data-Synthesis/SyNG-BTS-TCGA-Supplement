# Stage 02: attach a binary group label to every expression sample.
#
# For each row in config/datasets.tsv, reads expression sample barcodes and the
# BCR patient table from data/raw/, recodes one clinical column to groups 0/1
# (rules in recode.py / config/group_maps.tsv), and writes
# data/processed/grouped/<dataset_id>.csv with Sample, patient, and groups.
# Samples whose recoded label is missing are kept with groups=NA; stage 03
# drops them before marker filtering.
#
# This stage is deterministic and declares no seed.
#
# Run from the repository root:
#
#   uv run workflow/02_define_groups/run.py

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from recode import recode_column

ROOT = Path(__file__).resolve().parents[2]

DATASETS_TSV = ROOT / "config" / "datasets.tsv"
MAPS_TSV = ROOT / "config" / "group_maps.tsv"
PREPROCESS_TSV = ROOT / "config" / "preprocess.tsv"
CLINICAL_DIR = ROOT / "data" / "raw" / "clinical"
RESULTS_CSV = ROOT / "results" / "group_selection.csv"


def read_kv_tsv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["param"]: r["value"] for r in rows}


def read_datasets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_maps(path: Path) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            maps.setdefault(row["map_name"], {})[row["value"]] = row["group"]
    return maps


def split_levels(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [p for p in text.split("|") if p]


def patient_barcode(aliquot: str) -> str:
    return aliquot[:12]


def load_sample_barcodes(cohort: str, modality: str) -> list[str]:
    path = ROOT / "data" / "raw" / modality / f"TCGA-{cohort}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing expression matrix: {path}")
    # R write.csv: first column is the feature id, header cell is empty.
    return list(pd.read_csv(path, index_col=0, nrows=0).columns)


def load_clinical(cohort: str) -> pd.DataFrame:
    path = CLINICAL_DIR / f"TCGA-{cohort}.patient.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing clinical table: {path}")
    return pd.read_csv(path, dtype=str)


def define_one(row: dict[str, str], maps: dict[str, dict[str, str]], out_dir: Path) -> dict:
    dataset_id = row["dataset_id"]
    cohort = row["cohort"]
    modality = row["modality"]
    column = row["clinical_column"]
    recode = row["recode"]
    print(f"[{dataset_id}] recoding {column} ({recode})", flush=True)

    clin = load_clinical(cohort)
    if column not in clin.columns:
        raise KeyError(f"{dataset_id}: clinical column {column!r} not in TCGA-{cohort}.patient.csv")
    if "bcr_patient_barcode" not in clin.columns:
        raise KeyError(f"{dataset_id}: bcr_patient_barcode missing from clinical table")

    lookup = maps.get(row["map_name"]) if row.get("map_name") else None
    groups_patients, mapping = recode_column(
        clin[column].tolist(),
        recode=recode,
        lookup=lookup,
        keep_levels=split_levels(row.get("keep_levels", "")),
        drop_levels=split_levels(row.get("drop_levels", "")),
    )
    patient_to_group = dict(zip(clin["bcr_patient_barcode"].tolist(), groups_patients))

    samples = load_sample_barcodes(cohort, modality)
    groups = [patient_to_group.get(patient_barcode(s)) for s in samples]

    out = pd.DataFrame({
        "Sample": samples,
        "patient": [patient_barcode(s) for s in samples],
        "groups": groups,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}.csv"
    tmp = out_path.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(out_path)

    n_raw = len(groups)
    n_after = sum(g is not None for g in groups)
    counts = Counter(g for g in groups if g is not None)
    n_raw_ref = int(row["n_raw_ref"]) if row.get("n_raw_ref") else None
    n_after_ref = int(row["n_after_na_ref"]) if row.get("n_after_na_ref") else None

    note = []
    if n_raw_ref is not None and n_raw != n_raw_ref:
        note.append(f"n_raw {n_raw} vs ref {n_raw_ref}")
    if n_after_ref is not None and n_after != n_after_ref:
        note.append(f"n_after_na {n_after} vs ref {n_after_ref}")
    status = "OK" if not note else "MISMATCH: " + "; ".join(note)
    print(f"  samples {n_raw} -> {n_after} after NA  ({status})", flush=True)

    return {
        "dataset_id": dataset_id,
        "cohort": cohort,
        "modality": modality,
        "clinical_variable": row["clinical_variable"],
        "clinical_column": column,
        "recode": recode,
        "mapping": mapping,
        "n_raw": n_raw,
        "n_raw_ref": n_raw_ref,
        "n_after_na": n_after,
        "n_after_na_ref": n_after_ref,
        "n_group_0": counts.get("0", 0),
        "n_group_1": counts.get("1", 0),
        "match_n_after_na": n_after_ref is None or n_after == n_after_ref,
        "output": str(out_path.relative_to(ROOT)),
    }


def main() -> None:
    if not DATASETS_TSV.exists():
        raise SystemExit("config/datasets.tsv not found; run from the repository root")
    cfg = read_kv_tsv(PREPROCESS_TSV)
    out_dir = ROOT / cfg["grouped_dir"]
    maps = read_maps(MAPS_TSV)
    datasets = read_datasets(DATASETS_TSV)

    rows = [define_one(ds, maps, out_dir) for ds in datasets]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_bad = sum(not r["match_n_after_na"] for r in rows)
    print(f"wrote {RESULTS_CSV.relative_to(ROOT)}  ({len(rows)} datasets, {n_bad} N(after NA) mismatches)")


if __name__ == "__main__":
    main()
