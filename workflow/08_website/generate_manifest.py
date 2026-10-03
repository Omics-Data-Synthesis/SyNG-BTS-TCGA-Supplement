"""Scan ``website/plots/`` and write ``website/manifest.json``.

Layout (one data file per leaf directory):

    plots/<data_type>/<subtype>/<group_label>/<normalization>/<offaug>/<param>/data.json

Sample size and marker dimension come from ``config/datasets.tsv``
(``n_after_na_ref``, ``markers_final_ref``), not a separate TSV.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import re
from pathlib import Path

from dataset_keys import split_dataset_id

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workflow"))
from synthesize_results import provenance, validate_provenance, write_json
WEBSITE = REPO / "website"
PLOTS_ROOT = WEBSITE / "plots"
PLOT_FILENAME = "data.json"
OUTPUT = WEBSITE / "manifest.json"
DATASETS_TSV = REPO / "config" / "datasets.tsv"

DIMENSIONS = [
    {"key": "data_type", "label": "Data type"},
    {"key": "subtype", "label": "Cancer subtype"},
    {"key": "normalization", "label": "Normalization method"},
    {"key": "param", "label": "Augmentation model parameters"},
    {"key": "offaug", "label": "Offline augmentation"},
    {"key": "classifier", "label": "Classifier"},
    {"key": "metric", "label": "Metric"},
]

PREFERRED_ORDER = {
    "data_type": ["miRNA", "RNA"],
    "normalization": ["none", "TC", "DESeq"],
    "offaug": ["none", "AE"],
    "classifier": ["LOGIS", "SVM", "KNN", "RF", "XGB"],
    "metric": ["f1_score", "accuracy", "auc"],
}


def load_dataset_stats() -> dict[tuple[str, str, str], dict[str, str]]:
    """Map (data_type, subtype, group_label) -> sample size & marker dimension."""
    stats: dict[tuple[str, str, str], dict[str, str]] = {}
    if not DATASETS_TSV.exists():
        print(f"  [warn] {DATASETS_TSV.relative_to(REPO)} not found; dataset stats omitted.")
        return stats

    with DATASETS_TSV.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return stats
    if "n_after_na_ref" not in rows[0] or "markers_final_ref" not in rows[0]:
        print("  [warn] expected columns missing in datasets.tsv.")
        return stats

    for row in rows:
        dataset_id = (row.get("dataset_id") or "").strip()
        if not dataset_id:
            continue
        data_type, subtype, group_label = split_dataset_id(dataset_id, row.get("modality", ""))
        stats[(data_type, subtype, group_label)] = {
            "sample_size": (row.get("n_after_na_ref") or "").strip() or None,
            "marker_dim": (row.get("markers_final_ref") or "").strip() or None,
        }
    return stats


def order_values(key: str, values: set[str]) -> list[str]:
    vals = list(values)
    if key == "param":
        def num(v: str) -> tuple:
            m = re.search(r"(\d+)\s*$", v)
            return (int(m.group(1)) if m else 0, v)
        return sorted(vals, key=num)
    pref = PREFERRED_ORDER.get(key)
    if pref:
        rank = {v: i for i, v in enumerate(pref)}
        return sorted(vals, key=lambda v: (rank.get(v, len(pref)), v))
    return sorted(vals)


def validate_payload(payload: dict) -> None:
    validate_provenance(payload.get("provenance"), REPO)
    for classifier in payload["classifiers"]:
        for metric in payload["metrics"]:
            sources = payload["series"][classifier][metric]
            for source in ("real", "generated"):
                entry = sources[source]
                if entry is None:
                    continue
                if entry["fit_status"] not in {"ok", "failed"} or entry["interval_status"] not in {"ok", "unavailable"}:
                    raise ValueError("Unrecognized fit/interval status")
                groups = [("obs_n", "obs_y", "obs_sd")]
                if entry["fit_status"] == "ok":
                    groups.append(("fit_n", "fit_y"))
                elif any(key in entry for key in ("fit_n", "fit_y", "fit_lo", "fit_hi")):
                    raise ValueError("Failed fit contains fitted values")
                if entry["interval_status"] == "ok":
                    if entry["fit_status"] != "ok":
                        raise ValueError("Interval without a fitted curve")
                    groups.append(("fit_n", "fit_lo", "fit_hi"))
                elif "fit_lo" in entry or "fit_hi" in entry:
                    raise ValueError("Unavailable interval contains bounds")
                for fields in groups:
                    size = len(entry[fields[0]])
                    if not size or any(len(entry[key]) != size for key in fields):
                        raise ValueError("Empty or misaligned numerical arrays")
                    for key in fields:
                        if any(not (value is None and key == "obs_sd") and
                               (not isinstance(value, (int, float)) or not math.isfinite(value))
                               for value in entry[key]):
                            raise ValueError(f"Non-finite/missing {key}")


def build_manifest() -> dict:
    stats = load_dataset_stats()

    plots: list[dict] = []
    missing_stats: set[tuple[str, str, str]] = set()
    for data_path in sorted(PLOTS_ROOT.rglob(PLOT_FILENAME)):
        rel = data_path.relative_to(PLOTS_ROOT).parts
        # <data_type>/<subtype>/<group_label>/<norm>/<offaug>/<param>/data.json
        if len(rel) != 7:
            raise ValueError(f"Unexpected plot path: {data_path}")
        data_type, subtype, group_label, norm, offaug, param, _ = rel
        stat = stats.get((data_type, subtype, group_label))
        if stat is None:
            missing_stats.add((data_type, subtype, group_label))

        try:
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            validate_payload(payload)
            classifiers = payload["classifiers"]
            metrics = payload["metrics"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid or historical website data: {data_path}: {exc}") from exc

        base_entry = {
            "data_type": data_type,
            "subtype": subtype,
            "normalization": norm,
            "param": param,
            "offaug": offaug,
            "group_label": group_label,
            "sample_size": stat["sample_size"] if stat else None,
            "marker_dim": stat["marker_dim"] if stat else None,
            "path": str(data_path.relative_to(WEBSITE)).replace("\\", "/"),
        }
        for classifier in classifiers:
            for metric in metrics:
                sources = payload["series"][classifier][metric]
                if any(sources.values()):
                    plots.append({**base_entry, "classifier": classifier, "metric": metric})

    for key in sorted(missing_stats):
        print(f"  [warn] no datasets.tsv row for {key}")

    dimensions = []
    for dim in DIMENSIONS:
        values = order_values(dim["key"], {p[dim["key"]] for p in plots})
        dimensions.append({**dim, "values": values})

    return {
        "provenance": provenance(REPO),
        "status": "ready" if plots else "awaiting_rerun",
        "plot_type": "learning_curve",
        "count": len(plots),
        "dimensions": dimensions,
        "plots": plots,
    }


def main() -> None:
    OUTPUT.unlink(missing_ok=True)
    manifest = build_manifest()
    write_json(OUTPUT, manifest)
    print(f"Wrote {OUTPUT.relative_to(REPO)} with {manifest['count']} plots.")
    for dim in manifest["dimensions"]:
        print(f"  {dim['label']}: {dim['values']}")


if __name__ == "__main__":
    main()
