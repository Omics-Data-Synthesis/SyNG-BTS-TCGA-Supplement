"""Export manuscript tables from existing results; no analysis stages are rerun."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from agreement_data import KEYS, comparison_rows, select_agreement_configurations, validate_archive

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path, delimiter=","):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path.name}: {len(rows)} rows", flush=True)


def label_recoding(groups, datasets):
    rows = []
    for group in groups:
        dataset = datasets[group["dataset_id"]]
        exclusions = []
        if dataset["keep_levels"]:
            exclusions.append("Retain only " + dataset["keep_levels"].replace("|", ", "))
        if dataset["drop_levels"]:
            exclusions.append("Exclude " + dataset["drop_levels"].replace("|", ", "))
        recode = group["recode"]
        if recode == "map":
            exclusions.append("Unmapped categories excluded")
        elif recode == "median_split_na0":
            exclusions.append("Missing values set to 0 before splitting")
        elif recode != "none":
            exclusions.append("Missing labels excluded")
        rows.append({
            "Dataset identifier": group["dataset_id"],
            "Modality": "miRNA-seq" if group["modality"] == "mirna" else "RNA-seq",
            "Source clinical field": group["clinical_column"] or "None (one group)",
            "Recoding rule": group["mapping"] or "No binary recoding",
            "Exclusions or missing-value handling": "; ".join(exclusions) or "All samples retained",
            "n₀": group["n_group_0"] if recode != "none" else "",
            "n₁": group["n_group_1"] if recode != "none" else "",
            "Retained n": group["n_after_na"],
        })
    return rows


def marker_filtering(records):
    columns = {
        "Dataset (cancer_variable)": "dataset_id",
        "N (raw)": "n_samples_raw",
        "N (after NA)": "n_samples_after_na",
        "Markers (raw)": "n_markers_raw",
        "Markers (non-zero)": "n_markers_nonzero",
        "x* (mean)": "x_star",
        "y* (SD)": "y_star",
        "Markers (final)": "n_markers_final",
        "Removed (%)": "removed_pct",
    }
    return [{label: record[source] for label, source in columns.items()} for record in records]


def training_runtime(records):
    # Ignore skipped/error records; the current archive has one success per configuration.
    records = [r for r in records if r["status"] == "ok"
               and r["modality"] == "mirna" and r["off_aug"] == "none"]
    groups = {}
    for record in records:
        key = (record["dataset_id"], record["norm"])
        group = groups.setdefault(key, {})
        if record["norm"] not in ("raw", "TC", "DESeq"):
            raise ValueError(f"Unexpected runtime normalization: {key}")
        if record["model"] in group:
            raise ValueError(f"Duplicate successful runtime record: {key}, {record['model']}")
        group[record["model"]] = record
    rows = []
    for dataset, norm in sorted(groups, key=lambda key: (key[0], ("raw", "TC", "DESeq").index(key[1]))):
        if set(groups[dataset, norm]) != {"CVAE1-5", "CVAE1-10", "CVAE1-50"}:
            raise ValueError(f"Incomplete or unexpected runtime models: {dataset}, {norm}")
        row = {"Dataset (cancer_variable)": dataset, "Norm": norm}
        for model in ("CVAE1-5", "CVAE1-10", "CVAE1-50"):
            record = groups[dataset, norm][model]
            row[model + " Epochs"] = record["epochs_trained"]
            row[model + " Time (s)"] = record["elapsed_s"]
        rows.append(row)
    return rows


def agreement_summary(comparisons):
    rows = []
    scopes = ((None, "All {n} tasks"), ("mirna", "{n} miRNA tasks"), ("rnaseq", "{n} RNA-seq tasks"))
    for metric, label in (("auc", "AUC"), ("f1_score", "Macro-F1"), ("accuracy", "Accuracy")):
        for modality, scope in scopes:
            subset = [r for r in comparisons if r["metric_name"] == metric
                      and (modality is None or r["modality"] == modality)]
            rows.append({
                "Metric": label,
                "Task scope": scope.format(n=len({r["dataset_id"] for r in subset})),
                "Configuration–classifier comparisons": len(subset),
                "Median signed gap B": statistics.median(r["signed_gap"] for r in subset),
                "Median absolute gap D": statistics.median(r["absolute_gap"] for r in subset),
            })
    return rows


def selected_configurations(selections, datasets, head_settings):
    heads = {r["modality"]: r["AE_head_num"] for r in head_settings if r["off_aug"] == "AE_head"}
    return [{
        "Cohort": datasets[row["dataset_id"]]["cohort"],
        "Modality": "miRNA-seq" if row["modality"] == "mirna" else "RNA-seq",
        "Normalization": row["norm"],
        "Model": row["model"].replace("CVAE", "cVAE-").replace("1-", "1:"),
        "AE head": heads[row["modality"]] + " iterations" if row["off_aug"] == "AE_head" else "None",
        "Mean AUC D": row["mean_absolute_auc_gap"],
        "Maximum AUC D": row["max_absolute_auc_gap"],
        "Configurations evaluated": row["n_configurations_evaluated"],
    } for row in selections]


def main(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "tables"
    datasets = {r["dataset_id"]: r for r in read_csv(ROOT / "config/datasets.tsv", "\t")}
    archive = json.loads((ROOT / "results/learning_curve_fits.json").read_text())
    config = {r["param"]: r["value"] for r in read_csv(ROOT / "config/synthesize.tsv", "\t")}
    validate_archive(archive["fits"], config["n_sizes"], config["n_draws"])
    comparisons = comparison_rows(archive["fits"])
    selections = select_agreement_configurations(comparisons)
    selected_keys = {tuple(r[k] for k in KEYS[:5]) for r in selections}
    selected_rows = [r for r in comparisons if tuple(r[k] for k in KEYS[:5]) in selected_keys]
    tables = {
        "benchmark_label_recoding.csv": label_recoding(read_csv(ROOT / "results/group_selection.csv"), datasets),
        "marker_filtering.csv": marker_filtering(read_csv(ROOT / "results/marker_filtering.csv")),
        "training_runtime_mirna.csv": training_runtime(read_csv(ROOT / "results/training_time.csv")),
        "learning_curve_agreement_summary.csv": agreement_summary(comparisons),
        "learning_curve_selected_configurations.csv": selected_configurations(
            selections, datasets, read_csv(ROOT / "config/generate_off_aug.tsv", "\t")),
        "learning_curve_selected_agreement.csv": selected_rows,
        "learning_curve_agreement.csv": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        write_csv(output_dir / filename, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="CSV destination (default: tables/)")
    main(parser.parse_args().output_dir)
