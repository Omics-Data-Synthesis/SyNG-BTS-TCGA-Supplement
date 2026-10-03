"""Serialize stage 06's archived fits for the website; no numerical fitting."""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

from dataset_keys import grouping_of, split_dataset_id

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workflow"))
from synthesize_results import KEYS, read_archive, write_json

SOURCES = ("real", "generated")
NORM_LABELS = {"raw": "none"}
OFFAUG_LABELS = {"none": "none", "AE_head": "AE"}


def website_series(fit: dict) -> dict:
    """Keep numerical precision and explicit missing values from to_dict()."""
    obs, pred = fit["observed"], fit["predictions"]
    entry = {name: fit[name] for name in ("parameters", "covariance", "fit_status", "interval_status", "message")}
    entry.update(obs_n=[r["n"] for r in obs], obs_y=[r["observed_mean"] for r in obs],
                 obs_sd=[r["observed_std"] for r in obs])
    if fit["fit_status"] == "ok":
        entry.update(fit_n=[r["n"] for r in pred], fit_y=[r["predicted"] for r in pred])
    if fit["interval_status"] == "ok":
        entry.update(fit_lo=[r["ci_low"] for r in pred], fit_hi=[r["ci_high"] for r in pred])
    return entry


def build_payloads(root: Path) -> dict[Path, dict]:
    archive = read_archive(root)
    with (root / "config" / "datasets.tsv").open(newline="") as stream:
        datasets = {r["dataset_id"]: r for r in csv.DictReader(stream, delimiter="\t")}
    with (root / "config" / "synthesize.tsv").open(newline="") as stream:
        config = {r["param"]: r["value"] for r in csv.DictReader(stream, delimiter="\t")}
    methods = [m.strip() for m in config["methods"].split(",")]
    metrics = [m.strip() for m in config["metrics"].split(",")]
    payloads = {}
    for fit in archive["fits"]:
        ds = datasets[fit["dataset_id"]]
        if grouping_of(ds) == "none":
            raise ValueError(f"Unexpected unlabelled dataset: {fit['dataset_id']}")
        data_type, subtype, label = split_dataset_id(fit["dataset_id"], ds["modality"])
        dest = Path(data_type, subtype, label, NORM_LABELS.get(fit["norm"], fit["norm"]),
                    OFFAUG_LABELS.get(fit["off_aug"], fit["off_aug"]), fit["model"], "data.json")
        keys = {name: fit[name] for name in KEYS}
        if dest not in payloads:
            payloads[dest] = dict(
                provenance=archive["provenance"], configuration=keys, counts=archive["counts"],
                classifiers=methods, metrics=metrics,
                ylim=[float(config["plot_ylim_lo"]), float(config["plot_ylim_hi"])],
                series={m: {metric: {source: None for source in SOURCES} for metric in metrics} for m in methods},
            )
        payload = payloads[dest]
        sources = payload["series"][fit["method"]][fit["metric_name"]]
        if payload["configuration"] != keys or sources[fit["source"]] is not None:
            raise ValueError(f"Duplicate website series: {dest}")
        sources[fit["source"]] = website_series(fit)
    return payloads


def main() -> None:
    payloads = build_payloads(REPO)
    website = REPO / "website"
    website.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthesize-", dir=website) as tmp:
        staging = Path(tmp) / "plots"
        staging.mkdir()
        for relative, payload in payloads.items():
            write_json(staging / relative, payload)
        # Replace the complete generated tree, even if empty: no bootstrap fallback.
        (website / "manifest.json").unlink(missing_ok=True)
        destination = website / "plots"
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    print(f"Wrote {len(payloads)} configurations from the stage 06 archive.")


if __name__ == "__main__":
    main()
