"""Archive and cache bookkeeping for SyntheSize (standard library only).

All numerical results come from syng_bts.fit_sample_sizes in stage 06.
Consumers require the archive written last, after its companion CSV tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PACKAGE_VERSION = "3.6.0"
FORMAT_VERSION = 1
FIT_API = "syng_bts.fit_sample_sizes"
ARCHIVE_NAME = "learning_curve_fits.json"
KEYS = ["dataset_id", "modality", "norm", "model", "off_aug"]
POINT_COLUMNS = KEYS + ["source", "method", "metric", "kind", "n", "value",
                        "observed_std", "ci_low", "ci_high", "n_draws", "weight"]
FIT_COLUMNS = KEYS + ["source", "method", "metric", "fit_ok", "fit_status",
                      "interval_status", "message", "a", "b", "c"] + [
    f"cov_{a}{b}" for a in "abc" for b in "abc"
]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configuration_id(root: Path) -> str:
    """Identify the scientific configuration; plotting/skip settings do not refit."""
    config = {}
    for name in ("datasets", "generate_models", "generate_norms", "generate_off_aug"):
        config[name] = file_hash(root / "config" / f"{name}.tsv")
    with (root / "config" / "synthesize.tsv").open(newline="") as stream:
        config["synthesize"] = {
            row["param"]: row["value"] for row in csv.DictReader(stream, delimiter="\t")
            if row["param"] not in {"skip_existing", "static_metric", "plot_ylim_lo", "plot_ylim_hi"}
        }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def provenance(root: Path, seed: int = 42) -> dict:
    return dict(format_version=FORMAT_VERSION, syng_bts_version=PACKAGE_VERSION,
                fitting_api=FIT_API, configuration_id=configuration_id(root), seed=seed)


def input_identity(path: Path) -> dict | None:
    """Cheap input identity: changed size/mtime invalidates cached evaluations."""
    if not path.is_file():
        return None
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_completion(path: Path, identity: dict, files: list[Path], **details) -> None:
    write_json(path, {"identity": identity, "files": {p.name: file_hash(p) for p in files}, **details})


def cache_valid(path: Path, identity: dict, required: set[str]) -> bool:
    try:
        record = json.loads(path.read_text())
        return (isinstance(record, dict) and isinstance(record.get("files"), dict)
                and record["identity"] == identity and required <= record["files"].keys()
                and all(file_hash(path.parent / name) == digest
                        for name, digest in record["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_provenance(value: dict, root: Path) -> None:
    if value != provenance(root):
        raise ValueError("SyntheSize provenance is missing or incompatible; rerun stage 06 with the current configuration.")


def read_archive(root: Path) -> dict:
    path = root / "results" / ARCHIVE_NAME
    if not path.is_file():
        raise ValueError("No completed SyntheSize archive; run stage 06 first.")
    archive = json.loads(path.read_text())
    validate_provenance(archive.get("provenance"), root)
    for name in ("curves.csv", "curve_fits.csv"):
        if file_hash(path.parent / name) != archive["tables"][name]:
            raise ValueError(f"Incomplete or modified SyntheSize archive: {name}; rerun stage 06.")
    return archive


def point_rows(fit: dict) -> list[dict]:
    keys = {name: fit[name] for name in KEYS + ["source", "method"]}
    keys["metric"] = fit["metric_name"]
    observed = [dict(keys, kind="observed", n=r["n"], value=r["observed_mean"],
                     observed_std=r["observed_std"], n_draws=r["n_draws"], weight=r["weight"])
                for r in fit["observed"]]
    predicted = [dict(keys, kind="fitted", n=r["n"], value=r["predicted"],
                      ci_low=r["ci_low"], ci_high=r["ci_high"])
                 for r in fit["predictions"] if r["predicted"] is not None]
    return observed + predicted


def fit_row(fit: dict) -> dict:
    row = {name: fit[name] for name in KEYS + ["source", "method", "fit_status", "interval_status", "message"]}
    row.update(metric=fit["metric_name"], fit_ok=fit["fit_status"] == "ok")
    row.update(zip("abc", fit["parameters"] or [None] * 3))
    covariance = fit["covariance"] or [[None] * 3 for _ in range(3)]
    row.update({f"cov_{a}{b}": covariance[i][j]
                for i, a in enumerate("abc") for j, b in enumerate("abc")})
    return row
