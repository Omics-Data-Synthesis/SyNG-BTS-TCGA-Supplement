"""Map ``config/datasets.tsv`` rows onto website path segments."""

from __future__ import annotations


def grouping_of(row: dict[str, str]) -> str:
    text = (row.get("grouping") or "binary").strip()
    return text or "binary"


def data_type_of(modality: str) -> str:
    return "RNA" if (modality or "").strip() == "rnaseq" else "miRNA"


def split_dataset_id(dataset_id: str, modality: str) -> tuple[str, str, str]:
    """Return ``(data_type, subtype, group_label)`` for a ``dataset_id``.

    RNA ids are stored as ``RNA_COAD_tumor_status``; strip that prefix before
    splitting so the subtype is ``COAD``, not ``RNA``.
    """
    data_type = data_type_of(modality)
    rest = dataset_id[4:] if dataset_id.startswith("RNA_") else dataset_id
    subtype, _, group_label = rest.partition("_")
    return data_type, subtype, group_label or "unknown"
