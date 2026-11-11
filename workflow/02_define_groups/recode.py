"""Recode one BCR clinical column into a binary group label.

GDC missing codes become NA. The configured rule assigns binary labels by
median split, explicit mapping, or majority coding. Unmapped levels stay NA
and are dropped in stage 03.
"""

from __future__ import annotations

import math
from collections import Counter

NA_CODES = {
    "",
    "[Not Available]",
    "[Unknown]",
    "[Not Evaluated]",
    "[Not Applicable]",
    "[Not Reported]",
    "[Discrepancy]",
    "NA",
    "NaN",
    "None",
    "nan",
}


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text in NA_CODES


def clean_value(value) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def find_half(values: list[float]) -> tuple[float, bool]:
    """Notebook find_half: median, and whether ties should go to 1 (>=).

    equal=True means using >= median is closer to a 50/50 split than > median.
    """
    if not values:
        raise ValueError("find_half() needs at least one numeric value")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    ratio_gt = sum(x > median for x in values) / n
    ratio_ge = sum(x >= median for x in values) / n
    equal = abs(ratio_gt - 0.5) > abs(ratio_ge - 0.5)
    return median, equal


def median_split(raw: list[str | None], fill_na_zero: bool = False) -> tuple[list[str | None], str]:
    numeric: list[float | None] = []
    for v in raw:
        if fill_na_zero and v is None:
            numeric.append(0.0)
            continue
        if v is None:
            numeric.append(None)
            continue
        try:
            numeric.append(float(v))
        except ValueError:
            numeric.append(None)
    observed = [x for x in numeric if x is not None]
    median, equal = find_half(observed)
    out: list[str | None] = []
    for x in numeric:
        if x is None:
            out.append(None)
        elif x < median or (not equal and x == median):
            out.append("0")
        else:
            out.append("1")
    op = ">=" if equal else ">"
    mapping = f"median_split {op} {median:g} -> 1"
    return out, mapping


def binary_majority(raw: list[str | None]) -> tuple[list[str | None], str]:
    counts = Counter(v for v in raw if v is not None)
    if len(counts) == 0:
        return list(raw), "binary_majority: no non-missing values"
    if len(counts) != 2:
        levels = ", ".join(f"{k}={n}" for k, n in counts.most_common())
        raise ValueError(
            f"binary_majority expects exactly 2 levels, got {len(counts)}: {levels}"
        )
    majority, minority = [k for k, _ in counts.most_common()]
    mapping = {majority: "0", minority: "1"}
    out = [mapping.get(v) if v is not None else None for v in raw]
    desc = f"{majority} -> 0; {minority} -> 1"
    return out, desc


def apply_map(raw: list[str | None], lookup: dict[str, str]) -> tuple[list[str | None], str]:
    out = []
    unmapped = Counter()
    for v in raw:
        if v is None:
            out.append(None)
        elif v in lookup:
            out.append(str(lookup[v]))
        else:
            unmapped[v] += 1
            out.append(None)
    parts = [f"{k} -> {lookup[k]}" for k in sorted(lookup, key=lambda x: (lookup[x], x))]
    desc = "; ".join(parts)
    if unmapped:
        extra = ", ".join(f"{k} (n={n})" for k, n in unmapped.most_common())
        desc += f"; unmapped->NA: {extra}"
    return out, desc


def tgct_seminoma(raw: list[str | None]) -> tuple[list[str | None], str]:
    """Pure 'Seminoma; NOS' vs any other recorded histologic diagnosis."""
    out = []
    for v in raw:
        if v is None:
            out.append(None)
        elif v == "Seminoma; NOS":
            out.append("0")
        else:
            out.append("1")
    return out, "Seminoma; NOS -> 0; any other diagnosis -> 1"


def recode_column(
    raw_values: list,
    recode: str,
    lookup: dict[str, str] | None = None,
    keep_levels: list[str] | None = None,
    drop_levels: list[str] | None = None,
) -> tuple[list[str | None], str]:
    cleaned = [clean_value(v) for v in raw_values]
    if drop_levels:
        drop = set(drop_levels)
        cleaned = [None if v in drop else v for v in cleaned]
    if keep_levels:
        keep = set(keep_levels)
        cleaned = [v if v in keep else None for v in cleaned]

    if recode == "binary_majority":
        return binary_majority(cleaned)
    if recode == "median_split":
        return median_split(cleaned, fill_na_zero=False)
    if recode == "median_split_na0":
        return median_split(cleaned, fill_na_zero=True)
    if recode == "map":
        if not lookup:
            raise ValueError("recode=map requires a map_name with entries in group_maps.tsv")
        return apply_map(cleaned, lookup)
    if recode == "tgct_seminoma":
        return tgct_seminoma(cleaned)
    raise ValueError(f"unknown recode method: {recode}")
