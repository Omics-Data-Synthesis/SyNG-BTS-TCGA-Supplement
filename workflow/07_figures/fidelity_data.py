"""Read and select the stage 05 results used by the fidelity figures."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["dataset_id", "modality", "norm", "model", "off_aug", "AE_head_num"]
METRICS = [
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
]


def read_fidelity(root: Path = ROOT) -> pd.DataFrame:
    """Require complete, uniquely keyed draws; retain stored numerical precision."""
    path = root / "results/fidelity.csv"
    data = pd.read_csv(path, float_precision="round_trip")
    required = set(
        KEYS
        + METRICS
        + [
            "draw",
            "draw_seed",
            "seed",
            "n_real",
            "n_generated",
            "n_features",
        ]
    )
    if set(data.columns) != required or data.empty:
        raise ValueError(f"Invalid fidelity schema or empty table: {path}")
    config = (
        pd.read_csv(root / "config/fidelity.tsv", sep="\t").set_index("param").value
    )
    n_draws, seed_base = int(config["n_draws"]), int(config["draw_seed_base"])
    data["AE_head_num"] = data.AE_head_num.fillna(0)
    if data[KEYS].isna().any().any() or data.duplicated(KEYS + ["draw"]).any():
        raise ValueError("Missing configuration keys or duplicate fidelity draws")
    counts = data.groupby(KEYS, dropna=False).size()
    if (
        not counts.eq(n_draws).all()
        or not data.draw.isin(range(1, n_draws + 1)).all()
        or not data.draw_seed.eq(seed_base + data.draw).all()
    ):
        raise ValueError("Incomplete fidelity draws or inconsistent draw seeds")
    if not data.source_ari.between(-0.5, 1).all():
        raise ValueError("Missing or invalid signed source ARI")
    if not np.isfinite(data.seed).all() or not data.seed.mod(1).eq(0).all():
        raise ValueError("Missing or invalid workflow seeds")
    sizes = data[["n_real", "n_generated", "n_features"]]
    if (
        not np.isfinite(sizes).all().all()
        or not sizes.gt(0).all().all()
        or not sizes.mod(1).eq(0).all().all()
        or not data.n_real.eq(data.n_generated).all()
    ):
        raise ValueError("Invalid or unequal empirical/generated sample counts")
    if (
        data.groupby(KEYS)[["seed", *sizes.columns]]
        .nunique(dropna=False)
        .ne(1)
        .any()
        .any()
    ):
        raise ValueError("Inconsistent seeds or dimensions within a configuration")
    return data


def select_panel(data: pd.DataFrame, panel: str, root: Path = ROOT) -> pd.DataFrame:
    """Select every configured cohort using full configuration keys."""
    rules = pd.read_csv(root / "config/fidelity_figures.tsv", sep="\t")
    rules = rules.loc[rules.panel.eq(panel)].drop(columns="panel")
    if rules.empty:
        raise ValueError(f"No figure settings for {panel}")
    datasets = pd.read_csv(root / "config/datasets.tsv", sep="\t")
    datasets = datasets.loc[
        datasets.preprocess.ne("skip"), ["dataset_id", "cohort", "modality", "grouping"]
    ]
    settings = rules.merge(
        datasets, on=["modality", "grouping"], how="left", validate="one_to_many"
    )
    if settings.isna().any().any() or settings.duplicated(KEYS).any():
        raise ValueError(f"Invalid or unmatched figure settings for {panel}")
    return select_configurations(data, settings, panel)


def select_configurations(
    data: pd.DataFrame, settings: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Match complete configuration keys without silently dropping requested rows."""
    selected = settings.merge(
        data, on=KEYS, how="left", validate="one_to_many", indicator=True
    )
    missing = selected.loc[selected._merge.eq("left_only"), "dataset_id"].tolist()
    if missing:
        raise ValueError(f"Missing {label} configurations: {', '.join(missing)}")
    return selected.drop(columns="_merge")


def select_mirna_grid(data: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """Use the configured miRNA model/normalization grid and figure augmentation."""
    cohorts = select_panel(data, "mirna_umap", root)[
        ["dataset_id", "cohort", "modality", "grouping", "off_aug", "AE_head_num"]
    ].drop_duplicates()
    models = pd.read_csv(root / "config/generate_models.tsv", sep="\t")
    norms = pd.read_csv(root / "config/generate_norms.tsv", sep="\t")
    grid = norms[["label"]].rename(columns={"label": "norm"}).merge(models, how="cross")
    settings = cohorts.merge(grid, on=["modality", "grouping"])
    if settings.empty or settings.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate miRNA figure configurations")
    return select_configurations(data, settings, "miRNA fidelity")


def read_umap(root: Path = ROOT) -> pd.DataFrame:
    """Read stored stage 05 embeddings; no fitting or resampling happens here."""
    data = pd.read_csv(root / "results/umap.csv", float_precision="round_trip")
    required = set(KEYS[:-1] + ["UMAP1", "UMAP2", "datatype", "groups"])
    if set(data.columns) != required or data.empty:
        raise ValueError("Invalid UMAP schema or empty table")
    if (
        data[KEYS[:-1]].isna().any().any()
        or not np.isfinite(data[["UMAP1", "UMAP2"]]).all().all()
        or not data.datatype.isin(["Real", "Generated"]).all()
    ):
        raise ValueError("Missing or invalid UMAP coordinates/source labels")
    return data


def select_umap(data: pd.DataFrame, fidelity: pd.DataFrame) -> pd.DataFrame:
    """Require the selected embedding's source and group counts to match fidelity."""
    # The coordinate table does not identify AE heads. These figures use none.
    if not fidelity.off_aug.eq("none").all() or not fidelity.AE_head_num.eq(0).all():
        raise ValueError("UMAP figure selections require no AE-head augmentation")
    keys = KEYS[:-1]
    settings = fidelity[
        keys + ["cohort", "grouping", "n_real", "n_generated"]
    ].drop_duplicates()
    selected = settings.merge(data, on=keys, how="left", validate="one_to_many")
    if selected.datatype.isna().any():
        raise ValueError("Missing selected UMAP configuration")
    counts = selected.groupby(keys + ["datatype"]).size().unstack(fill_value=0)
    expected = settings.set_index(keys).reindex(counts.index)
    if (
        not counts.reindex(columns=["Real", "Generated"], fill_value=0)
        .eq(expected[["n_real", "n_generated"]].set_axis(["Real", "Generated"], axis=1))
        .all()
        .all()
    ):
        raise ValueError("UMAP source counts do not match fidelity sample counts")
    grouped = selected[selected.grouping.eq("binary")]
    if not grouped.groups.isin([0, 1]).all():
        raise ValueError("Binary UMAP configurations require group labels 0 and 1")
    if not grouped.empty:
        counts = (
            grouped.groupby(keys + ["groups", "datatype"]).size().unstack(fill_value=0)
        )
        if (
            not counts.Real.eq(counts.Generated).all()
            or not grouped.groupby(keys)["groups"].nunique().eq(2).all()
        ):
            raise ValueError("UMAP binary groups are missing or not source-matched")
    return selected


def summarize(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Means and observed draw ranges; missing selected metrics are errors."""
    if data.empty or not np.isfinite(data[metrics]).all().all():
        raise ValueError("Missing or nonfinite values in selected fidelity metrics")
    long = data.melt(
        id_vars=KEYS + ["cohort", "grouping"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return (
        long.groupby(KEYS + ["cohort", "grouping", "metric"], sort=False)
        .value.agg(mean="mean", low="min", high="max")
        .reset_index()
    )
