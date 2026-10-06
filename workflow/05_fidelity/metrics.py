"""Python-side fidelity helpers: sampling, source ARI, CCC, MAD, UMAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from umap import UMAP

META_COLS = ("groups", "samples", "Sample", "sample", "Unnamed: 0")


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    den = vx + vy + (mx - my) ** 2
    if den == 0:
        return float("nan")
    return 2.0 * sxy / den


def clean_groups(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(raw):
        return raw.apply(lambda x: str(int(round(x))) if pd.notna(x) else "NA")
    return raw.astype(str)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.select_dtypes(include=np.number).copy()
    for col in META_COLS:
        if col in feat.columns:
            feat = feat.drop(columns=[col])
    return feat


def load_table(path) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(path)
    groups = clean_groups(df["groups"]) if "groups" in df.columns else None
    return feature_frame(df), groups


def replace_zero_sd(real: np.ndarray, gen: np.ndarray) -> np.ndarray:
    out = gen.copy()
    zero = np.nanstd(out, axis=0, ddof=1) == 0
    if np.any(zero):
        out[:, zero] = real[:, zero]
    return out


def clip_nonfinite(mat: np.ndarray) -> np.ndarray:
    # pandas 3 CoW: DataFrame[cols].to_numpy() is a read-only view.
    # np.asarray(..., dtype=float) does not copy, so in-place writes fail.
    out = np.array(mat, dtype=float, copy=True)
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out


def stratified_sample(
    gen_feat: pd.DataFrame,
    gen_groups: pd.Series,
    real_groups: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    real_groups = real_groups.astype(str)
    gen_groups = gen_groups.astype(str)
    parts_x, parts_g = [], []
    for gid, n in real_groups.value_counts().items():
        n = int(n)
        pool = gen_feat.loc[gen_groups.to_numpy() == gid]
        if pool.empty or n == 0:
            continue
        replace = len(pool) < n
        idx = rng.choice(len(pool), size=n, replace=replace)
        parts_x.append(pool.iloc[idx])
        parts_g.append(pd.Series([gid] * n, dtype=str))
    if not parts_x:
        n = min(len(gen_feat), len(real_groups))
        idx = rng.choice(len(gen_feat), size=n, replace=False)
        return gen_feat.iloc[idx].reset_index(drop=True), gen_groups.iloc[idx].reset_index(drop=True)
    return (
        pd.concat(parts_x, ignore_index=True),
        pd.concat(parts_g, ignore_index=True),
    )


def sample_to_n(
    gen_feat: pd.DataFrame,
    gen_groups: pd.Series | None,
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Draw n generated rows without group stratification."""
    rng = np.random.default_rng(seed)
    n = min(int(n), len(gen_feat))
    if n <= 0:
        empty_g = None if gen_groups is None else gen_groups.iloc[0:0].reset_index(drop=True)
        return gen_feat.iloc[0:0].reset_index(drop=True), empty_g
    replace = len(gen_feat) < n
    idx = rng.choice(len(gen_feat), size=n, replace=replace)
    feat = gen_feat.iloc[idx].reset_index(drop=True)
    groups = None if gen_groups is None else gen_groups.iloc[idx].reset_index(drop=True)
    return feat, groups


def source_ari(real: np.ndarray, gen: np.ndarray) -> float:
    """Signed ARI of empirical/generated source labels versus Ward clusters.

    Keep full precision; zero is the chance-adjusted agreement baseline.
    Preserve the study's existing zero-SD replacement and clustering method.
    """
    gen = replace_zero_sd(real, gen)
    combined = np.vstack([real, gen])
    datatype = np.array(["Real"] * len(real) + ["Generated"] * len(gen))
    if combined.shape[0] < 4 or combined.shape[1] < 1:
        return float("nan")
    try:
        dist = pdist(combined, metric="euclidean")
        if not np.isfinite(dist).all() or dist.size == 0:
            return float("nan")
        z = linkage(dist, method="ward")
        clusters = fcluster(z, t=2, criterion="maxclust")
        return float(adjusted_rand_score(datatype, clusters))
    except ValueError:
        return float("nan")


def r_mad(x: np.ndarray) -> float:
    """R stats::mad default: 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def medae(x: np.ndarray) -> float:
    """Median absolute error: median_j |x_j|."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x)))


def mad_metrics(real: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    """Per-feature moment errors on the current (already log2) scale.

    For each feature, q in {mean, SD, CV}:
      Delta_j = q_j(generated) - q_j(real)
      MedAE = median_j |Delta_j|
      MAD   = 1.4826 * median_j |Delta_j - median(Delta)|   (R stats::mad)
    """
    mu_r, mu_g = real.mean(axis=0), gen.mean(axis=0)
    sd_r, sd_g = real.std(axis=0, ddof=1), gen.std(axis=0, ddof=1)
    cv_r = sd_r / (mu_r + 1e-9)
    cv_g = sd_g / (mu_g + 1e-9)
    d_mu = mu_g - mu_r
    d_sd = sd_g - sd_r
    d_cv = cv_g - cv_r
    return {
        "ccc_meandiff": round(lin_ccc(mu_r, mu_g), 4),
        "mad_mean": round(r_mad(d_mu), 6),
        "mad_sd": round(r_mad(d_sd), 6),
        "mad_cv": round(r_mad(d_cv), 6),
        "medae_mean": round(medae(d_mu), 6),
        "medae_sd": round(medae(d_sd), 6),
        "medae_cv": round(medae(d_cv), 6),
    }


def umap_coordinates(
    real: np.ndarray,
    gen: np.ndarray,
    groups_real: pd.Series | None,
    groups_gen: pd.Series | None,
    seed: int,
) -> pd.DataFrame:
    combined = np.vstack([real, gen])
    reducer = UMAP(random_state=seed, n_jobs=1)
    layout = reducer.fit_transform(combined)
    n_r = len(real)
    groups = (
        list(groups_real.astype(str)) + list(groups_gen.astype(str))
        if groups_real is not None and groups_gen is not None
        else ["NA"] * len(combined)
    )
    return pd.DataFrame(
        {
            "UMAP1": layout[:, 0],
            "UMAP2": layout[:, 1],
            "datatype": ["Real"] * n_r + ["Generated"] * (len(combined) - n_r),
            "groups": groups,
        }
    )
