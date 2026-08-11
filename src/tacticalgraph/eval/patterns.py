"""Module 4 evaluation: are the discovered patterns real, and do they precede shots?

There is no ground truth for "style of play", so the evaluation rests on three things that
can be measured:

1. **Shot lift.** Does a cluster's P(shot) differ from the corpus base rate (9.7%) by more
   than sampling noise? Wilson intervals rather than normal approximations, because several
   clusters sit near the 10% range where the normal interval misbehaves.
2. **Cross-season stability.** Fit the clustering on the training split, apply it to
   2017/18, and check whether each cluster keeps its shot rate and its share of chains. A
   pattern that only exists in one provider's data is an artefact.
3. **Separation.** Silhouette, for the baseline and learned representations on the same k
   sweep.

What is deliberately *not* claimed: that a cluster is tactically meaningful. That requires a
human, and `scripts/review_patterns.py` produces the sheet for one.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (4, 6, 8, 10, 12)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because cluster shot rates sit near 0.1 with
    cluster sizes ranging from a few hundred to tens of thousands, where the normal interval
    can extend below zero.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def fit_clustering(
    features: np.ndarray, train_mask: np.ndarray, k: int, seed: int = 0
) -> tuple[np.ndarray, KMeans, StandardScaler]:
    """Fit k-means on training rows only, then assign every row.

    Fitting the scaler and the centroids on the whole corpus would let the held-out season
    shape the clusters it is then evaluated in.
    """
    scaler = StandardScaler().fit(np.nan_to_num(features[train_mask]))
    scaled_all = scaler.transform(np.nan_to_num(features))
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled_all[train_mask])
    return model.predict(scaled_all), model, scaler


def shot_lift(
    chain_table: pd.DataFrame, labels: np.ndarray, subset_mask: np.ndarray | None = None
) -> pd.DataFrame:
    """P(shot | cluster) with Wilson intervals, against the base rate on the same subset."""
    frame = chain_table.copy()
    frame["cluster"] = labels
    if subset_mask is not None:
        frame = frame[subset_mask]
    if frame.empty:
        return pd.DataFrame()

    base_rate = float(frame["ends_in_shot"].mean())

    rows = []
    for cluster, group in frame.groupby("cluster"):
        successes = int(group["ends_in_shot"].sum())
        total = len(group)
        low, high = wilson_interval(successes, total)
        rate = successes / total
        rows.append(
            {
                "cluster": int(cluster),
                "n_chains": total,
                "share": round(total / len(frame), 4),
                "shot_rate": round(rate, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "base_rate": round(base_rate, 4),
                "lift": round(rate / base_rate, 3) if base_rate else float("nan"),
                # "significant" means the interval excludes the base rate, i.e. this cluster's
                # shot rate is not explained by sampling noise alone.
                "differs_from_base": bool(low > base_rate or high < base_rate),
            }
        )
    return pd.DataFrame(rows).sort_values("shot_rate", ascending=False).reset_index(drop=True)


def sweep_k(
    features: np.ndarray,
    chain_table: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    label: str = "",
    seed: int = 0,
) -> pd.DataFrame:
    """Cluster quality and shot discrimination across k, for one representation."""
    scaled = StandardScaler().fit_transform(np.nan_to_num(features))
    rows = []
    for k in k_values:
        labels, _, _ = fit_clustering(features, train_mask, k, seed=seed)

        # Silhouette on a subsample: exact silhouette over ~180k chains is O(n^2).
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(labels), size=min(8000, len(labels)), replace=False)
        try:
            separation = float(silhouette_score(scaled[sample], labels[sample]))
        except ValueError:
            separation = float("nan")

        train_lift = shot_lift(chain_table, labels, train_mask)
        test_lift = shot_lift(chain_table, labels, test_mask)

        rows.append(
            {
                "representation": label,
                "k": k,
                "silhouette": round(separation, 4),
                # Spread of shot rates across clusters: how much the clustering separates
                # dangerous possessions from harmless ones.
                "shot_rate_spread_train": round(
                    float(train_lift["shot_rate"].max() - train_lift["shot_rate"].min()), 4
                ),
                "shot_rate_spread_test": round(
                    float(test_lift["shot_rate"].max() - test_lift["shot_rate"].min()), 4
                ),
                "max_lift_test": round(float(test_lift["lift"].max()), 3),
                "clusters_differing_test": int(test_lift["differs_from_base"].sum()),
            }
        )
    return pd.DataFrame(rows)


def cross_season_stability(
    chain_table: pd.DataFrame, labels: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> pd.DataFrame:
    """Does each cluster keep its shot rate and its share across the season/provider change?

    A cluster whose share collapses or whose shot rate moves wildly is describing an
    annotation convention rather than a way of playing.
    """
    train_lift = shot_lift(chain_table, labels, train_mask).set_index("cluster")
    test_lift = shot_lift(chain_table, labels, test_mask).set_index("cluster")

    joined = train_lift.join(test_lift, lsuffix="_train", rsuffix="_test", how="outer")
    joined["shot_rate_delta"] = (
        joined["shot_rate_test"] - joined["shot_rate_train"]
    ).round(4)
    joined["share_ratio"] = (
        joined["share_test"] / joined["share_train"].replace(0, np.nan)
    ).round(3)
    # Overlapping Wilson intervals mean the shot rate is stable within noise.
    joined["rate_stable"] = (
        (joined["ci_low_test"] <= joined["ci_high_train"])
        & (joined["ci_low_train"] <= joined["ci_high_test"])
    )
    return joined[
        ["n_chains_train", "n_chains_test", "shot_rate_train", "shot_rate_test",
         "shot_rate_delta", "share_train", "share_test", "share_ratio", "rate_stable"]
    ].reset_index()


def compare_representations(sweeps: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(sweeps, ignore_index=True).sort_values(["k", "representation"]).reset_index(
        drop=True
    )
