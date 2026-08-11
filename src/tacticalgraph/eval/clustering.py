"""Module 2 evaluation: does the GNN embedding beat classical centrality?

Four comparisons, in increasing order of how much they actually prove:

1. **Agreement with the 4-class role** (ARI/NMI). Weakest: it is the label the model trained
   on, so a good score partly restates the training objective. Reported for completeness.

2. **Agreement with StatsBomb's 24-class position** -- the real external signal. Available
   on 2015/16 only, never used as a training label. If embedding clusters align with the 24
   fine-grained positions better than centrality clusters do, the embedding has recovered
   structure that nobody supervised.

3. **Within-player consistency.** Does the same player in different matches land in the same
   region of the space? Measured as mean same-player cosine similarity against a
   different-player baseline. A representation of *role* should be stable per player; one
   that mostly encodes match noise will not be.

4. **Cross-season stability.** Same, for players appearing in both seasons -- which also
   probes whether the representation survives the provider switch.

Every comparison runs on both representations with the same clustering procedure, so the
only thing varying is the feature space.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (4, 6, 8, 10, 12)


def cluster_and_score(
    features: np.ndarray,
    coarse_labels: pd.Series,
    fine_labels: pd.Series | None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    seed: int = 0,
    label: str = "",
) -> pd.DataFrame:
    """K-means over a feature space, scored against coarse and fine role labels."""
    matrix = StandardScaler().fit_transform(np.nan_to_num(features, nan=0.0))

    coarse = coarse_labels.to_numpy()
    coarse_ok = pd.notna(coarse)

    rows = []
    for k in k_values:
        assignments = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix)
        entry: dict[str, object] = {
            "representation": label,
            "k": k,
            "ari_coarse4": round(
                adjusted_rand_score(coarse[coarse_ok], assignments[coarse_ok]), 4
            ),
            "nmi_coarse4": round(
                normalized_mutual_info_score(coarse[coarse_ok], assignments[coarse_ok]), 4
            ),
            "silhouette": round(float(silhouette_score(matrix, assignments)), 4),
        }
        if fine_labels is not None:
            fine = fine_labels.to_numpy()
            fine_ok = pd.notna(fine)
            if fine_ok.sum() > k:
                entry["ari_fine24"] = round(
                    adjusted_rand_score(fine[fine_ok], assignments[fine_ok]), 4
                )
                entry["nmi_fine24"] = round(
                    normalized_mutual_info_score(fine[fine_ok], assignments[fine_ok]), 4
                )
        rows.append(entry)
    return pd.DataFrame(rows)


def _cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    unit = matrix / norms
    return unit @ unit.T


def within_player_consistency(
    features: np.ndarray,
    player_ids: pd.Series,
    seed: int = 0,
    max_players: int = 400,
    label: str = "",
) -> dict[str, float]:
    """Mean same-player similarity vs a random different-player baseline.

    The gap is the informative quantity: a representation of functional role should place a
    player's own matches closer together than two arbitrary players. Reported as a `lift`
    (same minus different) so the two representations can be compared directly despite
    living on different scales.
    """
    matrix = StandardScaler().fit_transform(np.nan_to_num(features, nan=0.0))
    ids = player_ids.to_numpy()

    rng = np.random.default_rng(seed)
    counts = pd.Series(ids).value_counts()
    repeated = counts[counts >= 2].index.to_numpy()
    if len(repeated) == 0:
        return {"representation": label, "n_players": 0}
    if len(repeated) > max_players:
        repeated = rng.choice(repeated, max_players, replace=False)

    same_scores, diff_scores = [], []
    for player in repeated:
        own = np.flatnonzero(ids == player)
        if len(own) < 2:
            continue
        if len(own) > 12:  # cap so prolific starters do not dominate the average
            own = rng.choice(own, 12, replace=False)
        similarity = _cosine_matrix(matrix[own])
        upper = similarity[np.triu_indices(len(own), k=1)]
        same_scores.append(float(upper.mean()))

        others = np.flatnonzero(ids != player)
        sample = rng.choice(others, min(len(own), len(others)), replace=False)
        cross = _cosine_matrix(np.vstack([matrix[own], matrix[sample]]))[
            : len(own), len(own) :
        ]
        diff_scores.append(float(cross.mean()))

    same = float(np.mean(same_scores))
    diff = float(np.mean(diff_scores))
    return {
        "representation": label,
        "n_players": len(same_scores),
        "same_player_cosine": round(same, 4),
        "diff_player_cosine": round(diff, 4),
        "lift": round(same - diff, 4),
    }


def cross_season_stability(
    features: np.ndarray,
    meta: pd.DataFrame,
    player_matches: pd.DataFrame,
    label: str = "",
    seed: int = 0,
) -> dict[str, float]:
    """Similarity of a player's mean embedding across the two seasons.

    `player_matches` maps StatsBomb player ids to Wyscout ids (from `aliases.match_players`).
    Because the seasons come from different providers, this measures role stability and
    provider robustness together -- the report says so explicitly rather than claiming it
    isolates either one.
    """
    matrix = StandardScaler().fit_transform(np.nan_to_num(features, nan=0.0))
    frame = meta.reset_index(drop=True)

    centroids: dict[tuple[str, int], np.ndarray] = {}
    for (provider, player_id), group in frame.groupby(["provider", "player_id"]):
        centroids[(provider, int(player_id))] = matrix[group.index.to_numpy()].mean(axis=0)

    pairs = []
    for row in player_matches.itertuples(index=False):
        left = centroids.get(("statsbomb", int(row.player_id_sb)))
        right = centroids.get(("wyscout", int(row.player_id_wy)))
        if left is not None and right is not None:
            pairs.append((left, right))

    if not pairs:
        return {"representation": label, "n_players": 0}

    left_stack = np.vstack([p[0] for p in pairs])
    right_stack = np.vstack([p[1] for p in pairs])

    def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        na[na < 1e-9] = 1.0
        nb[nb < 1e-9] = 1.0
        return np.sum(a * b, axis=1) / (na * nb)

    matched = _cos(left_stack, right_stack)

    # Baseline: the same players, shuffled, so any similarity from the space's global shape
    # is subtracted out rather than counted as stability.
    rng = np.random.default_rng(seed)
    shuffled = _cos(left_stack, right_stack[rng.permutation(len(pairs))])

    return {
        "representation": label,
        "n_players": len(pairs),
        "matched_cosine": round(float(matched.mean()), 4),
        "shuffled_cosine": round(float(shuffled.mean()), 4),
        "lift": round(float(matched.mean() - shuffled.mean()), 4),
    }


def compare_representations(results: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack per-representation clustering tables into one comparison."""
    return pd.concat(results, ignore_index=True).sort_values(
        ["k", "representation"]
    ).reset_index(drop=True)
