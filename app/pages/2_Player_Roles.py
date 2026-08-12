"""Module 2 — classical centrality baseline vs GraphSAGE functional-role embeddings."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _shared import (
    SEASON_LABEL,
    build_network,
    club_label,
    club_lookup,
    get_bundle,
    page_header,
    sidebar_provenance,
    status_banner,
    table,
)
from tacticalgraph.features.centrality import PLAYER_METRICS, player_centrality
from tacticalgraph.viz.pitch import draw_network

st.set_page_config(page_title="M2 · Player Roles", page_icon="🎯", layout="wide")

page_header(
    "🎯 Module 2 — Centrality & Functional Role",
    "Does a learned representation of a player's passing network describe their role better "
    "than classical centrality metrics? Yes — but read the caveat.",
)
sidebar_provenance()
status_banner(2)


# ============================================================== ablation + clustering
@st.cache_data(show_spinner=False)
def _module2_reports() -> dict[str, list[dict]]:
    """Collect the per-seed Module 2 reports from the bundle."""
    bundle = get_bundle()
    directory = bundle.root / "reports"
    ablation, clustering, consistency, stability = [], [], [], []
    for file in sorted(glob.glob(str(directory / "module2_roles_*_seed*.json"))):
        payload = json.loads(Path(file).read_text())
        split = payload["split"]["kind"]
        seed = Path(file).stem.split("seed")[-1]
        for row in payload.get("ablation", []):
            ablation.append({**row, "split": split, "seed": int(seed)})
        if seed == "0":  # clustering is seed-invariant enough; show one to stay readable
            for row in payload.get("clustering", []):
                clustering.append({**row, "split": split})
            for row in payload.get("within_player_consistency", []):
                consistency.append({**row, "split": split})
            # `stability` is the current key; `cross_season_stability` is what reports written
            # before the corpus refactor used. Both are read so an older bundle still renders.
            kind = payload.get("stability_kind", "cross_season")
            for row in payload.get("stability", payload.get("cross_season_stability", [])):
                stability.append({**row, "split": split, "stability_kind": kind})
    return {
        "ablation": ablation,
        "clustering": clustering,
        "consistency": consistency,
        "stability": stability,
    }


data = _module2_reports()
ablation = pd.DataFrame(data["ablation"])


def primary_split(frame: pd.DataFrame) -> str | None:
    """The split whose detail tables are shown.

    Not hardcoded to "cross_season": that split only exists on a two-season corpus, and on the
    Premier League corpus the only kind is "matchweek". Preference order puts the confounded
    split first when it exists, because the within-season control is meant to be read *against*
    it rather than instead of it.
    """
    if frame.empty or "split" not in frame:
        return None
    available = list(frame["split"].unique())
    for preferred in ("cross_season", "matchweek", "within_season"):
        if preferred in available:
            return preferred
    return available[0]

st.subheader("The leakage trap, and the ablation that addresses it")
st.markdown(
    "A player's mean pitch position nearly determines their coarse role (GK 8.8 m → forwards "
    "68.1 m on a 105 m pitch). A model handed `(x, y)` can therefore score well while learning "
    "nothing about passing structure. So the feature set is split three ways and **all three "
    "are always reported**."
)

if not ablation.empty:
    pivot = (
        ablation.groupby(["split", "feature_set"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    pivot["Test accuracy"] = pivot.apply(
        lambda r: f"{r['mean']:.4f} ± {0.0 if pd.isna(r['std']) else r['std']:.4f}", axis=1
    )
    order = {"position": 0, "topology": 1, "both": 2}
    pivot["_o"] = pivot["feature_set"].map(order)
    pivot = pivot.sort_values(["split", "_o"])

    left, right = st.columns([3, 2])
    with left:
        display = pivot[["split", "feature_set", "Test accuracy", "count"]].copy()
        display.columns = ["Split", "Feature set", "Test accuracy (mean ± std)", "Seeds"]
        st.dataframe(display, hide_index=True, width="stretch")
    with right:
        for split in pivot["split"].unique():
            subset = pivot[pivot["split"] == split].set_index("feature_set")
            if {"both", "position"} <= set(subset.index):
                gap = (subset.loc["both", "mean"] - subset.loc["position", "mean"]) * 100
                st.metric(
                    f"both − position ({split})",
                    f"{gap:+.2f} pp",
                    help="The actual claim of Module 2: what passing topology adds beyond "
                         "where a player stands.",
                )

clustering = pd.DataFrame(data["clustering"])
st.subheader("GNN embedding vs classical centrality")
if not clustering.empty:
    shown_split = primary_split(clustering)
    cross = clustering[clustering["split"] == shown_split]
    st.caption(f"Split shown: `{shown_split}`")
    k_choice = st.select_slider("Clusters (k)", sorted(cross["k"].unique()), value=4)
    subset = cross[cross["k"] == k_choice].drop(columns=["split", "k"])
    subset = subset.rename(
        columns={
            "representation": "Representation",
            "ari_coarse4": "ARI (4-class role)",
            "nmi_coarse4": "NMI (4-class role)",
            "silhouette": "Silhouette",
            "ari_fine24": "ARI (fine position)",
            "nmi_fine24": "NMI (fine position)",
        }
    )
    st.dataframe(subset, hide_index=True, width="stretch")
    st.caption(
        "`fine position` = StatsBomb's fine-grained position vocabulary, **never used as a "
        "training label** — the model is supervised with 4 coarse classes only. Agreement here "
        "means the embedding recovered structure nobody supervised."
    )

col_a, col_b = st.columns(2)
with col_a:
    consistency = pd.DataFrame(data["consistency"])
    if not consistency.empty:
        st.markdown("**Within-player consistency** — same player, different matches")
        display = consistency[consistency["split"] == primary_split(consistency)][
            ["representation", "same_player_cosine", "diff_player_cosine", "lift"]
        ]
        display.columns = ["Representation", "same player", "different player", "lift"]
        st.dataframe(display, hide_index=True, width="stretch")
with col_b:
    stability = pd.DataFrame(data["stability"])
    if not stability.empty:
        rows = stability[stability["split"] == primary_split(stability)]
        # Two different measures share this slot depending on the corpus: across providers on a
        # two-season corpus, across halves of one season otherwise. Labelling them the same
        # would misdescribe whichever one is actually shown.
        half_season = (rows["stability_kind"] == "half_season").any()
        if half_season:
            st.markdown("**Half-season stability** — same player, weeks 1-19 vs 20-38")
        else:
            st.markdown("**Cross-season stability** — players matched across providers")
        display = rows[["representation", "matched_cosine", "shuffled_cosine", "lift"]]
        display.columns = ["Representation", "matched", "shuffled", "lift"]
        st.dataframe(display, hide_index=True, width="stretch")
        if half_season:
            st.caption(
                "One provider, one competition — so a low score is the representation's "
                "fault, with no provider change to blame. Players appearing in only one half "
                "are excluded (that is a transfer window, not instability)."
            )
        else:
            st.caption("Conflates role stability with provider robustness — see Limitations.")

st.error(
    "**The honest headline.** The GNN embedding decisively beats classical centrality (ARI "
    "0.516 vs 0.051 at k=4) — centrality measures *how much* a player is involved, not *in "
    "what capacity*. **But passing topology is the minor contributor**: `topology` alone is the "
    "weakest variant everywhere, adding topology to position buys only ~1–1.5 pp, and at k ≥ 6 "
    "`position` alone matches or beats `both` at recovering fine positions. Much of the fine "
    "structure the embedding finds — left versus right — is available directly from `mean_y`."
)

# ============================================================== leaderboard
st.divider()
st.subheader("Most central players")

# `centrality_player_season.parquet` already carries player_name and coarse_role (joined by
# scripts/run_centrality.py), so re-merging the directory here would suffix both columns.
named = table("centrality_player_season.parquet")
missing = {"player_name", "coarse_role"} - set(named.columns)
if missing:
    named = named.merge(
        table("players.parquet")[
            ["season", "provider", "player_id", "player_name", "coarse_role"]
        ],
        on=["season", "provider", "player_id"],
        how="left",
    )

controls = st.columns(4)
season_pick = controls[0].selectbox(
    "Season", sorted(named["season"].unique()), format_func=lambda s: SEASON_LABEL.get(s, s),
    key="lb_season",
)
metric_pick = controls[1].selectbox("Metric", list(PLAYER_METRICS), index=list(PLAYER_METRICS).index("pagerank"))
role_pick = controls[2].multiselect("Role", ["GK", "DEF", "MID", "FWD"], default=["DEF", "MID", "FWD"])
min_matches = controls[3].slider("Minimum matches", 1, 38, 10)

filtered = named[
    (named["season"] == season_pick)
    & (named["coarse_role"].isin(role_pick))
    & (named["n_matches"] >= min_matches)
]
leaderboard = filtered.nlargest(20, metric_pick)[
    ["player_name", "coarse_role", "n_matches", metric_pick, "betweenness", "strength_out"]
].round(4)
leaderboard.columns = ["Player", "Role", "Matches", metric_pick, "betweenness", "strength_out"]
st.dataframe(leaderboard, hide_index=True, width="stretch")
st.caption(
    "Deep-lying playmakers dominating is the sanity check: Jorginho tops or near-tops both "
    "seasons, alongside Pjanić, Hamšík, Biglia, Badelj, Magnanelli and Cigarini."
)

# ============================================================== pitch by metric
st.divider()
st.subheader("Centrality on the pitch")

nodes = table("full_nodes.parquet")
edges = table("full_edges.parquet")
clubs = club_lookup()

pitch_controls = st.columns(3)
pitch_season = pitch_controls[0].selectbox(
    "Season", sorted(nodes["season"].unique()), format_func=lambda s: SEASON_LABEL.get(s, s),
    key="pitch_season",
)
season_clubs = clubs[clubs["season"] == pitch_season].sort_values("club")
pitch_club = pitch_controls[1].selectbox("Club", season_clubs["club"].tolist(), key="pitch_club")
pitch_metric = pitch_controls[2].selectbox(
    "Node size", list(PLAYER_METRICS), index=list(PLAYER_METRICS).index("betweenness")
)

pitch_team = int(season_clubs.loc[season_clubs["club"] == pitch_club, "team_id"].iloc[0])
pitch_provider = season_clubs.loc[season_clubs["club"] == pitch_club, "provider"].iloc[0]

network = build_network(nodes, edges, -1, pitch_team, pitch_season, pitch_provider)
metrics = player_centrality(network)
sizes = dict(zip(metrics["player_id"].astype(int), metrics[pitch_metric].fillna(0.0)))

figure, ax = plt.subplots(figsize=(8.5, 5.4))
draw_network(network, ax, node_metric=sizes, title=f"{pitch_club} — node size = {pitch_metric}")
st.pyplot(figure, width="content")
plt.close(figure)

# ============================================================== embedding + similarity
st.divider()
st.subheader("The learned role space")


@st.cache_data(show_spinner="Projecting embeddings…")
def _projection(use_umap: bool) -> np.ndarray:
    bundle = get_bundle()
    _, dims = bundle.embedding_matrix()
    matrix = StandardScaler().fit_transform(np.nan_to_num(dims.to_numpy(dtype=np.float64)))
    if use_umap:
        import umap

        return umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=0).fit_transform(matrix)
    return PCA(n_components=2, random_state=0).fit_transform(matrix)


@st.cache_data(show_spinner=False)
def _player_centroids() -> tuple[pd.DataFrame, np.ndarray]:
    """One embedding per (season, player): the mean over their team-match nodes."""
    bundle = get_bundle()
    identity, dims = bundle.embedding_matrix()
    frame = pd.concat([identity.reset_index(drop=True), dims.reset_index(drop=True)], axis=1)
    dim_columns = list(dims.columns)
    grouped = frame.groupby(["season", "provider", "player_id"], as_index=False)[dim_columns].mean()
    counts = (
        frame.groupby(["season", "provider", "player_id"]).size().rename("n_matches").reset_index()
    )
    grouped = grouped.merge(counts, on=["season", "provider", "player_id"])
    grouped = grouped.merge(
        table("players.parquet")[
            ["season", "provider", "player_id", "player_name", "coarse_role"]
        ],
        on=["season", "provider", "player_id"],
        how="left",
    )
    matrix = StandardScaler().fit_transform(
        np.nan_to_num(grouped[dim_columns].to_numpy(dtype=np.float64))
    )
    return grouped.drop(columns=dim_columns), matrix


tab_scatter, tab_similar = st.tabs(["Embedding map", "Find similar players"])

with tab_scatter:
    use_umap = st.checkbox(
        "Use UMAP instead of PCA (slower, better separation)", value=False
    )
    reduced = _projection(use_umap)
    identity, _ = get_bundle().embedding_matrix()
    plot_frame = identity.reset_index(drop=True).copy()
    plot_frame = plot_frame.merge(
        table("players.parquet")[["season", "provider", "player_id", "coarse_role"]],
        on=["season", "provider", "player_id"],
        how="left",
    )
    figure, ax = plt.subplots(figsize=(9, 6))
    for role in ("GK", "DEF", "MID", "FWD"):
        mask = (plot_frame["coarse_role"] == role).to_numpy()
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=3, alpha=0.45, label=role)
    ax.legend(markerscale=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{'UMAP' if use_umap else 'PCA'} of the GNN embedding, coloured by the 4-class role "
        "it was trained on"
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab_similar:
    st.markdown(
        "Nearest neighbours in the embedding space — the most direct demonstration of what the "
        "representation learned. Each player is the mean of their team-match embeddings."
    )
    centroids, matrix = _player_centroids()
    eligible = centroids[centroids["n_matches"] >= 10].copy()

    pick_columns = st.columns([2, 1, 1])
    labels = (
        eligible["player_name"].fillna("?")
        + "  ("
        + eligible["coarse_role"].fillna("?")
        + ", "
        + eligible["season"].str.replace("-", "/")
        + ")"
    )
    label_to_index = dict(zip(labels, eligible.index))
    chosen_label = pick_columns[0].selectbox("Player", sorted(labels.tolist()))
    top_n = pick_columns[1].slider("Neighbours", 3, 15, 8)
    cross_only = pick_columns[2].checkbox("Other season only", value=False)

    target_index = label_to_index[chosen_label]
    target_row = centroids.loc[target_index]
    target_vector = matrix[centroids.index.get_loc(target_index)]

    normalised = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    target_unit = target_vector / max(np.linalg.norm(target_vector), 1e-9)
    similarity = normalised @ target_unit

    result = centroids.copy()
    result["similarity"] = similarity
    result = result.drop(index=target_index)
    if cross_only:
        result = result[result["season"] != target_row["season"]]
    result = result[result["n_matches"] >= 10]

    top = result.nlargest(top_n, "similarity")[
        ["player_name", "coarse_role", "season", "n_matches", "similarity"]
    ].round(4)
    top.columns = ["Player", "Role", "Season", "Matches", "Cosine similarity"]
    st.dataframe(top, hide_index=True, width="stretch")
    st.caption(
        f"Reference: **{target_row['player_name']}** ({target_row['coarse_role']}, "
        f"{target_row['season']}, {int(target_row['n_matches'])} matches). Similarity is "
        "computed on standardised embeddings; 'Other season only' forces cross-provider "
        "matches, which is the harder test."
    )
