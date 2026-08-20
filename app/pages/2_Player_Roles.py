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
        # Every seed is collected and averaged below. An earlier version showed seed 0 only,
        # on the assumption clustering was "seed-invariant enough" -- it is not. On the Premier
        # League corpus gnn-both at k=4 scores 0.631 / 0.382 / 0.378 across seeds, so showing
        # seed 0 displayed 0.631 for a 0.463 mean and flattered the embedding by ~35%.
        for row in payload.get("clustering", []):
            clustering.append({**row, "split": split, "seed": int(seed)})
        for row in payload.get("within_player_consistency", []):
            consistency.append({**row, "split": split, "seed": int(seed)})
        # `stability` is the current key; `cross_season_stability` is what reports written
        # before the corpus refactor used. Both are read so an older bundle still renders.
        kind = payload.get("stability_kind", "cross_season")
        for row in payload.get("stability", payload.get("cross_season_stability", [])):
            stability.append({**row, "split": split, "stability_kind": kind, "seed": int(seed)})
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


def mean_over_seeds(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average per-seed rows down to one row per representation.

    Every table on this page is now built from all seeds, so anything displayed per
    representation has to be collapsed here rather than showing three near-duplicate rows.
    """
    return (
        frame.groupby("representation", as_index=False)[columns]
        .mean()
        .sort_values("representation")
    )


st.subheader("The leakage trap, and the ablation that addresses it")
st.markdown(
    "A player's mean pitch position nearly determines their coarse role (GK 8.8 m → forwards "
    "68.1 m on a 105 m pitch). A model handed `(x, y)` can therefore score well while learning "
    "nothing about passing structure. So the feature set is split into explicit variants and "
    "**all of them are always reported**. `position`, `topology` and `both` are the original "
    "three; `direction`, `topology+direction` and `all` add pass direction, which is what "
    "separates a target man from a recycler when volume cannot."
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
    # Explicit ordering so the ladder reads position -> volume -> union -> direction variants;
    # an unknown name sorts last rather than becoming NaN and landing in an arbitrary place.
    order = {"position": 0, "topology": 1, "both": 2,
             "direction": 3, "topology+direction": 4, "all": 5}
    pivot["_o"] = pivot["feature_set"].map(order).fillna(len(order))
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
                    help="What passing *volume* adds beyond where a player stands.",
                )
            if {"all", "position"} <= set(subset.index):
                gap_all = (subset.loc["all", "mean"] - subset.loc["position", "mean"]) * 100
                st.metric(
                    f"all − position ({split})",
                    f"{gap_all:+.2f} pp",
                    help="What volume *and direction* add. This is the larger result: direction "
                         "more than doubles the graph's contribution.",
                )

clustering = pd.DataFrame(data["clustering"])
st.subheader("GNN embedding vs classical centrality")
if not clustering.empty:
    shown_split = primary_split(clustering)
    cross = clustering[clustering["split"] == shown_split]
    st.caption(f"Split shown: `{shown_split}`")
    k_choice = st.select_slider("Clusters (k)", sorted(cross["k"].unique()), value=4)
    subset = cross[cross["k"] == k_choice]
    METRIC_LABELS = {
        "ari_coarse4": "ARI (4-class role)",
        "nmi_coarse4": "NMI (4-class role)",
        "silhouette": "Silhouette",
        "ari_fine24": "ARI (fine position)",
        "nmi_fine24": "NMI (fine position)",
    }
    # Mean ± std over seeds. k-means on a learned embedding is genuinely seed-sensitive here,
    # so a single seed's ARI is not a result -- the spread is part of the finding.
    n_seeds = subset["seed"].nunique()
    grouped = subset.groupby("representation", as_index=False)[list(METRIC_LABELS)].agg(
        ["mean", "std"]
    )
    grouped.columns = ["representation"] + [
        f"{m}_{stat}" for m in METRIC_LABELS for stat in ("mean", "std")
    ]
    display = pd.DataFrame({"Representation": grouped["representation"]})
    for metric, label in METRIC_LABELS.items():
        means, stds = grouped[f"{metric}_mean"], grouped[f"{metric}_std"].fillna(0.0)
        display[label] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(means, stds)]
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(f"Mean ± standard deviation over {n_seeds} seed(s).")
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
        display = mean_over_seeds(
            consistency[consistency["split"] == primary_split(consistency)],
            ["same_player_cosine", "diff_player_cosine", "lift"],
        )
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
        display = mean_over_seeds(rows, ["matched_cosine", "shuffled_cosine", "lift"])
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

# The headline is computed from the reports actually loaded, never written in prose: this page
# renders whichever corpus was last exported, and the two corpora disagree about how much
# topology contributes. Hardcoding Serie A's numbers here misdescribed a Premier League bundle.
def _headline() -> str:
    parts: list[str] = []
    if not clustering.empty:
        at_k4 = clustering[(clustering["split"] == primary_split(clustering)) & (clustering["k"] == 4)]
        by_rep = at_k4.groupby("representation")["ari_coarse4"].mean()
        gnn = by_rep[[r for r in by_rep.index if r.startswith("gnn-both")]]
        base = by_rep[[r for r in by_rep.index if "centrality" in r]]
        if not gnn.empty and not base.empty:
            g, b = float(gnn.iloc[0]), float(base.iloc[0])
            ratio = f" (~{g / b:.0f}×)" if b > 0 else ""
            parts.append(
                f"The GNN embedding beats classical centrality on role alignment — ARI "
                f"**{g:.3f} vs {b:.3f}** at k=4{ratio}. Centrality measures *how much* a player "
                f"is involved, not *in what capacity*."
            )
    if not ablation.empty:
        pos = ablation[ablation["feature_set"] == "position"]["test_acc"].mean()
        both = ablation[ablation["feature_set"] == "both"]["test_acc"].mean()
        topo = ablation[ablation["feature_set"] == "topology"]["test_acc"].mean()
        if pd.notna(pos) and pd.notna(both):
            parts.append(
                f"**Passing *volume* is the minor contributor**: adding it to position buys only "
                f"**{(both - pos) * 100:+.2f} pp** ({pos:.4f} → {both:.4f}), and volume alone "
                f"reaches just {topo:.4f}."
            )
        # The `all` variant adds pass direction, which is a different claim from volume and the
        # only graph feature group in this project that beats its own seed noise by a wide margin.
        # Reporting only the volume gap here would understate what the graph contributes.
        every = ablation[ablation["feature_set"] == "all"]["test_acc"].mean()
        if pd.notna(pos) and pd.notna(every) and pd.notna(both):
            parts.append(
                f"**But pass *direction* is not.** Adding progression made/received, pass length "
                f"and progressive share takes the graph's contribution to "
                f"**{(every - pos) * 100:+.2f} pp** ({every:.4f}) — more than double what volume "
                f"buys, and several times the seed spread rather than inside it. Volume measures "
                f"*how much* a player passes and is largely positional; direction measures *what "
                f"kind*, and separates a target man from a recycler who share a degree."
            )
    if not parts:
        return ""
    return "**The honest headline.** " + " ".join(parts)


headline = _headline()
if headline:
    st.error(headline)

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

# Three ways to rank the same graph, so the volume-proxy limitation is visible in the app
# rather than only in the docs. "pass volume" is the classical baseline; "xT-weighted" reweights
# every edge by the threat it created; "role-relative" z-scores within coarse role.
BASIS_SUFFIX = {
    "pass volume": "",
    "xT-weighted": "_xt",
    "role-relative (z)": "_z",
    "position-residual (r)": "_r",
}
THREAT_CHOICES = [
    m
    for m in ("xt_generated", "shot_involvement", "shot_conversion")
    if m in named.columns
]
METRIC_CHOICES = list(PLAYER_METRICS) + THREAT_CHOICES

controls = st.columns(5)
season_pick = controls[0].selectbox(
    "Season", sorted(named["season"].unique()), format_func=lambda s: SEASON_LABEL.get(s, s),
    key="lb_season",
)
metric_pick = controls[1].selectbox(
    "Metric", METRIC_CHOICES, index=METRIC_CHOICES.index("pagerank")
)
basis_pick = controls[2].selectbox("Weighting", list(BASIS_SUFFIX), index=0)
role_pick = controls[3].multiselect("Role", ["GK", "DEF", "MID", "FWD"], default=["DEF", "MID", "FWD"])
min_matches = controls[4].slider("Minimum matches", 1, 38, 10)

column = f"{metric_pick}{BASIS_SUFFIX[basis_pick]}"
if column not in named.columns:
    # `degree_*` count edges and ignore their weight, so they have no xT-weighted twin.
    st.info(
        f"`{metric_pick}` counts edges without regard to their weight, so it has no "
        f"{basis_pick} version. Showing the raw metric."
    )
    column = metric_pick

filtered = named[
    (named["season"] == season_pick)
    & (named["coarse_role"].isin(role_pick))
    & (named["n_matches"] >= min_matches)
]
extra = [c for c in ("betweenness", "strength_out") if c != column]
leaderboard = filtered.nlargest(20, column)[
    ["player_name", "coarse_role", "n_matches", column, *extra]
].round(4)
leaderboard.columns = ["Player", "Role", "Matches", column, *extra]
st.dataframe(leaderboard, hide_index=True, width="stretch")
st.caption(
    "Switch **Weighting** to see the limitation directly. On pass volume the list is almost all "
    "midfielders — 84% of the top 50 by `degree_total`, against a 31% population share, and no "
    "goalkeeper anywhere. xT-weighting does not fix that so much as invert it: `pagerank` swings "
    "to 90% forwards, because xThreat is a spatial surface and weighting by it makes centrality "
    "*more* positional, not less (R² against pitch position 0.42 → 0.77). Role-relative z-scores "
    "make every role rankable, but that is largely true by construction. "
    "**position-residual** subtracts a quadratic fit on mean pitch position, leaving whatever is "
    "not explained by where the player stands — the only option here that is neither a volume "
    "proxy nor representative by construction."
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
