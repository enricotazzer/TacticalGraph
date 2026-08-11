"""Module 3 — match result prediction. SPECIFICATION plus the real inputs that exist."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from _shared import (
    SEASON_LABEL,
    club_label,
    page_header,
    sidebar_provenance,
    status_banner,
    table,
)
from tacticalgraph.data.schema import PROVIDER_COMPARABLE_TYPES

st.set_page_config(page_title="M3 · Result Prediction", page_icon="📈", layout="wide")

page_header(
    "📈 Module 3 — Match Result Prediction (GNN + Transformer)",
    "Not implemented. This page states the design, names the baseline it must beat, and shows "
    "the real inputs already built for it.",
)
sidebar_provenance()
status_banner(3)

# ================================================================== the baseline ladder
st.subheader("The baseline ladder — and why B0 is the one that matters")

st.markdown(
    "In-match result prediction is **dominated by the current scoreline**. A graph model that "
    "beats *possession and shot counts* but not *the scoreline plus minutes remaining* has "
    "demonstrated nothing. So the ladder is explicit and B0 is mandatory:"
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Rung": "B0",
                "Features": "scoreline + minutes remaining only",
                "Model": "multinomial logistic regression",
                "Why it exists": "The dominant signal. Omitting it would make every later "
                                 "comparison dishonest.",
            },
            {
                "Rung": "B1",
                "Features": "B0 + possession, shots, accumulated xThreat",
                "Model": "logistic regression",
                "Why it exists": "Isolates what conventional aggregate stats add over the score.",
            },
            {
                "Rung": "B2",
                "Features": "full tabular set + rolling pre-match form",
                "Model": "gradient boosting (sklearn HistGradientBoosting)",
                "Why it exists": "The strong tabular baseline the graph model must beat.",
            },
            {
                "Rung": "Ours",
                "Features": "sequence of passing-network graphs",
                "Model": "GraphSAGE per window → small causal Transformer",
                "Why it exists": "Only worth keeping if it beats B2 at every checkpoint.",
            },
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.info(
    "**Metrics will be log-loss, Brier score, accuracy and reliability diagrams reported per "
    "checkpoint, not just at full time.** Calibration is a first-class result: a confidently "
    "wrong in-match probability is worse than an uncertain one."
)

# ================================================================== real inputs: graphs
st.divider()
st.subheader("Input 1 — the windowed graph sequences already exist")

nodes = table("windowed_sample_nodes.parquet")
edges = table("windowed_sample_edges.parquet")
games = table("games.parquet").set_index("game_id")

st.markdown(
    "**24,320 windowed networks** were built in Module 1: a 15-minute window sliding on a "
    "5-minute stride, 16 steps per match, averaging 11.5 nodes and 39.5 edges per window. The "
    "window length was chosen for exactly this purpose — a 5-minute window would leave ~13 "
    "edges per graph, mostly isolated nodes."
)
st.caption(
    f"The committed demo bundle carries a {nodes['game_id'].nunique()}-match subsample "
    "(the full windowed tables are ~17 MB, too large to ship)."
)

available = sorted(nodes["game_id"].unique())


def _label(game_id: int) -> str:
    if game_id not in games.index:
        return str(game_id)
    row = games.loc[game_id]
    return (
        f"MW{int(row['game_day']):02d} · "
        f"{club_label(row['provider'], row['home_team_id'])} vs "
        f"{club_label(row['provider'], row['away_team_id'])} · "
        f"{SEASON_LABEL.get(row['season'], row['season'])}"
    )


chosen = st.selectbox("Inspect a real window sequence", available, format_func=_label)

sequence = (
    edges[edges["game_id"] == chosen]
    .groupby(["window_index", "team_id"])
    .agg(edges=("weight", "size"), passes=("weight", "sum"))
    .reset_index()
)
node_sequence = (
    nodes[nodes["game_id"] == chosen]
    .groupby(["window_index", "team_id"])
    .size()
    .rename("nodes")
    .reset_index()
)
sequence = sequence.merge(node_sequence, on=["window_index", "team_id"], how="outer").fillna(0)

row = games.loc[chosen]
figure, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharex=True)
for team_id in sequence["team_id"].unique():
    subset = sequence[sequence["team_id"] == team_id].sort_values("window_index")
    name = club_label(row["provider"], team_id)
    axes[0].plot(subset["window_index"], subset["passes"], marker="o", label=name)
    axes[1].plot(subset["window_index"], subset["edges"], marker="o", label=name)
axes[0].set_title("completed passes per 15-min window")
axes[1].set_title("distinct pass connections per window")
for ax in axes:
    ax.set_xlabel("window index (0 = 0–15', 15 = 75–90')")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
st.pyplot(figure, width="stretch")
plt.close(figure)
st.caption(
    "This is real data — the graph sequence a Transformer would consume. It is **not** a "
    "prediction; no model exists yet."
)

# ================================================================== real inputs: labels
st.divider()
st.subheader("Input 2 — the labels exist for both seasons")

label_columns = st.columns(2)
with label_columns[0]:
    st.markdown("**Final result** (the target)")
    st.markdown(
        "- 2015/16: `home_score` / `away_score` present directly in the StatsBomb match index.\n"
        "- 2017/18: **null** in the store, but recoverable **380/380** from Wyscout's "
        "`matches_Italy.json` (`teamsData[*].score`, cross-checkable against the `label` and "
        "`winner` fields). Verified.\n"
        "- The exporter will need to backfill these before Module 3 can train."
    )
with label_columns[1]:
    st.markdown("**Running scoreline** (needed for B0 at each checkpoint)")
    st.markdown(
        "Derivable from SPADL: a shot with `result == success`, plus `owngoal` credited to the "
        "opposing side. Verified rates of **2.50** (2015/16) and **2.57** (2017/18) goals per "
        "match against a Serie A actual of ≈2.6 — so the derivation is sound for both providers."
    )

# ================================================================== constraints
st.divider()
st.subheader("Constraints inherited from Module 1")

st.warning(
    "**Features must come from `PROVIDER_COMPARABLE_TYPES` only.** Per-match action rates "
    "differ between providers by up to 296× (`bad_touch`) and 8.7× (`dribble`). A model given "
    "raw action counts would learn *which provider this is* and collapse on the 2017/18 test "
    "season."
)
st.code(
    "PROVIDER_COMPARABLE_TYPES = " + repr(sorted(PROVIDER_COMPARABLE_TYPES)),
    language="python",
)

st.subheader("Open decision to settle before implementing")
st.markdown(
    "**Checkpoint grid.** Two defensible options:\n\n"
    "1. **Align to the 16 window ends** (15', 20', … 90'). Every checkpoint then has a complete "
    "15-minute graph behind it and the baseline is directly comparable to the graph model. "
    "*Recommended.*\n"
    "2. **18 five-minute marks** (5', 10', … 90'). Finer resolution, but the first two "
    "checkpoints have no full window, so baseline and graph model would be evaluated on "
    "different supports.\n\n"
    "Also unsettled: whether rolling pre-match form for 2017/18 is computed within that season "
    "only (it must be — the squads differ, and using 2015/16 form would leak across the split)."
)
