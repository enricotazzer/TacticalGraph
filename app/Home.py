"""TacticalGraph demo — landing page.

Run with:  streamlit run app/Home.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import PHASES, get_bundle, page_header, reports, sidebar_provenance, table

st.set_page_config(page_title="TacticalGraph", page_icon="⚽", layout="wide")

page_header(
    "⚽ TacticalGraph",
    "Football tactical analysis on event data — graph neural networks, sequence models and "
    "reinforcement learning, sized to train inside Kaggle's free tier.",
)
sidebar_provenance()

# Which competition this bundle holds, stated before any number is shown. Two corpora in this
# project share a season key *and* a provider (Serie A 2015/16 and Premier League 2015/16 are
# both statsbomb/2015-2016), so an unlabelled page would be genuinely ambiguous.
st.info(
    f"**This demo shows the `{get_bundle().manifest.get('corpus', 'serie_a')}` corpus — "
    f"{get_bundle().corpus_label()}.** The project runs Modules 1–4 on two corpora: the "
    "**Premier League 2015/16** (380 matches, one provider, split by matchweek) is the primary "
    "one because it has no provider confound, and **Serie A 2015/16 + 2017/18** (760 matches, "
    "two providers) is kept as a cross-provider generalisation study. The Serie A bundle is "
    "the one committed here because it is the only one that can drive the harmonisation page; "
    "the Premier League numbers are in the README."
)

# --------------------------------------------------------------------------------- status
st.subheader("What is built")

status = pd.DataFrame(
    [
        {
            "Module": f"M{p.number}",
            "Phase": p.name,
            "Status": "✅ implemented" if p.built else "⬜ not implemented",
            "Notes": p.blurb,
        }
        for p in PHASES
    ]
)
st.dataframe(status, hide_index=True, width="stretch")

built = sum(p.built for p in PHASES)
st.caption(
    f"{built} of {len(PHASES)} modules implemented. Pages for the rest are **specifications** "
    "— they describe the design and show the real inputs that already exist, and say so in a "
    "banner at the top. Nothing on an unimplemented page is a model output."
)

# --------------------------------------------------------------------------------- corpus
st.divider()
st.subheader("The corpus")

left, right = st.columns([3, 2])

with left:
    games = table("games.parquet")
    nodes = table("full_nodes.parquet")
    edges = table("full_edges.parquet")

    summary = (
        games.groupby(["season", "provider"])
        .agg(matches=("game_id", "nunique"))
        .reset_index()
    )
    network_counts = (
        nodes.groupby("season")
        .agg(networks=("team_id", lambda s: len(s.drop_duplicates())), nodes=("player_id", "size"))
        .reset_index()
    )
    summary = summary.merge(
        nodes.groupby("season").size().rename("network_nodes").reset_index(), on="season"
    ).merge(
        edges.groupby("season").size().rename("network_edges").reset_index(), on="season"
    )
    summary.columns = ["Season", "Provider", "Matches", "Network nodes", "Network edges"]
    st.dataframe(summary, hide_index=True, width="stretch")

with right:
    st.metric("Matches", f"{games['game_id'].nunique():,}")
    st.metric("Team-match networks", f"{len(nodes.groupby(['game_id', 'team_id'])):,}")
    st.metric("Passing edges", f"{len(edges):,}")

# ------------------------------------------------------------------------------- honesty
st.divider()
st.subheader("Read this before the numbers")

st.warning(
    "**The two seasons come from different providers, and that is a confound.**\n\n"
    "StatsBomb open data contains only *one* usable Serie A season (2015/16). The second "
    "season (2017/18) comes from the Wyscout public dataset. Training on 2015/16 and testing "
    "on 2017/18 is therefore the strongest available anti-leakage split *and* it changes the "
    "data provider at the same moment it changes the season — so a performance drop cannot be "
    "attributed to football alone.\n\n"
    "This was a deliberate choice. The project's response is to **measure** the confound "
    "rather than hide it: Module 1's page quantifies the provider shift, and every headline "
    "metric is reported next to an unconfounded within-season control."
)

harmonisation = reports().get("harmonization_report", {})
if harmonisation:
    accuracy = pd.DataFrame(harmonisation.get("recipient_accuracy", []))
    columns = st.columns(3)
    if not accuracy.empty:
        native = accuracy[accuracy["context"] == "statsbomb-native"]
        degraded = accuracy[accuracy["context"] == "statsbomb-degraded"]
        if not native.empty:
            columns[0].metric(
                "Recipient inference (StatsBomb truth)", f"{native['accuracy'].iloc[0]:.2%}"
            )
        if not degraded.empty:
            columns[1].metric(
                "Same rule, Wyscout-like density", f"{degraded['accuracy'].iloc[0]:.2%}",
                help="Best available estimate for the 2017/18 season, which has no ground truth.",
            )
    possession = pd.DataFrame(harmonisation.get("possession", []))
    if not possession.empty:
        columns[2].metric(
            "Possession chains (ARI vs StatsBomb)",
            f"{possession['adjusted_rand_mean'].iloc[0]:.3f}",
        )

st.divider()
st.caption(
    "Navigate with the sidebar. Module 1 and Module 2 contain interactive demos computed from "
    "the real corpus; Modules 3–5 are specifications. Source: "
    "`github.com/…/TacticalGraph` — see README.md for full results, baselines and limitations."
)
