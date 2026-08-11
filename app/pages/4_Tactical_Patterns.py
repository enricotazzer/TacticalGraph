"""Module 4 — recurring tactical patterns. SPECIFICATION plus real possession-chain inputs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import page_header, reports, sidebar_provenance, status_banner

st.set_page_config(page_title="M4 · Tactical Patterns", page_icon="🔁", layout="wide")

page_header(
    "🔁 Module 4 — Recurring Tactical Patterns",
    "Not implemented. The possession chains this module needs are already reconstructed; the "
    "sequence clustering is not built.",
)
sidebar_provenance()
status_banner(4)

st.subheader("The design")
st.markdown(
    "Cluster possession sequences into recurring **styles of play** — build-up from the back, "
    "fast transition, slow circulation — and report which patterns precede a shot.\n\n"
    "- **Representation**: each possession chain is a sequence of SPADL actions (type, start/end "
    "location, outcome). Encoded either by the Module 3 sequence encoder or a dedicated small "
    "seq2seq model.\n"
    "- **Clustering**: k-means / GMM on the sequence embeddings, k chosen by silhouette and by "
    "whether the clusters are describable in football terms.\n"
    "- **Coach-facing output**: *\"this pattern precedes a shot in X% of cases\"*, with "
    "`game_id` + timestamp on every instance so it links straight to video.\n"
    "- **Validation** is deliberately qualitative but systematic: sample N instances per "
    "cluster, review them, and report the **proportion judged sensible by human review**. This "
    "is the honest option — there is no ground-truth label for \"style of play\"."
)

st.subheader("Input that already exists — reconstructed possession chains")

possession = pd.DataFrame(reports().get("harmonization_report", {}).get("possession", []))
if not possession.empty:
    row = possession.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Chains per match (ours)", f"{row['chains_ours_mean']:.0f}")
    columns[1].metric("Chains per match (StatsBomb)", f"{row['chains_statsbomb_mean']:.0f}")
    columns[2].metric("Adjusted Rand vs StatsBomb", f"{row['adjusted_rand_mean']:.3f}")
    columns[3].metric("Boundary Jaccard", f"{row['boundary_jaccard_mean']:.3f}")

st.markdown(
    "Chains are reconstructed identically for both providers (Wyscout has no possession "
    "counter at all), using: a new chain on game/period change, on any set-piece restart, or "
    "when the controlling team changes *and holds* the ball — with single opponent touches "
    "(clearances, failed interceptions) absorbed as contest rather than treated as a change of "
    "possession."
)

st.warning(
    "**A caveat this module must address rather than inherit.** The rule over-segments by "
    "~25% relative to StatsBomb's native counter (246 chains per match versus 196), because "
    "every set-piece is treated as a hard restart. For chain-level *pattern* modelling that is "
    "arguably wrong: a possession that restarts on a throw-in is often the same phase of play. "
    "Module 4 should re-examine the set-piece rule before clustering, since over-segmentation "
    "would split a single build-up pattern into several short fragments."
)

st.subheader("Why the possession chains are not in the demo bundle")
st.caption(
    "Chain ids live on the SPADL action rows (~55 MB), which are excluded from the committed "
    "8 MB bundle. The chain-length distribution shown above is therefore summarised from the "
    "harmonisation report rather than recomputed here. With `DATA_ROOT` mounted, the full "
    "action store is available and this page can be extended to plot the distribution directly."
)
