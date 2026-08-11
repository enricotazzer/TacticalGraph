"""Limitations — the things a reader should hold against every number in this app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import page_header, reports, sidebar_provenance

st.set_page_config(page_title="Limitations", page_icon="⚠️", layout="wide")

page_header(
    "⚠️ Limitations",
    "Stated here rather than buried, because most of these affect how the Module 1–2 results "
    "should be read.",
)
sidebar_provenance()

st.subheader("Data and harmonisation")
st.markdown(
    "- **The season/provider confound is real and undecomposable.** 2015/16 is StatsBomb and "
    "2017/18 is Wyscout, so the ~11 pp cross-season accuracy drop mixes a genuine seasonal "
    "effect with an annotation-convention effect. Every headline number is reported next to a "
    "within-season control for this reason. It is a documented trade-off, not an oversight.\n"
    "- **Pass recipients are inferred, not observed.** ~96% accurate at Wyscout-like action "
    "density, so roughly **1 edge in 25** in the 2017/18 networks is wrong or missing.\n"
    "- **Possession chains over-segment by ~25%** relative to StatsBomb's native counter.\n"
    "- **Minutes played are estimated** from a player's first-to-last action, symmetrically for "
    "both providers because Wyscout exposes no per-match position spells. A player with a "
    "single action is credited 0 minutes and filtered out of the network.\n"
    "- **Role labels are 4-class**, forced by Wyscout. The fine-grained validation signal "
    "exists for one of the two seasons only.\n"
    "- **Only 16 of 20 clubs** appear in both seasons, and cross-provider player linkage covers "
    "**199 players** matched on normalised names — so cross-season claims rest on a subset."
)

st.subheader("Modelling")
st.markdown(
    "- **Position dominates topology.** The stronger form of Module 2's thesis — that who you "
    "pass to reveals a functional role position cannot — is only weakly supported: topology "
    "adds ~1–1.5 pp, and position alone matches or beats the combination at recovering fine "
    "positions for k ≥ 6.\n"
    "- **Three seeds** per configuration. Enough to establish the ablation ordering; not enough "
    "for tight confidence intervals on a ~1 pp effect.\n"
    "- **Cross-season stability conflates two things**: role stability over time and robustness "
    "to the provider change. With one provider pair they cannot be separated.\n"
    "- **Reduced scale by design.** Two seasons of one competition, shallow models, short "
    "schedules — sized for Kaggle's free tier. These are proof-of-concept results, not "
    "state-of-the-art claims."
)

st.subheader("What event data cannot see")
st.markdown(
    "Off-ball movement, verbal communication, coaching instruction and tactical intent are not "
    "in this data. Even with 360 freeze-frames the representation is partial — and Serie A has "
    "no 360 in either season. No amount of modelling recovers what was never recorded."
)

st.divider()
st.subheader("Reproducibility notes")
manifest = reports()
bundle_reports = sorted(manifest.keys())
st.markdown(
    "This app reads a committed ~8 MB bundle rather than the full 1.5 GB store, so what you see "
    "does not depend on a drive being attached. Every metric shown is read from the report JSONs "
    "produced by the pipeline scripts — the app computes no headline numbers of its own, "
    "precisely so the demo cannot drift from the reported results."
)
st.caption(f"Reports in this bundle: {', '.join(bundle_reports)}")

st.dataframe(
    pd.DataFrame(
        [
            {"Script": "scripts/ingest.py", "Produces": "raw provider data (~1.4 GB)"},
            {"Script": "scripts/build_spadl.py", "Produces": "canonical action store + enrichment"},
            {"Script": "scripts/validate_harmonization.py", "Produces": "harmonization_report.json"},
            {"Script": "scripts/build_networks.py", "Produces": "passing networks + player directory"},
            {"Script": "scripts/run_centrality.py", "Produces": "centrality tables (Module 2 baseline)"},
            {"Script": "scripts/train_roles.py", "Produces": "role GNN, embeddings, module2 reports"},
            {"Script": "scripts/export_demo_bundle.py", "Produces": "demo_data/ (this app's input)"},
        ]
    ),
    hide_index=True,
    width="stretch",
)
