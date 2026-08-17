"""Limitations — the things a reader should hold against every number in this app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import (
    corpus_label,
    get_bundle,
    is_multi_provider,
    page_header,
    reports,
    sidebar_provenance,
)

st.set_page_config(page_title="Limitations", page_icon="⚠️", layout="wide")

page_header(
    "⚠️ Limitations",
    "Stated here rather than buried, because most of these affect how the Module 1–4 results "
    "should be read.",
)
sidebar_provenance()

# This page must describe the corpus actually loaded. It previously stated the Serie A
# cross-provider limitations unconditionally, which misdescribes a single-provider bundle --
# claiming a season/provider confound and a 96%-accurate recipient rule on a corpus that has
# neither. Limitations of the *other* corpus are still shown, but labelled as such.
MULTI_PROVIDER = is_multi_provider()

st.subheader("Data and harmonisation")
if MULTI_PROVIDER:
    st.markdown(
        "- **The season/provider confound is real and undecomposable.** 2015/16 is StatsBomb and "
        "2017/18 is Wyscout, so the ~11 pp cross-season accuracy drop mixes a genuine seasonal "
        "effect with an annotation-convention effect. Every headline number is reported next to a "
        "within-season control for this reason. It is a documented trade-off, not an oversight.\n"
        "- **Pass recipients are inferred, not observed.** ~96% accurate at Wyscout-like action "
        "density, so roughly **1 edge in 25** in the 2017/18 networks is wrong or missing.\n"
        "- **Possession chains over-segment by ~25%** relative to StatsBomb's native counter.\n"
        "- **Minutes played are estimated** from a player's first-to-last action, symmetrically "
        "for both providers because Wyscout exposes no per-match position spells. A player with a "
        "single action is credited 0 minutes and filtered out of the network.\n"
        "- **Role labels are 4-class**, forced by Wyscout. The fine-grained validation signal "
        "exists for one of the two seasons only.\n"
        "- **Only 16 of 20 clubs** appear in both seasons, and cross-provider player linkage "
        "covers **199 players** matched on normalised names — so cross-season claims rest on a "
        "subset."
    )
else:
    st.markdown(
        f"- **No provider confound, and no cross-season claim.** {corpus_label()} is one "
        "provider and one complete season, split by matchweek. That removes the confound the "
        "Serie A corpus carries, at the cost of saying nothing about generalisation across "
        "providers or seasons — a narrower claim, honestly scoped.\n"
        "- **Pass recipients are inferred, not observed.** The recipient is taken to be the next "
        "same-team player to act, applied identically on every corpus so networks are never "
        "systematically better on one provider than another. The rule is scored against "
        "StatsBomb's own recorded recipient — but that scoring has only been run on the Serie A "
        "StatsBomb season, **not separately on this corpus**, so treat >99% as the expected "
        "regime rather than a measurement of these networks.\n"
        "- **Minutes played are estimated** from a player's first-to-last action, not from "
        "substitution records. A player with a single action is credited 0 minutes and filtered "
        "out of the network.\n"
        "- **Role supervision is 4-class.** The fine-grained position vocabulary is held back "
        "entirely for validation, never used as a training label.\n"
        "- **A five-matchweek test fold is 50 matches.** Every interval on the Module 3 page is "
        "wide because of it, and that is a property of a single season, not of the method."
    )

st.subheader("Modelling")
st.markdown(
    "- **Position dominates topology.** The stronger form of Module 2's thesis — that who you "
    "pass to reveals a functional role position cannot — is only weakly supported. The exact "
    "contribution is computed on the Module 2 page from this bundle's own reports; it is under "
    "1.5 pp on both corpora.\n"
    "- **Three seeds** per configuration. Enough to establish the ablation ordering; not enough "
    "for tight confidence intervals on a ~1 pp effect. On this corpus k-means on the learned "
    "embedding is itself seed-sensitive, so cluster metrics are shown as mean ± std.\n"
    "- **Module 3's negative result was mostly an optimiser bug**, not a data limit: "
    "`optimiser.step()` ran once per match, so the batch size was 1. Fixing it moved "
    "\"significantly worse than the scoreline baseline\" from 9 of 9 runs to 1 of 9. Two further "
    "fixes are specified and unbuilt, so that result is open rather than settled.\n"
    + (
        "- **Cross-season stability conflates two things**: role stability over time and "
        "robustness to the provider change. With one provider pair they cannot be separated.\n"
        if MULTI_PROVIDER
        else "- **Stability is measured across halves of one season**, so it says nothing about "
        "robustness to a provider or season change.\n"
    )
    + "- **Reduced scale by design.** Shallow models, short schedules — sized for Kaggle's free "
    "tier. These are proof-of-concept results, not state-of-the-art claims."
)

st.subheader("What event data cannot see")
st.markdown(
    "Off-ball movement, verbal communication, coaching instruction and tactical intent are not "
    "in this data. Even with 360 freeze-frames the representation is partial (a real frame "
    "holds a mean of 14.9 of 22 players, never all 22) — and neither corpus has 360 on any "
    "match. No amount of modelling recovers what was never recorded."
)

st.divider()
st.subheader("Reproducibility notes")
manifest = reports()
bundle_reports = sorted(manifest.keys())
bundle_mb = sum(t.get("size_kb", 0.0) for t in get_bundle().manifest.get("tables", {}).values())
st.markdown(
    f"This app reads a committed bundle (~{bundle_mb / 1024:.0f} MB of tables) rather than the "
    "full 1.5 GB store, so what you see does not depend on a drive being attached. Every metric "
    "shown is read from the report JSONs produced by the pipeline scripts — the app computes no "
    "headline numbers of its own, precisely so the demo cannot drift from the reported results."
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
            {"Script": "scripts/train_outcome.py", "Produces": "module3 reports + test predictions"},
            {"Script": "scripts/estimate_ceiling.py", "Produces": "module3_ceiling.json (learning curve)"},
            {"Script": "scripts/train_patterns.py", "Produces": "module4 reports + chain clusters"},
            {"Script": "scripts/review_patterns.py", "Produces": "pattern review sheets + pitch figures"},
            {"Script": "scripts/export_demo_bundle.py", "Produces": "demo_data/ (this app's input)"},
        ]
    ),
    hide_index=True,
    width="stretch",
)
