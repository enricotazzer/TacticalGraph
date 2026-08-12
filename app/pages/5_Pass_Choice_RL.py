"""Module 5 — RL pass choice. SPECIFICATION only; blocked on 360 data neither corpus has."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import page_header, sidebar_provenance, status_banner

st.set_page_config(page_title="M5 · Pass Choice RL", page_icon="🤖", layout="wide")

page_header(
    "🤖 Module 5 — Tactical Simulation (RL Pass Choice)",
    "Not implemented, and unlike Modules 3–4 it is blocked on missing data rather than on "
    "effort.",
)
sidebar_provenance()
status_banner(5)

st.error(
    "**Neither corpus has 360 freeze-frame data.** Premier League 2015/16 has 0 of 380 matches; "
    "Serie A has none in either season (StatsBomb 2015/16 or Wyscout 2017/18). None of them "
    "records player positions off the ball, so this module cannot run on the project's data at "
    "all — it needs a different competition, which departs from the league framing the rest of "
    "the project maintains."
)

st.subheader("The formulation")
st.dataframe(
    pd.DataFrame(
        [
            {"Element": "State", "Definition": "Graph over the visible players in a 360 "
                                               "freeze-frame at the moment of a build-up pass"},
            {"Element": "Action", "Definition": "Which visible team-mate to pass to (discrete, ≤10)"},
            {"Element": "Reward", "Definition": "xThreat delta of the resulting ball position, "
                                                "discounted by a learned completion probability"},
            {"Element": "Baselines", "Definition": "most-advanced team-mate; nearest team-mate; "
                                                   "the pass actually played"},
            {"Element": "Evaluation", "Definition": "Off-policy (IPS / doubly-robust) on "
                                                    "held-out matches, never the training set"},
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.info(
    "**Framing matters here.** This is a one-step offline contextual bandit over a learned "
    "reward model — an *exploratory value estimator*, not a simulator and not a replacement for "
    "a coach's judgement. It can say \"passes into this region historically raised threat by "
    "X\"; it cannot say what would actually have happened."
)

st.subheader("The two hard constraints")

left, right = st.columns(2)
with left:
    st.markdown("**1. Wrong competition required**")
    st.markdown(
        "Candidate: **UEFA Euro 2024** — 51 matches, 360 available for all of them. Verified "
        "during planning. Alternatives with full 360 coverage: World Cup 2022 (64 matches), "
        "Women's World Cup 2023 (64), Euro 2020 (51).\n\n"
        "Cost: a tournament has no home/away symmetry, no league table, and heterogeneous "
        "opponents — so nothing from this module transfers back to the league analysis."
    )
with right:
    st.markdown("**2. Freeze-frames are anonymous and partial**")
    st.markdown(
        "A sampled Euro 2024 frame contained **18 of 22 players**, each carrying only "
        "`teammate` / `actor` / `keeper` flags and a location — **no player identity**. Players "
        "outside the broadcast `visible_area` are simply absent.\n\n"
        "Consequences: the action space is *position slots*, not named players; the true "
        "recipient must be matched to the nearest frame object by `pass.end_location`; and the "
        "chosen action can never be attributed to a specific footballer."
    )

st.subheader("What would need building")
st.markdown(
    "1. A 360 ingestion path (`three-sixty/{game_id}.json`, ~7.7 MB per match — ~390 MB for "
    "Euro 2024) plus a freeze-frame → graph encoder.\n"
    "2. An xThreat model fitted on the training split only, reused from `socceraction.xthreat`.\n"
    "3. A pass-completion model, for discounting the reward.\n"
    "4. The heuristic baselines and the off-policy evaluation harness.\n\n"
    "Everything else the module needs — SPADL conversion, the temporal-split guard, resource "
    "reporting — already exists and is reusable."
)
