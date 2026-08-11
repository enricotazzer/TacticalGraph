"""Module 1 — data ingestion, cross-provider harmonisation, passing networks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from _shared import (
    SEASON_LABEL,
    build_network,
    club_label,
    club_lookup,
    page_header,
    reports,
    sidebar_provenance,
    status_banner,
    table,
)
from tacticalgraph.data.aliases import CLUB_DISPLAY, club_to_team_id, clubs_in_both_seasons
from tacticalgraph.data.recipient import PASS_LIKE_TYPES
from tacticalgraph.viz.pitch import draw_network

st.set_page_config(page_title="M1 · Data & Harmonisation", page_icon="🧩", layout="wide")

page_header(
    "🧩 Module 1 — Data & Harmonisation",
    "Two providers, one canonical representation. This page shows how far apart the seasons "
    "are and why passing networks survive the switch anyway.",
)
sidebar_provenance()
status_banner(1)

report = reports().get("harmonization_report", {})

# =====================================================================  scorecard
st.subheader("Harmonisation scorecard")

st.markdown(
    "Both providers are missing things the project needs, so those are **reconstructed "
    "identically for both** — never *real value where available, estimate elsewhere*, which "
    "would make 2015/16 systematically better than 2017/18 and corrupt the cross-season test."
)

accuracy = pd.DataFrame(report.get("recipient_accuracy", []))
resolution = pd.DataFrame(report.get("recipient_resolution", []))
possession = pd.DataFrame(report.get("possession", []))

left, right = st.columns(2)

with left:
    st.markdown("**Pass recipient** — SPADL carries none, so it is inferred for both providers")
    if not accuracy.empty:
        display = accuracy[["context", "n", "accuracy", "wrong", "unresolved"]].copy()
        display["context"] = display["context"].map(
            {
                "statsbomb-native": "StatsBomb, native density",
                "statsbomb-degraded": "StatsBomb, degraded to Wyscout-like",
            }
        ).fillna(display["context"])
        for column in ("accuracy", "wrong", "unresolved"):
            display[column] = (display[column] * 100).round(2)
        display.columns = ["Measurement", "n passes", "correct %", "wrong %", "unresolved %"]
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption(
            "The second row is the important one. Wyscout has no ground truth, so accuracy "
            "there cannot be measured directly — degrading the StatsBomb stream to comparable "
            "action density and re-running the same rule is the honest estimate."
        )
    if not resolution.empty:
        coverage = resolution[["season", "provider", "completed_passes", "resolved_pct"]].copy()
        coverage["resolved_pct"] = (coverage["resolved_pct"] * 100).round(2)
        coverage.columns = ["Season", "Provider", "Completed passes", "Resolved %"]
        st.dataframe(coverage, hide_index=True, width="stretch")

with right:
    st.markdown("**Possession chains** — StatsBomb has a native counter, Wyscout has none")
    if not possession.empty:
        row = possession.iloc[0]
        st.metric("Adjusted Rand index", f"{row['adjusted_rand_mean']:.3f}")
        st.metric("Boundary Jaccard", f"{row['boundary_jaccard_mean']:.3f}")
        st.caption(
            f"Our rule produces {row['chains_ours_mean']:.0f} chains per match against "
            f"StatsBomb's {row['chains_statsbomb_mean']:.0f} — it **over-segments by ~25%** "
            "because every set-piece is treated as a hard restart. Documented rather than "
            "tuned away; Module 4 should revisit it."
        )

    shift = pd.DataFrame(report.get("distribution_shift", []))
    if not shift.empty:
        st.markdown("**Distribution shift** — KS statistic between seasons")
        shift.columns = ["Feature", "KS", "mean 2015/16", "mean 2017/18"]
        st.dataframe(shift, hide_index=True, width="stretch")

# =====================================================  the finding that licenses the project
st.divider()
st.subheader("Why passing networks survive the provider switch")

mix = pd.DataFrame(report.get("action_mix", []))
if not mix.empty:
    mix = mix.rename(columns={"type_name": "Action type"})
    mix["rate_ratio"] = pd.to_numeric(mix["rate_ratio"], errors="coerce")
    mix["comparable"] = mix["rate_ratio"].between(0.75, 1.33)
    mix["pass-like"] = mix["Action type"].isin(PASS_LIKE_TYPES)

    left, right = st.columns([3, 2])

    with left:
        chart = mix.dropna(subset=["rate_ratio"]).sort_values("rate_ratio", ascending=False)
        chart_display = chart[
            ["Action type", "per_game_2015_2016", "per_game_2017_2018", "rate_ratio", "comparable"]
        ].copy()
        chart_display.columns = ["Action type", "per match 15/16", "per match 17/18",
                                 "ratio", "comparable"]
        st.dataframe(
            chart_display.style.map(
                lambda v: "background-color: rgba(255,80,80,0.22)" if v is False else "",
                subset=["comparable"],
            ),
            hide_index=True,
            width="stretch",
            height=430,
        )

    with right:
        st.markdown(
            "Per-match **rates**, not shares. Shares are the wrong diagnostic here: "
            "StatsBomb's dribble inflation mechanically deflates every other type's share, "
            "making comparable types look divergent."
        )
        worst = chart.iloc[0]
        st.error(
            f"**Worst offenders — annotation convention, not football.**\n\n"
            f"`{worst['Action type']}` differs by **{worst['rate_ratio']:.0f}×**. "
            "`dribble` 8.7×, `tackle` 4.2×, `interception` 0.31×. Any feature that counts "
            "actions naively would encode *which provider this is* and collapse at test time."
        )
        aggregate = pd.DataFrame(report.get("aggregate_rates", []))
        if not aggregate.empty:
            passlike = aggregate.iloc[0]
            st.success(
                f"**But passes do not have this problem.** All pass-like types together: "
                f"{passlike['per_game_2015_2016']:.1f} vs {passlike['per_game_2017_2018']:.1f} "
                f"per match — ratio **{passlike['rate_ratio']:.3f}**.\n\n"
                "That is the quantitative licence for the whole project."
            )

# =====================================================================  match networks
st.divider()
st.subheader("Passing networks — pick a match")

nodes = table("full_nodes.parquet")
edges = table("full_edges.parquet")
games = table("games.parquet")
clubs = club_lookup()

control_left, control_mid, control_right = st.columns(3)
season = control_left.selectbox(
    "Season", sorted(nodes["season"].unique()), format_func=lambda s: SEASON_LABEL.get(s, s)
)
season_clubs = clubs[clubs["season"] == season].sort_values("club")
club_choice = control_mid.selectbox("Club", season_clubs["club"].tolist())
team_id = int(season_clubs.loc[season_clubs["club"] == club_choice, "team_id"].iloc[0])

team_games = sorted(nodes.loc[(nodes["season"] == season) & (nodes["team_id"] == team_id),
                              "game_id"].unique())
game_meta = games.set_index("game_id")


def _game_label(game_id: int) -> str:
    if game_id not in game_meta.index:
        return str(game_id)
    row = game_meta.loc[game_id]
    provider = row["provider"]
    home = club_label(provider, row["home_team_id"])
    away = club_label(provider, row["away_team_id"])
    date = str(row["game_date"])[:10]
    return f"MW{int(row['game_day']):02d} · {home} vs {away} · {date}"


game_choice = control_right.selectbox("Match", team_games, format_func=_game_label)

row = game_meta.loc[game_choice]
provider = row["provider"]
both_teams = [int(row["home_team_id"]), int(row["away_team_id"])]

figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, side_team in zip(axes, both_teams):
    network = build_network(nodes, edges, game_choice, side_team, season, provider)
    draw_network(
        network,
        ax,
        title=f"{club_label(provider, side_team)} — {network.n_nodes} nodes, "
              f"{network.n_edges} edges",
    )
st.pyplot(figure, width="stretch")
plt.close(figure)
st.caption(
    "Both teams attack left → right; the coordinate flip is applied at conversion time. Node "
    "labels are the last three digits of the player id. Edges below 3 passes are hidden for "
    "legibility."
)

# =====================================================================  the eyeball test
st.divider()
st.subheader("The harmonisation eyeball test")
st.markdown(
    "The same club's **season-aggregate** network from each provider, side by side. Summary "
    "statistics cannot reveal a mirrored coordinate flip or a provider-specific distortion; "
    "two pitches can. 16 of 20 clubs appear in both seasons (4 relegated, 4 promoted)."
)

overlap = clubs_in_both_seasons()
compare_club = st.selectbox(
    "Club present in both seasons", overlap, format_func=lambda k: CLUB_DISPLAY[k]
)
sb_ids, wy_ids = club_to_team_id("statsbomb"), club_to_team_id("wyscout")

figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (season_key, provider_key, team_ids) in zip(
    axes,
    [("2015-2016", "statsbomb", sb_ids), ("2017-2018", "wyscout", wy_ids)],
):
    network = build_network(
        nodes, edges, -1, team_ids[compare_club], season_key, provider_key
    )
    passes = int(network.edges["weight"].sum()) if not network.edges.empty else 0
    draw_network(
        network,
        ax,
        title=f"{CLUB_DISPLAY[compare_club]} — {SEASON_LABEL[season_key]}\n"
              f"{network.n_nodes} nodes, {network.n_edges} edges, {passes:,} passes",
    )
st.pyplot(figure, width="stretch")
plt.close(figure)

st.info(
    "Node positions are also correct in absolute terms, which matters because these plots "
    "*look* compressed — no player's **mean** position is in the attacking third, as it should "
    "be. Measured on the corpus: GK 8.8 m, centre-back 33.2, defenders 43.9, midfielders 56.4, "
    "forwards 68.1, centre-forwards 68.6 on a 105 m pitch; `mean_y ≈ 34` on a 68 m pitch for "
    "every role."
)
