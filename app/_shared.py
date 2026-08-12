"""Shared helpers for the demo app.

Everything here is presentation glue. The analysis lives in `tacticalgraph.*` and is reused
as-is -- the app must not become a second implementation of anything, or the demo and the
reported results can drift apart.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `streamlit run app/Home.py` from a plain checkout without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Force a headless matplotlib backend before *any* module imports pyplot. Streamlit runs page
# scripts on a worker thread, and on macOS the default interactive backend hangs when a figure
# is created off the main thread. Setting it here means every page inherits it regardless of
# its own import order.
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tacticalgraph.data.aliases import CLUB_DISPLAY, team_id_to_club  # noqa: E402
from tacticalgraph.demo.bundle import DemoBundle, load_bundle, verify_manifest  # noqa: E402
from tacticalgraph.graphs.passing_network import TeamNetwork  # noqa: E402

SEASON_LABEL = {"2015-2016": "2015/16 · StatsBomb", "2017-2018": "2017/18 · Wyscout"}


@dataclass(frozen=True)
class Phase:
    number: int
    name: str
    built: bool
    blurb: str


# Single source of truth for what is and is not implemented. Pages read this so the status
# shown on a page can never contradict the status shown on the home page.
PHASES: tuple[Phase, ...] = (
    Phase(1, "Data ingestion & graph representation", True,
          "760 matches, two providers, harmonised into SPADL; passing networks built."),
    Phase(2, "Player centrality & functional role", True,
          "Classical centrality baseline plus GraphSAGE role embeddings, with ablations."),
    Phase(3, "Match result prediction (GNN + Transformer)", True,
          "Baseline ladder built and beaten by B1; the graph model loses to B0 — reported as a "
          "negative result."),
    Phase(4, "Recurring tactical patterns", True,
          "Chains clustered two ways; shot lift up to 4.8x over base. Human review pending."),
    Phase(5, "Tactical simulation (RL pass choice)", False,
          "Blocked on 360 data, which Serie A does not have in either season."),
    Phase(6, "Coach-facing dashboard", False,
          "This app is its skeleton."),
)


@st.cache_resource(show_spinner="Loading demo bundle…")
def get_bundle() -> DemoBundle:
    return load_bundle()


@st.cache_data(show_spinner=False)
def _cached_table(name: str) -> pd.DataFrame:
    return get_bundle().table(name)


def table(name: str) -> pd.DataFrame:
    return _cached_table(name)


@st.cache_data(show_spinner=False)
def reports() -> dict:
    return get_bundle().reports()


def sidebar_provenance() -> None:
    """Show where the data came from, and warn loudly if the bundle is stale."""
    bundle = get_bundle()
    with st.sidebar:
        st.caption("**Data source**")
        st.caption(bundle.provenance())
        problems = verify_manifest(bundle)
        if problems:
            st.error(
                "Bundle does not match its manifest — the numbers below may be stale:\n\n"
                + "\n".join(f"- {p}" for p in problems)
            )


def status_banner(phase_number: int) -> None:
    """Unmissable banner distinguishing a built module from a specification.

    The single most important piece of honesty in the app: a reviewer must never mistake a
    design sketch for a result.
    """
    phase = next(p for p in PHASES if p.number == phase_number)
    if phase.built:
        st.success(
            f"**Module {phase.number} — implemented.** Everything below is computed from the "
            "real corpus. {}".format(phase.blurb)
        )
    else:
        st.warning(
            f"**Module {phase.number} — NOT IMPLEMENTED.** This page is a specification, not "
            f"a result. No model has been trained. {phase.blurb} Any numbers shown are "
            "properties of the *existing inputs*, never predictions."
        )


def club_label(provider: str, team_id: int) -> str:
    key = team_id_to_club(provider).get(int(team_id))
    return CLUB_DISPLAY.get(key, f"team {team_id}") if key else f"team {team_id}"


@st.cache_data(show_spinner=False)
def club_lookup() -> pd.DataFrame:
    """(provider, season, team_id, club) for every team that appears in the networks."""
    nodes = table("full_nodes.parquet")
    frame = nodes[["provider", "season", "team_id"]].drop_duplicates().reset_index(drop=True)
    frame["club"] = [club_label(p, t) for p, t in zip(frame["provider"], frame["team_id"])]
    return frame.sort_values("club").reset_index(drop=True)


def build_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    game_id: int,
    team_id: int,
    season: str,
    provider: str,
    window_index: int | None = None,
) -> TeamNetwork:
    """Rehydrate a `TeamNetwork` from the flat bundle tables."""
    def _filter(frame: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (frame["team_id"] == team_id)
            & (frame["season"] == season)
            & (frame["provider"] == provider)
        )
        if game_id >= 0:
            mask &= frame["game_id"] == game_id
        if window_index is not None and "window_index" in frame.columns:
            mask &= frame["window_index"] == window_index
        return frame[mask].reset_index(drop=True)

    node_rows = _filter(nodes)
    edge_rows = _filter(edges)

    # A season aggregate (game_id = -1) sums a player's rows across matches; a single match
    # needs no aggregation.
    if game_id < 0 and not node_rows.empty:
        node_rows = (
            node_rows.groupby("player_id", as_index=False)
            .agg(
                mean_x=("mean_x", "mean"),
                mean_y=("mean_y", "mean"),
                spread_x=("spread_x", "mean"),
                spread_y=("spread_y", "mean"),
                touches=("touches", "sum"),
                passes_attempted=("passes_attempted", "sum"),
                passes_completed=("passes_completed", "sum"),
            )
        )
        edge_rows = (
            edge_rows.groupby(["source", "target"], as_index=False)
            .agg(weight=("weight", "sum"), mean_length=("mean_length", "mean"),
                 mean_dx=("mean_dx", "mean"))
        )

    return TeamNetwork(
        game_id=int(game_id),
        team_id=int(team_id),
        season=str(season),
        provider=str(provider),
        nodes=node_rows,
        edges=edge_rows,
        window_index=window_index,
    )


@st.cache_data(show_spinner=False)
def player_names() -> dict[tuple[str, int], str]:
    """(season, player_id) -> display name."""
    players = table("players.parquet")
    return {
        (row.season, int(row.player_id)): row.player_name
        for row in players.itertuples(index=False)
    }


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
