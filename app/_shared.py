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

# One place naming the split kinds, so a page cannot label a `matchweek` run as a
# "within-season control" by falling through an if/else written when only two kinds existed.
SPLIT_LABELS = {
    "cross_season": "cross-season (train 2015/16 → test 2017/18, CONFOUNDED)",
    "within_season": "within-season control (2015/16 only, unconfounded)",
    "matchweek": "matchweek (wk1-26 train / 27-33 val / 34-38 test, unconfounded)",
}


@dataclass(frozen=True)
class Phase:
    number: int
    name: str
    built: bool
    blurb: str


# Single source of truth for what is and is not implemented. Pages read this so the status
# shown on a page can never contradict the status shown on the home page.
PHASES: tuple[Phase, ...] = (
    # Blurbs stay corpus-neutral: this table is shown for whichever bundle is loaded, so a
    # figure quoted here (760 matches, 4.8x lift) would be wrong on the other corpus. Exact
    # numbers belong on the module pages, which read them from that bundle's own reports.
    Phase(1, "Data ingestion & graph representation", True,
          "Provider data harmonised into SPADL; passing networks built per team-match and per "
          "15-minute window."),
    Phase(2, "Player centrality & functional role", True,
          "Classical centrality baseline plus GraphSAGE role embeddings, with ablations."),
    Phase(3, "Match result prediction (GNN + Transformer)", True,
          "Baseline ladder built; B1 wins. The graph model now starts as a frozen B1 and learns "
          "a correction to it — and still cannot improve on it."),
    Phase(4, "Recurring tactical patterns", True,
          "Chains clustered two ways; the interpretable baseline beats the learned encoder. "
          "Human review pending."),
    Phase(5, "Tactical simulation (RL pass choice)", False,
          "Blocked on 360 data, which neither corpus has on any match."),
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


@st.cache_data(show_spinner=False)
def corpus_providers() -> list[str]:
    """Distinct providers in this bundle, read from the data rather than the manifest.

    Pages branch on this instead of on a corpus slug: what makes the harmonisation sections
    meaningful is having *two providers to compare*, not being named "serie_a".
    """
    try:
        return sorted(table("games.parquet")["provider"].dropna().unique().tolist())
    except Exception:  # noqa: BLE001
        return []


def is_multi_provider() -> bool:
    return len(corpus_providers()) > 1


def corpus_label() -> str:
    return get_bundle().corpus_label()


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


@st.cache_data(show_spinner=False)
def _team_names_from_bundle() -> dict[tuple[str, int], str]:
    """(provider, team_id) -> name from the bundle's teams table, if it carries one.

    The app reads the exported bundle rather than DATA_ROOT, so it cannot call
    `aliases.club_labeller` (which needs a `Paths`). Same resolution order, different source.
    Keyed by provider because the two providers number teams independently.
    """
    try:
        teams = table("teams.parquet")
    except Exception:  # noqa: BLE001 - older bundles predate the teams table
        return {}
    return {
        (str(r.provider), int(r.team_id)): str(r.team_name)
        for r in teams.itertuples(index=False)
    }


def club_label(provider: str, team_id: int) -> str:
    """Display name for a team, corpus-agnostic.

    Prefers the bundle's own teams table so a non-Serie-A corpus gets real names; falls back
    to the hand-maintained alias table, which only covers the two Serie A providers.
    """
    team_id = int(team_id)
    name = _team_names_from_bundle().get((str(provider), team_id))
    if name:
        return name
    key = team_id_to_club(provider).get(team_id)
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
