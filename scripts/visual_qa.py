#!/usr/bin/env python
"""Phase 1.7 -- visual verification of the constructed passing networks.

Writes three things under DATA_ROOT/figures:

  networks_statsbomb.png   3 matches from 2015/16
  networks_wyscout.png     3 matches from 2017/18
  provider_comparison.png  the same clubs in both seasons, side by side

The third is the one that matters. Summary statistics can hide a mirrored coordinate flip
or a provider-specific distortion; two columns of pitches cannot.

    python scripts/visual_qa.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tacticalgraph.config import CORPORA, DEFAULT_CORPUS, Paths  # noqa: E402, SERIE_A_STATSBOMB, SERIE_A_WYSCOUT  # noqa: E402
from tacticalgraph.data.aliases import (  # noqa: E402
    CLUB_DISPLAY,
    club_to_team_id,
    clubs_in_both_seasons,
)
from tacticalgraph.data.spadl_store import read_actions, read_games  # noqa: E402
from tacticalgraph.features.centrality import player_centrality  # noqa: E402
from tacticalgraph.graphs.passing_network import build_team_network  # noqa: E402
from tacticalgraph.viz.pitch import (  # noqa: E402
    plot_match_networks,
    plot_provider_comparison,
)

log = logging.getLogger("visual_qa")


def _networks_for_games(
    actions: pd.DataFrame, game_ids: list[int], season: str, provider: str
) -> list:
    networks = []
    for game_id in game_ids:
        game_actions = actions[actions["game_id"] == game_id]
        for team_id in sorted(game_actions["team_id"].dropna().unique()):
            networks.append(
                build_team_network(
                    game_actions,
                    game_id=game_id,
                    team_id=int(team_id),
                    season=season,
                    provider=provider,
                )
            )
    return networks


def _pagerank_by_team(networks: list) -> dict[int, dict[int, float]]:
    """Node metric for sizing: PageRank on pass volume."""
    out = {}
    for network in networks:
        frame = player_centrality(network)
        if not frame.empty:
            out[network.team_id] = dict(
                zip(frame["player_id"].astype(int), frame["pagerank"].fillna(0.0))
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=3, help="matches per provider")
    parser.add_argument("--clubs", type=int, default=4, help="clubs in the comparison")
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
        help="which competition corpus to use (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    paths = Paths.load(args.corpus).ensure()
    games = read_games(paths)

    for season, provider in (
        (SERIE_A_STATSBOMB.key, SERIE_A_STATSBOMB.provider),
        (SERIE_A_WYSCOUT.key, SERIE_A_WYSCOUT.provider),
    ):
        actions = read_actions(paths, season=season, provider=provider)
        season_games = games[games["provider"] == provider].head(args.matches)
        ids = [int(g) for g in season_games["game_id"]]
        networks = _networks_for_games(actions, ids, season, provider)
        plot_match_networks(
            networks,
            paths.figures / f"networks_{provider}.png",
            node_metric=_pagerank_by_team(networks),
        )
        log.info(
            "%s: %d networks, mean %.1f nodes / %.1f edges",
            provider,
            len(networks),
            sum(n.n_nodes for n in networks) / len(networks),
            sum(n.n_edges for n in networks) / len(networks),
        )

    # The harmonisation eyeball test: aggregate each club's whole season into one network
    # per provider, so the comparison reflects structure rather than one match's noise.
    sb_actions = read_actions(paths, season=SERIE_A_STATSBOMB.key)
    wy_actions = read_actions(paths, season=SERIE_A_WYSCOUT.key)
    sb_ids, wy_ids = club_to_team_id("statsbomb"), club_to_team_id("wyscout")

    pairs = []
    for club in clubs_in_both_seasons()[: args.clubs]:
        left = build_team_network(
            sb_actions,
            game_id=-1,  # -1 marks a season aggregate rather than a single match
            team_id=sb_ids[club],
            season=SERIE_A_STATSBOMB.key,
            provider="statsbomb",
            apply_minute_filter=False,
        )
        right = build_team_network(
            wy_actions,
            game_id=-1,
            team_id=wy_ids[club],
            season=SERIE_A_WYSCOUT.key,
            provider="wyscout",
            apply_minute_filter=False,
        )
        pairs.append((CLUB_DISPLAY[club], left, right))
        log.info(
            "%-16s 2015/16: %2d nodes %3d edges | 2017/18: %2d nodes %3d edges",
            club,
            left.n_nodes,
            left.n_edges,
            right.n_nodes,
            right.n_edges,
        )

    plot_provider_comparison(pairs, paths.figures / "provider_comparison.png")
    print(f"\nfigures written to {paths.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
