#!/usr/bin/env python
"""Phase 2 -- classical centrality metrics: the interpretable baseline for Module 2.

Writes DATA_ROOT/networks/centrality_players.parquet and centrality_teams.parquet, and
prints the season's most central players -- a standalone deliverable, and the thing the GNN
embedding in Phase 3 has to beat.

    python scripts/run_centrality.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tacticalgraph.config import CORPORA, DEFAULT_CORPUS, Paths  # noqa: E402
from tacticalgraph.data.aliases import club_labeller  # noqa: E402
from tacticalgraph.data.players import load_player_directory  # noqa: E402
from tacticalgraph.eval.resources import ResourceMonitor  # noqa: E402
from tacticalgraph.features.centrality import (  # noqa: E402
    PLAYER_METRICS,
    aggregate_player_season,
    centrality_table,
)
from tacticalgraph.graphs.passing_network import TeamNetwork  # noqa: E402

log = logging.getLogger("centrality")


def networks_from_store(paths: Paths, kind: str = "full") -> list[TeamNetwork]:
    """Rehydrate persisted networks into TeamNetwork objects."""
    nodes = pd.read_parquet(paths.networks / f"{kind}_nodes.parquet")
    edges = pd.read_parquet(paths.networks / f"{kind}_edges.parquet")

    keys = ["game_id", "team_id", "season", "provider"]
    edge_groups = {k: g for k, g in edges.groupby(keys, sort=False)}

    networks = []
    for key, node_group in nodes.groupby(keys, sort=False):
        game_id, team_id, season, provider = key
        networks.append(
            TeamNetwork(
                game_id=int(game_id),
                team_id=int(team_id),
                season=str(season),
                provider=str(provider),
                nodes=node_group.reset_index(drop=True),
                edges=edge_groups.get(key, edges.iloc[0:0]).reset_index(drop=True),
            )
        )
    return networks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-matches", type=int, default=5)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
        help="which competition corpus to use (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    paths = Paths.load(args.corpus).ensure()

    networks = networks_from_store(paths, "full")
    log.info("rehydrated %d team-match networks", len(networks))

    with ResourceMonitor("centrality") as monitor:
        players, teams = centrality_table(networks)

    players.to_parquet(paths.networks / "centrality_players.parquet", index=False)
    teams.to_parquet(paths.networks / "centrality_teams.parquet", index=False)
    log.info(
        "wrote %d player-match rows, %d team-match rows", len(players), len(teams)
    )

    season_level = aggregate_player_season(players, min_matches=args.min_matches)
    directory = load_player_directory(paths)
    named = season_level.merge(
        directory[["season", "provider", "player_id", "player_name", "coarse_role"]],
        on=["season", "provider", "player_id"],
        how="left",
    )
    named.to_parquet(paths.networks / "centrality_player_season.parquet", index=False)

    print()
    print("=" * 78)
    print("MOST CENTRAL PLAYERS BY SEASON (PageRank on completed-pass volume)")
    print(f"min {args.min_matches} matches; classical baseline for Module 2")
    print("=" * 78)
    for season, frame in named.groupby("season"):
        # Competition comes from the corpus: two corpora share the season key 2015-2016.
        print(f"\n## {paths.spec.label.rsplit(' ', 1)[0]} {season}")
        top = frame.nlargest(args.top, "pagerank")
        print(
            top[
                [
                    "player_name",
                    "coarse_role",
                    "n_matches",
                    "pagerank",
                    "betweenness",
                    "strength_out",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    print()
    print("## Team structure by season (mean over team-matches)")
    label = club_labeller(paths)
    teams = teams.copy()
    teams["club"] = [
        label(p, t) for p, t in zip(teams["provider"], teams["team_id"])
    ]
    summary = (
        teams.groupby(["season", "club"])[
            ["team_density", "team_centralization", "team_avg_path_length", "team_total_passes"]
        ]
        .mean()
        .round(3)
        .sort_values("team_total_passes", ascending=False)
    )
    print(summary.head(12).to_string())

    print()
    print(f"resource footprint: {monitor.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
