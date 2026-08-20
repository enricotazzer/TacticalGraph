#!/usr/bin/env python
"""Phase 2 -- classical centrality metrics: the interpretable baseline for Module 2.

Writes DATA_ROOT/networks/centrality_players.parquet and centrality_teams.parquet, and
prints the season's most central players -- a standalone deliverable, and the thing the GNN
embedding in Phase 3 has to beat.

It also measures the limitation that baseline is known to have. Pass-only edges make
centrality a volume proxy: on the Premier League corpus midfielders are 31% of players with
>=10 matches and **84% of the top 50 by `degree_total`**, while goalkeepers take 0% on all ten
metrics. Three candidate fixes are computed here and scored against that baseline in one
report -- xT-weighted edges, shot-chain involvement, and role-relative z-scoring.

    python scripts/run_centrality.py --corpus premier_league
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tacticalgraph.config import ALL_SPLIT_KINDS, CORPORA, DEFAULT_CORPUS, Paths  # noqa: E402
from tacticalgraph.data.aliases import club_labeller  # noqa: E402
from tacticalgraph.data.players import load_player_directory  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions, read_games  # noqa: E402
from tacticalgraph.eval.resources import ResourceMonitor  # noqa: E402
from tacticalgraph.eval.splits import temporal_split  # noqa: E402
from tacticalgraph.features.centrality import (  # noqa: E402
    PLAYER_METRICS,
    aggregate_player_season,
    centrality_table,
    residualise_against_position,
    role_relative_metrics,
)
from tacticalgraph.features.chains import shot_chain_involvement  # noqa: E402
from tacticalgraph.features.xthreat import (  # noqa: E402
    attach_xt_edge_weights,
    fit_xthreat,
    player_threat,
)
from tacticalgraph.graphs.passing_network import TeamNetwork  # noqa: E402

log = logging.getLogger("centrality")

NETWORK_KEYS = ["game_id", "team_id", "season", "provider"]
PLAYER_KEYS = NETWORK_KEYS + ["player_id"]

# `degree_*` count edges and ignore their weight, so re-running the pipeline on xT weights
# reproduces them exactly. Only the seven weight-sensitive metrics get an `_xt` twin.
WEIGHTED_METRICS: tuple[str, ...] = tuple(
    m for m in PLAYER_METRICS if not m.startswith("degree_")
)
THREAT_METRICS: tuple[str, ...] = ("xt_generated", "shot_involvement", "shot_conversion")

# Mean pitch position, carried through to the season table so `residualise_against_position` can
# ask how much of each metric is just "where does this player stand".
POSITION_COLUMNS: tuple[str, ...] = ("mean_x", "mean_y")


def networks_from_frames(nodes: pd.DataFrame, edges: pd.DataFrame) -> list[TeamNetwork]:
    """Rehydrate persisted node/edge frames into TeamNetwork objects.

    Takes frames rather than reading parquet itself, because the caller has to join xT onto the
    edges after the split is known and before the networks are built.
    """
    edge_groups = {k: g for k, g in edges.groupby(NETWORK_KEYS, sort=False)}

    networks = []
    for key, node_group in nodes.groupby(NETWORK_KEYS, sort=False):
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


def top_role_composition(
    frame: pd.DataFrame, metric: str, top: int = 50, role_column: str = "coarse_role"
) -> dict[str, float]:
    """Share of the top-`top` players by `metric` falling in each coarse role.

    The measurement the volume-proxy limitation is stated in, so it is computed the same way
    for every candidate fix rather than eyeballed per metric.
    """
    ranked = frame.nlargest(top, metric)
    counts = ranked[role_column].value_counts()
    return {role: round(100.0 * counts.get(role, 0) / len(ranked), 1) for role in ("GK", "DEF", "MID", "FWD")}


def mean_pairwise_spearman(frame: pd.DataFrame, metrics: list[str]) -> float:
    """Mean off-diagonal Spearman rho across a family of metrics.

    `DATA_SOURCES.md` cites the metrics' *disagreement* -- `degree_total` reads 84% MID while
    `strength_out` reads 48% DEF -- as evidence that none of them measures a shared construct.
    This turns that observation into one number so the volume and xT families can be compared.
    """
    present = [m for m in metrics if m in frame.columns and frame[m].notna().any()]
    if len(present) < 2:
        return float("nan")
    corr = frame[present].corr(method="spearman").to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(upper))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-matches", type=int, default=5)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--composition-top", type=int, default=50,
                        help="cohort size for the role-composition report (default: %(default)s)")
    parser.add_argument("--composition-min-matches", type=int, default=10,
                        help="matches required to enter the composition report; 10 is the "
                             "threshold the published baseline in DATA_SOURCES.md uses")
    parser.add_argument(
        "--split", choices=ALL_SPLIT_KINDS, default=None,
        help="split kind whose TRAIN fold fits xThreat; defaults to the corpus's primary kind",
    )
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
        help="which competition corpus to use (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.split is None:
        args.split = CORPORA[args.corpus].split_kinds[0]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    paths = Paths.load(args.corpus).ensure()

    nodes = pd.read_parquet(paths.networks / "full_nodes.parquet")
    edges = pd.read_parquet(paths.networks / "full_edges.parquet")

    # This script predicts nothing, so an all-corpus xT fit would look harmless. It is not:
    # `centrality_players.parquet` is what `train_roles.py` loads as the *clustering baseline's*
    # feature matrix, and that baseline is scored on a test fold. Fitting the surface on the whole
    # corpus would leak the test fold into the baseline the GNN is compared against.
    games = read_games(paths)
    split = temporal_split(games, kind=args.split, corpus=args.corpus)
    actions = read_actions(paths)

    with ResourceMonitor("xthreat") as xt_monitor:
        xt = fit_xthreat(actions, split.train)
        edges = attach_xt_edge_weights(edges, actions, xt)
    log.info("xT edge weights: %s", xt_monitor.summary())

    networks = networks_from_frames(nodes, edges)
    log.info("rehydrated %d team-match networks", len(networks))

    with ResourceMonitor("centrality") as monitor:
        players, teams = centrality_table(networks)
        players_xt, _ = centrality_table(networks, weight_column="xt_weight")

    players = players.merge(
        players_xt[PLAYER_KEYS + list(WEIGHTED_METRICS)].rename(
            columns={m: f"{m}_xt" for m in WEIGHTED_METRICS}
        ),
        on=PLAYER_KEYS,
        how="left",
    )
    players = players.merge(
        player_threat(actions, xt), on=PLAYER_KEYS, how="left"
    ).merge(shot_chain_involvement(actions), on=PLAYER_KEYS, how="left")
    players[list(THREAT_METRICS)] = players[list(THREAT_METRICS)].fillna(0.0)

    players = players.merge(
        nodes[PLAYER_KEYS + list(POSITION_COLUMNS)], on=PLAYER_KEYS, how="left"
    )

    players.to_parquet(paths.networks / "centrality_players.parquet", index=False)
    teams.to_parquet(paths.networks / "centrality_teams.parquet", index=False)
    log.info("wrote %d player-match rows, %d team-match rows", len(players), len(teams))

    all_metrics = (
        list(PLAYER_METRICS)
        + [f"{m}_xt" for m in WEIGHTED_METRICS]
        + list(THREAT_METRICS)
        + list(POSITION_COLUMNS)
    )
    season_level = aggregate_player_season(
        players, min_matches=args.min_matches, metrics=tuple(all_metrics)
    )
    directory = load_player_directory(paths)
    named = season_level.merge(
        directory[["season", "provider", "player_id", "player_name", "coarse_role"]],
        on=["season", "provider", "player_id"],
        how="left",
    )
    # Raw centrality cannot express "unusually central *for a centre-back*", and sorts into a
    # list of midfielders. The z-scores are what make the leaderboard in the app rankable for
    # goalkeepers and forwards at all -- see the caveat in the report below.
    named = role_relative_metrics(named, metrics=tuple(all_metrics))
    # How much of each metric is position and nothing else? A quadratic fit, because pass volume
    # peaks in midfield and falls off toward both goals -- a linear one would leave that inverted
    # U in the residual and understate exactly what this is measuring.
    named, r2 = residualise_against_position(named, metrics=tuple(all_metrics))
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

    report = volume_proxy_report(named, args, r2)
    print_volume_proxy_report(report, paths)
    out = paths.reports / f"module2_volume_proxy_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out)

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


def volume_proxy_report(
    named: pd.DataFrame, args: argparse.Namespace, r2: pd.DataFrame
) -> dict:
    """Score the three candidate fixes against the published volume-proxy baseline."""
    cohort = named[named["n_matches"] >= args.composition_min_matches]
    cohort = cohort[cohort["coarse_role"].notna()]
    population = {
        role: round(100.0 * (cohort["coarse_role"] == role).mean(), 1)
        for role in ("GK", "DEF", "MID", "FWD")
    }

    # The like-for-like row is `volume_same_7` vs `xt_weighted`, not `volume_all_10` vs it.
    # `degree_total` is `degree_in + degree_out` by construction, so the three degree metrics are
    # near-collinear and inflate whichever family contains them. Both are reported so the
    # inflation is visible rather than hidden in a single headline number.
    families = {
        "volume_all_10": list(PLAYER_METRICS),
        "volume_same_7": list(WEIGHTED_METRICS),
        "xt_weighted": [f"{m}_xt" for m in WEIGHTED_METRICS],
    }
    headline = ["degree_total", "pagerank", "betweenness", "strength_out"]
    composition = {}
    for metric in headline:
        composition[metric] = top_role_composition(cohort, metric, args.composition_top)
        for variant in (f"{metric}_xt", f"{metric}_z", f"{metric}_xt_z"):
            if variant in cohort.columns:
                composition[variant] = top_role_composition(
                    cohort, variant, args.composition_top
                )
    for metric in THREAT_METRICS:
        composition[metric] = top_role_composition(cohort, metric, args.composition_top)
        if f"{metric}_r" in cohort.columns:
            composition[f"{metric}_r"] = top_role_composition(
                cohort, f"{metric}_r", args.composition_top
            )
    for metric in headline:
        if f"{metric}_r" in cohort.columns:
            composition[f"{metric}_r"] = top_role_composition(
                cohort, f"{metric}_r", args.composition_top
            )
    position_r2 = (
        r2.groupby("metric")["r2"].mean().round(4).to_dict() if not r2.empty else {}
    )

    return {
        "corpus": args.corpus,
        "split": args.split,
        "n_players": int(len(cohort)),
        "min_matches": args.composition_min_matches,
        "top": args.composition_top,
        "population_share": population,
        "top_composition": composition,
        "mean_pairwise_spearman": {
            name: round(mean_pairwise_spearman(cohort, metrics), 4)
            for name, metrics in families.items()
        },
        "position_r2": position_r2,
        "shot_conversion_vs_degree_total": round(
            float(cohort[["shot_conversion", "degree_total"]].corr(method="spearman").iloc[0, 1]),
            4,
        ),
        "shot_involvement_vs_degree_total": round(
            float(cohort[["shot_involvement", "degree_total"]].corr(method="spearman").iloc[0, 1]),
            4,
        ),
        "xt_generated_vs_degree_total": round(
            float(cohort[["xt_generated", "degree_total"]].corr(method="spearman").iloc[0, 1]),
            4,
        ),
    }


def print_volume_proxy_report(report: dict, paths: Paths) -> None:
    print()
    print("=" * 78)
    print("VOLUME-PROXY REPORT -- does the enriched graph stop ranking only midfielders?")
    print(
        f"{report['n_players']} players with >={report['min_matches']} matches; "
        f"share of the top {report['top']}"
    )
    print("=" * 78)
    rows = [{"metric": k, **v} for k, v in report["top_composition"].items()]
    rows.append({"metric": "*population*", **report["population_share"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Mean pairwise Spearman across each metric family (do they agree at all?):")
    print("  compare volume_same_7 against xt_weighted -- volume_all_10 is inflated by the")
    print("  three collinear degree metrics and is shown only to make that visible.")
    for name, value in report["mean_pairwise_spearman"].items():
        print(f"  {name:12s} {value:+.4f}")
    print()
    print(
        "Spearman vs degree_total -- is the new metric a third volume proxy?\n"
        f"  shot_involvement  {report['shot_involvement_vs_degree_total']:+.4f}\n"
        f"  shot_conversion   {report['shot_conversion_vs_degree_total']:+.4f}\n"
        f"  xt_generated      {report['xt_generated_vs_degree_total']:+.4f}"
    )
    if report["position_r2"]:
        print()
        print("Share of each metric explained by mean pitch position alone (quadratic fit):")
        for metric, value in sorted(
            report["position_r2"].items(), key=lambda kv: -kv[1]
        ):
            if metric in POSITION_COLUMNS:
                continue
            print(f"  {metric:20s} R2 = {value:.3f}")
    print()
    print(
        "Read the *_z rows with care: z-scoring within role forces the top-N mix toward the\n"
        "population mix, so their agreement is mechanical. They make 'central for a centre-back'\n"
        "expressible; they do not show the graph measures tactical importance."
    )


if __name__ == "__main__":
    raise SystemExit(main())
