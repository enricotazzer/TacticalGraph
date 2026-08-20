"""Classical graph centrality -- the interpretable baseline for Module 2.

This is what a tactical analyst can already compute with networkx and a passing network,
and it is the thing the GNN embedding has to justify itself against. It is also a useful
deliverable on its own: a ranked "who actually runs this team's build-up" table.

A subtlety that matters for football: networkx's shortest-path centralities treat edge
weight as *cost*, but a passing network's weight is *volume* -- a heavily used pass lane is
a short hop, not a long one. So betweenness and closeness are computed on inverted weights
(`distance = 1 / weight`), while degree/eigenvector/PageRank use the raw volume. Getting
this backwards silently inverts the interpretation, which is why it is done in one place.

The same routine runs a second time over xT-weighted edges: `weight_column` chooses which edge
column carries volume, and the inversion applies to it unchanged -- a lane that moves a lot of
threat is a short hop for the same reason a busy lane is. What xT weights add is a case pass
counts cannot produce, an edge worth exactly 0.0; see `ZERO_WEIGHT_DISTANCE`.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from tacticalgraph.graphs.passing_network import TeamNetwork

log = logging.getLogger(__name__)

# Order matters: this is the feature vector the clustering baseline consumes, and the
# Module 3 team-level features reuse the team_* metrics.
PLAYER_METRICS: tuple[str, ...] = (
    "degree_in",
    "degree_out",
    "degree_total",
    "strength_in",
    "strength_out",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
)

# The distance a zero-weight edge is given. With pass counts this never arises (an edge exists
# only because at least one pass made it, so weight >= 1), but an xT-weighted edge sums to
# exactly 0.0 whenever every pass down that lane went backwards -- only positive xT deltas are
# counted. Such a lane is real and must keep its degree, but it is worthless as a *route*, so
# the honest cost is infinite. `math.inf` is nonetheless the wrong value: networkx's weighted
# betweenness and closeness propagate it into NaN, which would silently void those two metrics
# for every player sitting behind such an edge. A large finite number keeps the arithmetic
# well-defined while ranking below every real path. 1e12 sits far above any inverted weight
# that can legitimately occur (even a 1e-6 xT lane inverts to just 1e6) and far below the scale
# at which summing a path's worth of them costs float64 precision.
ZERO_WEIGHT_DISTANCE = 1e12

TEAM_METRICS: tuple[str, ...] = (
    "team_density",
    "team_centralization",
    "team_avg_path_length",
    "team_n_nodes",
    "team_n_edges",
    "team_total_passes",
)


def to_networkx(network: TeamNetwork, weight_column: str = "weight") -> nx.DiGraph:
    """Directed, weighted graph with mean pitch position on each node.

    `weight_column` names the edge column carrying volume: `"weight"` (pass counts) for the
    Module 2 baseline, an xT column for the threat-weighted rerun. Whichever is chosen is
    attached to the edge under the attribute name `weight`, so every metric downstream reads
    one attribute and only the units change.
    """
    edges = network.edges
    if not edges.empty and weight_column not in edges.columns:
        raise KeyError(
            f"{weight_column!r} absent from the edge table; available: {list(edges.columns)}"
        )

    graph = nx.DiGraph()
    for row in network.nodes.itertuples(index=False):
        graph.add_node(
            int(row.player_id),
            mean_x=float(row.mean_x),
            mean_y=float(row.mean_y),
            touches=int(row.touches),
        )
    for row in edges.itertuples(index=False):
        value = float(getattr(row, weight_column))
        graph.add_edge(
            int(row.source),
            int(row.target),
            weight=value,
            # volume -> cost, see module docstring. The guard is `> 0` rather than `!= 0` so
            # that a non-positive weight can never invert into a negative distance, which the
            # shortest-path routines would either reject or quietly mis-solve.
            distance=1.0 / value if value > 0.0 else ZERO_WEIGHT_DISTANCE,
        )
    return graph


def player_centrality(network: TeamNetwork, weight_column: str = "weight") -> pd.DataFrame:
    """Per-player centrality metrics for one team-match network.

    `weight_column` selects the edge column to weight by. The metric column names are fixed
    either way: it is the caller that distinguishes the two runs, by renaming or suffixing the
    columns it gets back. Naming them here would fork `PLAYER_METRICS` and every consumer of it.

    Note that `degree_in`, `degree_out` and `degree_total` count edges, not their weight, so
    they come out bit-identical on both runs. Running the pipeline twice therefore produces
    three duplicate columns, and dropping them is the joining caller's job -- this function
    cannot suppress them, because it has no way to know which of the two runs it is in, and
    the first run needs them.
    """
    graph = to_networkx(network, weight_column=weight_column)
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["player_id", *PLAYER_METRICS])

    undirected = graph.to_undirected()

    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(graph, distance="distance")

    # Eigenvector centrality does not always converge on sparse/disconnected passing
    # networks. PageRank is the robust stand-in, so fall back rather than crash a run
    # halfway through 760 matches.
    try:
        eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except (nx.NetworkXException, np.linalg.LinAlgError) as exc:
        log.debug(
            "eigenvector failed for game=%s team=%s (%s); using NaN",
            network.game_id,
            network.team_id,
            exc,
        )
        eigenvector = {node: np.nan for node in graph.nodes}

    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {
        node: np.nan for node in graph.nodes
    }
    clustering = nx.clustering(undirected, weight="weight")

    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "player_id": node,
                "degree_in": graph.in_degree(node),
                "degree_out": graph.out_degree(node),
                "degree_total": graph.degree(node),
                "strength_in": graph.in_degree(node, weight="weight"),
                "strength_out": graph.out_degree(node, weight="weight"),
                "betweenness": betweenness.get(node, np.nan),
                "closeness": closeness.get(node, np.nan),
                "eigenvector": eigenvector.get(node, np.nan),
                "pagerank": pagerank.get(node, np.nan),
                "clustering": clustering.get(node, np.nan),
            }
        )

    frame = pd.DataFrame(rows)
    frame["game_id"] = network.game_id
    frame["team_id"] = network.team_id
    frame["season"] = network.season
    frame["provider"] = network.provider
    if network.window_index is not None:
        frame["window_index"] = network.window_index
    return frame


def team_metrics(network: TeamNetwork) -> dict[str, float]:
    """Team-level structural metrics for one network."""
    graph = to_networkx(network)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    total_passes = float(network.edges["weight"].sum()) if not network.edges.empty else 0.0

    if n_nodes < 2:
        return {
            "game_id": network.game_id,
            "team_id": network.team_id,
            "season": network.season,
            "provider": network.provider,
            "team_density": np.nan,
            "team_centralization": np.nan,
            "team_avg_path_length": np.nan,
            "team_n_nodes": n_nodes,
            "team_n_edges": n_edges,
            "team_total_passes": total_passes,
        }

    density = nx.density(graph)

    # Freeman centralisation on total degree: 0 = every player equally connected,
    # 1 = one player is the sole hub. Reads directly as "how star-shaped is this team".
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    max_degree = degrees.max()
    denominator = (n_nodes - 1) * (n_nodes - 2) if n_nodes > 2 else np.nan
    centralization = (
        float((max_degree - degrees).sum() / denominator) if denominator and denominator > 0 else np.nan
    )

    # Passing networks are frequently not strongly connected, so average shortest path is
    # taken over the largest weakly connected component rather than returning NaN.
    undirected = graph.to_undirected()
    if undirected.number_of_edges():
        component = max(nx.connected_components(undirected), key=len)
        subgraph = undirected.subgraph(component)
        avg_path = (
            nx.average_shortest_path_length(subgraph, weight="distance")
            if subgraph.number_of_nodes() > 1
            else np.nan
        )
    else:
        avg_path = np.nan

    return {
        "game_id": network.game_id,
        "team_id": network.team_id,
        "season": network.season,
        "provider": network.provider,
        "team_density": density,
        "team_centralization": centralization,
        "team_avg_path_length": avg_path,
        "team_n_nodes": n_nodes,
        "team_n_edges": n_edges,
        "team_total_passes": total_passes,
    }


def centrality_table(
    networks: list[TeamNetwork], weight_column: str = "weight"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute player and team metrics over many networks.

    `weight_column` is forwarded to `player_centrality` only. The team table is deliberately
    left on pass counts: `team_total_passes` is a count by definition, and the structural
    metrics beside it describe the same graph either way, so an xT rerun returns a team table
    identical to the baseline's and the caller can discard it.
    """
    player_frames, team_rows = [], []
    for index, network in enumerate(networks, start=1):
        frame = player_centrality(network, weight_column=weight_column)
        if not frame.empty:
            player_frames.append(frame)
        team_rows.append(team_metrics(network))
        if index % 200 == 0:
            log.info("centrality: %d/%d networks", index, len(networks))

    players = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    teams = pd.DataFrame(team_rows)
    return players, teams


def aggregate_player_season(
    players: pd.DataFrame,
    min_matches: int = 5,
    metrics: tuple[str, ...] = PLAYER_METRICS,
) -> pd.DataFrame:
    """Average a player's per-match centrality across a season.

    `min_matches` guards against ranking a player who happened to have one unusual game;
    5 is low enough to keep rotation players and high enough to damp single-match noise.

    `metrics` defaults to the ten classical ones. Callers that also computed the xT-weighted
    twins or the threat features pass the wider list; a name that is not a column raises here
    rather than being dropped, because a metric silently missing from the season table becomes a
    metric silently missing from the leaderboard.
    """
    missing = [m for m in metrics if m not in players.columns]
    if missing:
        raise KeyError(f"metrics {missing} absent from the player table")

    grouped = players.groupby(["season", "provider", "player_id"])
    aggregated = grouped[list(metrics)].mean()
    aggregated["n_matches"] = grouped.size()
    aggregated = aggregated.reset_index()
    return aggregated[aggregated["n_matches"] >= min_matches].reset_index(drop=True)


def role_relative_metrics(
    aggregated: pd.DataFrame,
    role_column: str = "coarse_role",
    metrics: tuple[str, ...] = PLAYER_METRICS,
    suffix: str = "_z",
) -> pd.DataFrame:
    """Add within-role z-scores for each centrality metric.

    Raw centrality is largely a positional artefact: on the Premier League corpus midfielders are
    31% of players with >=10 matches but **84% of the top 50 by `degree_total`**, and goalkeepers
    take 0% of the top 50 on all ten metrics. A leaderboard sorted on the raw value is therefore a
    list of midfielders, and it cannot express "unusually central *for a centre-back*" at all.

    Z-scoring within `coarse_role` makes that expressible and gives keepers and forwards a
    meaningful ranking, at the cost of no longer comparing across roles -- which raw centrality
    never did honestly anyway.

    A role with fewer than two players, or zero variance in a metric, gets 0.0 rather than NaN or
    an infinity: "no evidence this player is unusual for their role" is the correct reading, and a
    NaN would silently drop them from any downstream sort.
    """
    if role_column not in aggregated.columns:
        raise KeyError(
            f"{role_column!r} absent; join the player directory's role labels before z-scoring"
        )

    frame = aggregated.copy()
    present = [m for m in metrics if m in frame.columns]
    for metric in present:
        grouped = frame.groupby(role_column)[metric]
        mean = grouped.transform("mean")
        # Population std (ddof=0) so a two-player role still yields a finite spread.
        std = grouped.transform(lambda s: s.std(ddof=0))
        frame[f"{metric}{suffix}"] = ((frame[metric] - mean) / std.replace(0.0, np.nan)).fillna(0.0)
    return frame
