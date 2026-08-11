"""Classical graph centrality -- the interpretable baseline for Module 2.

This is what a tactical analyst can already compute with networkx and a passing network,
and it is the thing the GNN embedding has to justify itself against. It is also a useful
deliverable on its own: a ranked "who actually runs this team's build-up" table.

A subtlety that matters for football: networkx's shortest-path centralities treat edge
weight as *cost*, but a passing network's weight is *volume* -- a heavily used pass lane is
a short hop, not a long one. So betweenness and closeness are computed on inverted weights
(`distance = 1 / weight`), while degree/eigenvector/PageRank use the raw volume. Getting
this backwards silently inverts the interpretation, which is why it is done in one place.
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

TEAM_METRICS: tuple[str, ...] = (
    "team_density",
    "team_centralization",
    "team_avg_path_length",
    "team_n_nodes",
    "team_n_edges",
    "team_total_passes",
)


def to_networkx(network: TeamNetwork) -> nx.DiGraph:
    """Directed, weighted graph with mean pitch position on each node."""
    graph = nx.DiGraph()
    for row in network.nodes.itertuples(index=False):
        graph.add_node(
            int(row.player_id),
            mean_x=float(row.mean_x),
            mean_y=float(row.mean_y),
            touches=int(row.touches),
        )
    for row in network.edges.itertuples(index=False):
        graph.add_edge(
            int(row.source),
            int(row.target),
            weight=float(row.weight),
            distance=1.0 / float(row.weight),  # volume -> cost, see module docstring
        )
    return graph


def player_centrality(network: TeamNetwork) -> pd.DataFrame:
    """Per-player centrality metrics for one team-match network."""
    graph = to_networkx(network)
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


def centrality_table(networks: list[TeamNetwork]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute player and team metrics over many networks."""
    player_frames, team_rows = [], []
    for index, network in enumerate(networks, start=1):
        frame = player_centrality(network)
        if not frame.empty:
            player_frames.append(frame)
        team_rows.append(team_metrics(network))
        if index % 200 == 0:
            log.info("centrality: %d/%d networks", index, len(networks))

    players = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    teams = pd.DataFrame(team_rows)
    return players, teams


def aggregate_player_season(
    players: pd.DataFrame, min_matches: int = 5
) -> pd.DataFrame:
    """Average a player's per-match centrality across a season.

    `min_matches` guards against ranking a player who happened to have one unusual game;
    5 is low enough to keep rotation players and high enough to damp single-match noise.
    """
    grouped = players.groupby(["season", "provider", "player_id"])
    aggregated = grouped[list(PLAYER_METRICS)].mean()
    aggregated["n_matches"] = grouped.size()
    aggregated = aggregated.reset_index()
    return aggregated[aggregated["n_matches"] >= min_matches].reset_index(drop=True)
