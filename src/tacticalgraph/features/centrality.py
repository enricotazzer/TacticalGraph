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


def residualise_against_position(
    aggregated: pd.DataFrame,
    metrics: tuple[str, ...] = PLAYER_METRICS,
    position_columns: tuple[str, ...] = ("mean_x", "mean_y"),
    group_columns: tuple[str, ...] = ("season", "provider"),
    suffix: str = "_r",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each metric on the player's mean pitch position and keep only what the fit misses.

    `role_relative_metrics` works *around* the positional artefact by normalising inside a role.
    This measures it head-on. Four candidate fixes have been tried -- pass direction, role-relative
    z-scoring, xT-weighted edges, shot-chain involvement -- and none removed the positional signal;
    xT-weighting inverted it instead, taking `rho(pagerank_xt, mean_x)` to +0.927 against +0.298
    for `rho(pagerank, mean_x)`. Every one of those was judged by proxy, on how the leaderboard
    looked afterwards. The direct question is what share of a metric's variance the player's mean
    pitch position accounts for, and that share is an R^2: a metric that is only a statement about
    where a player stands has nothing left once position is fitted out, and `r2_table` reports it
    as a number instead of an impression. The residual is what a defensible metric would be built
    on; the R^2 beside it is what the caller actually reports.

    **The fit is quadratic, and that is the load-bearing decision.** Pass volume is not monotonic
    in `mean_x`: it peaks in midfield and falls away toward both goals, so degree-against-position
    is an inverted U. A straight line through a U fits nothing, leaves the whole arch standing in
    the residual, and reports a low R^2 -- it would *understate* how positional the metric is,
    which is the one direction of error a positionality detector must not make. The design matrix
    is therefore `[1, x, y, x^2, y^2, x*y]`: quadratic in both pitch axes, plus the cross term that
    lets the fitted surface tilt, since a wide player deep is not a wide player high.

    The fit runs once per `group_columns` because pitch-position distributions differ by provider
    and by season -- confounded in this corpus, as ever -- and a pooled fit would charge that
    difference to the player rather than to the coordinate frame it belongs to.

    A group that cannot be fitted gets a residual of 0.0 and an R^2 of 0.0 rather than a NaN or a
    fit. Two cases: fewer rows than the six design columns, where the system is underdetermined and
    would otherwise interpolate its way to a spurious R^2 of 1.0; and a metric with no variance
    inside the group, where there is no share of variance for position to explain. This is the
    convention `role_relative_metrics` already uses -- 0.0 reads as "no evidence", where a NaN
    would silently drop the player out of any downstream sort. Rows carrying a non-finite metric or
    position are held out of the fit and keep a 0.0 residual for the same reason; `eigenvector` and
    `pagerank` are NaN by design on networks where they do not converge.

    A metric absent from `aggregated` is skipped, following `role_relative_metrics`. A missing
    *position* column raises instead: there is no fit to be had without it, and handing back an
    unresidualised frame wearing residual column names is worse than stopping.

    Returns `(frame, r2_table)`. `frame` is `aggregated` plus one `f"{metric}{suffix}"` residual
    column per fitted metric; `r2_table` carries `[*group_columns, "metric", "r2", "n"]`, where `n`
    is the number of rows that entered that group's fit.
    """
    missing = [column for column in position_columns if column not in aggregated.columns]
    if missing:
        raise KeyError(
            f"position columns {missing} absent from the aggregated table; centrality cannot be "
            "residualised against a position the frame does not carry"
        )

    frame = aggregated.copy()
    present = [metric for metric in metrics if metric in frame.columns]

    positions = frame[list(position_columns)].to_numpy(dtype=float)
    n_axes = len(position_columns)
    # 1 intercept + n linear + n squared + one cross term per axis pair; 6 for the default (x, y).
    n_terms = 1 + 2 * n_axes + n_axes * (n_axes - 1) // 2

    residuals = {metric: np.zeros(len(frame)) for metric in present}
    r2_rows: list[dict[str, object]] = []

    groups = (
        frame.groupby(list(group_columns), dropna=False).indices
        if group_columns
        else {(): np.arange(len(frame))}
    )

    for key, rows in groups.items():
        key_values = key if isinstance(key, tuple) else (key,)
        block = positions[rows]
        placed = np.isfinite(block).all(axis=1)

        # Standardise within the group before squaring. `mean_x` is metres on a 0-105 pitch, so a
        # raw x^2 column runs to ~11000 while x sits near 50 and the intercept is 1 -- four orders
        # of column scale in one design, which costs lstsq precision on the linear terms and buys
        # nothing. Centring also decorrelates x from x^2, so the quadratic term reads as curvature
        # rather than quietly re-fitting the slope.
        scaled = np.zeros_like(block)
        if placed.any():
            mean = block[placed].mean(axis=0)
            std = block[placed].std(axis=0)
            scaled = (block - mean) / np.where(std > 0.0, std, 1.0)

        design_columns = [np.ones(len(rows)), *scaled.T, *(scaled.T**2)]
        for i in range(n_axes):
            for j in range(i + 1, n_axes):
                design_columns.append(scaled[:, i] * scaled[:, j])
        design = np.column_stack(design_columns)

        for metric in present:
            values = frame[metric].to_numpy(dtype=float)[rows]
            fitted = placed & np.isfinite(values)
            target = values[fitted]

            # The degeneracy test is on the spread relative to the level, not on SS_tot against 0.
            # A constant metric does not produce an SS_tot of exactly zero: `target.mean()` lands
            # an ulp off the value it is averaging and the squared deviations pile up as float
            # noise, so `SS_tot > 0` passes and R^2 comes back as noise over noise -- about -10 in
            # practice, which is worse than the NaN this function exists to avoid.
            level = float(np.abs(target).max()) if target.size else 0.0
            flat = target.size == 0 or float(np.ptp(target)) <= 1e-12 * level

            if target.size >= n_terms and not flat:
                ss_tot = float(((target - target.mean()) ** 2).sum())
                coefficients, *_ = np.linalg.lstsq(design[fitted], target, rcond=None)
                residual = target - design[fitted] @ coefficients
                residuals[metric][rows[fitted]] = residual
                r2 = 1.0 - float((residual**2).sum()) / ss_tot
            else:
                r2 = 0.0

            r2_rows.append(
                {
                    **dict(zip(group_columns, key_values, strict=True)),
                    "metric": metric,
                    "r2": r2,
                    "n": int(target.size),
                }
            )

    for metric in present:
        frame[f"{metric}{suffix}"] = residuals[metric]

    r2_table = pd.DataFrame(r2_rows, columns=[*group_columns, "metric", "r2", "n"])
    return frame, r2_table
