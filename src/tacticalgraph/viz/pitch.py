"""Passing-network visualisation on a pitch.

The purpose here is verification, not decoration. Two harmonisation failures would be
invisible in summary statistics but obvious in a picture:

* a broken coordinate flip -- a team attacking the wrong way, or a mirrored shape;
* a provider-specific distortion -- the same club looking structurally different in
  2015/16 and 2017/18 for reasons that are not football.

So the key output is `plot_provider_comparison`, which puts one club's network from each
season side by side.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these are written to files, never shown interactively

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mplsoccer import Pitch  # noqa: E402

from tacticalgraph.config import PITCH_LENGTH, PITCH_WIDTH  # noqa: E402
from tacticalgraph.graphs.passing_network import TeamNetwork  # noqa: E402

log = logging.getLogger(__name__)

# SPADL coordinates are metres on a 105x68 pitch, which is exactly mplsoccer's
# `pitch_type="custom"` with these dimensions -- no rescaling anywhere.
PITCH_KWARGS = dict(
    pitch_type="custom",
    pitch_length=PITCH_LENGTH,
    pitch_width=PITCH_WIDTH,
    line_color="#4a4a4a",
    pitch_color="#f7f7f7",
    linewidth=1.0,
)

MIN_EDGE_WEIGHT = 3  # below this, edges are noise and the plot becomes unreadable


def draw_network(
    network: TeamNetwork,
    ax: plt.Axes,
    node_metric: dict[int, float] | None = None,
    title: str | None = None,
    min_edge_weight: int = MIN_EDGE_WEIGHT,
) -> plt.Axes:
    """Draw one team's passing network onto an existing axis."""
    pitch = Pitch(**PITCH_KWARGS)
    pitch.draw(ax=ax)

    nodes = network.nodes
    if nodes.empty:
        ax.set_title(title or "(empty network)", fontsize=9)
        return ax

    positions = {
        int(row.player_id): (float(row.mean_x), float(row.mean_y))
        for row in nodes.itertuples(index=False)
    }

    edges = network.edges
    if not edges.empty:
        edges = edges[edges["weight"] >= min_edge_weight]
        max_weight = float(edges["weight"].max()) if not edges.empty else 1.0
        for row in edges.itertuples(index=False):
            start, end = positions.get(int(row.source)), positions.get(int(row.target))
            if start is None or end is None:
                continue
            pitch.lines(
                start[0],
                start[1],
                end[0],
                end[1],
                lw=0.6 + 3.4 * float(row.weight) / max_weight,
                color="#1f6feb",
                alpha=0.35,
                zorder=1,
                ax=ax,
            )

    # Node size encodes the supplied metric (centrality by default caller's choice), so a
    # visual check can confirm that hubs sit where a coach would expect them.
    if node_metric:
        values = np.array([node_metric.get(pid, 0.0) for pid in positions], dtype=float)
        finite = values[np.isfinite(values)]
        spread = finite.max() - finite.min() if finite.size and finite.max() > finite.min() else 1.0
        low = finite.min() if finite.size else 0.0
        sizes = 90 + 620 * np.clip((values - low) / spread, 0, 1)
    else:
        sizes = np.full(len(positions), 220.0)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pitch.scatter(
        xs, ys, s=sizes, color="#ff7b00", edgecolors="#22223b", linewidth=1.1, zorder=3, ax=ax
    )

    for pid, (x, y) in positions.items():
        ax.annotate(
            str(pid)[-3:],  # last 3 digits keep the label readable
            (x, y),
            fontsize=5.5,
            ha="center",
            va="center",
            zorder=4,
            color="#22223b",
        )

    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_match_networks(
    networks: list[TeamNetwork],
    dest: Path,
    labels: dict[int, str] | None = None,
    node_metric: dict[int, dict[int, float]] | None = None,
) -> Path:
    """Grid of team networks, two per row (one match per row)."""
    n = len(networks)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, network in zip(axes, networks):
        team_label = (labels or {}).get(network.team_id, f"team {network.team_id}")
        metric = (node_metric or {}).get(network.team_id)
        window = "" if network.window_index is None else f" w{network.window_index}"
        draw_network(
            network,
            ax,
            node_metric=metric,
            title=(
                f"{team_label} -- {network.season} ({network.provider})"
                f"{window}\n{network.n_nodes} nodes, {network.n_edges} edges"
            ),
        )
    for ax in axes[n:]:
        ax.axis("off")

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", dest)
    return dest


def plot_provider_comparison(
    pairs: list[tuple[str, TeamNetwork, TeamNetwork]], dest: Path
) -> Path:
    """The harmonisation eyeball test: one club, both seasons, side by side.

    `pairs` is [(club_name, network_2015_16, network_2017_18), ...]. If the two columns
    look structurally alike, the SPADL conversion and the coordinate flip agree across
    providers; if one column is mirrored or squashed, something is wrong upstream.
    """
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13, 4.6 * len(pairs)))
    axes = np.atleast_2d(axes)

    for row, (club, left, right) in enumerate(pairs):
        for col, network in enumerate((left, right)):
            draw_network(
                network,
                axes[row, col],
                title=(
                    f"{club} -- {network.season} ({network.provider})\n"
                    f"{network.n_nodes} nodes, {network.n_edges} edges, "
                    f"{int(network.edges['weight'].sum()) if not network.edges.empty else 0} passes"
                ),
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "Harmonisation check: same club, both providers (attacking left to right)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", dest)
    return dest
