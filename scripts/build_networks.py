#!/usr/bin/env python
"""Phase 1.6 -- build and persist passing networks, plus the player directory.

Writes under DATA_ROOT/networks:

    full_nodes.parquet / full_edges.parquet          one network per team-match (Module 2)
    windowed_nodes.parquet / windowed_edges.parquet  15-min window, 5-min stride (Module 3)

and DATA_ROOT/spadl/players.parquet (the cross-provider player directory).

    python scripts/build_networks.py --all
    python scripts/build_networks.py --full-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tacticalgraph.config import Paths  # noqa: E402
from tacticalgraph.data.players import build_player_directory  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions  # noqa: E402
from tacticalgraph.graphs.passing_network import (  # noqa: E402
    DEFAULT_MIN_MINUTES,
    build_match_networks,
    build_windowed_networks,
    networks_to_frames,
    window_bounds,
)

log = logging.getLogger("build_networks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--windowed-only", action="store_true")
    parser.add_argument("--min-minutes", type=float, default=DEFAULT_MIN_MINUTES)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    if not (args.all or args.full_only or args.windowed_only):
        args.all = True

    paths = Paths.load().ensure()
    actions = read_actions(paths)
    log.info("loaded %d actions / %d games", len(actions), actions["game_id"].nunique())

    directory = build_player_directory(paths, actions=actions)
    directory.to_parquet(paths.spadl / "players.parquet", index=False)
    log.info("wrote player directory -> %s", paths.spadl / "players.parquet")

    if args.all or args.full_only:
        started = time.perf_counter()
        networks = build_match_networks(actions, min_minutes=args.min_minutes)
        nodes, edges = networks_to_frames(networks)
        nodes.to_parquet(paths.networks / "full_nodes.parquet", index=False)
        edges.to_parquet(paths.networks / "full_edges.parquet", index=False)
        log.info(
            "full networks: %d networks, %d nodes, %d edges in %.1fs "
            "(mean %.1f nodes, %.1f edges per network)",
            len(networks),
            len(nodes),
            len(edges),
            time.perf_counter() - started,
            len(nodes) / len(networks),
            len(edges) / len(networks),
        )

    if args.all or args.windowed_only:
        started = time.perf_counter()
        bounds = window_bounds()
        log.info("windowed: %d windows per match %s", len(bounds), bounds[:3])
        networks = build_windowed_networks(actions)
        nodes, edges = networks_to_frames(networks)
        nodes.to_parquet(paths.networks / "windowed_nodes.parquet", index=False)
        edges.to_parquet(paths.networks / "windowed_edges.parquet", index=False)
        log.info(
            "windowed networks: %d networks, %d nodes, %d edges in %.1fs "
            "(mean %.1f nodes, %.1f edges per window)",
            len(networks),
            len(nodes),
            len(edges),
            time.perf_counter() - started,
            len(nodes) / len(networks),
            len(edges) / len(networks),
        )

    summary = pd.DataFrame(
        [
            {
                "file": path.name,
                "rows": len(pd.read_parquet(path)),
                "size_mb": round(path.stat().st_size / 1e6, 2),
            }
            for path in sorted(paths.networks.glob("*.parquet"))
            if not path.name.startswith("._")
        ]
    )
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
