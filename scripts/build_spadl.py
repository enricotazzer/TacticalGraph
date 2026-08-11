#!/usr/bin/env python
"""Convert raw provider data into the canonical SPADL store.

    python scripts/build_spadl.py --all
    python scripts/build_spadl.py --provider statsbomb --limit 20

Also builds the StatsBomb-only enrichment tables used to validate the harmonisation
(true recipients, native possession ids, 24-class positions). Those are validation data,
never model inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tacticalgraph.config import Paths  # noqa: E402
from tacticalgraph.data.adapters import (  # noqa: E402
    BINDINGS,
    convert_game,
    get_binding,
    load_games,
)
from tacticalgraph.data.download import statsbomb_match_ids  # noqa: E402
from tacticalgraph.data.enrichment import build_enrichment  # noqa: E402
from tacticalgraph.data.spadl_store import (  # noqa: E402
    read_games,
    store_summary,
    write_actions,
    write_games,
)

log = logging.getLogger("build_spadl")


def build_provider(
    paths: Paths, provider: str, limit: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert every available game for one provider."""
    binding = get_binding(provider)
    loader = binding.make_loader(paths)
    games = load_games(paths, provider)

    # Only convert games whose raw payload actually landed. The StatsBomb pull is 760
    # files and may be mid-flight or partially failed.
    if provider == "statsbomb":
        available = set(statsbomb_match_ids(paths))
        games = games[games["game_id"].astype(int).isin(available)].reset_index(drop=True)
    if limit:
        games = games.head(limit)

    log.info("%s: converting %d games", provider, len(games))
    frames, failures = [], []
    started = time.perf_counter()

    for index, (_, game) in enumerate(games.iterrows(), start=1):
        try:
            frames.append(convert_game(paths, provider, game, loader=loader))
        except Exception as exc:  # noqa: BLE001 - a bad game must not kill the season
            failures.append({"game_id": int(game["game_id"]), "error": str(exc)})
            log.warning("game %s failed: %s", game["game_id"], exc)
        if index % 50 == 0:
            elapsed = time.perf_counter() - started
            log.info("%s: %d/%d games (%.1fs)", provider, index, len(games), elapsed)

    if not frames:
        raise RuntimeError(f"{provider}: no games converted")

    actions = pd.concat(frames, ignore_index=True)
    # action_id is per-game from the adapter; make it unique across the season so it can
    # serve as a stable key in the store.
    actions["action_id"] = (
        actions.groupby("game_id").cumcount()
        + actions["game_id"].astype("int64") * 100_000
    )

    games = games.copy()
    games["season"] = binding.spec.key
    games["provider"] = provider

    elapsed = time.perf_counter() - started
    log.info(
        "%s: %d actions from %d games in %.1fs (%d failures)",
        provider,
        len(actions),
        actions["game_id"].nunique(),
        elapsed,
        len(failures),
    )
    if failures:
        dest = paths.reports / f"conversion_failures_{provider}.csv"
        pd.DataFrame(failures).to_csv(dest, index=False)
        log.warning("wrote failure list -> %s", dest)

    return actions, games


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--provider", choices=sorted(BINDINGS), action="append")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # socceraction's converters are chatty about pandas downcasting internals.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    providers = sorted(BINDINGS) if args.all else (args.provider or [])
    if not providers:
        parser.error("pick --all or --provider {statsbomb,wyscout}")

    paths = Paths.load().ensure()
    all_games = []

    for provider in providers:
        actions, games = build_provider(paths, provider, limit=args.limit)
        write_actions(paths, actions, season=games["season"].iloc[0], provider=provider)
        all_games.append(games)

        if provider == "statsbomb" and not args.skip_enrichment:
            game_ids = sorted(actions["game_id"].astype(int).unique())
            log.info("building StatsBomb enrichment for %d games", len(game_ids))
            build_enrichment(paths, game_ids)

    # Merge with any previously-built season so a single-provider rebuild does not drop
    # the other one from the index.
    try:
        existing = read_games(paths)
        keep = existing[~existing["provider"].isin(providers)]
        all_games.append(keep)
    except FileNotFoundError:
        pass

    write_games(paths, pd.concat(all_games, ignore_index=True))

    print()
    print(store_summary(paths).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
