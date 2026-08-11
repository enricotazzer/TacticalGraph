#!/usr/bin/env python
"""Raw data acquisition entry point.

    python scripts/ingest.py --all
    python scripts/ingest.py --statsbomb --limit 20     # small dev subset
    python scripts/ingest.py --wyscout

Resumable: re-running skips anything already on disk and valid.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacticalgraph.config import Paths  # noqa: E402
from tacticalgraph.data.download import (  # noqa: E402
    download_statsbomb_season,
    download_wyscout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="both providers")
    parser.add_argument("--statsbomb", action="store_true", help="Serie A 2015/16")
    parser.add_argument("--wyscout", action="store_true", help="Serie A 2017/18")
    parser.add_argument("--limit", type=int, default=None, help="first N matches only")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not (args.all or args.statsbomb or args.wyscout):
        parser.error("pick at least one of --all / --statsbomb / --wyscout")

    paths = Paths.load().ensure()
    logging.info("DATA_ROOT = %s", paths.root)

    # Each provider is independent: a failure in one must not discard the other's work,
    # since between them they are ~1.2 GB of downloads.
    failed = []
    if args.all or args.statsbomb:
        try:
            download_statsbomb_season(paths, force=args.force, limit=args.limit)
        except Exception:  # noqa: BLE001
            logging.exception("StatsBomb download failed")
            failed.append("statsbomb")
    if args.all or args.wyscout:
        try:
            download_wyscout(paths, force=args.force)
        except Exception:  # noqa: BLE001
            logging.exception("Wyscout download failed")
            failed.append("wyscout")

    if failed:
        logging.error("failed providers: %s (re-run to resume)", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
