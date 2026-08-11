#!/usr/bin/env python
"""Export a small, self-contained demo bundle into the repo.

The full data store is ~1.5 GB on an external drive, which makes the project unshowable:
the demo app would only run on the machine that built it. This script copies the minimum
needed to drive every demo page into `demo_data/` (~8 MB), so `streamlit run app/Home.py`
works on any checkout with no external drive and no re-ingestion.

What makes 8 MB sufficient: `engineer_node_features()` derives every model input from the
node/edge tables alone, so the 55 MB SPADL action store and the 1.4 GB of raw provider JSON
are not needed for anything the demo shows.

    python scripts/export_demo_bundle.py
    python scripts/export_demo_bundle.py --windowed-matches 30
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tacticalgraph.config import Paths  # noqa: E402

log = logging.getLogger("export_demo_bundle")

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "demo_data"

# (source-relative-path, destination-name). Sources are resolved against DATA_ROOT.
COPY_TABLES: tuple[tuple[str, str], ...] = (
    ("spadl/games.parquet", "games.parquet"),
    ("spadl/players.parquet", "players.parquet"),
    ("networks/full_nodes.parquet", "full_nodes.parquet"),
    ("networks/full_edges.parquet", "full_edges.parquet"),
    ("networks/centrality_players.parquet", "centrality_players.parquet"),
    ("networks/centrality_player_season.parquet", "centrality_player_season.parquet"),
    ("networks/centrality_teams.parquet", "centrality_teams.parquet"),
    ("models/role_embeddings.parquet", "role_embeddings.parquet"),
)

COPY_FILES: tuple[tuple[str, str], ...] = (("models/role_gnn_both.pt", "role_gnn_both.pt"),)


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # no commits yet, or git unavailable


def _subsample_windowed(paths: Paths, n_matches: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windowed networks for a handful of matches.

    The full windowed tables are ~17 MB across 24,320 networks -- too large to commit for
    what Module 3's page needs, which is one real window sequence to look at. Matches are
    taken from both seasons so the page can show either provider.
    """
    nodes = pd.read_parquet(paths.networks / "windowed_nodes.parquet")
    edges = pd.read_parquet(paths.networks / "windowed_edges.parquet")

    per_season = max(n_matches // 2, 1)
    keep: list[int] = []
    for _, group in nodes.groupby("season"):
        keep.extend(sorted(group["game_id"].unique())[:per_season])

    return (
        nodes[nodes["game_id"].isin(keep)].reset_index(drop=True),
        edges[edges["game_id"].isin(keep)].reset_index(drop=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windowed-matches", type=int, default=20)
    parser.add_argument("--clean", action="store_true", help="wipe demo_data/ first")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    paths = Paths.load()

    if args.clean and BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    (BUNDLE_DIR / "reports").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "source": str(paths.root),
        "tables": {},
        "files": {},
        "reports": [],
    }

    missing: list[str] = []

    for source_rel, dest_name in COPY_TABLES:
        source = paths.root / source_rel
        if not source.exists():
            missing.append(source_rel)
            continue
        frame = pd.read_parquet(source)
        dest = BUNDLE_DIR / dest_name
        frame.to_parquet(dest, index=False)
        manifest["tables"][dest_name] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "size_kb": round(dest.stat().st_size / 1e3, 1),
            "source": source_rel,
        }
        log.info("%-38s %7d rows  %8.1f KB", dest_name, len(frame), dest.stat().st_size / 1e3)

    for source_rel, dest_name in COPY_FILES:
        source = paths.root / source_rel
        if not source.exists():
            missing.append(source_rel)
            continue
        shutil.copy2(source, BUNDLE_DIR / dest_name)
        manifest["files"][dest_name] = {
            "size_kb": round((BUNDLE_DIR / dest_name).stat().st_size / 1e3, 1),
            "source": source_rel,
        }
        log.info("%-38s %18.1f KB", dest_name, (BUNDLE_DIR / dest_name).stat().st_size / 1e3)

    # Windowed sample, for Module 3's spec page.
    if (paths.networks / "windowed_nodes.parquet").exists():
        window_nodes, window_edges = _subsample_windowed(paths, args.windowed_matches)
        for name, frame in (
            ("windowed_sample_nodes.parquet", window_nodes),
            ("windowed_sample_edges.parquet", window_edges),
        ):
            dest = BUNDLE_DIR / name
            frame.to_parquet(dest, index=False)
            manifest["tables"][name] = {
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "size_kb": round(dest.stat().st_size / 1e3, 1),
                "source": "networks/windowed_*.parquet (subsampled)",
                "matches": int(frame["game_id"].nunique()),
            }
            log.info("%-38s %7d rows  %8.1f KB", name, len(frame), dest.stat().st_size / 1e3)
    else:
        missing.append("networks/windowed_nodes.parquet")

    # Report JSONs: small, and they carry every published metric.
    for report in sorted(paths.reports.glob("*.json")):
        if report.name.startswith("._"):  # exFAT sidecar
            continue
        shutil.copy2(report, BUNDLE_DIR / "reports" / report.name)
        manifest["reports"].append(report.name)

    if missing:
        log.warning(
            "missing %d source(s): %s -- run the earlier pipeline stages first",
            len(missing),
            missing,
        )
        manifest["missing_sources"] = missing

    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_mb = sum(f.stat().st_size for f in BUNDLE_DIR.rglob("*") if f.is_file()) / 1e6
    log.info("bundle: %d tables, %d reports, %.2f MB total",
             len(manifest["tables"]), len(manifest["reports"]), total_mb)
    print(f"\nwrote {BUNDLE_DIR} ({total_mb:.2f} MB)")
    if missing:
        print(f"WARNING: {len(missing)} source(s) missing -- bundle is incomplete")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
