"""Loader for the demo bundle.

Resolves demo data from one of two places, in order:

1. `demo_data/` in the repo -- the ~8 MB committed bundle. Works on any checkout with no
   external drive, which is what makes the app shareable.
2. `DATA_ROOT` -- the full store, when the drive happens to be mounted.

Deliberately free of any `streamlit` import so it can be unit-tested and reused by scripts.
Caching belongs in the app layer (`st.cache_data`), not here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tacticalgraph.config import REPO_ROOT, clean_glob

log = logging.getLogger(__name__)

BUNDLE_DIR = REPO_ROOT / "demo_data"

# Bundle file name -> path relative to DATA_ROOT, for the fallback path.
_FALLBACK = {
    "games.parquet": "spadl/games.parquet",
    "players.parquet": "spadl/players.parquet",
    "full_nodes.parquet": "networks/full_nodes.parquet",
    "full_edges.parquet": "networks/full_edges.parquet",
    "centrality_players.parquet": "networks/centrality_players.parquet",
    "centrality_player_season.parquet": "networks/centrality_player_season.parquet",
    "centrality_teams.parquet": "networks/centrality_teams.parquet",
    "role_embeddings.parquet": "models/role_embeddings.parquet",
    "role_gnn_both.pt": "models/role_gnn_both.pt",
    "windowed_sample_nodes.parquet": "networks/windowed_nodes.parquet",
    "windowed_sample_edges.parquet": "networks/windowed_edges.parquet",
    # Modules 3 and 4.
    "module3_test_predictions.parquet": "models/module3_test_predictions.parquet",
    "module4_chains_sample.parquet": "models/module4_chains.parquet",
    "chain_encoder.pt": "models/chain_encoder.pt",
}


class BundleMissingError(FileNotFoundError):
    """Neither the committed bundle nor DATA_ROOT could satisfy a request."""


@dataclass
class DemoBundle:
    """Everything the demo app reads, plus provenance."""

    source: str  # "bundle" | "data_root"
    root: Path
    manifest: dict = field(default_factory=dict)
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- resolution
    def path(self, name: str) -> Path:
        """Locate one bundle member, falling back to DATA_ROOT's layout."""
        direct = self.root / name
        if direct.exists():
            return direct

        if self.source == "data_root" and name in _FALLBACK:
            candidate = self.root / _FALLBACK[name]
            if candidate.exists():
                return candidate

        raise BundleMissingError(
            f"{name!r} not found under {self.root}. Run "
            "`python scripts/export_demo_bundle.py` to build the demo bundle "
            "(requires DATA_ROOT with the full pipeline output)."
        )

    def table(self, name: str) -> pd.DataFrame:
        """Read a parquet member, memoised per bundle instance."""
        if name not in self._cache:
            self._cache[name] = pd.read_parquet(self.path(name))
        return self._cache[name]

    def has(self, name: str) -> bool:
        try:
            self.path(name)
            return True
        except BundleMissingError:
            return False

    def reports(self) -> dict[str, dict]:
        """All report JSONs, keyed by filename stem."""
        directory = self.root / "reports" if self.source == "bundle" else self.root / "reports"
        if not directory.exists():
            return {}
        out = {}
        for file in clean_glob(directory, "*.json"):
            try:
                out[file.stem] = json.loads(file.read_text())
            except json.JSONDecodeError:
                log.warning("skipping unparseable report %s", file)
        return out

    # ---------------------------------------------------------------- convenience
    @property
    def games(self) -> pd.DataFrame:
        return self.table("games.parquet")

    @property
    def players(self) -> pd.DataFrame:
        return self.table("players.parquet")

    @property
    def nodes(self) -> pd.DataFrame:
        return self.table("full_nodes.parquet")

    @property
    def edges(self) -> pd.DataFrame:
        return self.table("full_edges.parquet")

    @property
    def centrality(self) -> pd.DataFrame:
        return self.table("centrality_players.parquet")

    @property
    def centrality_season(self) -> pd.DataFrame:
        return self.table("centrality_player_season.parquet")

    @property
    def centrality_teams(self) -> pd.DataFrame:
        return self.table("centrality_teams.parquet")

    @property
    def embeddings(self) -> pd.DataFrame:
        return self.table("role_embeddings.parquet")

    @property
    def outcome_predictions(self) -> pd.DataFrame:
        """Module 3 per-(match, checkpoint) probabilities on the test fold."""
        return self.table("module3_test_predictions.parquet")

    @property
    def chains(self) -> pd.DataFrame:
        """Module 4 possession chains -- a cluster-stratified sample, not the full table.

        Headline numbers come from the report JSONs, which are computed over all 109,912
        chains; this table exists so individual examples can be browsed.
        """
        return self.table("module4_chains_sample.parquet")

    def embedding_matrix(self) -> tuple[pd.DataFrame, "pd.DataFrame"]:
        """Split the embedding table into (identity columns, embedding columns)."""
        frame = self.embeddings
        dims = [c for c in frame.columns if c.startswith("e") and c[1:].isdigit()]
        identity = [c for c in ("game_id", "team_id", "season", "provider", "player_id") if c in frame]
        return frame[identity], frame[dims]

    def provenance(self) -> str:
        if self.source == "bundle" and self.manifest:
            generated = self.manifest.get("generated_at", "unknown")
            sha = self.manifest.get("git_sha") or "no-commit"
            return f"committed bundle · generated {generated} · {sha}"
        return f"live DATA_ROOT · {self.root}"


def load_bundle(prefer_bundle: bool = True) -> DemoBundle:
    """Load the demo bundle, or fall back to the full store.

    `prefer_bundle=True` (the default) is what makes the app deterministic and portable: it
    reads the committed 8 MB snapshot even when the external drive happens to be attached, so
    what a reviewer sees does not depend on local state.
    """
    manifest_path = BUNDLE_DIR / "manifest.json"
    if prefer_bundle and manifest_path.exists():
        return DemoBundle(
            source="bundle", root=BUNDLE_DIR, manifest=json.loads(manifest_path.read_text())
        )

    # Fall back to the full store. Imported lazily so a missing/unmounted DATA_ROOT does not
    # break the bundle path.
    try:
        from tacticalgraph.config import Paths

        paths = Paths.load()
        if paths.root.exists():
            return DemoBundle(source="data_root", root=paths.root)
    except RuntimeError as exc:
        log.debug("DATA_ROOT unavailable: %s", exc)

    if manifest_path.exists():
        return DemoBundle(
            source="bundle", root=BUNDLE_DIR, manifest=json.loads(manifest_path.read_text())
        )

    raise BundleMissingError(
        f"no demo data found. Expected either {BUNDLE_DIR} (run "
        "`python scripts/export_demo_bundle.py`) or a mounted DATA_ROOT with the pipeline "
        "output."
    )


def verify_manifest(bundle: DemoBundle) -> list[str]:
    """Check every table's row count against the manifest.

    Guards against a stale bundle being shipped after the pipeline changed -- the app shows
    these discrepancies rather than silently presenting outdated numbers.
    """
    problems: list[str] = []
    for name, expected in (bundle.manifest.get("tables") or {}).items():
        try:
            actual = len(bundle.table(name))
        except BundleMissingError:
            problems.append(f"{name}: listed in manifest but missing from bundle")
            continue
        if actual != expected.get("rows"):
            problems.append(
                f"{name}: manifest says {expected.get('rows')} rows, file has {actual}"
            )
    return problems
