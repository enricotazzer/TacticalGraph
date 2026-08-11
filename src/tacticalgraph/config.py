"""Path and dataset configuration.

All bulk data lives outside the repo, on an external drive, because the internal disk
has under 2 GB free. `DATA_ROOT` is read from the environment (or a `.env` file at the
repo root) and every path in the project is derived from it -- nothing is hardcoded.

The drive is removable, so `data_root()` fails loudly rather than silently creating an
empty tree at a mountpoint that is not actually mounted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# exFAT volumes (the T7) accumulate AppleDouble sidecar files: for every `spadl` dir
# macOS also writes `._spadl`. They are not parquet and must never reach a reader.
APPLEDOUBLE_PREFIX = "._"


def _load_dotenv() -> None:
    """Minimal .env loader. Avoids importing python-dotenv just for one variable."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def data_root() -> Path:
    """Resolve DATA_ROOT, verifying the external drive is actually mounted."""
    _load_dotenv()
    raw = os.environ.get("DATA_ROOT")
    if not raw:
        raise RuntimeError(
            "DATA_ROOT is not set. Copy .env.example to .env and point DATA_ROOT at a "
            "location with ~5 GB free (the internal disk does not have it)."
        )
    root = Path(raw).expanduser()

    # A path under /Volumes that does not exist almost always means "drive unplugged",
    # which is worth a clearer message than a downstream FileNotFoundError.
    if not root.exists():
        parents = list(root.parents)
        if len(parents) >= 2 and parents[-2] == Path("/Volumes"):
            volume = parents[-3] if len(parents) >= 3 else root
            if not volume.exists():
                raise RuntimeError(
                    f"DATA_ROOT={root} is on volume {volume}, which is not mounted. "
                    "Plug the drive in (or repoint DATA_ROOT) and retry."
                )
        root.mkdir(parents=True, exist_ok=True)
    return root


def clean_glob(directory: Path, pattern: str) -> list[Path]:
    """Glob that drops macOS AppleDouble sidecars written onto exFAT volumes."""
    return sorted(
        p for p in directory.glob(pattern) if not p.name.startswith(APPLEDOUBLE_PREFIX)
    )


@dataclass(frozen=True)
class Paths:
    """Derived layout under DATA_ROOT."""

    root: Path

    @classmethod
    def load(cls) -> "Paths":
        return cls(root=data_root())

    @property
    def raw_statsbomb(self) -> Path:
        return self.root / "raw" / "statsbomb"

    @property
    def raw_wyscout(self) -> Path:
        return self.root / "raw" / "wyscout"

    @property
    def spadl(self) -> Path:
        return self.root / "spadl"

    @property
    def enrichment(self) -> Path:
        """StatsBomb-only fields, quarantined: validation use only, never model input."""
        return self.root / "enrichment"

    @property
    def networks(self) -> Path:
        return self.root / "networks"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "Paths":
        for path in (
            self.raw_statsbomb,
            self.raw_wyscout,
            self.spadl,
            self.enrichment,
            self.networks,
            self.models,
            self.figures,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------------------
# Dataset definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonSpec:
    """One competition-season, and where it comes from."""

    key: str  # partition value, e.g. "2015-2016"
    provider: str  # "statsbomb" | "wyscout"
    label: str
    n_matches_expected: int


# Serie A is the target competition. StatsBomb open data contains exactly one usable
# Serie A season (2015/16; the 1986/87 entry has a single match), so the second season
# comes from the Wyscout public dataset. This is the whole reason a harmonization layer
# exists in this project -- see README.
SERIE_A_STATSBOMB = SeasonSpec(
    key="2015-2016",
    provider="statsbomb",
    label="Serie A 2015/2016",
    n_matches_expected=380,
)
SERIE_A_WYSCOUT = SeasonSpec(
    key="2017-2018",
    provider="wyscout",
    label="Serie A 2017/2018",
    n_matches_expected=380,
)
SEASONS: tuple[SeasonSpec, ...] = (SERIE_A_STATSBOMB, SERIE_A_WYSCOUT)

# StatsBomb open-data identifiers for Serie A 2015/16.
STATSBOMB_COMPETITION_ID = 12
STATSBOMB_SEASON_ID = 27
STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Wyscout public dataset (Pappalardo et al. 2019), figshare collection 4415000.
WYSCOUT_COLLECTION_URL = "https://api.figshare.com/v2/collections/4415000/articles"
WYSCOUT_COUNTRY = "Italy"  # selects Serie A 2017/18 out of the 7 bundled competitions

# Pitch dimensions SPADL normalises to (metres).
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
