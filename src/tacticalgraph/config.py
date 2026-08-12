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

# Defined before `Paths` because it is a dataclass field default, evaluated at class
# creation. The registry it indexes (`CORPORA`) is built further down and only read inside
# methods, so it may be defined later.
DEFAULT_CORPUS = "serie_a"


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
    """Derived layout under DATA_ROOT, namespaced by corpus.

    Two competitions can share a provider and a season key -- Serie A 2015/16 and Premier
    League 2015/16 are both `statsbomb` / `2015-2016`. Partitioning only by season and
    provider would therefore collide, silently merging two competitions into one table.

    Rather than thread a `competition` column through every table, reader and split, each
    corpus gets its own subtree and every derived path hangs off it. `raw/` is deliberately
    *outside* that namespace: StatsBomb event files are keyed by globally unique match id, so
    two corpora can share one download cache instead of duplicating 1.4 GB.
    """

    root: Path
    corpus: str = DEFAULT_CORPUS

    @classmethod
    def load(cls, corpus: str = DEFAULT_CORPUS) -> "Paths":
        if corpus not in CORPORA:
            raise ValueError(
                f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}"
            )
        return cls(root=data_root(), corpus=corpus)

    @property
    def spec(self) -> "CorpusSpec":
        return CORPORA[self.corpus]

    @property
    def derived(self) -> Path:
        """Everything this project computes, isolated per corpus."""
        return self.root / "corpora" / self.corpus

    # -- shared across corpora (keyed by globally unique match id) --
    @property
    def raw_statsbomb(self) -> Path:
        return self.root / "raw" / "statsbomb"

    @property
    def raw_wyscout(self) -> Path:
        return self.root / "raw" / "wyscout"

    # -- per corpus --
    @property
    def spadl(self) -> Path:
        return self.derived / "spadl"

    @property
    def enrichment(self) -> Path:
        """StatsBomb-only fields, quarantined: validation use only, never model input."""
        return self.derived / "enrichment"

    @property
    def networks(self) -> Path:
        return self.derived / "networks"

    @property
    def models(self) -> Path:
        return self.derived / "models"

    @property
    def figures(self) -> Path:
        return self.derived / "figures"

    @property
    def reports(self) -> Path:
        return self.derived / "reports"

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

# Premier League 2015/16: a *complete* 380-match season from a single provider. This is the
# corpus the event-based modules (1-4) belong on, because it removes the confound that limits
# the Serie A results -- there, the season change and the provider change are the same event,
# so a drop on the test fold cannot be attributed to either. Here train/val/test are
# matchweeks of one season logged by one provider, so a drop is the model's fault.
#
# It has no 360 data (0 of 380 matches), which is why it cannot host the phase/formation work.
PREMIER_LEAGUE_2015 = SeasonSpec(
    key="2015-2016",
    provider="statsbomb",
    label="Premier League 2015/2016",
    n_matches_expected=380,
)

STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# StatsBomb open-data identifiers for Serie A 2015/16, kept as module constants for
# backwards compatibility with the original single-corpus scripts.
STATSBOMB_COMPETITION_ID = 12
STATSBOMB_SEASON_ID = 27


@dataclass(frozen=True)
class CorpusSpec:
    """A competition-level corpus: its seasons, its provenance, and how it may be split.

    `split_kinds` is authoritative. A single-season corpus has no cross-season split, and
    asking for one must fail loudly rather than silently return empty folds.
    """

    slug: str
    label: str
    seasons: tuple[SeasonSpec, ...]
    split_kinds: tuple[str, ...]
    # StatsBomb (competition_id, season_id) pairs to ingest; empty for non-StatsBomb sources.
    statsbomb_ids: tuple[tuple[int, int], ...] = ()
    uses_wyscout: bool = False
    has_360: bool = False
    notes: str = ""

    @property
    def n_matches_expected(self) -> int:
        return sum(s.n_matches_expected for s in self.seasons)


SERIE_A_CORPUS = CorpusSpec(
    slug="serie_a",
    label="Serie A 2015/16 + 2017/18",
    seasons=(SERIE_A_STATSBOMB, SERIE_A_WYSCOUT),
    split_kinds=("cross_season", "within_season"),
    statsbomb_ids=((12, 27),),
    uses_wyscout=True,
    has_360=False,
    notes=(
        "Two providers by necessity: StatsBomb open data has exactly one usable Serie A "
        "season. Retained as a cross-provider generalisation study."
    ),
)

PREMIER_LEAGUE_CORPUS = CorpusSpec(
    slug="premier_league",
    label="Premier League 2015/16",
    seasons=(PREMIER_LEAGUE_2015,),
    split_kinds=("matchweek",),
    statsbomb_ids=((2, 27),),
    uses_wyscout=False,
    has_360=False,
    notes=(
        "Complete 380-match single-provider season. No provider confound; split by "
        "matchweek. No 360 data on any match."
    ),
)

CORPORA: dict[str, CorpusSpec] = {
    SERIE_A_CORPUS.slug: SERIE_A_CORPUS,
    PREMIER_LEAGUE_CORPUS.slug: PREMIER_LEAGUE_CORPUS,
}

# Union of split kinds across all corpora, for `--split` argparse choices. The *valid* set
# for a given run is narrower and is enforced by `eval.splits.temporal_split` against the
# corpus spec -- argparse only rejects typos.
ALL_SPLIT_KINDS: tuple[str, ...] = tuple(
    dict.fromkeys(k for spec in CORPORA.values() for k in spec.split_kinds)
)

# Wyscout public dataset (Pappalardo et al. 2019), figshare collection 4415000.
WYSCOUT_COLLECTION_URL = "https://api.figshare.com/v2/collections/4415000/articles"
WYSCOUT_COUNTRY = "Italy"  # selects Serie A 2017/18 out of the 7 bundled competitions

# Pitch dimensions SPADL normalises to (metres).
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
