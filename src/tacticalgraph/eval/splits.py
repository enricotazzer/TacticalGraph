"""Temporal splits, centrally enforced.

Every predictive model in this project must be split chronologically. A random split over
football event data leaks badly: the same match appears as many rows (players, windows,
possessions), so a random partition puts the *same match* on both sides and inflates every
metric. That failure is invisible in the numbers -- they simply look good.

So splitting is not left to call sites. `temporal_split` is the only sanctioned way to
partition, and `assert_no_overlap` is a cheap assertion that call sites can use to prove
they did not reintroduce leakage downstream.

The canonical design for Modules 2-3:

    train  Serie A 2015/16, matchweeks 1-30      (StatsBomb)
    val    Serie A 2015/16, matchweeks 31-38     (StatsBomb)
    test   Serie A 2017/18, all matchweeks       (Wyscout)

plus a `within_season` control that stays inside 2015/16, so a drop on the real test set can
be attributed to season-vs-provider shift rather than to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tacticalgraph.config import SERIE_A_STATSBOMB, SERIE_A_WYSCOUT

log = logging.getLogger(__name__)

TRAIN_MATCHWEEK_MAX = 30  # of 38


class LeakageError(AssertionError):
    """Raised when a split would place the same match on both sides."""


@dataclass(frozen=True)
class Split:
    """Game ids per fold, plus a human-readable description."""

    name: str
    train: set[int]
    val: set[int]
    test: set[int]
    description: str

    def assign(self, game_ids: pd.Series) -> pd.Series:
        """Label each row train/val/test/unused by its game id."""
        def _fold(game_id: int) -> str:
            if game_id in self.train:
                return "train"
            if game_id in self.val:
                return "val"
            if game_id in self.test:
                return "test"
            return "unused"

        return game_ids.astype("int64").map(_fold)

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.train)} train / {len(self.val)} val / "
            f"{len(self.test)} test games -- {self.description}"
        )


def _matchweek_column(games: pd.DataFrame) -> str:
    for candidate in ("game_day", "match_week", "matchweek"):
        if candidate in games.columns:
            return candidate
    raise KeyError(
        f"no matchweek column in games frame (have {list(games.columns)}); "
        "a chronological split cannot be built without one"
    )


def temporal_split(
    games: pd.DataFrame, kind: str = "cross_season", train_max_week: int = TRAIN_MATCHWEEK_MAX
) -> Split:
    """Build a chronological split.

    kind="cross_season"  -- train/val on 2015/16, test on 2017/18. The headline split.
    kind="within_season" -- train/val/test all inside 2015/16, single provider. The control
                            that isolates model quality from the provider shift.
    """
    week = _matchweek_column(games)
    games = games.copy()
    games[week] = pd.to_numeric(games[week], errors="coerce")

    left = games[games["season"] == SERIE_A_STATSBOMB.key]
    right = games[games["season"] == SERIE_A_WYSCOUT.key]

    if kind == "cross_season":
        train = set(left.loc[left[week] <= train_max_week, "game_id"].astype(int))
        val = set(left.loc[left[week] > train_max_week, "game_id"].astype(int))
        test = set(right["game_id"].astype(int))
        description = (
            f"train=2015/16 wk1-{train_max_week}, val=2015/16 wk{train_max_week + 1}-38, "
            "test=2017/18 (all) -- CONFOUNDED: season change coincides with provider change"
        )
    elif kind == "within_season":
        # 26 / 6 / 6 weeks: keeps the test fold strictly later in time than train.
        train = set(left.loc[left[week] <= 26, "game_id"].astype(int))
        val = set(left.loc[left[week].between(27, 32), "game_id"].astype(int))
        test = set(left.loc[left[week] >= 33, "game_id"].astype(int))
        description = (
            "train=2015/16 wk1-26, val=wk27-32, test=wk33-38 -- single provider, "
            "unconfounded control"
        )
    else:
        raise ValueError(f"unknown split kind {kind!r}")

    split = Split(name=kind, train=train, val=val, test=test, description=description)
    assert_no_overlap(split)
    log.info(split.summary())
    return split


def assert_no_overlap(split: Split) -> None:
    """Verify the three folds are disjoint at the match level."""
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = getattr(split, a) & getattr(split, b)
        if shared:
            raise LeakageError(
                f"{split.name}: {len(shared)} game(s) in both {a} and {b}, e.g. "
                f"{sorted(shared)[:5]}"
            )


def reject_random_split(fold: pd.Series, game_ids: pd.Series) -> None:
    """Assert that a fold assignment does not split any single match across folds.

    The guard Phase 3 (and later Modules) call to prove no accidental random split slipped
    in: a match must belong to exactly one fold, whatever the row granularity.
    """
    frame = pd.DataFrame({"fold": fold.to_numpy(), "game_id": game_ids.to_numpy()})
    frame = frame[frame["fold"] != "unused"]
    per_game = frame.groupby("game_id")["fold"].nunique()
    offenders = per_game[per_game > 1]
    if not offenders.empty:
        raise LeakageError(
            f"{len(offenders)} match(es) appear in more than one fold "
            f"(e.g. {list(offenders.index[:5])}). This is the signature of a random split "
            "over per-player or per-window rows; use temporal_split instead."
        )


def stratified_report(frame: pd.DataFrame, fold_column: str = "fold") -> pd.DataFrame:
    """Rows, matches and label balance per fold -- printed before every training run."""
    rows = []
    for fold, group in frame.groupby(fold_column):
        entry: dict[str, object] = {
            "fold": fold,
            "rows": len(group),
            "games": group["game_id"].nunique() if "game_id" in group else np.nan,
        }
        if "coarse_role" in group:
            counts = group["coarse_role"].value_counts(normalize=True)
            for role in ("GK", "DEF", "MID", "FWD"):
                entry[f"pct_{role}"] = round(100 * counts.get(role, 0.0), 1)
        rows.append(entry)
    return pd.DataFrame(rows)
