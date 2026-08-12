#!/usr/bin/env python
"""Harmonisation report -- the gate between Phase 1 and Phase 2.

The project trains on Serie A 2015/16 (StatsBomb) and tests on Serie A 2017/18 (Wyscout).
That is the strongest available anti-leakage split, but it confounds the season effect with
a provider effect. This script exists so the confound is measured and published rather than
hidden. It answers four questions:

1. How good is the inferred pass recipient? Measured directly against StatsBomb ground
   truth, and again on an event stream degraded to Wyscout-like density -- the latter is the
   only honest estimate of accuracy on the season where no ground truth exists.
2. How good is the reconstructed possession chain? Scored against StatsBomb's native
   possession counter.
3. How far apart are the two seasons on the features models will actually consume?
   Reported as a two-sample KS statistic per feature.
4. Does the 24 -> 4 role collapse cover everything present in the data?

    python scripts/validate_harmonization.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from tacticalgraph.config import CORPORA, DEFAULT_CORPUS, Paths  # noqa: E402, SERIE_A_STATSBOMB, SERIE_A_WYSCOUT  # noqa: E402
from tacticalgraph.data.enrichment import load_enrichment  # noqa: E402
from tacticalgraph.data.possession import (  # noqa: E402
    evaluate_possessions,
    reconstruct_possessions,
)
from tacticalgraph.data.recipient import (  # noqa: E402
    PASS_LIKE_TYPES,
    evaluate_inference,
    infer_recipients,
)
from tacticalgraph.data.roles import verify_mapping_coverage  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions  # noqa: E402

log = logging.getLogger("harmonization")

# Action types SPADL derives from provider-specific "carry"-like events. StatsBomb logs
# carries explicitly and Wyscout does not, so dribble frequency differs by ~12x between
# providers. Removing them is how we simulate Wyscout density on StatsBomb data.
DENSITY_ASYMMETRIC_TYPES = ("dribble",)


def basic_counts(actions: pd.DataFrame) -> pd.DataFrame:
    """Per-provider volumes: the first-order description of the provider shift."""
    rows = []
    for (season, provider), frame in actions.groupby(["season", "provider"]):
        n_games = frame["game_id"].nunique()
        is_pass = frame["type_name"].isin(PASS_LIKE_TYPES)
        passes = frame[is_pass]
        rows.append(
            {
                "season": season,
                "provider": provider,
                "games": n_games,
                "actions": len(frame),
                "actions_per_game": round(len(frame) / n_games, 1),
                "passes_per_game": round(len(passes) / n_games, 1),
                "pass_completion": round((passes["result_name"] == "success").mean(), 4),
                "dribbles_per_game": round(
                    (frame["type_name"] == "dribble").sum() / n_games, 1
                ),
            }
        )
    return pd.DataFrame(rows)


def recipient_resolution(actions: pd.DataFrame) -> pd.DataFrame:
    """Share of completed passes that got a recipient, per provider.

    This is a *coverage* number, available for both providers. Accuracy needs ground truth
    and is handled separately.
    """
    rows = []
    for (season, provider), frame in actions.groupby(["season", "provider"]):
        eligible = frame["type_name"].isin(PASS_LIKE_TYPES) & (
            frame["result_name"] == "success"
        )
        subset = frame[eligible]
        rows.append(
            {
                "season": season,
                "provider": provider,
                "completed_passes": int(len(subset)),
                "resolved": int(subset["recipient_confident"].sum()),
                "resolved_pct": round(float(subset["recipient_confident"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def recipient_accuracy_vs_truth(
    paths: Paths, actions: pd.DataFrame, degrade: bool = False
) -> dict[str, float]:
    """Score inferred recipients against StatsBomb ground truth.

    With `degrade=True`, carry-derived `dribble` actions are dropped before inference so the
    action stream resembles Wyscout's density. The resulting accuracy is our best estimate
    of how the same rule performs on the 2017/18 season, where truth does not exist.
    """
    truth = load_enrichment(paths, "statsbomb_event_truth")[
        ["original_event_id", "true_recipient_id"]
    ]

    frame = actions[actions["provider"] == "statsbomb"].copy()
    if degrade:
        frame = frame[~frame["type_name"].isin(DENSITY_ASYMMETRIC_TYPES)].copy()
        # Re-infer per game on the thinned stream; the rule inspects neighbouring actions,
        # so it must see the degraded ordering rather than the original one.
        frame = pd.concat(
            [infer_recipients(g) for _, g in frame.groupby("game_id", sort=False)],
            ignore_index=True,
        )

    merged = frame.merge(truth, on="original_event_id", how="left")
    result = evaluate_inference(
        merged,
        merged["true_recipient_id"].astype("Float64").astype("Int64"),
        context="statsbomb-degraded" if degrade else "statsbomb-native",
    )
    result["actions_per_game"] = round(len(frame) / frame["game_id"].nunique(), 1)
    return result


def possession_agreement(
    paths: Paths, actions: pd.DataFrame, n_games: int = 60
) -> pd.DataFrame:
    """Score reconstructed chains against StatsBomb's native possession counter."""
    truth = load_enrichment(paths, "statsbomb_event_truth")[
        ["original_event_id", "statsbomb_possession"]
    ]
    frame = actions[actions["provider"] == "statsbomb"]
    game_ids = sorted(frame["game_id"].unique())[:n_games]

    rows = []
    for game_id in game_ids:
        game = frame[frame["game_id"] == game_id].merge(
            truth, on="original_event_id", how="left"
        )
        game = game[game["statsbomb_possession"].notna()]
        if game.empty:
            continue
        rebuilt = reconstruct_possessions(game)
        rows.append(
            evaluate_possessions(
                rebuilt, rebuilt["statsbomb_possession"], context=str(game_id)
            )
        )
    return pd.DataFrame(rows)


def distribution_shift(actions: pd.DataFrame) -> pd.DataFrame:
    """Two-sample KS statistic per modelling feature, 2015/16 vs 2017/18.

    This is the number that makes the accepted confound honest: a large KS on a feature
    means any cross-season performance drop on that feature cannot be attributed to
    football alone.
    """
    left = actions[actions["season"] == SERIE_A_STATSBOMB.key]
    right = actions[actions["season"] == SERIE_A_WYSCOUT.key]

    passes_left = left[left["type_name"].isin(PASS_LIKE_TYPES)]
    passes_right = right[right["type_name"].isin(PASS_LIKE_TYPES)]

    features = {
        "start_x": (left["start_x"], right["start_x"]),
        "start_y": (left["start_y"], right["start_y"]),
        "pass_length": (
            np.hypot(
                passes_left["end_x"] - passes_left["start_x"],
                passes_left["end_y"] - passes_left["start_y"],
            ),
            np.hypot(
                passes_right["end_x"] - passes_right["start_x"],
                passes_right["end_y"] - passes_right["start_y"],
            ),
        ),
        "pass_dx": (
            passes_left["end_x"] - passes_left["start_x"],
            passes_right["end_x"] - passes_right["start_x"],
        ),
    }

    rows = []
    rng = np.random.default_rng(0)
    for name, (a, b) in features.items():
        a = pd.Series(a).dropna().to_numpy()
        b = pd.Series(b).dropna().to_numpy()
        # KS p-values are meaningless at n~750k (everything is "significant"); the
        # statistic is the effect size we care about. Subsample for tractability.
        cap = 200_000
        if len(a) > cap:
            a = rng.choice(a, cap, replace=False)
        if len(b) > cap:
            b = rng.choice(b, cap, replace=False)
        ks = stats.ks_2samp(a, b)
        rows.append(
            {
                "feature": name,
                "ks_statistic": round(float(ks.statistic), 4),
                "mean_2015_2016": round(float(a.mean()), 2),
                "mean_2017_2018": round(float(b.mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def action_mix(actions: pd.DataFrame) -> pd.DataFrame:
    """Per-game rate of each action type by season, with the ratio between them.

    Deliberately a *rate*, not a share. Shares are the wrong diagnostic here: StatsBomb
    logs 790 dribbles per game against Wyscout's 90, and that single inflation deflates
    every other type's share, making comparable types look divergent. Per-game rates
    isolate each type. This is the table that decides which action types may be counted in
    a model feature (see `schema.PROVIDER_COMPARABLE_TYPES`).
    """
    games = actions.groupby("season")["game_id"].nunique()
    counts = actions.groupby(["season", "type_name"]).size().unstack("season")
    rates = (counts / games).round(2)

    left, right = SERIE_A_STATSBOMB.key, SERIE_A_WYSCOUT.key
    rates = rates.rename(
        columns={left: "per_game_2015_2016", right: "per_game_2017_2018"}
    )
    rates["rate_ratio"] = (
        rates["per_game_2015_2016"] / rates["per_game_2017_2018"].replace(0.0, np.nan)
    ).round(2)
    rates["provider_comparable"] = rates["rate_ratio"].between(0.75, 1.33)
    return rates.sort_values("rate_ratio", ascending=False)


def role_coverage(paths: Paths) -> dict[str, object]:
    """Positions present in the data but missing from the 24 -> 4 mapping."""
    out: dict[str, object] = {}
    try:
        lineups = load_enrichment(paths, "statsbomb_lineup_positions")
        observed = set(lineups["position_name_24"].dropna().unique())
        out["statsbomb_positions_observed"] = len(observed)
        out["statsbomb_unmapped"] = verify_mapping_coverage(observed, "statsbomb")
    except FileNotFoundError:
        out["statsbomb_unmapped"] = ["<enrichment missing>"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--possession-games", type=int, default=60)
    parser.add_argument("--skip-degraded", action="store_true", help="skip the slow proxy")
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
        help="which competition corpus to use (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = Paths.load(args.corpus).ensure()
    actions = read_actions(paths)
    log.info(
        "loaded %d actions across %d games", len(actions), actions["game_id"].nunique()
    )

    report: dict[str, object] = {}
    sections: list[tuple[str, object]] = []

    counts = basic_counts(actions)
    sections.append(("Volumes per provider", counts))
    report["counts"] = counts.to_dict(orient="records")

    resolution = recipient_resolution(actions)
    sections.append(("Recipient resolution COVERAGE (both providers)", resolution))
    report["recipient_resolution"] = resolution.to_dict(orient="records")

    accuracy_rows = [recipient_accuracy_vs_truth(paths, actions, degrade=False)]
    if not args.skip_degraded:
        log.info("running degraded-density proxy (re-infers on thinned stream)...")
        accuracy_rows.append(recipient_accuracy_vs_truth(paths, actions, degrade=True))
    accuracy = pd.DataFrame(accuracy_rows)
    sections.append(("Recipient ACCURACY vs StatsBomb ground truth", accuracy))
    report["recipient_accuracy"] = accuracy.to_dict(orient="records")

    possession = possession_agreement(paths, actions, n_games=args.possession_games)
    if not possession.empty:
        summary = pd.DataFrame(
            [
                {
                    "games_scored": len(possession),
                    "boundary_jaccard_mean": round(possession["boundary_jaccard"].mean(), 4),
                    "adjusted_rand_mean": round(possession["adjusted_rand"].mean(), 4),
                    "chains_ours_mean": round(possession["n_chains_ours"].mean(), 1),
                    "chains_statsbomb_mean": round(
                        possession["n_chains_statsbomb"].mean(), 1
                    ),
                }
            ]
        )
        sections.append(("Possession reconstruction vs StatsBomb native", summary))
        report["possession"] = summary.to_dict(orient="records")

    shift = distribution_shift(actions)
    sections.append(("Distribution shift 2015/16 vs 2017/18 (KS statistic)", shift))
    report["distribution_shift"] = shift.to_dict(orient="records")

    mix = action_mix(actions)
    sections.append(("Per-game action rates by season (all types)", mix))
    report["action_mix"] = mix.reset_index().to_dict(orient="records")

    passlike = mix[mix.index.isin(PASS_LIKE_TYPES)]
    aggregate = pd.DataFrame(
        [
            {
                "group": "all pass-like types (passing-network edges)",
                "per_game_2015_2016": round(passlike["per_game_2015_2016"].sum(), 1),
                "per_game_2017_2018": round(passlike["per_game_2017_2018"].sum(), 1),
                "rate_ratio": round(
                    passlike["per_game_2015_2016"].sum()
                    / passlike["per_game_2017_2018"].sum(),
                    3,
                ),
            },
            {
                "group": "provider-comparable types only",
                "per_game_2015_2016": round(
                    mix.loc[mix["provider_comparable"], "per_game_2015_2016"].sum(), 1
                ),
                "per_game_2017_2018": round(
                    mix.loc[mix["provider_comparable"], "per_game_2017_2018"].sum(), 1
                ),
                "rate_ratio": round(
                    mix.loc[mix["provider_comparable"], "per_game_2015_2016"].sum()
                    / mix.loc[mix["provider_comparable"], "per_game_2017_2018"].sum(),
                    3,
                ),
            },
        ]
    )
    sections.append(
        ("Aggregate rates -- why passing networks survive the provider switch", aggregate)
    )
    report["aggregate_rates"] = aggregate.to_dict(orient="records")

    coverage = role_coverage(paths)
    sections.append(("Role mapping coverage", coverage))
    report["role_coverage"] = coverage

    print()
    print("=" * 78)
    print("HARMONISATION REPORT -- Serie A 2015/16 (StatsBomb) vs 2017/18 (Wyscout)")
    print("=" * 78)
    for title, payload in sections:
        print()
        print(f"## {title}")
        if isinstance(payload, pd.DataFrame):
            print(payload.to_string(index=payload.index.name is not None))
        else:
            print(json.dumps(payload, indent=2, default=str))

    dest = paths.reports / "harmonization_report.json"
    dest.write_text(json.dumps(report, indent=2, default=str))
    print()
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
