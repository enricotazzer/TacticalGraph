#!/usr/bin/env python
"""Is in-match result prediction near its ceiling, or is it data-limited?

    python scripts/estimate_ceiling.py
    python scripts/estimate_ceiling.py --fractions 0.125 0.25 0.5 1.0 --draws 5

Module 3's GNN+Transformer loses to B0 on both corpora, and the stated diagnosis is "not
enough independent training matches". That diagnosis is only worth believing if more data
would actually help *the baselines* -- so this script measures it directly, before any
further model work.

Two measurements, on the pooled 1,140-match corpus (Premier League 2015/16 + Serie A 2015/16
+ Serie A 2017/18):

1. **A learning curve.** Fit B0/B1/B2 on random subsets of the training matches (subsampled
   BY MATCH, never by row) and evaluate on a fixed held-out fold. If test log-loss has
   flattened by the time all the data is used, the task is near its ceiling and no
   architecture will recover much -- which turns "our GNN lost" into the much stronger claim
   "there is little left to win here".

2. **Pooled vs single-corpus training**, evaluated on each corpus's own test fold. If pooling
   three competition-seasons does not beat training on one, the extra data is not usable
   without domain adaptation, and that is worth knowing before collecting more of it.

Every fold boundary is a whole match, and every bootstrap resamples matches rather than rows,
because one match contributes 16 heavily correlated checkpoint rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tacticalgraph.config import CORPORA, Paths  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions, read_games, write_games  # noqa: E402
from tacticalgraph.eval.outcome_metrics import (  # noqa: E402
    bootstrap_by_match,
    class_prior_predictions,
    log_loss,
)
from tacticalgraph.eval.resources import ResourceMonitor, device_label  # noqa: E402
from tacticalgraph.eval.splits import reject_random_split, temporal_split  # noqa: E402
from tacticalgraph.features.match_state import (  # noqa: E402
    B0_FEATURES,
    B1_FEATURES,
    B2_FEATURES,
    backfill_wyscout_scores,
    build_state_table,
    match_outcomes,
)
from tacticalgraph.features.xthreat import fit_xthreat  # noqa: E402
from tacticalgraph.models.outcome_baselines import make_model  # noqa: E402

log = logging.getLogger("estimate_ceiling")

RUNGS = {"B0": B0_FEATURES, "B1": B1_FEATURES, "B2": B2_FEATURES}
DEFAULT_FRACTIONS = (0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)


def prepare_corpus(slug: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """State table + games for one corpus, with its fold assignment attached."""
    paths = Paths.load(slug).ensure()
    games = read_games(paths)
    if (
        "wyscout" in {s.provider for s in paths.spec.seasons}
        and games.loc[games["provider"] == "wyscout", "home_score"].isna().any()
    ):
        games = backfill_wyscout_scores(paths, games)
        write_games(paths, games)

    outcomes = match_outcomes(games)
    actions = read_actions(paths)

    # The primary split for this corpus: `matchweek` for a single season, `cross_season`
    # otherwise. Folds are reused verbatim so ceiling numbers sit on the same test matches the
    # Module 3 report used, and remain directly comparable to it.
    kind = paths.spec.split_kinds[0]
    split = temporal_split(games, kind=kind, corpus=slug)

    xt = fit_xthreat(actions, split.train)
    state = build_state_table(actions, outcomes, xt_values=xt)
    state["fold"] = split.assign(state["game_id"])
    state["corpus"] = slug
    reject_random_split(state["fold"], state["game_id"])

    log.info(
        "%s: %d rows, folds %s",
        slug,
        len(state),
        state["fold"].value_counts().to_dict(),
    )
    return state, games, kind


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rung: str,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Fit one rung and return its test log-loss plus its probabilities."""
    features = list(RUNGS[rung])
    model = make_model(rung, seed=seed)
    model.fit(train[features].to_numpy(), train["outcome_index"].to_numpy())

    raw = model.predict_proba(test[features].to_numpy())
    # Map class-ordered output onto the fixed 3-class layout: a subsample can be missing a
    # class entirely, and sklearn would then emit a 2-column matrix.
    probabilities = np.zeros((len(test), 3))
    for position, class_index in enumerate(model.classes_):
        probabilities[:, int(class_index)] = raw[:, position]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities, row_sums, out=np.full_like(probabilities, 1 / 3), where=row_sums > 0
    )
    return log_loss(test["outcome_index"].to_numpy(), probabilities), probabilities


def learning_curve(
    train: pd.DataFrame,
    test_sets: dict[str, pd.DataFrame],
    fractions: tuple[float, ...],
    draws: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Test log-loss against the number of training *matches*, per test fold.

    Subsampling is by match, not by row: dropping rows would leave every match partially
    represented and make 16 correlated checkpoints look like 16 independent samples.

    Scored on each corpus's test fold *separately* as well as pooled, because the pooled fold
    is not balanced -- Serie A's cross-season test is its whole 2017/18 season (380 matches)
    against the Premier League's 50, so a single pooled curve would mostly be measuring
    cross-provider transfer and would be read as a general data-scaling result.
    """
    match_ids = np.array(sorted(train["game_id"].unique()))
    rows = []
    for fraction in fractions:
        n_matches = min(max(int(round(fraction * len(match_ids))), 20), len(match_ids))
        for draw in range(draws if n_matches < len(match_ids) else 1):
            rng = np.random.default_rng(seed + draw)
            picked = rng.choice(match_ids, n_matches, replace=False)
            subset = train[train["game_id"].isin(picked)]
            for rung in RUNGS:
                for test_name, test in test_sets.items():
                    try:
                        score, _ = fit_and_score(subset, test, rung, seed=seed)
                    except Exception as exc:  # noqa: BLE001 - a degenerate subsample must not stop the sweep
                        log.warning(
                            "%s/%s at n=%d draw=%d failed: %s",
                            rung, test_name, n_matches, draw, exc,
                        )
                        continue
                    rows.append(
                        {
                            "rung": rung,
                            "test_fold": test_name,
                            "n_train_matches": n_matches,
                            "fraction": round(fraction, 4),
                            "draw": draw,
                            "test_log_loss": round(score, 5),
                        }
                    )
        log.info("learning curve: n_train_matches=%d done", n_matches)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora", nargs="+", default=sorted(CORPORA),
        help="corpora to pool (default: all)",
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument(
        "--draws", type=int, default=3,
        help="random match subsets per fraction, to separate trend from sampling noise",
    )
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    log.info("device: %s", device_label())

    with ResourceMonitor("ceiling-state-tables") as state_monitor:
        per_corpus = {}
        for slug in args.corpora:
            state, _, kind = prepare_corpus(slug)
            per_corpus[slug] = {"state": state, "split_kind": kind}

    pooled = pd.concat([v["state"] for v in per_corpus.values()], ignore_index=True)
    log.info(
        "pooled: %d rows / %d matches across %d corpora",
        len(pooled),
        pooled["game_id"].nunique(),
        len(per_corpus),
    )

    report: dict[str, object] = {
        "corpora": args.corpora,
        "seed": args.seed,
        "pooled_matches": int(pooled["game_id"].nunique()),
        "split_kinds": {k: v["split_kind"] for k, v in per_corpus.items()},
    }

    # ------------------------------------------------------------------ learning curve
    print()
    print("=" * 78)
    print("LEARNING CURVE — does test log-loss still fall as training matches are added?")
    print("=" * 78)

    pooled_train = pooled[pooled["fold"] == "train"]
    pooled_test = pooled[pooled["fold"] == "test"]

    test_sets = {"pooled": pooled_test}
    for slug, payload in per_corpus.items():
        own_test = payload["state"][payload["state"]["fold"] == "test"]
        if not own_test.empty:
            test_sets[slug] = own_test
    log.info(
        "test folds: %s",
        {name: int(frame["game_id"].nunique()) for name, frame in test_sets.items()},
    )

    with ResourceMonitor("ceiling-learning-curve") as curve_monitor:
        curve = learning_curve(
            pooled_train, test_sets, tuple(args.fractions), args.draws, seed=args.seed
        )

    if not curve.empty:
        summary = (
            curve.groupby(["rung", "test_fold", "n_train_matches"])["test_log_loss"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["test_fold", "rung", "n_train_matches"])
        )
        summary["std"] = summary["std"].fillna(0.0)
        print(summary.round(4).to_string(index=False))
        report["learning_curve"] = curve.to_dict(orient="records")
        report["learning_curve_summary"] = summary.round(5).to_dict(orient="records")

        print()
        print("Marginal gain from the last doubling of data (negative = still improving):")
        verdicts = []
        for (rung, test_fold), group in summary.groupby(["rung", "test_fold"]):
            group = group.sort_values("n_train_matches")
            if len(group) < 2:
                continue
            largest = group.iloc[-1]
            half = group.iloc[
                (group["n_train_matches"] - largest["n_train_matches"] / 2).abs().argmin()
            ]
            delta = float(largest["mean"] - half["mean"])
            # "Noise" here is the spread across random match subsets at the same size. A gain
            # smaller than that spread is not evidence of anything.
            noise = float(max(largest["std"], half["std"]))
            plateaued = abs(delta) <= noise
            verdicts.append(
                {
                    "rung": rung,
                    "test_fold": test_fold,
                    "n_half": int(half["n_train_matches"]),
                    "n_full": int(largest["n_train_matches"]),
                    "log_loss_half": round(float(half["mean"]), 5),
                    "log_loss_full": round(float(largest["mean"]), 5),
                    "delta": round(delta, 5),
                    "subsample_noise": round(noise, 5),
                    "plateaued": bool(plateaued),
                }
            )
            print(
                f"  {rung:3s} on {test_fold:15s} {half['n_train_matches']:4.0f} -> "
                f"{largest['n_train_matches']:4.0f} matches  {half['mean']:.4f} -> "
                f"{largest['mean']:.4f}  delta {delta:+.4f} (subsample noise {noise:.4f})  "
                f"{'PLATEAUED' if plateaued else 'STILL IMPROVING'}"
            )
        report["plateau_verdict"] = verdicts

    # ------------------------------------------------- pooled vs single-corpus training
    print()
    print("=" * 78)
    print("POOLED vs SINGLE-CORPUS TRAINING, on each corpus's own test fold")
    print("=" * 78)

    transfer = []
    for slug, payload in per_corpus.items():
        state = payload["state"]
        own_train = state[state["fold"] == "train"]
        own_test = state[state["fold"] == "test"]
        if own_test.empty:
            continue
        # Pooled training is the union of every corpus's *train* fold. That is already safe:
        # each corpus assigns its own matches, so no corpus's test or val matches can appear
        # here. Asserted rather than assumed, because getting it wrong would leak silently.
        pooled_train_safe = pooled[pooled["fold"] == "train"]
        leaked = set(pooled_train_safe["game_id"]) & set(own_test["game_id"])
        if leaked:
            raise AssertionError(
                f"{len(leaked)} test match(es) of {slug} present in the pooled training set"
            )
        for rung in RUNGS:
            own_score, own_probabilities = fit_and_score(own_train, own_test, rung, args.seed)
            pooled_score, pooled_probabilities = fit_and_score(
                pooled_train_safe, own_test, rung, args.seed
            )
            _, own_low, own_high = bootstrap_by_match(
                own_test["outcome_index"].to_numpy(), own_probabilities,
                own_test["game_id"].to_numpy(), n_boot=args.n_boot, seed=args.seed,
            )
            _, pooled_low, pooled_high = bootstrap_by_match(
                own_test["outcome_index"].to_numpy(), pooled_probabilities,
                own_test["game_id"].to_numpy(), n_boot=args.n_boot, seed=args.seed,
            )
            transfer.append(
                {
                    "corpus": slug,
                    "rung": rung,
                    "n_train_own": int(own_train["game_id"].nunique()),
                    "n_train_pooled": int(pooled_train_safe["game_id"].nunique()),
                    "log_loss_own": round(own_score, 5),
                    "ci_own": [round(own_low, 4), round(own_high, 4)],
                    "log_loss_pooled": round(pooled_score, 5),
                    "ci_pooled": [round(pooled_low, 4), round(pooled_high, 4)],
                    "delta_pooled_minus_own": round(pooled_score - own_score, 5),
                }
            )

    if transfer:
        frame = pd.DataFrame(transfer)
        print(
            frame[
                ["corpus", "rung", "n_train_own", "n_train_pooled", "log_loss_own",
                 "log_loss_pooled", "delta_pooled_minus_own"]
            ].to_string(index=False)
        )
        print("\nNegative delta = pooling helped. Positive = the extra corpora hurt.")
        report["pooled_vs_single"] = transfer

    # ------------------------------------------------------------------ prior reference
    prior_probabilities = class_prior_predictions(
        pooled_train["outcome_index"].to_numpy(), len(pooled_test)
    )
    prior_score = log_loss(pooled_test["outcome_index"].to_numpy(), prior_probabilities)
    report["pooled_prior_log_loss"] = round(prior_score, 5)
    print(f"\nPooled class-prior floor: {prior_score:.4f} log-loss")

    report["resources"] = [state_monitor.as_dict(), curve_monitor.as_dict()]
    print()
    for entry in report["resources"]:
        print(
            f"resource: {entry['name']:26s} {entry['seconds']:7.1f}s "
            f"peak_rss={entry['peak_rss_mb']:.0f}MB device={entry['device']}"
        )

    destination = Paths.load(args.corpora[0]).reports / "module3_ceiling.json"
    destination.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
