#!/usr/bin/env python
"""Module 3 -- in-match result prediction: baseline ladder vs GNN+Transformer.

    python scripts/train_outcome.py --split cross_season
    python scripts/train_outcome.py --split within_season --seed 1
    python scripts/train_outcome.py --skip-gnn          # ladder only, fast

Writes DATA_ROOT/reports/module3_outcome_<split>_seed<n>.json plus a compact predictions
table for the demo app.

Reads honestly: **a graph model that does not beat B0 is reported as not beating B0.** With
300 independent training matches that is the likely outcome, so the paired bootstrap CI on
the log-loss difference is the headline, not the raw metric ordering.
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

from tacticalgraph.config import ALL_SPLIT_KINDS, CORPORA, DEFAULT_CORPUS, Paths  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions, read_games, write_games  # noqa: E402
from tacticalgraph.eval.outcome_metrics import (  # noqa: E402
    compare_models,
    paired_difference,
    per_checkpoint,
    reliability_curve,
)
from tacticalgraph.eval.resources import ResourceMonitor  # noqa: E402
from tacticalgraph.eval.splits import (  # noqa: E402
    reject_random_split,
    stratified_report,
    temporal_split,
)
from tacticalgraph.features.match_state import (  # noqa: E402
    backfill_wyscout_scores,
    build_state_table,
    derive_goals,
    match_outcomes,
)
from tacticalgraph.models.outcome_baselines import feature_importance, fit_ladder  # noqa: E402

log = logging.getLogger("train_outcome")


def prepare_games(paths: Paths) -> pd.DataFrame:
    """Load the match index, backfilling Wyscout scores if they are still missing."""
    games = read_games(paths)
    if games.loc[games["provider"] == "wyscout", "home_score"].isna().any():
        log.info("Wyscout scores missing; backfilling from the raw match file")
        games = backfill_wyscout_scores(paths, games)
        write_games(paths, games)
    return games


def validate_goal_derivation(actions: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, object]:
    """Assert the derived scoreline reproduces the recorded final score.

    Run on every invocation rather than only in tests: the running scoreline is B0's dominant
    feature, and a silent regression here would corrupt every number the script prints.
    """
    goals = derive_goals(actions)
    counts = goals.groupby(["game_id", "scoring_team_id"]).size()

    checked = mismatched = 0
    examples = []
    for row in outcomes.itertuples(index=False):
        home = int(counts.get((row.game_id, int(row.home_team_id)), 0))
        away = int(counts.get((row.game_id, int(row.away_team_id)), 0))
        checked += 1
        if (home, away) != (int(row.home_score), int(row.away_score)):
            mismatched += 1
            if len(examples) < 5:
                examples.append(
                    {"game_id": int(row.game_id), "derived": [home, away],
                     "recorded": [int(row.home_score), int(row.away_score)]}
                )

    rate = (checked - mismatched) / max(checked, 1)
    log.info("goal derivation matches the recorded score for %d/%d games (%.1f%%)",
             checked - mismatched, checked, 100 * rate)
    if examples:
        log.warning("scoreline mismatches, e.g. %s", examples)
    return {"games_checked": checked, "exact_match": checked - mismatched,
            "match_rate": round(rate, 4), "examples": examples}


def fit_xthreat(actions: pd.DataFrame, train_game_ids: set[int]) -> pd.Series:
    """Fit xThreat on the training games only and rate every action.

    Fitting on the whole corpus would let the 2017/18 test season shape the value surface that
    its own features are then built from.
    """
    from socceraction.xthreat import ExpectedThreat

    train_actions = actions[actions["game_id"].isin(train_game_ids)]
    model = ExpectedThreat(l=16, w=12)
    model.fit(train_actions)
    values = model.rate(actions)
    log.info(
        "xT fitted on %d train games; rated %d/%d actions (rest are non-move actions)",
        train_actions["game_id"].nunique(),
        int(np.sum(~np.isnan(values))),
        len(values),
    )
    return pd.Series(values, index=actions.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=ALL_SPLIT_KINDS, default=None,
        help="split kind; defaults to the corpus's primary kind",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
        help="which competition corpus to use (default: %(default)s)",
    )
    args = parser.parse_args()
    # Resolve after parsing so the default follows --corpus: "cross_season" is
    # meaningless for a single-season corpus.
    if args.split is None:
        args.split = CORPORA[args.corpus].split_kinds[0]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    paths = Paths.load(args.corpus).ensure()

    games = prepare_games(paths)
    outcomes = match_outcomes(games)
    log.info("outcomes: %d games | %s", len(outcomes),
             outcomes["outcome"].value_counts(normalize=True).round(3).to_dict())

    actions = read_actions(paths)
    goal_check = validate_goal_derivation(actions, outcomes)

    split = temporal_split(games, kind=args.split, corpus=args.corpus)
    xt = fit_xthreat(actions, split.train)

    window_nodes = pd.read_parquet(paths.networks / "windowed_nodes.parquet")
    window_edges = pd.read_parquet(paths.networks / "windowed_edges.parquet")

    with ResourceMonitor("state-table") as state_monitor:
        state = build_state_table(
            actions, outcomes, window_nodes=window_nodes, window_edges=window_edges, xt_values=xt
        )

    state["fold"] = split.assign(state["game_id"])
    reject_random_split(state["fold"], state["game_id"])
    print()
    print(stratified_report(state).to_string(index=False))

    folds = {name: state[state["fold"] == name].reset_index(drop=True)
             for name in ("train", "val", "test")}
    for name, frame in folds.items():
        if frame.empty:
            raise RuntimeError(f"fold {name!r} is empty; check the split definition")

    # ------------------------------------------------------------------ ladder
    with ResourceMonitor("ladder") as ladder_monitor:
        ladder, ladder_diagnostics = fit_ladder(folds["train"], folds, seed=args.seed)

    predictions = {rung: probabilities["test"] for rung, probabilities in ladder.items()}

    # ------------------------------------------------------------------ graph model
    gnn_history: dict[str, list[float]] = {}
    gnn_resources = None
    sweep: list[dict] = []
    best_config: dict | None = None
    if not args.skip_gnn:
        from tacticalgraph.models.outcome_gnn_transformer import (
            SEQUENCE_STATE_FEATURES,
            WINDOW_NODE_FEATURES,
            build_window_features,
            make_sequences,
            predict_proba,
            train_outcome_model,
        )

        window_features = build_window_features(window_nodes, window_edges)
        train_sequences, scaler = make_sequences(
            folds["train"], window_features, window_edges, outcomes
        )
        val_sequences, _ = make_sequences(
            folds["val"], window_features, window_edges, outcomes, scaler=scaler
        )
        test_sequences, _ = make_sequences(
            folds["test"], window_features, window_edges, outcomes, scaler=scaler
        )
        log.info(
            "sequences: %d train / %d val / %d test matches",
            len(train_sequences), len(val_sequences), len(test_sequences),
        )

        # Capacity sweep chosen on the validation fold only. 300 independent training labels
        # (the 16 checkpoints of a match are one observation) will not support a large model,
        # so trying a range is basic fairness rather than tuning -- and the winner is picked
        # by val log-loss, never by test.
        capacity_grid = [
            {"graph_out": 8, "d_model": 16, "n_heads": 2, "n_layers": 1, "dropout": 0.4},
            {"graph_out": 16, "d_model": 32, "n_heads": 2, "n_layers": 1, "dropout": 0.3},
            {"graph_out": 32, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.2},
        ]
        from tacticalgraph.models.outcome_gnn_transformer import evaluate_loss

        sweep, best_config, best_model, best_val = [], None, None, float("inf")
        with ResourceMonitor("gnn-transformer") as gnn_monitor:
            for config in capacity_grid:
                candidate, history = train_outcome_model(
                    train_sequences,
                    val_sequences,
                    node_in_channels=len(WINDOW_NODE_FEATURES),
                    state_in_channels=len(SEQUENCE_STATE_FEATURES),
                    epochs=args.epochs,
                    device=args.device,
                    seed=args.seed,
                    **config,
                )
                val_loss = evaluate_loss(candidate, val_sequences, device=args.device)
                n_params = sum(p.numel() for p in candidate.parameters())
                sweep.append({**config, "params": n_params, "val_log_loss": round(val_loss, 4),
                              "epochs_run": len(history["train_loss"])})
                log.info("capacity %s -> val log-loss %.4f (%d params)", config, val_loss, n_params)
                if val_loss < best_val:
                    best_val, best_config, best_model, gnn_history = (
                        val_loss, config, candidate, history
                    )
        gnn_resources = gnn_monitor.as_dict()
        model = best_model
        log.info("selected capacity %s (val log-loss %.4f)", best_config, best_val)
        print()
        print("## GNN capacity sweep (selected on validation, never on test)")
        print(pd.DataFrame(sweep).to_string(index=False))

        tidy = predict_proba(model, test_sequences, device=args.device)
        merged = folds["test"][["game_id", "window_index"]].merge(
            tidy, on=["game_id", "window_index"], how="left"
        )
        if merged[["p_home_win", "p_draw", "p_away_win"]].isna().any().any():
            raise RuntimeError("graph model produced no prediction for some test rows")
        predictions["gnn_transformer"] = merged[["p_home_win", "p_draw", "p_away_win"]].to_numpy()

    # ------------------------------------------------------------------ scoring
    test = folds["test"]
    comparison = compare_models(test, predictions, n_boot=args.n_boot, seed=args.seed)

    print()
    print("=" * 82)
    print(f"MODULE 3 -- test-set comparison ({args.split}, seed {args.seed})")
    print("log-loss lower is better; CI is a 95% bootstrap resampled BY MATCH")
    print("=" * 82)
    print(comparison.to_string(index=False))

    # The honest headline: is anything actually better than B0?
    print()
    print("## Paired log-loss difference vs B0 (negative = better than B0)")
    paired_rows = []
    for name, probabilities in predictions.items():
        if name == "B0":
            continue
        result = paired_difference(
            test, probabilities, predictions["B0"], n_boot=args.n_boot, seed=args.seed
        )
        paired_rows.append({"model": name, **result})
    paired = pd.DataFrame(paired_rows)
    print(paired.to_string(index=False))
    print()
    print(
        "A CI containing zero means the model is statistically indistinguishable from B0 on "
        "this corpus -- which is a result, not a failure to report."
    )

    checkpoint_tables = {}
    for name, probabilities in predictions.items():
        checkpoint_tables[name] = per_checkpoint(test, probabilities)
    print()
    print("## Log-loss by checkpoint (test)")
    curve = pd.DataFrame({"checkpoint": checkpoint_tables["B0"]["checkpoint_minute"]})
    for name, frame in checkpoint_tables.items():
        curve[name] = frame["log_loss"].to_numpy()
    print(curve.round(4).to_string(index=False))

    best = comparison.iloc[0]["model"]
    reliability = reliability_curve(
        test["outcome_index"].to_numpy(), predictions[best]
    )
    print()
    print(f"## Reliability of the best model ({best})")
    print(reliability.to_string(index=False))

    importance = feature_importance(folds["train"], seed=args.seed)
    print()
    print("## B2 permutation importance (top 8)")
    print(importance.head(8).to_string(index=False))

    # ------------------------------------------------------------------ persistence
    report = {
        "split": {"kind": split.name, "description": split.description},
        "seed": args.seed,
        "goal_derivation_check": goal_check,
        "outcome_balance": outcomes.groupby("season")["outcome"]
        .value_counts(normalize=True)
        .round(4)
        .unstack()
        .to_dict(orient="index"),
        "ladder_diagnostics": ladder_diagnostics,
        "comparison": comparison.to_dict(orient="records"),
        "paired_vs_b0": paired.to_dict(orient="records"),
        "per_checkpoint": {k: v.to_dict(orient="records") for k, v in checkpoint_tables.items()},
        "reliability_best": reliability.to_dict(orient="records"),
        "b2_importance": importance.to_dict(orient="records"),
        "gnn_history": gnn_history,
        "gnn_capacity_sweep": sweep,
        "gnn_selected_capacity": best_config,
        "resources": [r for r in (state_monitor.as_dict(), ladder_monitor.as_dict(), gnn_resources) if r],
    }
    destination = paths.reports / f"module3_outcome_{args.split}_seed{args.seed}.json"
    destination.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {destination}")

    # Compact predictions table for the demo app (best model + B0 for contrast).
    tidy_predictions = test[["game_id", "season", "provider", "window_index",
                             "checkpoint_minute", "outcome_index", "goal_diff"]].copy()
    for name in ("B0", "B2") + (("gnn_transformer",) if "gnn_transformer" in predictions else ()):
        probabilities = predictions[name]
        tidy_predictions[f"{name}_p_home"] = probabilities[:, 0]
        tidy_predictions[f"{name}_p_draw"] = probabilities[:, 1]
        tidy_predictions[f"{name}_p_away"] = probabilities[:, 2]
    if args.split == "cross_season" and args.seed == 0:
        out = paths.models / "module3_test_predictions.parquet"
        tidy_predictions.to_parquet(out, index=False)
        log.info("wrote %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
