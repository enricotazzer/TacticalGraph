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

# Duplicated from `models.outcome_gnn_transformer` on purpose: argparse needs the choices at
# module import time, and importing the model module here would pull in torch even for
# `--skip-gnn`, which exists precisely to avoid that. The GNN block asserts the two agree, so a
# drift fails loudly instead of silently offering a scheme the model does not implement.
CHECKPOINT_WEIGHT_SCHEMES: tuple[str, ...] = ("uniform", "linear", "b0_signal")


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
        "--batch-size", type=int, default=16,
        help="matches per optimiser step. 1 reproduces the original one-step-per-match "
             "behaviour, which is the 'before' side of the batching change.",
    )
    parser.add_argument(
        "--lr-grid", nargs="+", type=float, default=None,
        help="learning rates to sweep on val (default depends on --batch-size)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="suffix for the report filename, so before/after runs do not overwrite",
    )
    parser.add_argument(
        "--baseline-residual", choices=("none", "b1"), default="b1",
        help="'b1' adds a fitted, frozen B1's logits and zero-inits the head, so the model "
             "starts as B1 and learns a correction (B1 becomes the floor). 'none' reproduces "
             "the original parallel learned state_head, where the floor is chance.",
    )
    parser.add_argument(
        "--checkpoint-weights", nargs="+", default=None,
        choices=CHECKPOINT_WEIGHT_SCHEMES,
        help="per-checkpoint training weight schemes to sweep on validation "
             f"(default: all of {', '.join(CHECKPOINT_WEIGHT_SCHEMES)})",
    )
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
            CHECKPOINT_WEIGHT_SCHEMES as MODEL_SCHEMES,
        )
        from tacticalgraph.models.outcome_gnn_transformer import (
            SEQUENCE_STATE_FEATURES,
            WINDOW_NODE_FEATURES,
            build_window_features,
            checkpoint_weights,
            make_sequences,
            predict_proba,
            train_outcome_model,
        )

        assert tuple(MODEL_SCHEMES) == CHECKPOINT_WEIGHT_SCHEMES, (
            "checkpoint weight schemes drifted between this script and the model module: "
            f"{MODEL_SCHEMES} vs {CHECKPOINT_WEIGHT_SCHEMES}"
        )

        # ---------------------------------------------------------- frozen baseline logits
        #
        # B1 is already fitted -- `fit_ladder` above fitted it on the train fold and predicted on
        # every fold, row-aligned to `folds[name]`. Reusing that means the frozen baseline is
        # provably the same B1 the report compares against, and adds no new leakage surface.
        #
        # `log` of the probabilities, clipped: softmax(log p) == p, so a zero-initialised head
        # leaves the model reproducing B1 exactly. Unclipped, a zero probability would give -inf
        # and poison the first backward pass.
        base_columns: tuple[str, ...] | None = None
        residual = args.baseline_residual == "b1"
        if residual:
            base_columns = ("b1_logit_home", "b1_logit_draw", "b1_logit_away")
            for name, frame in folds.items():
                probabilities = np.clip(ladder["B1"][name], 1e-6, 1.0)
                frame[list(base_columns)] = np.log(probabilities)
            log.info("residual mode: head zero-initialised on top of frozen B1 logits")

        window_features = build_window_features(window_nodes, window_edges)
        train_sequences, scaler = make_sequences(
            folds["train"], window_features, window_edges, outcomes,
            base_logit_columns=base_columns,
        )
        val_sequences, _ = make_sequences(
            folds["val"], window_features, window_edges, outcomes, scaler=scaler,
            base_logit_columns=base_columns,
        )
        test_sequences, _ = make_sequences(
            folds["test"], window_features, window_edges, outcomes, scaler=scaler,
            base_logit_columns=base_columns,
        )
        log.info(
            "sequences: %d train / %d val / %d test matches",
            len(train_sequences), len(val_sequences), len(test_sequences),
        )

        # ---------------------------------------------------------- checkpoint weight schemes
        #
        # `b0_signal` needs B0's per-checkpoint log-loss on the *training* fold. Taking it from
        # the val or test fold would let the objective be shaped by data it is scored on.
        schemes = tuple(args.checkpoint_weights or CHECKPOINT_WEIGHT_SCHEMES)
        train_b0 = per_checkpoint(folds["train"], ladder["B0"]["train"])
        b0_train_loss = train_b0.sort_values("checkpoint_minute")["log_loss"].to_numpy()
        weight_schemes = {
            scheme: checkpoint_weights(scheme, b0_log_loss=b0_train_loss) for scheme in schemes
        }
        for scheme, weights in weight_schemes.items():
            log.info(
                "checkpoint weights %-9s first=%.3f last=%.3f (mean %.3f)",
                scheme, weights[0], weights[-1], weights.mean(),
            )

        # Capacity sweep chosen on the validation fold only -- the winner is picked by val
        # log-loss, never by test.
        #
        # This comment used to justify the range by asserting the corpus could not support a
        # large model. That was withdrawn: the sweep's preference for small models was an
        # artefact of training at batch size 1, and once batching worked the spread between
        # capacities collapsed to ~0.04 log-loss. The range stays because selecting capacity on
        # validation is basic fairness, not because the answer is known in advance.
        capacity_grid = [
            {"graph_out": 8, "d_model": 16, "n_heads": 2, "n_layers": 1, "dropout": 0.4},
            {"graph_out": 16, "d_model": 32, "n_heads": 2, "n_layers": 1, "dropout": 0.3},
            {"graph_out": 32, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.2},
        ]
        # Learning rate is swept jointly with capacity because it is not independent of
        # --batch-size: averaging the gradient over 16 matches instead of 1 shrinks every step,
        # so the rate that suited batch size 1 systematically underfits at batch size 16.
        # Selection is on the validation fold only, as for capacity.
        learning_rates = args.lr_grid or ([3e-4] if args.batch_size == 1 else [1e-3, 3e-3])
        from tacticalgraph.models.outcome_gnn_transformer import evaluate_loss

        sweep, best_config, best_model, best_val = [], None, None, float("inf")

        def try_config(capacity: dict, learning_rate: float, scheme: str) -> float:
            """Train one configuration, record it, and keep it if it wins on validation."""
            nonlocal best_val, best_config, best_model, gnn_history
            candidate, history = train_outcome_model(
                train_sequences,
                val_sequences,
                node_in_channels=len(WINDOW_NODE_FEATURES),
                state_in_channels=len(SEQUENCE_STATE_FEATURES),
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
                batch_size=args.batch_size,
                baseline_residual=residual,
                checkpoint_weights=weight_schemes[scheme],
                lr=learning_rate,
                **capacity,
            )
            # Scored on the *unweighted* validation loss whatever the scheme, so a reweighting
            # that helps only its own objective cannot win the selection.
            val_loss = evaluate_loss(candidate, val_sequences, device=args.device)
            config = {**capacity, "lr": learning_rate, "weights": scheme}
            # -1 means no epoch beat the untrained model. In residual mode that is the
            # informative case: the model fell back to B1 rather than improving on it.
            best_epoch = int(history["best_epoch"])
            sweep.append({
                **config,
                "batch_size": args.batch_size,
                "residual": args.baseline_residual,
                "params": sum(p.numel() for p in candidate.parameters()),
                "val_log_loss": round(val_loss, 4),
                "val_log_loss_at_init": round(float(history["val_loss_at_init"]), 4),
                "epochs_run": len(history["train_loss"]),
                "best_val_epoch": best_epoch,
            })
            log.info("config %s -> val log-loss %.4f (best epoch %d)",
                     config, val_loss, best_epoch)
            if val_loss < best_val:
                best_val, best_config, best_model, gnn_history = (
                    val_loss, config, candidate, history
                )
            return val_loss

        # Two stages rather than one full grid, and the reason is the validation fold's size.
        #
        # Capacity x learning rate x weighting is 3 x 2 x 3 = 18 configurations, all selected on
        # ~70 validation matches. That is enough selection pressure for the winner to be noise.
        # Stage 1 sweeps capacity and learning rate under the *control* weighting; stage 2 then
        # tries the other schemes only at the winning capacity/learning rate. 8 configurations
        # instead of 18.
        #
        # This is greedy and could miss a scheme that only pays off at some other capacity. It is
        # also biased *against* the schemes -- the capacity was chosen while they were switched
        # off -- and since the schemes are my own proposed change, that is the direction the bias
        # should point.
        control = schemes[0]
        with ResourceMonitor("gnn-transformer") as gnn_monitor:
            for capacity in capacity_grid:
                for learning_rate in learning_rates:
                    try_config(capacity, learning_rate, control)

            stage1_capacity = {k: best_config[k] for k in capacity_grid[0]}
            stage1_lr = best_config["lr"]
            for scheme in schemes[1:]:
                try_config(stage1_capacity, stage1_lr, scheme)
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

    # Paired against B0 (does anything beat the scoreline?) and against B1 (does anything beat
    # the best tabular model?).
    #
    # B1 is the reference that matters once the graph model runs as a residual on it: with B1's
    # frozen logits as the starting point, "worse than B0" stops being reachable by construction,
    # so a Δ-vs-B0 headline would flatter the model for free. Δ vs B1 is the question that
    # survives the change, and both are reported so the older runs remain comparable.
    paired_rows, paired_b1_rows = [], []
    for name, probabilities in predictions.items():
        if name != "B0":
            paired_rows.append({
                "model": name,
                **paired_difference(
                    test, probabilities, predictions["B0"], n_boot=args.n_boot, seed=args.seed
                ),
            })
        if name != "B1":
            paired_b1_rows.append({
                "model": name,
                **paired_difference(
                    test, probabilities, predictions["B1"], n_boot=args.n_boot, seed=args.seed
                ),
            })
    paired = pd.DataFrame(paired_rows)
    paired_b1 = pd.DataFrame(paired_b1_rows)
    print()
    print("## Paired log-loss difference vs B0 (negative = better than B0)")
    print(paired.to_string(index=False))
    print()
    print("## Paired log-loss difference vs B1 (negative = better than B1)")
    print(paired_b1.to_string(index=False))
    print()
    print(
        "A CI containing zero means the model is statistically indistinguishable from the "
        "reference on this corpus -- which is a result, not a failure to report."
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
        "paired_vs_b1": paired_b1.to_dict(orient="records"),
        "per_checkpoint": {k: v.to_dict(orient="records") for k, v in checkpoint_tables.items()},
        "reliability_best": reliability.to_dict(orient="records"),
        "b2_importance": importance.to_dict(orient="records"),
        "gnn_history": gnn_history,
        "gnn_capacity_sweep": sweep,
        "gnn_selected_capacity": best_config,
        # Which arm produced this report. Without it a reader cannot tell a residual-on-B1 run
        # from a parallel-state_head one, and the two are not comparable.
        "gnn_arm": {
            "baseline_residual": args.baseline_residual,
            "checkpoint_weight_schemes": list(schemes) if not args.skip_gnn else [],
            "batch_size": args.batch_size,
        },
        "resources": [r for r in (state_monitor.as_dict(), ladder_monitor.as_dict(), gnn_resources) if r],
    }
    suffix = f"_{args.tag}" if args.tag else ""
    destination = (
        paths.reports / f"module3_outcome_{args.split}_seed{args.seed}{suffix}.json"
    )
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
    # The corpus's primary split, not the literal "cross_season": that string does not exist on
    # a single-season corpus, so gating on it left the Premier League bundle with no
    # per-checkpoint predictions and a blank match-timeline chart.
    if args.split == CORPORA[args.corpus].split_kinds[0] and args.seed == 0:
        out = paths.models / "module3_test_predictions.parquet"
        tidy_predictions.to_parquet(out, index=False)
        log.info("wrote %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
