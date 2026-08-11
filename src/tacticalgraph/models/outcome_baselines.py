"""The baseline ladder Module 3's graph model has to beat.

The rungs are nested (B0 ⊂ B1 ⊂ B2), so each comparison isolates exactly what the added
features buy.

**B0 is the rung that matters.** In-match result prediction is dominated by the current
scoreline and the time left; a model that beats "possession and shot counts" but not "the
score" has demonstrated nothing. Reporting B1/B2 without B0 would be the easy way to claim a
win, so B0 is mandatory and so is the `prior` floor beneath it.

`HistGradientBoostingClassifier` rather than LightGBM: competitive on tabular data, already
present, and adds no dependency to an environment pinned at `numpy<2` / `pandas<3`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tacticalgraph.eval.outcome_metrics import N_CLASSES
from tacticalgraph.features.match_state import FEATURE_LADDER

log = logging.getLogger(__name__)


def make_model(rung: str, seed: int = 0):
    """Instantiate the estimator for one rung of the ladder."""
    if rung in ("B0", "B1"):
        # Scaled logistic regression: the relationship between goal difference, time left and
        # outcome is close to linear in the log-odds, and a linear model cannot manufacture an
        # advantage the features do not contain.
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    # Multinomial is the default in modern scikit-learn; the explicit
                    # `multi_class` argument was removed in 1.9.
                    "model",
                    LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
                ),
            ]
        )
    if rung == "B2":
        # `early_stopping=False` is deliberate. sklearn's internal validation split is drawn
        # at *row* level, which puts the same match on both sides (16 checkpoint rows per
        # match) and fools the stopping criterion into training far too long. Capacity is
        # instead chosen on the project's match-disjoint validation fold -- see
        # `_select_max_iter`.
        return HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"unknown rung {rung!r}; expected one of {sorted(FEATURE_LADDER)}")


# Capacity grid for B2, searched on the match-disjoint validation fold.
MAX_ITER_GRID: tuple[int, ...] = (25, 50, 100, 200)


def _select_max_iter(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    label_column: str,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    """Pick B2's boosting rounds by validation log-loss on held-out *matches*.

    With 300 training matches and 24 features the model overfits quickly, and the overfitting
    is invisible to any row-level split.
    """
    from sklearn.metrics import log_loss as sklearn_log_loss

    X_train = train[columns].to_numpy(dtype=np.float64)
    y_train = train[label_column].to_numpy()
    X_val = validation[columns].to_numpy(dtype=np.float64)
    y_val = validation[label_column].to_numpy()

    trace, best_iter, best_loss = [], MAX_ITER_GRID[0], float("inf")
    for max_iter in MAX_ITER_GRID:
        model = make_model("B2", seed=seed)
        model.set_params(max_iter=max_iter)
        model.fit(X_train, y_train)
        probabilities = _align(model.predict_proba(X_val), model.classes_)
        loss = float(sklearn_log_loss(y_val, probabilities, labels=list(range(N_CLASSES))))
        trace.append({"max_iter": max_iter, "val_log_loss": round(loss, 4)})
        if loss < best_loss:
            best_loss, best_iter = loss, max_iter

    log.info("B2 capacity selected on val: max_iter=%d (val log-loss %.4f) | %s",
             best_iter, best_loss, trace)
    return best_iter, trace


def _align(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map an estimator's class-ordered output onto the fixed 3-class layout.

    sklearn orders columns by the classes it actually saw. If a fold happened to miss a class
    the columns would silently shift, so the mapping is made explicit.
    """
    aligned = np.zeros((len(probabilities), N_CLASSES), dtype=np.float64)
    for position, label in enumerate(classes):
        aligned[:, int(label)] = probabilities[:, position]
    return aligned


def fit_ladder(
    train: pd.DataFrame,
    folds: dict[str, pd.DataFrame],
    label_column: str = "outcome_index",
    seed: int = 0,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    """Fit every rung on `train` and predict on each fold.

    Returns ({rung: {fold_name: probabilities}}, diagnostics), including a "prior" entry that
    acts as the floor every rung must clear.
    """
    y_train = train[label_column].to_numpy()
    results: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, object] = {}

    counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
    prior = counts / counts.sum()
    results["prior"] = {
        name: np.tile(prior, (len(frame), 1)) for name, frame in folds.items()
    }
    log.info("class prior from train: %s", np.round(prior, 4).tolist())

    for rung, feature_names in FEATURE_LADDER.items():
        columns = list(feature_names)
        model = make_model(rung, seed=seed)

        if rung == "B2" and "val" in folds and not folds["val"].empty:
            best_iter, trace = _select_max_iter(
                train, folds["val"], columns, label_column, seed
            )
            model.set_params(max_iter=best_iter)
            diagnostics["b2_capacity"] = {"selected_max_iter": best_iter, "trace": trace}

        model.fit(train[columns].to_numpy(dtype=np.float64), y_train)
        classes = getattr(model, "classes_", None)
        if classes is None:  # Pipeline exposes it on the final estimator
            classes = model[-1].classes_

        results[rung] = {
            name: _align(
                model.predict_proba(frame[columns].to_numpy(dtype=np.float64)), classes
            )
            for name, frame in folds.items()
        }
        log.info("fitted %s on %d rows / %d features", rung, len(train), len(columns))

    return results, diagnostics


def feature_importance(train: pd.DataFrame, label_column: str = "outcome_index", seed: int = 0) -> pd.DataFrame:
    """Permutation importance for B2, to show which features carry the signal.

    Useful mostly as a sanity check: if anything outranks goal difference late in a match,
    something is wrong with the feature layer.
    """
    from sklearn.inspection import permutation_importance

    columns = list(FEATURE_LADDER["B2"])
    model = make_model("B2", seed=seed)
    X = train[columns].to_numpy(dtype=np.float64)
    y = train[label_column].to_numpy()
    model.fit(X, y)

    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=seed, scoring="neg_log_loss"
    )
    return (
        pd.DataFrame(
            {
                "feature": columns,
                "importance": result.importances_mean.round(5),
                "std": result.importances_std.round(5),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
