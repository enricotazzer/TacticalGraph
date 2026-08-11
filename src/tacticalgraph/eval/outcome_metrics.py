"""Scoring for Module 3's 3-class outcome predictions.

Two things here are deliberate and matter more than the metric formulas:

**Uncertainty is resampled by match, not by row.** Each match contributes 16 highly
correlated checkpoint rows. Bootstrapping rows would treat them as 16 independent
observations, shrinking the confidence intervals by roughly 4x and manufacturing
"significant" differences between models that are indistinguishable. With 300 training and
380 test *matches*, honest intervals are wide, and saying so is the point.

**Calibration is reported alongside accuracy.** For in-match probabilities a confidently
wrong forecast is worse than an uncertain one, so log-loss, Brier and ECE all appear.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_CLASSES = 3
EPS = 1e-15


def _clip(probabilities: np.ndarray) -> np.ndarray:
    """Normalise and clip so log-loss cannot see a zero."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    p = _clip(probabilities)
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot truth."""
    p = _clip(probabilities)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(np.argmax(probabilities, axis=1) == y_true))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """ECE on the top-1 predicted probability."""
    p = _clip(probabilities)
    confidence = p.max(axis=1)
    correct = (np.argmax(p, axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error, total = 0.0, len(y_true)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def reliability_curve(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Binned confidence vs observed accuracy, for a reliability diagram."""
    p = _clip(probabilities)
    confidence = p.max(axis=1)
    correct = (np.argmax(p, axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": round(float(low), 3),
                "bin_high": round(float(high), 3),
                "n": int(mask.sum()),
                "mean_confidence": round(float(confidence[mask].mean()), 4),
                "observed_accuracy": round(float(correct[mask].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


ALL_METRICS = {
    "log_loss": log_loss,
    "brier": brier_score,
    "accuracy": accuracy,
    "ece": expected_calibration_error,
}


def score(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {name: round(function(y_true, probabilities), 4) for name, function in ALL_METRICS.items()}


def bootstrap_by_match(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    game_ids: np.ndarray,
    metric: str = "log_loss",
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and 95% CI, resampling whole matches.

    Resampling rows instead would treat one match's 16 checkpoints as 16 independent
    observations and produce intervals roughly 4x too narrow.
    """
    function = ALL_METRICS[metric]
    point = function(y_true, probabilities)

    unique_games = np.unique(game_ids)
    index_by_game = {game: np.flatnonzero(game_ids == game) for game in unique_games}

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([index_by_game[game] for game in drawn])
        samples.append(function(y_true[rows], probabilities[rows]))

    low, high = np.percentile(samples, [2.5, 97.5])
    return float(point), float(low), float(high)


def per_checkpoint(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    label_column: str = "outcome_index",
    checkpoint_column: str = "checkpoint_minute",
) -> pd.DataFrame:
    """Metrics computed separately at each checkpoint.

    Reporting only the full-match number would hide the interesting part: early-match
    prediction is nearly uninformative and late-match prediction is nearly determined by the
    scoreline, so a single average blurs two very different regimes.
    """
    y_true = frame[label_column].to_numpy()
    rows = []
    for checkpoint, group in frame.groupby(checkpoint_column):
        index = group.index.to_numpy()
        positions = frame.index.get_indexer(index)
        rows.append(
            {
                "checkpoint_minute": float(checkpoint),
                "n": len(group),
                **score(y_true[positions], probabilities[positions]),
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint_minute").reset_index(drop=True)


def compare_models(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    label_column: str = "outcome_index",
    game_column: str = "game_id",
    n_boot: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per model: overall metrics plus a match-level bootstrap CI on log-loss."""
    y_true = frame[label_column].to_numpy()
    game_ids = frame[game_column].to_numpy()

    rows = []
    for name, probabilities in predictions.items():
        metrics = score(y_true, probabilities)
        point, low, high = bootstrap_by_match(
            y_true, probabilities, game_ids, metric="log_loss", n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "model": name,
                **metrics,
                "log_loss_ci_low": round(low, 4),
                "log_loss_ci_high": round(high, 4),
                "n_rows": len(frame),
                "n_matches": int(len(np.unique(game_ids))),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def paired_difference(
    frame: pd.DataFrame,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    label_column: str = "outcome_index",
    game_column: str = "game_id",
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI on the log-loss *difference* between two models.

    The paired form is the right test for "does the graph model beat B0": the two models see
    identical matches, so resampling them together removes the between-match variance that
    dominates the individual intervals. If this CI contains zero, the honest statement is
    that the models are indistinguishable on this corpus.
    """
    y_true = frame[label_column].to_numpy()
    game_ids = frame[game_column].to_numpy()

    unique_games = np.unique(game_ids)
    index_by_game = {game: np.flatnonzero(game_ids == game) for game in unique_games}

    point = log_loss(y_true, probabilities_a) - log_loss(y_true, probabilities_b)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([index_by_game[game] for game in drawn])
        samples.append(
            log_loss(y_true[rows], probabilities_a[rows])
            - log_loss(y_true[rows], probabilities_b[rows])
        )

    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "delta_log_loss": round(float(point), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "significant": bool(low > 0 or high < 0),
    }


def class_prior_predictions(train_labels: np.ndarray, n_rows: int) -> np.ndarray:
    """The floor: predict the training class frequencies for every row."""
    counts = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
    prior = counts / counts.sum()
    return np.tile(prior, (n_rows, 1))
