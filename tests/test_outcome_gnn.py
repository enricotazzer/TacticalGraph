"""Invariants of Module 3's graph + sequence model.

The model had no tests at all before the batching change, which is indefensible for the two
properties below -- both are silent when broken:

* **Causality.** If the mask leaks, a prediction at minute 20 can see minute 90 and every
  reported metric is void while looking excellent. This was previously checked by hand once.
* **Batched == unbatched.** `forward_batch` reshapes a flat pool of `n_matches x 16 x 2`
  graphs back into per-match home/away vectors. An ordering mistake there would mix one
  match's graphs into another's tokens and still train, still converge, and still report
  plausible numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tacticalgraph.models.outcome_gnn_transformer import (
    N_CLASSES,
    N_WINDOWS,
    MatchSequence,
    OutcomeGNNTransformer,
    causal_mask,
    evaluate_loss,
    predict_proba,
)

NODE_FEATURES = 6
STATE_FEATURES = 4


def _sequence(game_id: int, seed: int, n_players: int = 11, empty_window: int | None = None):
    """A synthetic match: random small graphs per window, random scalar state."""
    rng = np.random.default_rng(seed)

    def graphs():
        out = []
        for window in range(N_WINDOWS):
            if window == empty_window:
                out.append(
                    (np.zeros((0, NODE_FEATURES), dtype=np.float32),
                     np.zeros((2, 0), dtype=np.int64))
                )
                continue
            x = rng.normal(size=(n_players, NODE_FEATURES)).astype(np.float32)
            n_edges = rng.integers(5, 20)
            edges = rng.integers(0, n_players, size=(2, n_edges)).astype(np.int64)
            out.append((x, edges))
        return out

    return MatchSequence(
        game_id=game_id,
        label=int(rng.integers(0, N_CLASSES)),
        state=rng.normal(size=(N_WINDOWS, STATE_FEATURES)).astype(np.float32),
        home_graphs=graphs(),
        away_graphs=graphs(),
    )


def _model(seed: int = 0) -> OutcomeGNNTransformer:
    torch.manual_seed(seed)
    model = OutcomeGNNTransformer(
        node_in_channels=NODE_FEATURES,
        state_in_channels=STATE_FEATURES,
        graph_out=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,  # deterministic, so equality checks are meaningful
    )
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------------------


def test_causal_mask_shape_and_values():
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    assert torch.isneginf(mask[0, 1:]).all(), "row 0 must not attend to any later step"
    assert (mask[3, :] == 0).all(), "last row attends to everything"
    assert (torch.tril(mask) == 0).all(), "no masking at or below the diagonal"


def test_future_windows_cannot_change_earlier_predictions():
    """The load-bearing property: perturb windows 12-15, predictions 0-11 must be identical."""
    model = _model()
    original = _sequence(1, seed=7)

    with torch.no_grad():
        before = model(original).clone()

    rng = np.random.default_rng(99)
    perturbed = MatchSequence(
        game_id=original.game_id,
        label=original.label,
        state=original.state.copy(),
        home_graphs=list(original.home_graphs),
        away_graphs=list(original.away_graphs),
    )
    for window in range(12, N_WINDOWS):
        x = rng.normal(size=(11, NODE_FEATURES)).astype(np.float32) * 50.0
        edges = rng.integers(0, 11, size=(2, 15)).astype(np.int64)
        perturbed.home_graphs[window] = (x, edges)
        perturbed.away_graphs[window] = (x * -1.0, edges)
        perturbed.state[window] = perturbed.state[window] + 25.0

    with torch.no_grad():
        after = model(perturbed)

    assert torch.allclose(before[:12], after[:12], atol=1e-6), (
        "FUTURE LEAK: perturbing windows 12-15 changed predictions for windows 0-11"
    )
    assert not torch.allclose(before[12:], after[12:]), (
        "perturbation had no effect at all, so the test proves nothing"
    )


# --------------------------------------------------------------------------------------
# Batched == unbatched
# --------------------------------------------------------------------------------------


def test_batched_matches_per_sequence_forward():
    model = _model()
    sequences = [_sequence(i, seed=i + 3) for i in range(5)]

    with torch.no_grad():
        batched = model.forward_batch(sequences)
        individual = torch.stack([model(s) for s in sequences])

    assert batched.shape == (5, N_WINDOWS, N_CLASSES)
    assert torch.allclose(batched, individual, atol=1e-5), (
        "batched and per-sequence forward disagree -- the reshape in forward_batch is "
        "mixing matches together"
    )


def test_batching_is_order_preserving():
    """A match's logits must not depend on its position in the batch."""
    model = _model()
    sequences = [_sequence(i, seed=i + 11) for i in range(4)]

    with torch.no_grad():
        forward = model.forward_batch(sequences)
        reversed_batch = model.forward_batch(list(reversed(sequences)))

    assert torch.allclose(forward[0], reversed_batch[-1], atol=1e-5)
    assert torch.allclose(forward[-1], reversed_batch[0], atol=1e-5)


def test_empty_windows_pool_to_zero_not_nan():
    """A team with no completed passes in a window has an empty graph. `global_mean_pool`
    over zero rows must give zeros, not NaN, or the whole match's loss becomes NaN."""
    model = _model()
    sequence = _sequence(1, seed=5, empty_window=3)

    with torch.no_grad():
        logits = model.forward_batch([sequence])

    assert torch.isfinite(logits).all(), "empty window produced non-finite logits"


def test_empty_window_agrees_between_paths():
    model = _model()
    sequences = [_sequence(1, seed=5, empty_window=3), _sequence(2, seed=6)]
    with torch.no_grad():
        batched = model.forward_batch(sequences)
        individual = torch.stack([model(s) for s in sequences])
    assert torch.allclose(batched, individual, atol=1e-5)


# --------------------------------------------------------------------------------------
# Evaluation helpers must not depend on their chunking
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 32])
def test_evaluate_loss_is_independent_of_batch_size(batch_size: int):
    """A short final chunk must not be over-weighted -- the reason the mean is weighted."""
    model = _model()
    sequences = [_sequence(i, seed=i + 21) for i in range(7)]
    reference = evaluate_loss(model, sequences, batch_size=1)
    assert evaluate_loss(model, sequences, batch_size=batch_size) == pytest.approx(
        reference, abs=1e-5
    )


@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_predict_proba_is_independent_of_batch_size(batch_size: int):
    model = _model()
    sequences = [_sequence(i, seed=i + 31) for i in range(5)]
    reference = predict_proba(model, sequences, batch_size=1)
    frame = predict_proba(model, sequences, batch_size=batch_size)

    assert list(frame["game_id"]) == list(reference["game_id"])
    for column in ("p_home_win", "p_draw", "p_away_win"):
        np.testing.assert_allclose(frame[column], reference[column], atol=1e-5)


def test_predict_proba_rows_sum_to_one():
    model = _model()
    sequences = [_sequence(i, seed=i + 41) for i in range(3)]
    frame = predict_proba(model, sequences)
    totals = frame[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    np.testing.assert_allclose(totals, 1.0, atol=1e-5)
    assert len(frame) == 3 * N_WINDOWS
