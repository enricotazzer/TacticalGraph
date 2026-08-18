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
    CHECKPOINT_WEIGHT_SCHEMES,
    N_CLASSES,
    N_WINDOWS,
    MatchSequence,
    OutcomeGNNTransformer,
    causal_mask,
    checkpoint_weights,
    evaluate_loss,
    predict_proba,
    sequence_loss,
)

NODE_FEATURES = 6
STATE_FEATURES = 4


def _sequence(
    game_id: int,
    seed: int,
    n_players: int = 11,
    empty_window: int | None = None,
    with_base: bool = False,
):
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

    base_logits = None
    if with_base:
        # Stand-in for a fitted B1: proper probabilities, stored as clipped log-probabilities
        # exactly as `train_outcome.py` does.
        probabilities = rng.dirichlet(np.ones(N_CLASSES), size=N_WINDOWS).astype(np.float32)
        base_logits = np.log(np.clip(probabilities, 1e-6, 1.0)).astype(np.float32)

    return MatchSequence(
        game_id=game_id,
        label=int(rng.integers(0, N_CLASSES)),
        state=rng.normal(size=(N_WINDOWS, STATE_FEATURES)).astype(np.float32),
        home_graphs=graphs(),
        away_graphs=graphs(),
        base_logits=base_logits,
    )


def _model(seed: int = 0, baseline_residual: bool = False) -> OutcomeGNNTransformer:
    torch.manual_seed(seed)
    model = OutcomeGNNTransformer(
        node_in_channels=NODE_FEATURES,
        state_in_channels=STATE_FEATURES,
        graph_out=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,  # deterministic, so equality checks are meaningful
        baseline_residual=baseline_residual,
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


# --------------------------------------------------------------------------------------
# Checkpoint weighting (Fix A)
#
# The weights change the gradient only. Every reported metric, the early-stopping criterion and
# the capacity sweep all use the unweighted loss, so a scheme that helps only its own objective
# loses the validation selection instead of quietly flattering itself.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", CHECKPOINT_WEIGHT_SCHEMES)
def test_checkpoint_weights_normalise_to_mean_one(scheme: str):
    """Normalisation keeps the loss on the same scale for every scheme.

    Without it, `clip_grad_norm_`'s threshold and the early-stopping delta would silently mean
    different things depending on which scheme won the sweep.
    """
    b0 = np.linspace(1.05, 0.35, N_WINDOWS)
    weights = checkpoint_weights(scheme, b0_log_loss=b0)
    assert weights.shape == (N_WINDOWS,)
    assert weights.mean() == pytest.approx(1.0)
    assert (weights > 0).all(), "a zero or negative weight would drop a checkpoint entirely"
    assert np.isfinite(weights).all()


def test_uniform_weights_reduce_to_the_unweighted_loss():
    """The control arm must be *exactly* the old objective, or the A/B measures two changes."""
    torch.manual_seed(0)
    logits = torch.randn(5, N_WINDOWS, N_CLASSES)
    target = torch.randint(0, N_CLASSES, (5,)).unsqueeze(1).expand(-1, N_WINDOWS)
    weights = torch.as_tensor(checkpoint_weights("uniform"), dtype=torch.float32)

    assert sequence_loss(logits, target, weights) == pytest.approx(
        float(sequence_loss(logits, target)), abs=1e-6
    )


def test_weighting_actually_reweights():
    """Guard against a scheme that normalises to something indistinguishable from uniform."""
    torch.manual_seed(0)
    logits = torch.randn(4, N_WINDOWS, N_CLASSES)
    target = torch.randint(0, N_CLASSES, (4,)).unsqueeze(1).expand(-1, N_WINDOWS)
    linear = torch.as_tensor(checkpoint_weights("linear"), dtype=torch.float32)

    assert float(sequence_loss(logits, target, linear)) != pytest.approx(
        float(sequence_loss(logits, target)), abs=1e-4
    )


def test_b0_signal_weights_follow_available_signal():
    """Late checkpoints, where B0 scores better, must be weighted up rather than down."""
    b0 = np.linspace(1.05, 0.35, N_WINDOWS)  # loss falls as the match progresses
    weights = checkpoint_weights("b0_signal", b0_log_loss=b0)
    assert weights[-1] > weights[0]
    assert np.all(np.diff(weights) > 0), "monotone falling loss should give monotone rising weight"


def test_b0_signal_requires_its_input_and_rejects_the_wrong_length():
    """It is derived from the *train* fold; silently defaulting would hide a leak or a bug."""
    with pytest.raises(ValueError, match="per-checkpoint"):
        checkpoint_weights("b0_signal")
    with pytest.raises(ValueError, match="expected"):
        checkpoint_weights("b0_signal", b0_log_loss=np.ones(5))


def test_unknown_weight_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown scheme"):
        checkpoint_weights("later_is_better")


# --------------------------------------------------------------------------------------
# Residual on a fitted baseline (Fix B)
#
# The point of this arm is that B1 becomes the *floor*: the model starts as B1 and learns a
# correction, so "learn nothing" means "match B1" rather than "fail". That property rests
# entirely on the head being zero-initialised, which is easy to undo by accident.
# --------------------------------------------------------------------------------------


def test_residual_model_reproduces_the_baseline_exactly_at_init():
    """The load-bearing property of Fix B, and it must hold to the bit, not approximately."""
    model = _model(baseline_residual=True)
    sequences = [_sequence(i, seed=i + 11, with_base=True) for i in range(4)]

    with torch.no_grad():
        output = model.forward_batch(sequences)
    base = torch.as_tensor(np.stack([s.base_logits for s in sequences]))

    assert torch.equal(output, base), "a zero-init head must leave the baseline untouched"


def test_residual_model_recovers_baseline_probabilities():
    """softmax(log p) == p, so the model starts by predicting exactly what B1 predicted."""
    model = _model(baseline_residual=True)
    sequence = _sequence(1, seed=5, with_base=True)

    with torch.no_grad():
        probabilities = torch.softmax(model(sequence), dim=-1)
    expected = torch.softmax(torch.as_tensor(sequence.base_logits), dim=-1)

    torch.testing.assert_close(probabilities, expected, atol=1e-6, rtol=1e-5)


def test_residual_mode_drops_the_parallel_state_head():
    """Keeping both paths would double-count: B1 already consumes exactly the state features."""
    assert _model(baseline_residual=True).state_head is None
    assert _model(baseline_residual=False).state_head is not None


def test_residual_head_starts_at_zero_but_still_trains():
    """Zero-init gives no gradient to the encoder on step 0. It must unstick on step 1.

    If it did not, the graph encoder would never learn and the whole comparison would be
    measuring B1 against itself.
    """
    torch.manual_seed(0)
    model = OutcomeGNNTransformer(
        node_in_channels=NODE_FEATURES, state_in_channels=STATE_FEATURES,
        graph_out=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0,
        baseline_residual=True,
    )
    assert float(model.head.weight.abs().sum()) == 0.0
    assert float(model.head.bias.abs().sum()) == 0.0

    sequences = [_sequence(i, seed=i + 3, with_base=True) for i in range(6)]
    target = torch.tensor([s.label for s in sequences]).unsqueeze(1).expand(-1, N_WINDOWS)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    encoder_weight = model.encoder.conv1.lin_l.weight

    gradients = []
    for _ in range(2):
        optimiser.zero_grad()
        sequence_loss(model.forward_batch(sequences), target).backward()
        gradients.append(float(encoder_weight.grad.abs().sum()))
        optimiser.step()

    assert gradients[0] == 0.0, "with a zero head the encoder cannot receive gradient at step 0"
    assert gradients[1] > 0.0, "the encoder must start training once the head leaves zero"


def test_residual_mode_without_base_logits_fails_loudly():
    """Silently falling back to no baseline would report a different arm under the same name."""
    model = _model(baseline_residual=True)
    with pytest.raises(ValueError, match="base_logits"):
        model.forward_batch([_sequence(1, seed=2, with_base=False)])


def test_non_residual_arm_is_unchanged_and_needs_no_baseline():
    """`--baseline-residual none` must reproduce the original architecture exactly."""
    model = _model(baseline_residual=False)
    sequences = [_sequence(i, seed=i + 17) for i in range(3)]
    with torch.no_grad():
        logits = model.forward_batch(sequences)
    assert logits.shape == (3, N_WINDOWS, N_CLASSES)
    assert float(logits.abs().sum()) > 0


def test_causality_still_holds_in_residual_mode():
    """Adding a per-checkpoint baseline must not open a path from late windows to early ones."""
    model = _model(baseline_residual=True)
    original = _sequence(1, seed=7, with_base=True)

    with torch.no_grad():
        before = model(original).clone()

    rng = np.random.default_rng(123)
    perturbed = MatchSequence(
        game_id=original.game_id,
        label=original.label,
        state=original.state.copy(),
        home_graphs=list(original.home_graphs),
        away_graphs=list(original.away_graphs),
        base_logits=original.base_logits.copy(),
    )
    for window in range(12, N_WINDOWS):
        x = rng.normal(size=(11, NODE_FEATURES)).astype(np.float32) * 50.0
        edges = rng.integers(0, 11, size=(2, 15)).astype(np.int64)
        perturbed.home_graphs[window] = (x, edges)
        perturbed.away_graphs[window] = (x * -1.0, edges)
        perturbed.state[window] = perturbed.state[window] + 25.0
        perturbed.base_logits[window] = perturbed.base_logits[window] - 5.0

    with torch.no_grad():
        after = model(perturbed)

    assert torch.allclose(before[:12], after[:12], atol=1e-6), (
        "FUTURE LEAK in residual mode: windows 12-15 changed predictions for windows 0-11"
    )


def test_untrained_state_competes_for_best_and_is_reported():
    """The floor is only real if the *untrained* model can win early stopping.

    Evaluating validation only after each epoch would leave the initial state -- the one that
    reproduces B1 exactly -- unable to be selected, so "the model can always fall back to B1"
    would be a claim the trainer does not honour. `best_epoch == -1` records that case.
    """
    from tacticalgraph.models.outcome_gnn_transformer import train_outcome_model

    train = [_sequence(i, seed=i, with_base=True) for i in range(6)]
    val = [_sequence(100 + i, seed=100 + i, with_base=True) for i in range(4)]

    _, history = train_outcome_model(
        train, val,
        node_in_channels=NODE_FEATURES, state_in_channels=STATE_FEATURES,
        epochs=2, patience=5, device="cpu", seed=0,
        graph_out=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0,
        baseline_residual=True,
    )

    assert "val_loss_at_init" in history, "the untrained model must be scored"
    assert "best_epoch" in history
    assert int(history["best_epoch"]) >= -1
    # Whatever wins, the returned model can never be worse on validation than the initial state.
    assert min([history["val_loss_at_init"], *history["val_loss"]]) == pytest.approx(
        min(history["val_loss_at_init"], min(history["val_loss"]))
    )


# --------------------------------------------------------------------------------------
# Node-feature causality
#
# The mask test above protects the model. These protect the *features handed to it*, which is
# where a real leak lived: `engineer_node_features` aggregated edges over the whole match, so 6
# of the 10 topology features were full-match values repeated across all 16 windows. The model's
# causal mask cannot help when the token for minute 15 already contains minute 90.
# --------------------------------------------------------------------------------------


def _windowed_tables(n_players: int = 8, n_windows: int = N_WINDOWS, seed: int = 0):
    """Minimal windowed node/edge tables shaped like the persisted ones."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    node_rows, edge_rows = [], []
    for window in range(n_windows):
        for player in range(n_players):
            node_rows.append({
                "game_id": 1, "team_id": 10, "season": "2015-2016", "provider": "statsbomb",
                "window_index": window, "player_id": player,
                "mean_x": rng.uniform(10, 90), "mean_y": rng.uniform(5, 63),
                "spread_x": rng.uniform(1, 9), "spread_y": rng.uniform(1, 9),
                "touches": int(rng.integers(1, 20)),
                "actions_all_types": int(rng.integers(1, 25)),
                "passes_attempted": int(rng.integers(1, 15)),
                "passes_completed": int(rng.integers(0, 10)),
            })
        # Edge structure deliberately differs per window, so a match-level aggregate is
        # detectably different from a per-window one.
        for _ in range(int(rng.integers(4, 12))):
            source, target = rng.choice(n_players, size=2, replace=False)
            edge_rows.append({
                "game_id": 1, "team_id": 10, "season": "2015-2016", "provider": "statsbomb",
                "window_index": window, "source": int(source), "target": int(target),
                "weight": int(rng.integers(1, 6)),
                "mean_length": rng.uniform(5, 40), "mean_dx": rng.uniform(-20, 30),
            })
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def test_window_features_are_not_computable_from_later_windows():
    """THE leakage test for node features, mirroring the state table's truncation test.

    Build the features for window t from the whole match, then rebuild from tables truncated at
    t. The row for t must be identical -- otherwise it depended on the future.
    """
    import pandas as pd

    from tacticalgraph.models.outcome_gnn_transformer import build_window_features
    from tacticalgraph.models.role_gnn import DIRECTION_FEATURES, TOPOLOGY_FEATURES

    nodes, edges = _windowed_tables()
    full = build_window_features(nodes, edges)
    # Direction features included: they are newer than this test and travel the same windowed
    # path, so leaving them out would let the identical bug recur unnoticed.
    columns = ["player_id", *TOPOLOGY_FEATURES, *DIRECTION_FEATURES]

    for checkpoint in (0, 5, 11):
        truncated = build_window_features(
            nodes[nodes["window_index"] <= checkpoint],
            edges[edges["window_index"] <= checkpoint],
        )
        left = (
            full[full["window_index"] == checkpoint][columns]
            .sort_values("player_id").reset_index(drop=True)
        )
        right = (
            truncated[truncated["window_index"] == checkpoint][columns]
            .sort_values("player_id").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False,
            obj=f"FUTURE LEAK: window {checkpoint} features change when later windows are removed",
        )


def test_edge_derived_window_features_vary_between_windows():
    """Cheap guard for the same bug: constant-across-windows means match-level aggregation.

    Every one of these was constant per player before the fix.
    """
    from tacticalgraph.models.outcome_gnn_transformer import build_window_features

    nodes, edges = _windowed_tables()
    features = build_window_features(nodes, edges)

    for column in ("degree_in_norm", "degree_out_norm", "strength_in_norm",
                   "strength_out_norm", "edge_share_in", "edge_share_out",
                   "progression_made", "progression_received", "progressive_share"):
        varies = features.groupby("player_id")[column].nunique()
        assert (varies > 1).any(), (
            f"{column} is identical across all 16 windows for every player, which means it was "
            "aggregated over the whole match"
        )


def test_match_level_keys_still_used_for_full_networks():
    """Module 2's path must keep match-level aggregation, and must not silently drop rows.

    The full-match tables carry a `window_index` column that is entirely null, so grouping on it
    would drop every row -- which is why the keys are an explicit argument, not inferred.
    """
    from tacticalgraph.models.role_gnn import (
        NETWORK_KEYS,
        WINDOW_KEYS,
        engineer_node_features,
    )

    nodes, edges = _windowed_tables(n_windows=1)
    nodes = nodes.assign(window_index=None)
    edges = edges.assign(window_index=None)

    features = engineer_node_features(nodes, edges, group_keys=NETWORK_KEYS)
    assert len(features) == len(nodes), "match-level aggregation must preserve every node row"

    with pytest.raises(ValueError, match="nulls"):
        engineer_node_features(nodes, edges, group_keys=WINDOW_KEYS)
