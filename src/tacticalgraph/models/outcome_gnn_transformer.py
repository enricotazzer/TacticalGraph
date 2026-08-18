"""Module 3's graph + sequence model.

Architecture, per match:

    for each of 16 windows:
        GraphSAGE(home passing network) -> mean-pool -> home_vec
        GraphSAGE(away passing network) -> mean-pool -> away_vec
        token_t = [home_vec | away_vec | scalar match state at t]
    causal Transformer encoder over the 16 tokens
    per-timestep 3-class head

Three details do the real work:

**The causal mask is load-bearing, not decorative.** Without it, the prediction at minute 20
attends to minute 90 and every reported number is void. It is built once and asserted in
`tests`.

**Where the tabular state enters decides what a null result means.** In residual mode the model
adds a fitted, frozen B1's logits and zero-initialises its own head, so it starts *as* B1 and
learns only a correction -- making B1 the floor. See `OutcomeGNNTransformer.__init__`.

**The training objective need not weight all 16 checkpoints equally**, since minute 15 is close
to irreducible while minute 90 is nearly determined. See `checkpoint_weights`. Only the gradient
changes; every reported metric stays unweighted.

**On capacity.** The model is small (2 GraphSAGE layers, up to 2 Transformer layers, d_model 64)
to fit Kaggle's free tier, and the sweep picks between 5k/13k/77k parameters on validation. Note
that an earlier version of this docstring asserted the corpus "cannot support capacity" -- that
was wrong. It rested on the sweep preferring small models, which turned out to be an artefact of
training at batch size 1; once batching worked, the spread between capacities collapsed from
~0.25 to ~0.04 log-loss and the largest model became competitive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv, global_mean_pool

from tacticalgraph.features.match_state import B1_FEATURES
from tacticalgraph.models.role_gnn import (
    DIRECTION_FEATURES,
    TOPOLOGY_FEATURES,
    WINDOW_KEYS,
    engineer_node_features,
)

log = logging.getLogger(__name__)

# Node features for the windowed graphs. Position plus topology: within a window we want the
# shape of the network, and mean location is what makes it a *passing shape* rather than an
# abstract graph.
WINDOW_NODE_FEATURES: tuple[str, ...] = ("mean_x", "mean_y") + TOPOLOGY_FEATURES

# The same, plus pass direction. Every feature in TOPOLOGY_FEATURES measures *how much* a player
# passes, and on Module 2's ablation that volume buys only ~+1 pp over pitch position alone --
# adding direction takes it to +2.65 pp, more than doubling the graph's contribution. Whether
# that carries over to a sequence model is a separate question, which is why this is a selectable
# variant rather than a silent change to the default.
WINDOW_NODE_FEATURES_WITH_DIRECTION: tuple[str, ...] = (
    WINDOW_NODE_FEATURES + DIRECTION_FEATURES
)

NODE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "volume": WINDOW_NODE_FEATURES,
    "volume+direction": WINDOW_NODE_FEATURES_WITH_DIRECTION,
}

# Scalar state appended to each token. B1 rather than B2: the graph is meant to supply the
# structural information, so handing it B2's network summaries too would double-count.
SEQUENCE_STATE_FEATURES: tuple[str, ...] = B1_FEATURES

N_WINDOWS = 16
N_CLASSES = 3

# Schemes for how much each of the 16 checkpoints contributes to the training objective.
#
# Weighting them equally -- the original behaviour, kept here as `uniform` -- spends as much of
# the objective on minute 15, where the outcome is close to irreducible (B0 scores ~1.06 log-loss
# there), as on minute 90, where it is nearly determined. A large share of the loss is therefore
# spent fitting noise.
CHECKPOINT_WEIGHT_SCHEMES: tuple[str, ...] = ("uniform", "linear", "b0_signal")


def checkpoint_weights(
    scheme: str,
    b0_log_loss: np.ndarray | None = None,
    n_windows: int = N_WINDOWS,
) -> np.ndarray:
    """Per-checkpoint training weights, normalised to mean 1.

    The normalisation is not cosmetic: it keeps the loss on the same scale as the unweighted
    objective, so `clip_grad_norm_`'s threshold and the early-stopping delta mean the same thing
    whichever scheme is selected.

    - `uniform`   -- every checkpoint counts equally (the control).
    - `linear`    -- proportional to elapsed match time.
    - `b0_signal` -- inversely proportional to B0's per-checkpoint log-loss on the **training**
      fold, so weight follows how much signal is actually there. Fit anywhere but train and the
      test fold would be shaping its own objective.
    """
    if scheme == "uniform":
        weights = np.ones(n_windows, dtype=np.float64)
    elif scheme == "linear":
        weights = np.arange(1, n_windows + 1, dtype=np.float64)
    elif scheme == "b0_signal":
        if b0_log_loss is None:
            raise ValueError("scheme 'b0_signal' needs B0's per-checkpoint train log-loss")
        losses = np.asarray(b0_log_loss, dtype=np.float64)
        if losses.shape != (n_windows,):
            raise ValueError(f"expected {n_windows} per-checkpoint losses, got {losses.shape}")
        weights = 1.0 / np.clip(losses, 1e-6, None)
    else:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {CHECKPOINT_WEIGHT_SCHEMES}")
    return weights / weights.mean()


def sequence_loss(
    logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy over (batch, timestep), optionally weighted per timestep.

    `reduction="none"` then a manual weighted mean -- `F.cross_entropy`'s own `weight=` argument
    is per *class*, not per timestep, and using it here would silently reweight home/draw/away
    instead of early/late.

    With weights normalised to mean 1 this reduces exactly to the unweighted mean when the scheme
    is `uniform`, which is what makes the before/after a single flag.
    """
    per_step = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).view(target.shape)
    if weights is None:
        return per_step.mean()
    return (per_step * weights).sum() / (per_step.shape[0] * weights.sum())


@dataclass
class MatchSequence:
    """One match as a padded sequence of window graphs plus scalar state."""

    game_id: int
    label: int
    # (16,) node-slice bookkeeping is held in the batch, not here.
    state: np.ndarray  # (16, n_state)
    home_graphs: list[tuple[np.ndarray, np.ndarray]]  # per window: (x, edge_index)
    away_graphs: list[tuple[np.ndarray, np.ndarray]]
    # Frozen per-checkpoint logits from a fitted baseline (B1), shape (16, 3). Present only when
    # the model runs in residual mode; `None` reproduces the original parallel-`state_head`
    # architecture.
    base_logits: np.ndarray | None = None


class WindowGraphEncoder(nn.Module):
    """2-layer GraphSAGE producing one vector per window graph via mean pooling."""

    def __init__(self, in_channels: int, hidden_channels: int = 48, out_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros(self.out_channels, device=x.device, dtype=torch.float32)
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h.mean(dim=0)  # graph-level vector

    def forward_pooled(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        """Encode many disjoint graphs in one pass and mean-pool each separately.

        Mathematically identical to calling `forward` per graph -- SAGEConv only aggregates
        along edges, and a `Batch` introduces none between its members -- but it replaces
        `n_graphs` kernel launches with one. With 16 windows x 2 teams per match, the
        per-graph version spent nearly all its time in launch overhead on ~11-node graphs.

        `size=n_graphs` matters: an empty window (a team with no completed passes in it) has
        no rows in `batch`, and without an explicit size the output would silently be shorter
        than the number of graphs and every subsequent reshape would misalign. With it, empty
        graphs pool to zero, matching `forward`'s explicit empty case.
        """
        if x.numel() == 0:
            return torch.zeros(n_graphs, self.out_channels, device=x.device)
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return global_mean_pool(h, batch, size=n_graphs)


def causal_mask(size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Additive float mask forbidding attention to any later timestep.

    True/-inf above the diagonal: token t sees 0..t only.
    """
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class OutcomeGNNTransformer(nn.Module):
    """GraphSAGE per window + causal Transformer over the window sequence."""

    def __init__(
        self,
        node_in_channels: int,
        state_in_channels: int,
        graph_out: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
        baseline_residual: bool = False,
    ) -> None:
        super().__init__()
        self.baseline_residual = baseline_residual
        self.encoder = WindowGraphEncoder(node_in_channels, out_channels=graph_out)
        self.project = nn.Linear(2 * graph_out + state_in_channels, d_model)
        # Learned positional embedding: 16 fixed positions, so a table is simpler and no
        # worse than a sinusoid here.
        self.positions = nn.Parameter(torch.zeros(N_WINDOWS, d_model))
        nn.init.normal_(self.positions, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

        # How the model gets access to the tabular match state decides what a null result means.
        #
        # The question Module 3 asks is "does the graph sequence add anything *over* the tabular
        # state?", so the model must be able to recover the tabular baseline trivially -- exactly
        # as Module 2's `both` variant contains `position`. Otherwise the comparison conflates
        # "graphs do not help" with "the network failed to relearn goal difference".
        #
        # Two ways to grant that, and they differ in where the floor sits:
        #
        # `state_head` (baseline_residual=False) is a parallel *learned* linear path. The model
        # can in principle rediscover B1, but it has to do so by gradient descent from a random
        # start -- so "learn nothing" means "fail", and the floor is chance.
        #
        # `baseline_residual=True` instead adds B1's own frozen logits and zero-initialises
        # `head`, so at step 0 the model *is* B1 and only ever learns a correction to it. The
        # floor becomes B1: early stopping can always fall back to it, and "learn nothing" means
        # "match B1". That makes a null result interpretable, which the parallel path never did.
        # Zero-initialising `head` means the graph encoder and Transformer receive *no* gradient
        # on the very first step: dL/d(encoded) is scaled by `head.weight`, which is zero. That
        # looks alarming and is harmless -- `head` itself gets a large gradient immediately, so it
        # leaves zero after one step and everything upstream trains from step 1 onward. The same
        # trick is standard in zero-init residual blocks and LoRA. It is asserted in the tests so
        # nobody "fixes" it back into a random head and loses the match-B1-at-init property.
        if baseline_residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.state_head = None
        else:
            self.state_head = nn.Linear(state_in_channels, n_classes)
        self.d_model = d_model

    def encode_tokens(self, sequence: MatchSequence, device: str = "cpu") -> torch.Tensor:
        """Build the (16, d_model) token matrix for one match."""
        tokens = []
        for window in range(N_WINDOWS):
            home_x, home_edges = sequence.home_graphs[window]
            away_x, away_edges = sequence.away_graphs[window]
            home_vec = self.encoder(
                torch.as_tensor(home_x, dtype=torch.float32, device=device),
                torch.as_tensor(home_edges, dtype=torch.long, device=device),
            )
            away_vec = self.encoder(
                torch.as_tensor(away_x, dtype=torch.float32, device=device),
                torch.as_tensor(away_edges, dtype=torch.long, device=device),
            )
            state = torch.as_tensor(
                sequence.state[window], dtype=torch.float32, device=device
            )
            tokens.append(torch.cat([home_vec, away_vec, state]))
        return self.project(torch.stack(tokens))

    def forward_batch(
        self, sequences: list[MatchSequence], device: str = "cpu"
    ) -> torch.Tensor:
        """Logits for a minibatch of matches: (n_matches, 16, 3).

        This is the only implementation; `forward` wraps it for a single sequence, so the
        batched and unbatched paths cannot drift apart.
        """
        n_matches = len(sequences)
        graphs = []
        for sequence in sequences:
            for window in range(N_WINDOWS):
                # Order matters and is relied on by the reshape below:
                # match-major, then window, then (home, away).
                for side in (sequence.home_graphs, sequence.away_graphs):
                    x, edge_index = side[window]
                    graphs.append(
                        Data(
                            x=torch.as_tensor(x, dtype=torch.float32),
                            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                        )
                    )
        n_graphs = len(graphs)
        batch = Batch.from_data_list(graphs).to(device)
        pooled = self.encoder.forward_pooled(
            batch.x, batch.edge_index, batch.batch, n_graphs
        )
        # (n_matches, 16, 2, graph_out) -> home/away vectors per window
        pooled = pooled.view(n_matches, N_WINDOWS, 2, -1)
        home_vectors, away_vectors = pooled[:, :, 0], pooled[:, :, 1]

        state = torch.as_tensor(
            np.stack([s.state for s in sequences]), dtype=torch.float32, device=device
        )
        tokens = self.project(torch.cat([home_vectors, away_vectors, state], dim=-1))
        encoded = self.transformer(
            tokens + self.positions, mask=causal_mask(N_WINDOWS, device=device)
        )
        # Both extra paths are per-timestep and already causal by construction (B1's features at
        # checkpoint t use only actions up to t, enforced by the truncation test), so adding them
        # here cannot leak the future.
        if self.baseline_residual:
            missing = [s.game_id for s in sequences if s.base_logits is None]
            if missing:
                raise ValueError(
                    f"baseline_residual=True needs base_logits on every sequence; "
                    f"{len(missing)} lack them (e.g. game {missing[0]})"
                )
            base = torch.as_tensor(
                np.stack([s.base_logits for s in sequences]),
                dtype=torch.float32,
                device=device,
            )
            return base + self.head(encoded)  # (n_matches, 16, 3)
        return self.head(encoded) + self.state_head(state)  # (n_matches, 16, 3)

    def forward(self, sequence: MatchSequence, device: str = "cpu") -> torch.Tensor:
        return self.forward_batch([sequence], device=device).squeeze(0)  # (16, 3)


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------


def build_window_features(
    window_nodes: pd.DataFrame, window_edges: pd.DataFrame
) -> pd.DataFrame:
    """Node features for the windowed graphs, reusing Module 2's definition.

    Sharing `engineer_node_features` matters: two separate definitions of "what a node knows"
    would let Module 2's and Module 3's conclusions drift apart for reasons nobody could trace.

    `WINDOW_KEYS` is the load-bearing argument. Without it the edge aggregates are computed over
    the whole match and repeated across all 16 windows, so the token for minute 15 carries the
    player's minute-90 passing structure. Every Module 3 number reported before this argument
    existed was produced that way.
    """
    return engineer_node_features(window_nodes, window_edges, group_keys=WINDOW_KEYS)


def make_sequences(
    state_table: pd.DataFrame,
    window_features: pd.DataFrame,
    window_edges: pd.DataFrame,
    outcomes: pd.DataFrame,
    state_columns: tuple[str, ...] = SEQUENCE_STATE_FEATURES,
    node_columns: tuple[str, ...] = WINDOW_NODE_FEATURES,
    scaler: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    base_logit_columns: tuple[str, ...] | None = None,
) -> tuple[list[MatchSequence], tuple[np.ndarray, ...]]:
    """Assemble one `MatchSequence` per match.

    Standardisation statistics are fit on whatever is passed first (the training fold) and
    reused for val/test via `scaler`; fitting per fold would leak the test distribution.

    `base_logit_columns` names three columns holding a fitted baseline's per-checkpoint log
    probabilities. They are carried through **unstandardised** -- they are logits to be added to
    the model's output, not features to be normalised, and scaling them would destroy the property
    that the residual model reproduces the baseline exactly at initialisation.
    """
    node_matrix = window_features[list(node_columns)].to_numpy(dtype=np.float32)
    state_matrix = state_table[list(state_columns)].to_numpy(dtype=np.float32)

    if scaler is None:
        node_mean, node_std = node_matrix.mean(axis=0), node_matrix.std(axis=0)
        state_mean, state_std = state_matrix.mean(axis=0), state_matrix.std(axis=0)
        node_std[node_std < 1e-6] = 1.0
        state_std[state_std < 1e-6] = 1.0
    else:
        node_mean, node_std, state_mean, state_std = scaler

    features = window_features.copy()
    features[list(node_columns)] = (node_matrix - node_mean) / node_std

    teams = outcomes.set_index("game_id")[["home_team_id", "away_team_id"]]

    # Index once; per-match lookup in a loop over 760 games would otherwise dominate runtime.
    node_groups = {
        key: group for key, group in features.groupby(["game_id", "team_id", "window_index"])
    }
    edge_groups = {
        key: group for key, group in window_edges.groupby(["game_id", "team_id", "window_index"])
    }

    sequences: list[MatchSequence] = []
    for game_id, game_rows in state_table.groupby("game_id", sort=False):
        if game_id not in teams.index:
            continue
        game_rows = game_rows.sort_values("window_index")
        home = int(teams.loc[game_id, "home_team_id"])
        away = int(teams.loc[game_id, "away_team_id"])

        state = (game_rows[list(state_columns)].to_numpy(dtype=np.float32) - state_mean) / state_std
        base_logits = (
            game_rows[list(base_logit_columns)].to_numpy(dtype=np.float32)
            if base_logit_columns is not None
            else None
        )

        def graphs_for(team: int) -> list[tuple[np.ndarray, np.ndarray]]:
            out = []
            for window in range(N_WINDOWS):
                nodes = node_groups.get((game_id, team, window))
                if nodes is None or nodes.empty:
                    out.append(
                        (np.zeros((0, len(node_columns)), dtype=np.float32),
                         np.zeros((2, 0), dtype=np.int64))
                    )
                    continue
                nodes = nodes.reset_index(drop=True)
                position = {int(pid): i for i, pid in enumerate(nodes["player_id"])}
                x = nodes[list(node_columns)].to_numpy(dtype=np.float32)

                edges = edge_groups.get((game_id, team, window))
                if edges is None or edges.empty:
                    edge_index = np.zeros((2, 0), dtype=np.int64)
                else:
                    pairs = [
                        (position[int(s)], position[int(t)])
                        for s, t in zip(edges["source"], edges["target"])
                        if int(s) in position and int(t) in position
                    ]
                    edge_index = (
                        np.asarray(pairs, dtype=np.int64).T
                        if pairs
                        else np.zeros((2, 0), dtype=np.int64)
                    )
                out.append((x, edge_index))
            return out

        sequences.append(
            MatchSequence(
                game_id=int(game_id),
                label=int(game_rows["outcome_index"].iloc[0]),
                state=state,
                home_graphs=graphs_for(home),
                away_graphs=graphs_for(away),
                base_logits=base_logits,
            )
        )

    return sequences, (node_mean, node_std, state_mean, state_std)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train_outcome_model(
    train: list[MatchSequence],
    val: list[MatchSequence],
    node_in_channels: int,
    state_in_channels: int,
    epochs: int = 150,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 25,
    device: str = "cpu",
    seed: int = 0,
    graph_out: int = 8,
    d_model: int = 16,
    n_heads: int = 2,
    n_layers: int = 1,
    dropout: float = 0.4,
    batch_size: int = 16,
    baseline_residual: bool = False,
    checkpoint_weights: np.ndarray | None = None,
) -> tuple[OutcomeGNNTransformer, dict[str, list[float]]]:
    """Train with early stopping on validation log-loss.

    Log-loss rather than accuracy for the stopping criterion: with three imbalanced classes
    and a draw class that is barely predictable, accuracy plateaus while calibration is still
    changing, and calibration is what this module reports.

    **`checkpoint_weights` changes the gradient only.** Early stopping, the capacity sweep and
    every reported metric all use the *unweighted* loss via `evaluate_loss`. That split is
    deliberate: reweighting is a statement about where the model should spend capacity, not a
    change to what counts as a good model, and the reported number has to stay comparable to
    B0/B1/B2. A weighting scheme that hurts the unweighted metric therefore simply loses the
    validation selection instead of quietly flattering itself.

    **`batch_size` is the substantive knob here.** The original loop called
    `optimiser.step()` once per match, i.e. batch size 1: ~260-300 updates per epoch, each
    from a single match's heavily-correlated 16 checkpoints. Across eight seeded runs on two
    corpora the best validation epoch was 0, 1, 1, 0, 1, 4, 17, 3 -- the model was almost
    never better than initialisation-plus-one-step, which is what an optimiser fed pure
    gradient noise looks like. `batch_size=1` reproduces that behaviour exactly, so the
    before/after comparison is a single flag.

    The defaults were set from the *validation* curve, never the test set. An earlier run at
    lr=1e-3 / patience=8 oscillated violently between epochs (val log-loss 0.81 -> 1.16 ->
    0.89 -> 1.25) and stopped at epoch 9 with its best score at epoch 0, while training loss
    was still falling. Note that a learning rate tuned for batch size 1 is not the right one
    for batch size 16 -- averaging over 16 matches shrinks each step -- so `lr` belongs in the
    validation sweep whenever `batch_size` changes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = OutcomeGNNTransformer(
        node_in_channels=node_in_channels,
        state_in_channels=state_in_channels,
        graph_out=graph_out,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        baseline_residual=baseline_residual,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    weights = (
        torch.as_tensor(checkpoint_weights, dtype=torch.float32, device=device)
        if checkpoint_weights is not None
        else None
    )

    history: dict[str, object] = {"train_loss": [], "val_loss": []}
    rng = np.random.default_rng(seed)

    # Score the *untrained* model before the first epoch, and let it compete for `best_state`.
    #
    # This is what actually makes a residual baseline a floor. Evaluating only after each epoch
    # means the initial state -- the one that reproduces the frozen baseline exactly -- is never a
    # candidate, so "the model can always fall back to B1" would be a claim the code does not
    # honour. With it, `best_epoch == -1` reports "training never improved on the baseline", which
    # is a result worth being able to state.
    best_loss = evaluate_loss(model, val, device=device)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, stale = -1, 0
    history["val_loss_at_init"] = best_loss

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        total, n_batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            chunk = [train[i] for i in order[start : start + batch_size]]
            optimiser.zero_grad()
            logits = model.forward_batch(chunk, device=device)  # (B, 16, 3)
            target = torch.tensor(
                [s.label for s in chunk], dtype=torch.long, device=device
            ).unsqueeze(1).expand(-1, N_WINDOWS)
            loss = sequence_loss(logits, target, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss)
            n_batches += 1

        train_loss = total / max(n_batches, 1)
        val_loss = evaluate_loss(model, val, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best val log-loss %.4f)", epoch, best_loss)
                break

    history["best_epoch"] = best_epoch
    if best_epoch < 0:
        log.info(
            "no epoch improved on the initial state (val log-loss %.4f); returning it unchanged",
            best_loss,
        )
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_loss(
    model: OutcomeGNNTransformer,
    sequences: list[MatchSequence],
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    """Mean per-match cross-entropy, always **unweighted**.

    Batched in chunks and weighted by chunk size, so the result is identical to the
    per-sequence mean regardless of `batch_size` (a plain mean of chunk means would be wrong
    whenever the last chunk is short).

    Deliberately takes no `checkpoint_weights`: this is the number early stopping, the sweep and
    the report all use, and it has to stay on the same footing as the tabular ladder's log-loss.
    """
    model.eval()
    total, n = 0.0, 0
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        logits = model.forward_batch(chunk, device=device)
        target = torch.tensor(
            [s.label for s in chunk], dtype=torch.long, device=device
        ).unsqueeze(1).expand(-1, N_WINDOWS)
        loss = sequence_loss(logits, target)
        total += float(loss) * len(chunk)
        n += len(chunk)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(
    model: OutcomeGNNTransformer,
    sequences: list[MatchSequence],
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Per (game, window) class probabilities, tidy so they can be joined to the state table."""
    model.eval()
    rows = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        probabilities = F.softmax(
            model.forward_batch(chunk, device=device), dim=-1
        ).cpu().numpy()
        for position, sequence in enumerate(chunk):
            for window in range(N_WINDOWS):
                rows.append(
                    {
                        "game_id": sequence.game_id,
                        "window_index": window,
                        "p_home_win": probabilities[position, window, 0],
                        "p_draw": probabilities[position, window, 1],
                        "p_away_win": probabilities[position, window, 2],
                    }
                )
    return pd.DataFrame(rows)
