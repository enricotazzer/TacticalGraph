"""Module 3's graph + sequence model.

Architecture, per match:

    for each of 16 windows:
        GraphSAGE(home passing network) -> mean-pool -> home_vec
        GraphSAGE(away passing network) -> mean-pool -> away_vec
        token_t = [home_vec | away_vec | scalar match state at t]
    causal Transformer encoder over the 16 tokens
    per-timestep 3-class head

Two details do the real work:

**The causal mask is load-bearing, not decorative.** Without it, the prediction at minute 20
attends to minute 90 and every reported number is void. It is built once and asserted in
`tests`.

**The model is deliberately tiny.** 300 independent training labels (the 16 checkpoints of a
match are one observation, not sixteen) cannot support capacity, and it has to fit Kaggle's
free tier. 2 GraphSAGE layers, 2 Transformer layers, d_model 64.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from tacticalgraph.features.match_state import B1_FEATURES
from tacticalgraph.models.role_gnn import TOPOLOGY_FEATURES, engineer_node_features

log = logging.getLogger(__name__)

# Node features for the windowed graphs. Position plus topology: within a window we want the
# shape of the network, and mean location is what makes it a *passing shape* rather than an
# abstract graph.
WINDOW_NODE_FEATURES: tuple[str, ...] = ("mean_x", "mean_y") + TOPOLOGY_FEATURES

# Scalar state appended to each token. B1 rather than B2: the graph is meant to supply the
# structural information, so handing it B2's network summaries too would double-count.
SEQUENCE_STATE_FEATURES: tuple[str, ...] = B1_FEATURES

N_WINDOWS = 16
N_CLASSES = 3


@dataclass
class MatchSequence:
    """One match as a padded sequence of window graphs plus scalar state."""

    game_id: int
    label: int
    # (16,) node-slice bookkeeping is held in the batch, not here.
    state: np.ndarray  # (16, n_state)
    home_graphs: list[tuple[np.ndarray, np.ndarray]]  # per window: (x, edge_index)
    away_graphs: list[tuple[np.ndarray, np.ndarray]]


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
    ) -> None:
        super().__init__()
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
        # Direct linear path from the scalar match state to the logits.
        #
        # This is the experimental design, not a crutch. The question Module 3 asks is
        # "does the graph sequence add anything *over* the tabular state?" -- so the model
        # must be able to recover the tabular baseline trivially, exactly as Module 2's
        # `both` variant contains `position`. Without this path the comparison conflates
        # "graphs do not help" with "the network failed to relearn goal difference".
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

    def forward(self, sequence: MatchSequence, device: str = "cpu") -> torch.Tensor:
        tokens = self.encode_tokens(sequence, device=device) + self.positions
        encoded = self.transformer(
            tokens.unsqueeze(0), mask=causal_mask(N_WINDOWS, device=device)
        ).squeeze(0)
        state = torch.as_tensor(sequence.state, dtype=torch.float32, device=device)
        # Scalar state is per-timestep and already causal by construction, so adding it here
        # cannot leak the future.
        return self.head(encoded) + self.state_head(state)  # (16, 3)


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------


def build_window_features(
    window_nodes: pd.DataFrame, window_edges: pd.DataFrame
) -> pd.DataFrame:
    """Node features for the windowed graphs, reusing Module 2's definition.

    Sharing `engineer_node_features` matters: two separate definitions of "what a node knows"
    would let Module 2's and Module 3's conclusions drift apart for reasons nobody could trace.
    """
    return engineer_node_features(window_nodes, window_edges)


def make_sequences(
    state_table: pd.DataFrame,
    window_features: pd.DataFrame,
    window_edges: pd.DataFrame,
    outcomes: pd.DataFrame,
    state_columns: tuple[str, ...] = SEQUENCE_STATE_FEATURES,
    node_columns: tuple[str, ...] = WINDOW_NODE_FEATURES,
    scaler: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[MatchSequence], tuple[np.ndarray, ...]]:
    """Assemble one `MatchSequence` per match.

    Standardisation statistics are fit on whatever is passed first (the training fold) and
    reused for val/test via `scaler`; fitting per fold would leak the test distribution.
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
) -> tuple[OutcomeGNNTransformer, dict[str, list[float]]]:
    """Train with early stopping on validation log-loss.

    Log-loss rather than accuracy for the stopping criterion: with three imbalanced classes
    and a draw class that is barely predictable, accuracy plateaus while calibration is still
    changing, and calibration is what this module reports.

    The defaults were set from the *validation* curve, never the test set. An earlier run at
    lr=1e-3 / patience=8 oscillated violently between epochs (val log-loss 0.81 -> 1.16 ->
    0.89 -> 1.25) and stopped at epoch 9 with its best score at epoch 0, while training loss
    was still falling -- classic too-high learning rate plus a patience budget consumed by
    noise on an 80-match validation fold.
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
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state, best_loss, stale = None, float("inf"), 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for index in order:
            sequence = train[index]
            optimiser.zero_grad()
            logits = model(sequence, device=device)
            target = torch.full(
                (N_WINDOWS,), sequence.label, dtype=torch.long, device=device
            )
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss)

        train_loss = total / max(len(train), 1)
        val_loss = evaluate_loss(model, val, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best val log-loss %.4f)", epoch, best_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_loss(
    model: OutcomeGNNTransformer, sequences: list[MatchSequence], device: str = "cpu"
) -> float:
    model.eval()
    total = 0.0
    for sequence in sequences:
        logits = model(sequence, device=device)
        target = torch.full((N_WINDOWS,), sequence.label, dtype=torch.long, device=device)
        total += float(F.cross_entropy(logits, target))
    return total / max(len(sequences), 1)


@torch.no_grad()
def predict_proba(
    model: OutcomeGNNTransformer, sequences: list[MatchSequence], device: str = "cpu"
) -> pd.DataFrame:
    """Per (game, window) class probabilities, tidy so they can be joined to the state table."""
    model.eval()
    rows = []
    for sequence in sequences:
        probabilities = F.softmax(model(sequence, device=device), dim=1).cpu().numpy()
        for window in range(N_WINDOWS):
            rows.append(
                {
                    "game_id": sequence.game_id,
                    "window_index": window,
                    "p_home_win": probabilities[window, 0],
                    "p_draw": probabilities[window, 1],
                    "p_away_win": probabilities[window, 2],
                }
            )
    return pd.DataFrame(rows)
