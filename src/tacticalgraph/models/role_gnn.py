"""Module 2 -- GraphSAGE functional-role embeddings.

The claim under test: a player's *functional* role is richer than their nominal position,
and the structure of who they pass to carries that information.

Design decisions that make the claim testable rather than decorative:

**Supervision.** Node classification of the 4-class coarse role (GK/DEF/MID/FWD) -- the only
vocabulary both providers express. The embedding is then evaluated against StatsBomb's
24-class positions, which the model never sees.

**The leakage trap.** A player's mean pitch position nearly determines their coarse role
(measured on this corpus: GK 8.8, DEF 43.9, MID 56.4, FWD 68.1 metres on a 105 m pitch). A
model handed (x, y) can score well while learning nothing about passing structure. So the
feature set is split into three explicit variants and all three are reported:

    position   mean/spread of pitch location only    -- how much is trivially positional?
    topology   connectivity and volume only, no x/y  -- what does structure alone carry?
    both       the union                             -- does topology add over position?

`both` vs `position` is the actual result. Reporting only `both` would be the easy way to
claim success without evidence.

**Size.** 2 layers, 64 hidden units, ~19k nodes over 1,520 graphs. This trains in seconds on
a GPU and under a minute on CPU, which is the point: the whole project has to fit in Kaggle's
free tier, and a shallow GraphSAGE with neighbour sampling is the cheap choice that suffices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from tacticalgraph.data.roles import COARSE_ROLES

log = logging.getLogger(__name__)

# Feature groups. Keeping these as data (not scattered literals) is what makes the ablation
# honest -- there is exactly one definition of what "position" and "topology" mean.
POSITION_FEATURES: tuple[str, ...] = ("mean_x", "mean_y", "spread_x", "spread_y")

TOPOLOGY_FEATURES: tuple[str, ...] = (
    "touches",
    "passes_attempted",
    "passes_completed",
    "pass_completion_rate",
    "degree_in_norm",
    "degree_out_norm",
    "strength_in_norm",
    "strength_out_norm",
    "edge_share_in",
    "edge_share_out",
)

# A pass is "progressive" if it gains this many metres towards the opponent goal. 5 m is a low
# bar deliberately: the aim is to separate forward intent from square/backward recycling, not to
# reproduce a broadcaster's definition of a line-breaking pass.
PROGRESSIVE_DX_METRES: float = 5.0

# What kind of pass, rather than how many. Every feature in TOPOLOGY_FEATURES is a volume
# measure, which is why centrality built on them is a positional proxy -- midfielders take 84% of
# the top 50 by degree against a 31% population share, and goalkeepers 0% on all ten. These four
# are computed from `mean_dx`/`mean_length`, which have sat unused on the edge table since
# Module 1, and they separate players volume cannot: a target man and a deep recycler can have
# identical degree while one receives +14 m passes and the other -1 m.
DIRECTION_FEATURES: tuple[str, ...] = (
    "progression_made",
    "progression_received",
    "pass_length_made",
    "progressive_share",
)

# Who moves the ball somewhere dangerous, and who is there when the shot arrives. Volume and
# direction both describe a player's *passing*; these two describe what the passing was worth.
# `xt_generated` and `shot_involvement` come from the actions (see `features/xthreat.py` and
# `features/chains.py`); `xt_strength_*_norm` come from the edge table once its weights are xT
# sums rather than pass counts. Together they are the last of the three candidate fixes listed
# against the volume-proxy limitation in docs/DATA_SOURCES.md.
#
# `xt_generated` and the two `xt_strength_*` features are fitted quantities: xT is learned from
# data and MUST come from a surface fitted on the training fold only. `shot_involvement` is not
# -- it counts observed possessions and has no fitting step at all.
THREAT_FEATURES: tuple[str, ...] = (
    "xt_generated",
    "shot_involvement",
    "xt_strength_in_norm",
    "xt_strength_out_norm",
)

# `shot_involvement` divides by the team's shot count, which is constant within a team-match and
# therefore never normalises away the player's own touch frequency -- it correlates +0.71 with
# `degree_total`. `shot_conversion` conditions on the player's own involvement instead and drops
# that to +0.04. Added as a fifth feature in its own set rather than folded into THREAT_FEATURES,
# so the published `all+threat` number stays reproducible against the four it was measured on.
CONVERSION_FEATURES: tuple[str, ...] = ("shot_conversion",)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "position": POSITION_FEATURES,
    "topology": TOPOLOGY_FEATURES,
    "both": POSITION_FEATURES + TOPOLOGY_FEATURES,
    # Added later; the three above are left exactly as they were so the published ablation and
    # its +0.73 pp topology contribution stay comparable rather than being silently restated.
    "direction": DIRECTION_FEATURES,
    "topology+direction": TOPOLOGY_FEATURES + DIRECTION_FEATURES,
    "all": POSITION_FEATURES + TOPOLOGY_FEATURES + DIRECTION_FEATURES,
    "threat": THREAT_FEATURES,
    "all+threat": (
        POSITION_FEATURES + TOPOLOGY_FEATURES + DIRECTION_FEATURES + THREAT_FEATURES
    ),
    "all+threat+conv": (
        POSITION_FEATURES
        + TOPOLOGY_FEATURES
        + DIRECTION_FEATURES
        + THREAT_FEATURES
        + CONVERSION_FEATURES
    ),
}


class GraphSAGERoleModel(nn.Module):
    """Two-layer GraphSAGE encoder with a linear role head.

    `forward` returns (logits, embedding) so a single pass serves both the training loss and
    the downstream clustering evaluation -- the embedding is the deliverable, the logits are
    only the training signal.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        n_classes: int = len(COARSE_ROLES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, embedding_dim, aggr="mean")
        self.head = nn.Linear(embedding_dim, n_classes)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        embedding = self.conv2(h, edge_index)
        return self.head(F.relu(embedding)), embedding


@dataclass
class GraphBundle:
    """One PyG graph per team-match, plus the row metadata to join results back."""

    data: list[Data]
    meta: pd.DataFrame
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scaler_std: np.ndarray = field(default_factory=lambda: np.ones(0))


NETWORK_KEYS: tuple[str, ...] = ("game_id", "team_id", "season", "provider")
# A windowed network is identified by its window as well as its team-match. Aggregating without
# `window_index` silently produces full-match values -- see `engineer_node_features`.
WINDOW_KEYS: tuple[str, ...] = NETWORK_KEYS + ("window_index",)


def engineer_node_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    group_keys: tuple[str, ...] = NETWORK_KEYS,
    node_threat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive the topology features from the persisted node/edge tables.

    All features are *within-network normalised* (a player's share of their team's passing)
    rather than raw counts. Raw counts would carry the provider's action-density signature;
    shares are comparable across providers, which is what the 2015/16 -> 2017/18 test needs.

    **`group_keys` decides what "their team's passing" means, and getting it wrong is a silent
    future leak.** Every aggregate below is computed within a group. On the full-match tables the
    default is right. On the *windowed* tables the caller must pass `WINDOW_KEYS`: with the
    match-level default, a player's edge-derived features become their whole-match values repeated
    across all 16 windows, so a model predicting at minute 15 sees passing structure from minute
    90. That was live in Module 3 until a truncation test caught it -- 6 of the 10 topology
    features were full-match.

    It is an explicit argument rather than something inferred from the presence of a
    `window_index` column, because the full-match tables *do* carry that column (entirely null),
    so any auto-detection would have to guess and would fail silently in one direction or the
    other.

    `node_threat` carries the per-player quantities that cannot be derived from the network
    tables alone -- `xt_generated` and `shot_involvement`, both keyed on `group_keys +
    ["player_id"]`. Pass None (the default) and THREAT_FEATURES are simply absent from the
    result; requesting a feature set that needs them then fails on the missing column, which is
    the intended behaviour. Filling them with zeros instead would hand `build_graphs` a
    plausible-looking constant column and produce an ablation row that measures nothing.
    The same applies to `xt_strength_*_norm`, which appear only when `edges` carries an
    `xt_weight` column.
    """
    keys = list(group_keys)
    missing = [k for k in keys if k not in nodes.columns or k not in edges.columns]
    if missing:
        raise KeyError(f"group_keys {missing} absent from the node/edge tables")
    if nodes[keys].isna().any().any():
        raise ValueError(
            f"group_keys {keys} contain nulls, which pandas would silently drop from every "
            "aggregate; pass keys that are fully populated for this table"
        )

    out_agg = (
        edges.groupby(keys + ["source"])
        .agg(strength_out=("weight", "sum"), degree_out=("weight", "size"))
        .reset_index()
        .rename(columns={"source": "player_id"})
    )
    in_agg = (
        edges.groupby(keys + ["target"])
        .agg(strength_in=("weight", "sum"), degree_in=("weight", "size"))
        .reset_index()
        .rename(columns={"target": "player_id"})
    )

    frame = nodes.merge(out_agg, on=keys + ["player_id"], how="left").merge(
        in_agg, on=keys + ["player_id"], how="left"
    )
    for column in ("strength_out", "degree_out", "strength_in", "degree_in"):
        frame[column] = frame[column].fillna(0.0)

    totals = frame.groupby(keys).agg(
        team_strength_out=("strength_out", "sum"),
        team_strength_in=("strength_in", "sum"),
        team_touches=("touches", "sum"),
        team_nodes=("player_id", "size"),
        team_degree_out=("degree_out", "sum"),
        team_degree_in=("degree_in", "sum"),
    ).reset_index()
    frame = frame.merge(totals, on=keys, how="left")

    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return (numerator / denominator.replace(0.0, np.nan)).fillna(0.0)

    # Degree normalised by the maximum possible (team size - 1) so it reads as "what share
    # of team-mates does this player connect to", independent of squad rotation.
    max_degree = (frame["team_nodes"] - 1).clip(lower=1)
    frame["degree_out_norm"] = _safe_ratio(frame["degree_out"], max_degree)
    frame["degree_in_norm"] = _safe_ratio(frame["degree_in"], max_degree)
    frame["strength_out_norm"] = _safe_ratio(frame["strength_out"], frame["team_strength_out"])
    frame["strength_in_norm"] = _safe_ratio(frame["strength_in"], frame["team_strength_in"])
    frame["edge_share_out"] = _safe_ratio(frame["degree_out"], frame["team_degree_out"])
    frame["edge_share_in"] = _safe_ratio(frame["degree_in"], frame["team_degree_in"])
    frame["pass_completion_rate"] = _safe_ratio(
        frame["passes_completed"], frame["passes_attempted"]
    )
    frame["touches"] = _safe_ratio(frame["touches"], frame["team_touches"])

    # ------------------------------------------------------------------ direction features
    #
    # Vectorised on purpose. A groupby-apply computing a weighted mean per player is minutes on
    # the windowed edge table (~473k rows on the Premier League); pre-multiplying and summing is
    # seconds. The weighted mean of an edge attribute is sum(weight * attr) / sum(weight).
    #
    # Note these are the only features here NOT expressed as a share of the team's total:
    # `progression_*` and `pass_length_made` are in metres. The scaler standardises them on the
    # training fold, so absolute scale is handled, but any provider difference in how pass end
    # points are annotated will show up here as a distribution shift rather than being normalised
    # away. `progressive_share` is a share and is the provider-robust member of the group.
    if edges.empty:
        # A team with no completed passes in the window. Zero is the honest reading of "no
        # evidence of direction", and matches how the volume features treat the same case.
        return _attach_threat_features(
            frame.assign(**dict.fromkeys(DIRECTION_FEATURES, 0.0)), edges, keys, node_threat
        )

    weighted = edges[keys + ["source", "target", "weight"]].copy()
    weighted["_w_dx"] = edges["weight"] * edges["mean_dx"]
    weighted["_w_len"] = edges["weight"] * edges["mean_length"]
    weighted["_w_prog"] = edges["weight"] * (edges["mean_dx"] > PROGRESSIVE_DX_METRES)

    made = (
        weighted.groupby(keys + ["source"])
        .agg(_w=("weight", "sum"), _wdx=("_w_dx", "sum"),
             _wlen=("_w_len", "sum"), _wprog=("_w_prog", "sum"))
        .reset_index()
        .rename(columns={"source": "player_id"})
    )
    made["progression_made"] = _safe_ratio(made["_wdx"], made["_w"])
    made["pass_length_made"] = _safe_ratio(made["_wlen"], made["_w"])
    made["progressive_share"] = _safe_ratio(made["_wprog"], made["_w"])

    received = (
        weighted.groupby(keys + ["target"])
        .agg(_w=("weight", "sum"), _wdx=("_w_dx", "sum"))
        .reset_index()
        .rename(columns={"target": "player_id"})
    )
    received["progression_received"] = _safe_ratio(received["_wdx"], received["_w"])

    frame = frame.merge(
        made[keys + ["player_id", "progression_made", "pass_length_made", "progressive_share"]],
        on=keys + ["player_id"], how="left",
    ).merge(
        received[keys + ["player_id", "progression_received"]],
        on=keys + ["player_id"], how="left",
    )
    # A player who received but never made a pass (or vice versa) has no row on one side.
    frame[list(DIRECTION_FEATURES)] = frame[list(DIRECTION_FEATURES)].fillna(0.0)

    return _attach_threat_features(frame, edges, keys, node_threat)


def _attach_threat_features(
    frame: pd.DataFrame,
    edges: pd.DataFrame,
    keys: list[str],
    node_threat: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add whichever of THREAT_FEATURES the inputs actually support, and no more.

    Deliberately silent when an input is absent: the caller gets a frame without those columns
    rather than a frame full of zeros. `build_graphs` then raises on the missing column, which is
    the loud failure this project prefers to a silently useless feature.
    """
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return (numerator / denominator.replace(0.0, np.nan)).fillna(0.0)

    if "xt_weight" in edges.columns and not edges.empty:
        out_xt = (
            edges.groupby(keys + ["source"])["xt_weight"].sum()
            .rename("xt_strength_out").reset_index().rename(columns={"source": "player_id"})
        )
        in_xt = (
            edges.groupby(keys + ["target"])["xt_weight"].sum()
            .rename("xt_strength_in").reset_index().rename(columns={"target": "player_id"})
        )
        frame = frame.merge(out_xt, on=keys + ["player_id"], how="left").merge(
            in_xt, on=keys + ["player_id"], how="left"
        )
        for column in ("xt_strength_out", "xt_strength_in"):
            frame[column] = frame[column].fillna(0.0)

        # A share of the team's xT-weighted strength, for the same reason every volume feature is
        # a share: raw xT sums carry the provider's action-density signature and the match's
        # overall danger, neither of which is a property of the player.
        totals = frame.groupby(keys).agg(
            team_xt_out=("xt_strength_out", "sum"),
            team_xt_in=("xt_strength_in", "sum"),
        ).reset_index()
        frame = frame.merge(totals, on=keys, how="left")
        frame["xt_strength_out_norm"] = _safe_ratio(frame["xt_strength_out"], frame["team_xt_out"])
        frame["xt_strength_in_norm"] = _safe_ratio(frame["xt_strength_in"], frame["team_xt_in"])

    if node_threat is not None:
        wanted = [
            c
            for c in ("xt_generated", "shot_involvement", "shot_conversion")
            if c in node_threat.columns
        ]
        missing_keys = [k for k in keys + ["player_id"] if k not in node_threat.columns]
        if missing_keys:
            raise KeyError(
                f"node_threat is missing join keys {missing_keys}; it must be keyed the same way "
                "as the node table, or the merge silently drops every row"
            )
        before = len(frame)
        frame = frame.merge(
            node_threat[keys + ["player_id"] + wanted], on=keys + ["player_id"], how="left"
        )
        if len(frame) != before:
            raise ValueError(
                f"node_threat join changed the row count ({before} -> {len(frame)}); it has "
                "duplicate rows for some (network, player)"
            )
        # A player with no qualifying action generated no threat and appeared in no shot chain.
        # Here zero IS the measurement, unlike the absent-input case handled above.
        frame[wanted] = frame[wanted].fillna(0.0)

    return frame


def build_graphs(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    feature_set: str,
    label_column: str = "role_index",
    scaler: tuple[np.ndarray, np.ndarray] | None = None,
) -> GraphBundle:
    """Assemble one PyG `Data` per team-match network.

    Standardisation statistics are fit on the training fold only and passed in for val/test
    -- fitting them on the whole corpus would leak the test season's distribution.
    """
    names = FEATURE_SETS[feature_set]
    keys = ["game_id", "team_id", "season", "provider"]

    if scaler is None:
        matrix = features[list(names)].to_numpy(dtype=np.float32)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-6] = 1.0
    else:
        mean, std = scaler

    edge_groups = {k: g for k, g in edges.groupby(keys, sort=False)}

    graphs: list[Data] = []
    meta_rows = []
    for key, group in features.groupby(keys, sort=False):
        group = group.reset_index(drop=True)
        index_of = {int(pid): i for i, pid in enumerate(group["player_id"])}

        matrix = group[list(names)].to_numpy(dtype=np.float32)
        matrix = (matrix - mean) / std

        edge_group = edge_groups.get(key)
        if edge_group is None or edge_group.empty:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            pairs = [
                (index_of[int(s)], index_of[int(t)])
                for s, t in zip(edge_group["source"], edge_group["target"])
                if int(s) in index_of and int(t) in index_of
            ]
            edge_index = (
                torch.tensor(pairs, dtype=torch.long).t().contiguous()
                if pairs
                else torch.zeros((2, 0), dtype=torch.long)
            )

        labels = group[label_column].fillna(-1).to_numpy(dtype=np.int64)
        graphs.append(
            Data(
                x=torch.from_numpy(matrix),
                edge_index=edge_index,
                y=torch.from_numpy(labels),
                num_nodes=len(group),
            )
        )
        meta_rows.append(group.assign(_graph_index=len(graphs) - 1))

    return GraphBundle(
        data=graphs,
        meta=pd.concat(meta_rows, ignore_index=True),
        feature_names=names,
        scaler_mean=mean,
        scaler_std=std,
    )


def train_model(
    train_graphs: list[Data],
    val_graphs: list[Data],
    in_channels: int,
    epochs: int = 60,
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    hidden_channels: int = 64,
    embedding_dim: int = 32,
    patience: int = 12,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[GraphSAGERoleModel, dict[str, list[float]]]:
    """Train with early stopping on validation accuracy.

    Full-batch per graph rather than NeighborLoader: these graphs have ~13 nodes each, so
    neighbour sampling would add overhead and approximation for no memory benefit. The
    sampling machinery earns its place in Module 3, where graph sequences get large.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GraphSAGERoleModel(
        in_channels=in_channels, hidden_channels=hidden_channels, embedding_dim=embedding_dim
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_batch = [g.to(device) for g in train_graphs]
    val_batch = [g.to(device) for g in val_graphs]

    history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}
    best_state, best_acc, stale = None, -1.0, 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_nodes = 0.0, 0
        for graph in train_batch:
            mask = graph.y >= 0
            if not bool(mask.any()):
                continue
            optimiser.zero_grad()
            logits, _ = model(graph.x, graph.edge_index)
            loss = F.cross_entropy(logits[mask], graph.y[mask])
            loss.backward()
            optimiser.step()
            total_loss += float(loss) * int(mask.sum())
            total_nodes += int(mask.sum())

        train_loss = total_loss / max(total_nodes, 1)
        val_acc = evaluate_accuracy(model, val_batch)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best val acc %.4f)", epoch, best_acc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_accuracy(model: GraphSAGERoleModel, graphs: list[Data]) -> float:
    model.eval()
    correct, total = 0, 0
    for graph in graphs:
        mask = graph.y >= 0
        if not bool(mask.any()):
            continue
        logits, _ = model(graph.x, graph.edge_index)
        correct += int((logits[mask].argmax(dim=1) == graph.y[mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


@torch.no_grad()
def extract_embeddings(
    model: GraphSAGERoleModel, bundle: GraphBundle, device: str = "cpu"
) -> np.ndarray:
    """Embedding per node, ordered to match `bundle.meta` row order."""
    model.eval()
    chunks = []
    for graph in bundle.data:
        graph = graph.to(device)
        _, embedding = model(graph.x, graph.edge_index)
        chunks.append(embedding.cpu().numpy())
    return np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def save_checkpoint(
    path, model: GraphSAGERoleModel, bundle: GraphBundle, feature_set: str
) -> None:
    """Persist weights *and* everything needed to reproduce an embedding.

    A bare `state_dict` is not enough. The features are standardised with statistics fit on
    the training fold only, so without those statistics a later forward pass silently uses a
    different input scaling and produces embeddings that do not match the ones the reported
    metrics were computed from. Storing the scaler alongside the weights is what makes the
    saved model reproducible rather than merely loadable.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_set": feature_set,
            "feature_names": list(bundle.feature_names),
            "scaler_mean": np.asarray(bundle.scaler_mean, dtype=np.float32),
            "scaler_std": np.asarray(bundle.scaler_std, dtype=np.float32),
            "hyperparameters": {
                "in_channels": model.conv1.in_channels,
                "hidden_channels": model.conv2.in_channels,
                "embedding_dim": model.head.in_features,
                "n_classes": model.head.out_features,
                "dropout": model.dropout,
            },
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> tuple[GraphSAGERoleModel, dict]:
    """Restore a model plus its scaler and feature names.

    Returns (model in eval mode, metadata) where metadata carries `feature_names`,
    `scaler_mean` and `scaler_std` for passing straight into `build_graphs`.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(
            f"{path} looks like a bare state_dict saved by an older version. Re-run "
            "`python scripts/train_roles.py` to write a checkpoint that includes the "
            "feature scaler, without which embeddings cannot be reproduced."
        )

    hyperparameters = payload["hyperparameters"]
    model = GraphSAGERoleModel(
        in_channels=hyperparameters["in_channels"],
        hidden_channels=hyperparameters["hidden_channels"],
        embedding_dim=hyperparameters["embedding_dim"],
        n_classes=hyperparameters["n_classes"],
        dropout=hyperparameters["dropout"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    metadata = {
        "feature_set": payload["feature_set"],
        "feature_names": tuple(payload["feature_names"]),
        "scaler_mean": np.asarray(payload["scaler_mean"], dtype=np.float32),
        "scaler_std": np.asarray(payload["scaler_std"], dtype=np.float32),
    }
    return model, metadata
