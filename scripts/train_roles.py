#!/usr/bin/env python
"""Phase 3 -- GraphSAGE functional-role embeddings vs the classical centrality baseline.

Trains the three feature ablations (position / topology / both), then evaluates every
representation -- including the Phase 2 centrality vector -- on the same clustering protocol
and the same external signals.

    python scripts/train_roles.py
    python scripts/train_roles.py --split within_season   # unconfounded control

Outputs under DATA_ROOT/reports and DATA_ROOT/figures.
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
from tacticalgraph.data.aliases import match_players  # noqa: E402
from tacticalgraph.data.enrichment import load_enrichment  # noqa: E402
from tacticalgraph.data.players import load_player_directory  # noqa: E402
from tacticalgraph.data.roles import ROLE_TO_INDEX  # noqa: E402
from tacticalgraph.data.schema import assert_no_enrichment_leakage  # noqa: E402
from tacticalgraph.data.spadl_store import read_games  # noqa: E402
from tacticalgraph.eval.clustering import (  # noqa: E402
    cluster_and_score,
    compare_representations,
    cross_season_stability,
    half_season_stability,
    within_player_consistency,
)
from tacticalgraph.eval.resources import ResourceMonitor, device_label  # noqa: E402
from tacticalgraph.eval.splits import (  # noqa: E402
    reject_random_split,
    stratified_report,
    temporal_split,
)
from tacticalgraph.features.centrality import PLAYER_METRICS  # noqa: E402
from tacticalgraph.models.role_gnn import (  # noqa: E402
    FEATURE_SETS,
    build_graphs,
    engineer_node_features,
    evaluate_accuracy,
    extract_embeddings,
    save_checkpoint,
    train_model,
)

log = logging.getLogger("train_roles")


def load_features(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Node features with role labels attached, plus the edge table."""
    nodes = pd.read_parquet(paths.networks / "full_nodes.parquet")
    edges = pd.read_parquet(paths.networks / "full_edges.parquet")
    features = engineer_node_features(nodes, edges)

    directory = load_player_directory(paths)
    features = features.merge(
        directory[["season", "provider", "player_id", "player_name", "coarse_role"]],
        on=["season", "provider", "player_id"],
        how="left",
    )
    features["role_index"] = features["coarse_role"].map(ROLE_TO_INDEX).astype("Int64")

    # The 24-class position is validation-only. Attached under an explicit name and
    # excluded from every feature set; `assert_no_enrichment_leakage` is the mechanical
    # guarantee that it never reaches the model.
    lineups = load_enrichment(paths, "statsbomb_lineup_positions")
    features = features.merge(
        lineups[["game_id", "player_id", "position_name_24"]],
        on=["game_id", "player_id"],
        how="left",
    )

    features = features[features["role_index"].notna()].reset_index(drop=True)
    log.info(
        "features: %d nodes, %d graphs, roles %s",
        len(features),
        features.groupby(["game_id", "team_id"]).ngroups,
        features["coarse_role"].value_counts().to_dict(),
    )
    return features, edges


def run_ablation(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    feature_set: str,
    folds: dict[str, pd.DataFrame],
    epochs: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    """Train one feature-set variant and return metrics plus its test embedding."""
    # Fit standardisation on train only; val/test reuse those statistics.
    train_bundle = build_graphs(folds["train"], edges, feature_set)
    scaler = (train_bundle.scaler_mean, train_bundle.scaler_std)
    val_bundle = build_graphs(folds["val"], edges, feature_set, scaler=scaler)
    test_bundle = build_graphs(folds["test"], edges, feature_set, scaler=scaler)

    for name, bundle in (("train", train_bundle), ("val", val_bundle), ("test", test_bundle)):
        assert_no_enrichment_leakage(
            bundle.meta[list(bundle.feature_names)], context=f"{feature_set}/{name} features"
        )

    with ResourceMonitor(f"gnn-{feature_set}") as monitor:
        model, history = train_model(
            train_bundle.data,
            val_bundle.data,
            in_channels=len(train_bundle.feature_names),
            epochs=epochs,
            device=device,
            seed=seed,
        )

    metrics = {
        "feature_set": feature_set,
        "n_features": len(train_bundle.feature_names),
        "features": list(train_bundle.feature_names),
        "train_acc": round(evaluate_accuracy(model, [g.to(device) for g in train_bundle.data]), 4),
        "val_acc": round(evaluate_accuracy(model, [g.to(device) for g in val_bundle.data]), 4),
        "test_acc": round(evaluate_accuracy(model, [g.to(device) for g in test_bundle.data]), 4),
        "epochs_run": len(history["train_loss"]),
        "resources": monitor.as_dict(),
    }
    log.info(
        "%-9s train %.3f | val %.3f | test %.3f (%d features, %.1fs)",
        feature_set,
        metrics["train_acc"],
        metrics["val_acc"],
        metrics["test_acc"],
        metrics["n_features"],
        monitor.seconds,
    )

    # One bundle over every node, reusing the train-fold scaler. Serves both the clustering
    # evaluation and the persisted embedding the demo app reads.
    all_bundle = build_graphs(features, edges, feature_set, scaler=scaler)

    return {
        "metrics": metrics,
        "model": model,
        "bundle": all_bundle,
        "test_embedding": extract_embeddings(model, test_bundle, device=device),
        "test_meta": test_bundle.meta,
        "all_embedding": extract_embeddings(model, all_bundle, device=device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=ALL_SPLIT_KINDS, default=None,
        help="split kind; defaults to the corpus's primary kind",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
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
    log.info("device: %s (requested %s)", device_label(), args.device)

    features, edges = load_features(paths)
    games = read_games(paths)
    split = temporal_split(games, kind=args.split, corpus=args.corpus)

    features["fold"] = split.assign(features["game_id"])
    # Prove no match straddles two folds. With ~13 rows per match, a random split would be
    # invisible in the metrics but would inflate every one of them.
    reject_random_split(features["fold"], features["game_id"])
    print()
    print(stratified_report(features).to_string(index=False))

    folds = {
        name: features[features["fold"] == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    for name, frame in folds.items():
        if frame.empty:
            raise RuntimeError(f"fold {name!r} is empty; check the split definition")

    # ---------------------------------------------------------------- ablations
    results = {}
    for feature_set in FEATURE_SETS:
        results[feature_set] = run_ablation(
            features, edges, feature_set, folds, args.epochs, args.device, args.seed
        )

    ablation = pd.DataFrame([r["metrics"] for r in results.values()])
    print()
    print("=" * 78)
    print("ABLATION -- role classification accuracy (4-class)")
    print("=" * 78)
    print(
        ablation[["feature_set", "n_features", "train_acc", "val_acc", "test_acc", "epochs_run"]]
        .to_string(index=False)
    )
    print()
    print(
        "topology-vs-position gap on test: "
        f"{ablation.set_index('feature_set').loc['both', 'test_acc'] - ablation.set_index('feature_set').loc['position', 'test_acc']:+.4f} "
        "(both - position)"
    )

    # ------------------------------------------------- clustering comparison
    # `engineer_node_features` and the centrality table both define degree_*/strength_*,
    # so namespace the baseline columns instead of letting pandas silently suffix them.
    centrality = pd.read_parquet(paths.networks / "centrality_players.parquet")
    keys = ["game_id", "team_id", "season", "provider", "player_id"]
    centrality = centrality[keys + list(PLAYER_METRICS)].rename(
        columns={metric: f"cent_{metric}" for metric in PLAYER_METRICS}
    )
    baseline_columns = [f"cent_{metric}" for metric in PLAYER_METRICS]

    joined = features.merge(centrality, on=keys, how="left")
    if len(joined) != len(features):
        raise RuntimeError(
            f"centrality join changed row count ({len(features)} -> {len(joined)}); "
            "duplicate keys in centrality_players.parquet"
        )
    baseline_matrix = joined[baseline_columns].to_numpy(dtype=np.float64)

    tables = [
        cluster_and_score(
            baseline_matrix,
            joined["coarse_role"],
            joined["position_name_24"],
            label="centrality (baseline)",
        )
    ]
    for feature_set, payload in results.items():
        tables.append(
            cluster_and_score(
                payload["all_embedding"],
                features["coarse_role"],
                features["position_name_24"],
                label=f"gnn-{feature_set}",
            )
        )
    comparison = compare_representations(tables)

    print()
    print("=" * 78)
    print("CLUSTERING: GNN embedding vs classical centrality")
    print("ari/nmi_fine24 = agreement with StatsBomb 24-class position (NEVER trained on)")
    print("=" * 78)
    print(comparison.to_string(index=False))

    # ------------------------------------------------- stability diagnostics
    consistency = [
        within_player_consistency(
            baseline_matrix, joined["player_id"], label="centrality (baseline)"
        )
    ]
    for feature_set, payload in results.items():
        consistency.append(
            within_player_consistency(
                payload["all_embedding"], features["player_id"], label=f"gnn-{feature_set}"
            )
        )
    consistency_frame = pd.DataFrame(consistency)
    print()
    print("## Within-player consistency (same player, different matches)")
    print(consistency_frame.to_string(index=False))

    # A two-provider corpus can compare seasons; a single-season one compares the two halves
    # of its own season instead. The latter is the cleaner measure -- no provider change to
    # confound it -- so the corpus picks the check rather than one being skipped as "n/a".
    multi_provider = len(paths.spec.seasons) > 1
    if multi_provider:
        directory = load_player_directory(paths)
        pairs = match_players(
            directory[directory["provider"] == "statsbomb"],
            directory[directory["provider"] == "wyscout"],
            sb_name_col="player_name",
            wy_name_col="player_name",
        )
        stability = [
            cross_season_stability(
                baseline_matrix, joined, pairs, label="centrality (baseline)"
            )
        ]
        for feature_set, payload in results.items():
            stability.append(
                cross_season_stability(
                    payload["all_embedding"], features, pairs, label=f"gnn-{feature_set}"
                )
            )
        heading = "## Cross-season stability (2015/16 StatsBomb vs 2017/18 Wyscout)"
        caveat = "   NB: conflates role stability with provider robustness -- see README"
        stability_kind = "cross_season"
    else:
        stability = [
            half_season_stability(
                baseline_matrix, joined, games, label="centrality (baseline)"
            )
        ]
        for feature_set, payload in results.items():
            stability.append(
                half_season_stability(
                    payload["all_embedding"], features, games, label=f"gnn-{feature_set}"
                )
            )
        heading = f"## Half-season stability ({paths.spec.label}, wk1-19 vs wk20-38)"
        caveat = (
            "   One provider, one competition: a low score is the representation's fault, "
            "not a provider change's"
        )
        stability_kind = "half_season"

    stability_frame = pd.DataFrame(stability)
    print()
    print(heading)
    print(caveat)
    print(stability_frame.to_string(index=False))

    # ------------------------------------------------------------- persistence
    report = {
        "split": {"kind": split.name, "description": split.description},
        "ablation": ablation.drop(columns=["features"]).to_dict(orient="records"),
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "clustering": comparison.to_dict(orient="records"),
        "within_player_consistency": consistency_frame.to_dict(orient="records"),
        "stability_kind": stability_kind,
        "stability": stability_frame.to_dict(orient="records"),
        "resources": [r["metrics"]["resources"] for r in results.values()],
    }
    dest = paths.reports / f"module2_roles_{args.split}_seed{args.seed}.json"
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {dest}")

    best = max(results, key=lambda k: results[k]["metrics"]["val_acc"])
    checkpoint_path = paths.models / f"role_gnn_{best}.pt"
    save_checkpoint(checkpoint_path, results[best]["model"], results[best]["bundle"], best)
    log.info("saved best model (%s) -> %s", best, checkpoint_path)

    # Persist the embedding so the demo app never needs a forward pass at page-load time.
    # `all_embedding` is row-aligned with `features` by construction (build_graphs preserves
    # group order in bundle.meta), so join on the identity columns rather than on position.
    embedding = results[best]["all_embedding"]
    meta = results[best]["bundle"].meta
    if len(meta) != len(embedding):
        raise RuntimeError(
            f"embedding rows ({len(embedding)}) != meta rows ({len(meta)}); cannot persist"
        )
    embedding_frame = meta[
        ["game_id", "team_id", "season", "provider", "player_id"]
    ].reset_index(drop=True)
    for dimension in range(embedding.shape[1]):
        embedding_frame[f"e{dimension:02d}"] = embedding[:, dimension].astype("float32")
    embedding_frame["feature_set"] = best
    embedding_frame.to_parquet(paths.models / "role_embeddings.parquet", index=False)
    log.info(
        "saved %d x %d embeddings -> %s",
        len(embedding_frame),
        embedding.shape[1],
        paths.models / "role_embeddings.parquet",
    )

    _plot_embedding(paths, results, features, args.split)
    return 0


def _plot_embedding(
    paths: Paths, results: dict, features: pd.DataFrame, split_name: str
) -> None:
    """UMAP of the `both` embedding, coloured by coarse role and by 24-class position."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError as exc:
        log.warning("skipping embedding plot: %s", exc)
        return

    embedding = results["both"]["all_embedding"]
    reduced = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=0).fit_transform(
        np.nan_to_num(embedding)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for role in ("GK", "DEF", "MID", "FWD"):
        mask = (features["coarse_role"] == role).to_numpy()
        axes[0].scatter(reduced[mask, 0], reduced[mask, 1], s=3, alpha=0.45, label=role)
    axes[0].legend(markerscale=4, fontsize=9)
    axes[0].set_title("GNN embedding (feature set: both)\ncoloured by 4-class role (trained)")

    fine = features["position_name_24"]
    top = fine.value_counts().head(10).index
    for position in top:
        mask = (fine == position).to_numpy()
        axes[1].scatter(reduced[mask, 0], reduced[mask, 1], s=3, alpha=0.5, label=position)
    axes[1].legend(markerscale=4, fontsize=6.5, ncol=2)
    axes[1].set_title(
        "same embedding\ncoloured by StatsBomb 24-class position (never trained on)"
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    dest = paths.figures / f"role_embedding_{split_name}.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", dest)


if __name__ == "__main__":
    raise SystemExit(main())
