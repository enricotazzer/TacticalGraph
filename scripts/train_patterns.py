#!/usr/bin/env python
"""Module 4 -- recurring tactical patterns from possession chains.

    python scripts/train_patterns.py
    python scripts/train_patterns.py --k 8 --split within_season

Clusters possession chains two ways -- hand-crafted features (the mandatory baseline) and a
self-supervised GRU autoencoder latent -- and measures both against the corpus shot base rate
with Wilson intervals, plus cross-season stability.

Writes DATA_ROOT/reports/module4_patterns_<split>.json, the labelled chain table for the demo
app, and the encoder checkpoint.
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
import torch  # noqa: E402

from tacticalgraph.config import ALL_SPLIT_KINDS, CORPORA, DEFAULT_CORPUS, Paths  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions, read_games  # noqa: E402
from tacticalgraph.eval.patterns import (  # noqa: E402
    compare_representations,
    cross_season_stability,
    fit_clustering,
    shot_lift,
    sweep_k,
)
from tacticalgraph.eval.resources import ResourceMonitor  # noqa: E402
from tacticalgraph.eval.splits import temporal_split  # noqa: E402
from tacticalgraph.features.chains import (  # noqa: E402
    CHAIN_FEATURES,
    build_chain_table,
    chain_sequences,
    cluster_profiles,
)
from tacticalgraph.models.chain_encoder import (  # noqa: E402
    encode_all,
    train_chain_encoder,
)

log = logging.getLogger("train_patterns")


def fit_xthreat(actions: pd.DataFrame, train_game_ids: set[int]) -> pd.Series:
    """xThreat fitted on training games only, so chain xT gain carries no test-season info."""
    from socceraction.xthreat import ExpectedThreat

    model = ExpectedThreat(l=16, w=12)
    model.fit(actions[actions["game_id"].isin(train_game_ids)])
    return pd.Series(model.rate(actions), index=actions.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=ALL_SPLIT_KINDS, default=None,
        help="split kind; defaults to the corpus's primary kind",
    )
    parser.add_argument("--k", type=int, default=8, help="k for the reported clustering")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-encoder", action="store_true")
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

    games = read_games(paths)
    actions = read_actions(paths)
    split = temporal_split(games, kind=args.split, corpus=args.corpus)

    xt = fit_xthreat(actions, split.train)

    with ResourceMonitor("chain-table") as chain_monitor:
        chains = build_chain_table(actions, xt_values=xt)

    fold = split.assign(chains["game_id"])
    chains["fold"] = fold.to_numpy()
    train_mask = (chains["fold"] == "train").to_numpy()
    test_mask = (chains["fold"] == "test").to_numpy()

    print()
    print("=" * 82)
    print(f"MODULE 4 -- possession patterns ({args.split})")
    print("=" * 82)
    print(f"chains: {len(chains):,} (>=3 comparable actions)")
    print(f"base shot rate: {chains['ends_in_shot'].mean():.4f}")
    print(
        f"folds: {int(train_mask.sum()):,} train / {int(test_mask.sum()):,} test chains "
        f"({int(chains['game_id'].nunique())} games)"
    )
    print()
    print("chain length by season (comparable types only -- provider ratio ~0.90):")
    print(
        chains.groupby("season")["n_actions"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    baseline_features = chains[list(CHAIN_FEATURES)].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ learned encoder
    latent = None
    encoder_history: dict[str, list[float]] = {}
    encoder_resources = None
    if not args.skip_encoder:
        sequences, lengths, vocabulary = chain_sequences(actions, chains)
        with ResourceMonitor("chain-encoder") as encoder_monitor:
            encoder, encoder_history = train_chain_encoder(
                sequences,
                lengths,
                train_mask,
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
            )
            latent = encode_all(encoder, sequences, lengths, device=args.device)
        encoder_resources = encoder_monitor.as_dict()
        torch.save(
            {
                "state_dict": encoder.state_dict(),
                "n_token_features": sequences.shape[2],
                "max_length": sequences.shape[1],
                "latent_dim": encoder.latent_dim,
                "vocabulary": vocabulary,
            },
            paths.models / "chain_encoder.pt",
        )
        log.info("saved %s", paths.models / "chain_encoder.pt")

    # ------------------------------------------------------------------ k sweep
    sweeps = [
        sweep_k(baseline_features, chains, train_mask, test_mask,
                label="hand-crafted (baseline)", seed=args.seed)
    ]
    if latent is not None:
        sweeps.append(
            sweep_k(latent, chains, train_mask, test_mask,
                    label="GRU autoencoder", seed=args.seed)
        )
    sweep = compare_representations(sweeps)

    print()
    print("## k sweep -- separation and shot discrimination (test fold)")
    print(sweep.to_string(index=False))

    # ------------------------------------------------------------------ reported clustering
    results: dict[str, object] = {}
    for label, features in [("hand-crafted", baseline_features)] + (
        [("gru-autoencoder", latent)] if latent is not None else []
    ):
        labels, _, _ = fit_clustering(features, train_mask, args.k, seed=args.seed)
        chains[f"cluster_{label}"] = labels

        profile = cluster_profiles(chains, labels)
        lift_test = shot_lift(chains, labels, test_mask)
        stability = cross_season_stability(chains, labels, train_mask, test_mask)

        print()
        print(f"### {label}, k={args.k} -- cluster profiles")
        print(
            profile[["cluster", "label", "n_chains", "share_of_chains", "shot_rate",
                     "n_actions", "duration_s", "start_x", "directness", "xt_gain"]]
            .to_string(index=False)
        )
        print()
        print(f"### {label} -- P(shot | cluster) on the TEST fold vs base rate")
        print(lift_test.to_string(index=False))
        print()
        print(f"### {label} -- cross-season stability")
        print(stability.to_string(index=False))

        results[label] = {
            "profiles": profile.to_dict(orient="records"),
            "shot_lift_test": lift_test.to_dict(orient="records"),
            "stability": stability.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------ persistence
    report = {
        "split": {"kind": split.name, "description": split.description},
        "k": args.k,
        "seed": args.seed,
        "n_chains": int(len(chains)),
        "base_shot_rate": round(float(chains["ends_in_shot"].mean()), 4),
        "chain_length_by_season": chains.groupby("season")["n_actions"]
        .agg(["count", "mean", "median"])
        .round(3)
        .to_dict(orient="index"),
        "k_sweep": sweep.to_dict(orient="records"),
        "clusterings": results,
        "encoder_history": encoder_history,
        "resources": [
            r for r in (chain_monitor.as_dict(), encoder_resources) if r
        ],
    }
    destination = paths.reports / f"module4_patterns_{args.split}.json"
    destination.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {destination}")

    # Written for the corpus's *primary* split rather than the literal "cross_season", which
    # only exists on a two-season corpus -- gating on the string meant a single-season corpus
    # produced a report but no chain table, and the demo bundle then silently lost its
    # pattern-browsing page.
    if args.split == CORPORA[args.corpus].split_kinds[0]:
        keep = [
            "game_id", "possession_id", "season", "provider", "team_id", "fold",
            "start_minute", "end_minute", "period_id", "n_actions", "duration_seconds",
            "start_x", "start_y", "end_x", "end_y", "net_dx", "directness", "xt_gain",
            "started_with_set_piece", "width_used", "share_final_third",
            "ends_in_shot", "start_zone", "end_zone",
        ] + [c for c in chains.columns if c.startswith("cluster_")]
        out = paths.models / "module4_chains.parquet"
        chains[keep].to_parquet(out, index=False)
        log.info("wrote %s (%d chains)", out, len(chains))

    print()
    print(
        "NOTE: cluster names are generated from the same profile numbers shown beside them, "
        "so they describe rather than validate. The human review step is "
        "`scripts/review_patterns.py`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
