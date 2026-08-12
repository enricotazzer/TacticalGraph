"""Tests for the committed demo bundle and model checkpointing.

These guard the two ways the demo can quietly become wrong: a bundle that no longer matches
what the pipeline produces, and a checkpoint that loads but cannot reproduce its own
embeddings.

Every test skips cleanly when `demo_data/` is absent, so a fresh clone that has not run
`scripts/export_demo_bundle.py` still gets a green suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tacticalgraph.config import CORPORA
from tacticalgraph.demo.bundle import (
    BUNDLE_DIR,
    BundleMissingError,
    DemoBundle,
    load_bundle,
    verify_manifest,
)

pytestmark = pytest.mark.skipif(
    not (BUNDLE_DIR / "manifest.json").exists(),
    reason="demo bundle not built; run scripts/export_demo_bundle.py",
)


@pytest.fixture(scope="module")
def bundle() -> DemoBundle:
    return load_bundle()


def test_bundle_is_preferred_over_data_root(bundle: DemoBundle):
    """The app must read the committed snapshot even when the drive is attached.

    Otherwise what a reviewer sees depends on local state.
    """
    assert bundle.source == "bundle"
    assert bundle.root == BUNDLE_DIR


def test_manifest_matches_actual_row_counts(bundle: DemoBundle):
    """Catches a stale bundle shipped after the pipeline changed."""
    assert verify_manifest(bundle) == []


def test_manifest_records_provenance(bundle: DemoBundle):
    assert "generated_at" in bundle.manifest
    assert bundle.manifest.get("tables")
    assert "missing_sources" not in bundle.manifest, (
        f"bundle is incomplete: {bundle.manifest.get('missing_sources')}"
    )


def test_core_tables_present_and_joinable(bundle: DemoBundle):
    """Node and edge tables must share the keys the app joins them on.

    Counts are checked against the *declared corpus* rather than hardcoded: the bundle can
    hold either corpus, and a fixed 760 would fail on the Premier League while telling us
    nothing about whether the bundle is self-consistent.
    """
    keys = {"game_id", "team_id", "season", "provider"}
    assert keys <= set(bundle.nodes.columns)
    assert keys <= set(bundle.edges.columns)

    spec = CORPORA[bundle.manifest["corpus"]]
    assert len(bundle.games) == spec.n_matches_expected
    assert bundle.games["provider"].nunique() == len({s.provider for s in spec.seasons})
    assert bundle.nodes["season"].nunique() == len(spec.seasons)
    # Every node's match must exist in the games index, or the app's joins drop rows silently.
    assert set(bundle.nodes["game_id"]) <= set(bundle.games["game_id"])


def test_embeddings_align_with_players(bundle: DemoBundle):
    identity, dims = bundle.embedding_matrix()
    assert len(identity) == len(dims)
    assert dims.shape[1] >= 8
    # Every embedded node should resolve to a known player in the directory.
    players = bundle.players[["season", "provider", "player_id"]].drop_duplicates()
    merged = identity.merge(players, on=["season", "provider", "player_id"], how="left",
                            indicator=True)
    unmatched = (merged["_merge"] == "left_only").sum()
    assert unmatched == 0, f"{unmatched} embedded nodes have no player directory entry"


def test_player_names_have_no_escaped_unicode(bundle: DemoBundle):
    """Regression: the Wyscout dump stores names with literal ``\\uXXXX`` text.

    Left unrepaired, the demo displays 'M. Pjani\\u0107' instead of 'M. Pjanić'. 94 of 1,083
    names were affected before `players._fix_double_encoded` was applied.
    """
    names = bundle.players["player_name"].dropna()
    offenders = names[names.str.contains(r"\\u", regex=True)]
    assert offenders.empty, f"escaped unicode survives in: {offenders.head(5).tolist()}"


def test_missing_bundle_raises_actionable_error(tmp_path, monkeypatch):
    """A missing bundle must name the script that builds it."""
    import tacticalgraph.demo.bundle as module

    monkeypatch.setattr(module, "BUNDLE_DIR", tmp_path / "nope")
    stub = DemoBundle(source="bundle", root=tmp_path / "nope")
    with pytest.raises(BundleMissingError, match="export_demo_bundle"):
        stub.path("games.parquet")


def test_checkpoint_round_trip_reproduces_embeddings(bundle: DemoBundle):
    """A checkpoint must carry its scaler, not just its weights.

    Without the training-fold standardisation statistics a later forward pass silently uses a
    different input scaling, so the embeddings stop matching the reported metrics.
    """
    from tacticalgraph.models.role_gnn import load_checkpoint

    model, metadata = load_checkpoint(bundle.path("role_gnn_both.pt"))
    n_features = len(metadata["feature_names"])
    assert n_features == metadata["scaler_mean"].shape[0]
    assert metadata["scaler_std"].shape[0] == n_features
    assert np.all(metadata["scaler_std"] > 0)

    # Deterministic forward pass on a fixed toy graph.
    torch.manual_seed(0)
    x = torch.arange(3 * n_features, dtype=torch.float32).reshape(3, n_features) / 10.0
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with torch.no_grad():
        first_logits, first_embedding = model(x, edge_index)

    reloaded, _ = load_checkpoint(bundle.path("role_gnn_both.pt"))
    with torch.no_grad():
        second_logits, second_embedding = reloaded(x, edge_index)

    torch.testing.assert_close(first_embedding, second_embedding)
    torch.testing.assert_close(first_logits, second_logits)


def test_bare_state_dict_is_rejected_with_guidance(tmp_path):
    """An old-format checkpoint must fail loudly, not load and mislead."""
    from tacticalgraph.models.role_gnn import GraphSAGERoleModel, load_checkpoint

    path = tmp_path / "legacy.pt"
    torch.save(GraphSAGERoleModel(in_channels=4).state_dict(), path)
    with pytest.raises(ValueError, match="train_roles"):
        load_checkpoint(path)


def test_windowed_sample_has_full_sequences(bundle: DemoBundle):
    """Module 3's page needs complete 16-step sequences to display."""
    nodes = bundle.table("windowed_sample_nodes.parquet")
    assert "window_index" in nodes.columns
    per_match = nodes.groupby("game_id")["window_index"].nunique()
    assert per_match.max() == 16, f"expected 16 windows per match, saw {per_match.max()}"
    spec = CORPORA[bundle.manifest["corpus"]]
    assert nodes["season"].nunique() == len(spec.seasons), (
        "windowed sample should cover every season the corpus declares"
    )


def test_reports_carry_the_published_metrics(bundle: DemoBundle):
    reports = bundle.reports()

    # The harmonisation report only exists for a multi-provider corpus -- it *is* the
    # cross-provider comparison. Requiring it unconditionally would demand a measurement that
    # is undefined on a single-provider corpus.
    spec = CORPORA[bundle.manifest["corpus"]]
    providers = {season.provider for season in spec.seasons}
    if len(providers) > 1:
        assert "harmonization_report" in reports
        harmonisation = reports["harmonization_report"]
        for section in ("recipient_accuracy", "possession", "distribution_shift", "action_mix"):
            assert section in harmonisation, f"missing {section}"

        accuracy = pd.DataFrame(harmonisation["recipient_accuracy"])
        native = accuracy[accuracy["context"] == "statsbomb-native"]["accuracy"].iloc[0]
        assert native > 0.99, f"recipient accuracy regressed to {native}"
    else:
        assert "harmonization_report" not in reports, (
            "a single-provider bundle must not ship a cross-provider harmonisation report -- "
            "it would be describing a different corpus"
        )

    assert any(k.startswith("module2_roles_") for k in reports), "no Module 2 report in bundle"
    assert any(k.startswith("module3_outcome_") for k in reports), "no Module 3 report"
    assert any(k.startswith("module4_patterns_") for k in reports), "no Module 4 report"
