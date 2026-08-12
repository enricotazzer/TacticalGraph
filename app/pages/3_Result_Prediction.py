"""Module 3 -- in-match result prediction. Baseline ladder vs GNN+Transformer."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from _shared import (
    SEASON_LABEL,
    club_label,
    get_bundle,
    page_header,
    sidebar_provenance,
    status_banner,
    table,
)

st.set_page_config(page_title="M3 · Result Prediction", page_icon="📈", layout="wide")

page_header(
    "📈 Module 3 — In-Match Result Prediction",
    "A baseline ladder against a GNN+Transformer over passing-network sequences. The headline "
    "is a negative result, reported as such.",
)
sidebar_provenance()
status_banner(3)

MODEL_ORDER = ["prior", "B0", "B1", "B2", "gnn_transformer"]
MODEL_LABEL = {
    "prior": "prior (class frequencies)",
    "B0": "B0 · scoreline + minutes left",
    "B1": "B1 · + shots, passes, xT",
    "B2": "B2 · + form, network (gbm)",
    "gnn_transformer": "GNN + Transformer",
}


@st.cache_data(show_spinner=False)
def load_reports() -> dict[str, list[dict]]:
    """Module 3 reports across seeds and splits."""
    directory = get_bundle().root / "reports"
    out: dict[str, list[dict]] = {}
    for file in sorted(glob.glob(str(directory / "module3_outcome_*.json"))):
        payload = json.loads(Path(file).read_text())
        out.setdefault(payload["split"]["kind"], []).append(payload)
    return out


reports = load_reports()
if not reports:
    st.error("No Module 3 reports in the bundle. Run `python scripts/train_outcome.py`.")
    st.stop()

split_choice = st.radio(
    "Split",
    sorted(reports.keys()),
    horizontal=True,
    format_func=lambda k: (
        "cross-season (train 2015/16 → test 2017/18, CONFOUNDED)"
        if k == "cross_season"
        else "within-season control (2015/16 only, unconfounded)"
    ),
)
runs = reports[split_choice]
primary = runs[0]

st.caption(primary["split"]["description"])

# ============================================================== headline
st.subheader("The result")

comparison = pd.DataFrame(primary["comparison"])
paired = pd.DataFrame(primary["paired_vs_b0"])

# Average the metric over seeds; the tabular rungs are deterministic, the GNN is not.
stacked = pd.concat([pd.DataFrame(r["comparison"]).assign(seed=r["seed"]) for r in runs])
averaged = (
    stacked.groupby("model")
    .agg(log_loss=("log_loss", "mean"), log_loss_std=("log_loss", "std"),
         brier=("brier", "mean"), accuracy=("accuracy", "mean"), ece=("ece", "mean"),
         seeds=("seed", "nunique"))
    .reindex([m for m in MODEL_ORDER if m in stacked["model"].unique()])
    .reset_index()
)
averaged["log_loss_std"] = averaged["log_loss_std"].fillna(0.0)
averaged["model"] = averaged["model"].map(MODEL_LABEL).fillna(averaged["model"])

left, right = st.columns([3, 2])
with left:
    st.dataframe(
        averaged.round(4).rename(
            columns={"model": "Model", "log_loss": "log-loss", "log_loss_std": "± std",
                     "brier": "Brier", "accuracy": "accuracy", "ece": "ECE",
                     "seeds": "seeds"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Mean over {averaged['seeds'].max()} seeds. Lower log-loss is better. The tabular "
        "rungs are deterministic, so their std is 0."
    )

with right:
    best = averaged.iloc[0]
    st.metric("Best model", str(best["model"]), f"log-loss {best['log_loss']:.4f}")
    gnn_row = paired[paired["model"] == "gnn_transformer"]
    if not gnn_row.empty:
        row = gnn_row.iloc[0]
        st.metric(
            "GNN vs B0 (paired Δ log-loss)",
            f"{row['delta_log_loss']:+.4f}",
            f"CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
            delta_color="inverse",
        )

st.error(
    "**The GNN+Transformer is significantly worse than B0, in every seed and both splits.** "
    "Diagnosis is in the training curves below: with **300 independent training matches** (the "
    "16 checkpoints of one match are one observation, not sixteen) a 13k–77k parameter "
    "sequence model memorises the training set. Train loss fell from 1.30 to 0.59 while "
    "validation loss rose from 0.89 to 2.55. Three fixes were tried — lower learning rate with "
    "longer patience, a residual path so the model can trivially recover the tabular baseline, "
    "and a capacity sweep selected on validation — and none changed the conclusion."
)

st.info(
    "**B1 is the champion on the cross-season split** (log-loss 0.7075 vs B0's 0.7504, paired "
    "Δ −0.043, CI [−0.067, −0.018]): shots, passes and accumulated xThreat do add real "
    "information over the scoreline.\n\n"
    "**But that advantage disappears on the unconfounded within-season control** (Δ +0.018, CI "
    "[−0.048, +0.088] — B0 actually wins). With 60 test matches the intervals are wide, so the "
    "honest reading is that B1's gain is specific to the cross-season setting rather than a "
    "general improvement. Only the GNN's deficit is consistent across both splits."
)

# ============================================================== paired test
st.subheader("Is anything actually better than B0?")
st.markdown(
    "The paired bootstrap resamples **whole matches**, not rows. Resampling rows would treat "
    "one match's 16 correlated checkpoints as 16 independent observations and shrink every "
    "interval by roughly 4×, manufacturing significance."
)
paired_display = paired.copy()
paired_display["model"] = paired_display["model"].map(MODEL_LABEL).fillna(paired_display["model"])
paired_display = paired_display.rename(
    columns={"delta_log_loss": "Δ log-loss vs B0", "ci_low": "CI low", "ci_high": "CI high",
             "significant": "CI excludes 0"}
)
st.dataframe(paired_display, hide_index=True, width="stretch")
st.caption("Negative Δ = better than B0. A CI containing zero means indistinguishable from B0.")

# ============================================================== per checkpoint
st.subheader("Log-loss over the course of a match")

per_checkpoint = primary["per_checkpoint"]
curve = pd.DataFrame(
    {"checkpoint": [r["checkpoint_minute"] for r in per_checkpoint["B0"]]}
)
for model in MODEL_ORDER:
    if model in per_checkpoint:
        curve[model] = [r["log_loss"] for r in per_checkpoint[model]]

figure, ax = plt.subplots(figsize=(9, 4.2))
for model in [m for m in MODEL_ORDER if m in curve.columns]:
    ax.plot(curve["checkpoint"], curve[model], marker="o", ms=3.5, label=MODEL_LABEL[model])
ax.set_xlabel("checkpoint (minute the 15-minute window closes)")
ax.set_ylabel("log-loss (lower is better)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title("Prediction sharpens as the match progresses")
st.pyplot(figure, width="stretch")
plt.close(figure)
st.caption(
    "Reporting only a full-match average would blur two regimes: early prediction is nearly "
    "uninformative, late prediction is nearly determined by the scoreline."
)

# ============================================================== calibration
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Calibration of the best model")
    reliability = pd.DataFrame(primary["reliability_best"])
    if not reliability.empty:
        figure, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
        ax.plot(reliability["mean_confidence"], reliability["observed_accuracy"],
                marker="o", color="#1f6feb", label="observed")
        for _, row in reliability.iterrows():
            ax.annotate(f"n={int(row['n'])}", (row["mean_confidence"], row["observed_accuracy"]),
                        fontsize=6, xytext=(3, -8), textcoords="offset points")
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("observed accuracy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(figure, width="content")
        plt.close(figure)

with col_b:
    st.subheader("What carries the signal")
    importance = pd.DataFrame(primary["b2_importance"]).head(8)
    st.dataframe(
        importance.rename(columns={"feature": "Feature", "importance": "Permutation importance"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Sanity check: `goal_diff` dominating is exactly right. Anything outranking it would "
        "mean the feature layer is broken."
    )

# ============================================================== training curves
if primary.get("gnn_history", {}).get("train_loss"):
    st.subheader("Why the graph model fails: the overfitting is unambiguous")
    history = primary["gnn_history"]
    figure, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(history["train_loss"], label="train", color="#2ca02c")
    ax.plot(history["val_loss"], label="validation", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Training loss falls while validation loss rises — 300 independent labels")
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    sweep = primary.get("gnn_capacity_sweep") or []
    if sweep:
        st.markdown("**Capacity sweep, selected on validation and never on test:**")
        st.dataframe(pd.DataFrame(sweep), hide_index=True, width="stretch")

# ============================================================== match timeline
st.divider()
st.subheader("Probability timeline for a single match")

try:
    predictions = table("module3_test_predictions.parquet")
except Exception:
    predictions = pd.DataFrame()

if predictions.empty:
    st.caption("Per-match predictions are not in this bundle.")
else:
    games = table("games.parquet").set_index("game_id")

    def label(game_id: int) -> str:
        if game_id not in games.index:
            return str(game_id)
        row = games.loc[game_id]
        return (
            f"MW{int(row['game_day']):02d} · {club_label(row['provider'], row['home_team_id'])} "
            f"{int(row['home_score'])}-{int(row['away_score'])} "
            f"{club_label(row['provider'], row['away_team_id'])}"
        )

    available = sorted(predictions["game_id"].unique())
    chosen = st.selectbox("Match (test fold, 2017/18)", available, format_func=label)
    match = predictions[predictions["game_id"] == chosen].sort_values("window_index")

    model_pick = st.radio(
        "Model", [c[:-7] for c in match.columns if c.endswith("_p_home")],
        horizontal=True, format_func=lambda m: MODEL_LABEL.get(m, m),
    )

    figure, ax = plt.subplots(figsize=(9, 3.8))
    for outcome, colour in (("home", "#1f6feb"), ("draw", "#888888"), ("away", "#d62728")):
        ax.plot(match["checkpoint_minute"], match[f"{model_pick}_p_{outcome}"],
                marker="o", ms=3.5, color=colour, label=outcome)
    ax.set_ylim(0, 1)
    ax.set_xlabel("minute")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    truth = ["home win", "draw", "away win"][int(match["outcome_index"].iloc[0])]
    ax.set_title(f"{label(chosen)} — actual result: {truth}")
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.caption(
        "Goal difference at each checkpoint: "
        + ", ".join(
            f"{int(r.checkpoint_minute)}′ {int(r.goal_diff):+d}" for r in match.itertuples()
        )
    )

# ============================================================== method
st.divider()
with st.expander("Method and the leakage rule"):
    st.markdown(
        "**Checkpoints** are the 16 window ends (15′, 20′ … 90′), identical to the graph "
        "model's windows so both are scored on the same support.\n\n"
        "**The leakage rule**: every feature at checkpoint *t* is computable from actions with "
        "`minute <= t` and nothing else. That rules out full-match aggregates *and* full-match "
        "network metrics — B2's structural features come from the window that has just closed. "
        "`tests/test_match_state.py` enforces this by rebuilding each feature row from a "
        "truncated action stream and asserting it is unchanged.\n\n"
        "**Causality** in the Transformer is enforced by an additive `-inf` mask and verified "
        "empirically: perturbing windows 12–15 leaves predictions 0–11 bit-identical.\n\n"
        "**xThreat** is fitted on training games only.\n\n"
        "**One caveat on the labels**: the derived scoreline reproduces the recorded final "
        f"score for {primary['goal_derivation_check']['exact_match']}/"
        f"{primary['goal_derivation_check']['games_checked']} games "
        f"({primary['goal_derivation_check']['match_rate']:.1%}). The two failures are Wyscout "
        "matches where a goal is simply absent from the event stream; no phantom goal was "
        "inserted to paper over it."
    )
