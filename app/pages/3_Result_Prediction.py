"""Module 3 -- in-match result prediction. Baseline ladder vs GNN+Transformer."""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from _shared import (
    SPLIT_LABELS,
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
    "A baseline ladder against a GNN+Transformer over passing-network sequences. Originally a "
    "clear negative result; an optimiser fix turned most of it into a draw.",
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


#: Canonical report filename: module3_outcome_<split>_seed<n>.json and nothing more. Ablation
#: runs carry a `--tag` suffix (e.g. `..._seed0_batch1.json`), and those must NOT be mixed in:
#: they share a `split.kind` with the canonical run, so a plain glob would silently average two
#: different training configurations into one set of "the" results.
CANONICAL_REPORT = re.compile(r"^module3_outcome_(?P<split>[a-z_]+)_seed\d+\.json$")


@st.cache_data(show_spinner=False)
def load_reports() -> dict[str, list[dict]]:
    """Module 3 reports across seeds and splits, canonical runs only."""
    directory = get_bundle().root / "reports"
    out: dict[str, list[dict]] = {}
    for file in sorted(glob.glob(str(directory / "module3_outcome_*.json"))):
        if not CANONICAL_REPORT.match(Path(file).name):
            continue
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
    format_func=lambda k: SPLIT_LABELS.get(k, k),
)
runs = reports[split_choice]
primary = runs[0]

st.caption(primary["split"]["description"])

# ============================================================== headline
st.subheader("The result")

comparison = pd.DataFrame(primary["comparison"])


# Every number in the two banners below is read from the loaded reports. They used to be
# hardcoded from the Serie A run, which would state Serie A figures over Premier League data
# once a second corpus existed -- the exact misattribution this app is supposed to prevent.
def _paired_row(
    runs_for_split: list[dict], model: str, key: str = "paired_vs_b0"
) -> dict | None:
    """Mean paired delta for one model across seeds, with the widest CI seen.

    `key` selects the reference: `paired_vs_b0` (the scoreline) or `paired_vs_b1` (the best
    tabular model). Reports written before the residual arm existed carry only the former.
    """
    rows = [
        r for run in runs_for_split for r in run.get(key, []) if r["model"] == model
    ]
    if not rows:
        return None
    return {
        "delta": float(np.mean([r["delta_log_loss"] for r in rows])),
        "ci_low": float(min(r["ci_low"] for r in rows)),
        "ci_high": float(max(r["ci_high"] for r in rows)),
        "n_significant": sum(bool(r["significant"]) for r in rows),
        "n_runs": len(rows),
    }


# Which arm produced these reports. A residual-on-B1 run is initialised as B1, so B1 -- not B0 --
# is the reference every headline on this page has to quote.
arm = primary.get("gnn_arm", {})
residual = arm.get("baseline_residual") == "b1"

# Every paired figure on this page goes through `_paired_row`/`_paired_table`, which aggregate
# across all seeds of the split. Reading `primary["paired_vs_b0"]` directly used to show seed 0
# alone beside a 3-seed mean comparison table, printing two different deltas for one quantity.

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
    # Lowest log-loss, not the first row. `averaged` is reindexed by MODEL_ORDER, which is ladder
    # order starting at `prior` -- so `iloc[0]` labelled the class prior as the best model on
    # every split.
    best = averaged.loc[averaged["log_loss"].idxmin()]
    st.metric("Best model", str(best["model"]), f"log-loss {best['log_loss']:.4f}")
    headline_reference = "B1" if residual else "B0"
    row = _paired_row(runs, "gnn_transformer", key=f"paired_vs_{headline_reference.lower()}")
    if row is None:  # a pre-residual report has only the B0 comparison
        headline_reference, row = "B0", _paired_row(runs, "gnn_transformer")
    if row is not None:
        st.metric(
            f"GNN vs {headline_reference} (paired Δ log-loss)",
            f"{row['delta']:+.4f}",
            f"CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
            delta_color="inverse",
        )

gnn_paired = _paired_row(runs, "gnn_transformer")
history = primary.get("gnn_history", {})

gnn_vs_b1 = _paired_row(runs, "gnn_transformer", key="paired_vs_b1")

if gnn_paired and history.get("train_loss"):
    train_curve, val_curve = history["train_loss"], history["val_loss"]
    # `best_epoch` is recorded by the trainer, where -1 means "no epoch beat the untrained
    # model". Recomputing it from the curve here would report a trained epoch in exactly the
    # case that matters most -- a residual run that never improved on its frozen baseline.
    best_epoch = int(history.get("best_epoch", val_curve.index(min(val_curve))))

    # B1 is the reference that matters whenever the model runs as a residual on it: it starts
    # *as* B1, so "better than B0" comes almost for free and would flatter the model. Fall back
    # to the B0 comparison only for reports written before that arm existed.
    reference = gnn_vs_b1 if (residual and gnn_vs_b1) else gnn_paired
    reference_name = "B1" if reference is gnn_vs_b1 else "B0"
    n_sig, n_runs = reference["n_significant"], reference["n_runs"]
    beats = reference["delta"] < 0

    if n_sig == n_runs:
        verdict, box = (
            f"**significantly worse than {reference_name} in {n_sig}/{n_runs} runs**", st.error
        )
    elif n_sig:
        verdict, box = (
            f"**worse than {reference_name} in {n_sig}/{n_runs} runs**, indistinguishable in "
            "the rest",
            st.warning,
        )
    elif beats:
        verdict, box = (
            f"**indistinguishable from {reference_name}** — the point estimate favours the graph "
            "model, but the interval spans zero",
            st.info,
        )
    else:
        verdict, box = (
            f"**indistinguishable from {reference_name}** — on the wrong side of zero, but not "
            "significantly so",
            st.warning,
        )

    message = (
        f"The GNN+Transformer is {verdict} on this split: paired Δ "
        f"{reference['delta']:+.4f} log-loss, CI [{reference['ci_low']:+.3f}, "
        f"{reference['ci_high']:+.3f}] over {n_runs} seed(s). Train loss "
        f"{train_curve[0]:.2f} → {min(train_curve):.2f}; validation best at epoch "
        f"{best_epoch}, then rising to {max(val_curve):.2f}."
    )
    if residual:
        message += (
            "\n\n**The model starts *as* B1 and learns a correction to it.** A fitted B1 is "
            "frozen, its log-probabilities are added to the output, and the graph head is "
            "zero-initialised — so at step 0 the prediction is exactly B1's, and early stopping "
            "can always fall back to it. That makes B1 the floor rather than something the "
            "network has to rediscover by gradient descent, and it is why Δ vs B1 is the number "
            "quoted here: against B0 the residual model would look good for free."
        )
        if best_epoch < 0:
            message += (
                " **On this split no epoch improved on that starting point**, so what is "
                "reported is B1 itself."
            )
    box(message)

b1_here = _paired_row(runs, "B1")
if b1_here:
    beats = b1_here["delta"] < 0
    others = []
    for other_split, other_runs in sorted(reports.items()):
        if other_split == split_choice:
            continue
        row = _paired_row(other_runs, "B1")
        if row:
            others.append(
                f"On `{other_split}` it is Δ {row['delta']:+.4f}, CI "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
                f"{' (significant)' if row['n_significant'] else ' (spans zero)'}."
            )
    st.info(
        f"**B1 vs B0 on this split: Δ {b1_here['delta']:+.4f} log-loss, CI "
        f"[{b1_here['ci_low']:+.3f}, {b1_here['ci_high']:+.3f}]** — "
        + (
            "B1 is ahead"
            if beats
            else "B0 is ahead"
        )
        + (
            ", and the interval excludes zero, so shots, passes and accumulated xThreat add "
            "real information over the scoreline."
            if b1_here["n_significant"]
            else ", but the interval spans zero, so the two are indistinguishable here."
        )
        + ("\n\n" + " ".join(others) if others else "")
        + "\n\nRead the splits against each other: an advantage that appears only on the "
        "confounded cross-season split is a property of that setting, not a general "
        "improvement. The GNN's deficit is the one result consistent everywhere."
    )

# ============================================================== paired test
st.subheader("Is anything actually better than the baselines?")
st.markdown(
    "The paired bootstrap resamples **whole matches**, not rows. Resampling rows would treat "
    "one match's 16 correlated checkpoints as 16 independent observations and shrink every "
    "interval by roughly 4×, manufacturing significance."
)


def _paired_table(reference: str) -> pd.DataFrame | None:
    """One row per model: mean Δ against `reference`, widest CI, and how many seeds were sig."""
    key = f"paired_vs_{reference.lower()}"
    rows = [
        {"model": model, **row}
        for model in MODEL_ORDER
        if model != reference and (row := _paired_row(runs, model, key=key)) is not None
    ]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["model"] = frame["model"].map(MODEL_LABEL).fillna(frame["model"])
    frame["significant runs"] = (
        frame["n_significant"].astype(str) + " / " + frame["n_runs"].astype(str)
    )
    return frame[["model", "delta", "ci_low", "ci_high", "significant runs"]].rename(
        columns={"model": "Model", "delta": f"Δ log-loss vs {reference} (mean)",
                 "ci_low": "CI low", "ci_high": "CI high"}
    )


# Both references are shown, and which one *matters* depends on the arm. A residual-on-B1 model
# starts at B1, so beating B0 costs it nothing and only Δ vs B1 tests the graph's contribution.
b1_table = _paired_table("B1")
tabs = st.tabs(["vs B1 (the best tabular model)", "vs B0 (the scoreline)"]) if b1_table is not None \
    else [st.container()]
with tabs[0]:
    if b1_table is not None:
        st.dataframe(b1_table.round(4), hide_index=True, width="stretch")
        if residual:
            st.caption(
                "**This is the comparison that matters for this arm.** The model is initialised "
                "as B1, so a favourable Δ vs B0 would be inherited rather than earned."
            )
    else:
        st.caption("This report predates the Δ-vs-B1 comparison.")
if len(tabs) > 1:
    with tabs[1]:
        b0_table = _paired_table("B0")
        if b0_table is not None:
            st.dataframe(b0_table.round(4), hide_index=True, width="stretch")

st.caption(
    "Negative Δ = better than the reference. Δ is the mean over seeds and the interval is the "
    "**widest** seen across them, so it is deliberately conservative. `significant runs` counts "
    "how many seeds produced a CI excluding zero — one run out of three is weak evidence, not a "
    "result."
)

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
