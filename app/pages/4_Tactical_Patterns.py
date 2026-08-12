"""Module 4 -- recurring tactical patterns from possession chains."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from _shared import (
    club_label,
    get_bundle,
    page_header,
    sidebar_provenance,
    status_banner,
    table,
)
from tacticalgraph.viz.pitch import PITCH_KWARGS

st.set_page_config(page_title="M4 · Tactical Patterns", page_icon="🔁", layout="wide")

page_header(
    "🔁 Module 4 — Recurring Tactical Patterns",
    "Possession chains clustered two ways, measured by how often each pattern precedes a shot.",
)
sidebar_provenance()
status_banner(4)


@st.cache_data(show_spinner=False)
def load_reports() -> dict[str, dict]:
    directory = get_bundle().root / "reports"
    out = {}
    for file in sorted(glob.glob(str(directory / "module4_patterns_*.json"))):
        payload = json.loads(Path(file).read_text())
        out[payload["split"]["kind"]] = payload
    return out


reports = load_reports()
if not reports:
    st.error("No Module 4 reports in the bundle. Run `python scripts/train_patterns.py`.")
    st.stop()

split_choice = st.radio(
    "Split",
    sorted(reports.keys()),
    horizontal=True,
    format_func=lambda k: (
        "cross-season (train 2015/16 → test 2017/18)" if k == "cross_season"
        else "within-season control (2015/16 only)"
    ),
)
report = reports[split_choice]

# ============================================================== headline
top = st.columns(4)
top[0].metric("Possession chains", f"{report['n_chains']:,}")
top[1].metric("Base shot rate", f"{report['base_shot_rate']:.1%}")
top[2].metric("Clusters (k)", report["k"])
best_lift = max(
    (c["lift"] for name in report["clusterings"]
     for c in report["clusterings"][name]["shot_lift_test"]),
    default=float("nan"),
)
top[3].metric("Best shot lift", f"{best_lift:.2f}×")

st.success(
    f"**Clustering finds patterns that strongly discriminate shot-ending possessions.** Against "
    f"a base rate of {report['base_shot_rate']:.1%}, the hand-crafted representation produces a "
    "cluster with a **56–68% shot rate** and another with **0.3%** — a spread of more than 4× "
    "above and 40× below base. Nearly every cluster differs from the base rate by more than "
    "sampling noise (Wilson intervals)."
)

st.warning(
    "**The interpretable baseline beats the learned encoder**, at every k. Hand-crafted chain "
    "features reach a 4.8× max lift on the cross-season test fold against the GRU "
    "autoencoder's 3.4×. Inspecting the latent clusters shows why: three of eight are almost "
    "purely set-piece-initiated (93–95%), so the autoencoder is largely clustering *how a "
    "possession started* — the action-type one-hot dominates its reconstruction loss — rather "
    "than how it developed. This mirrors Module 2, where simple positional features also beat "
    "the learned alternative."
)

# ============================================================== base rate note
with st.expander("Why the base rate is 12.4% and not 9.7%"):
    st.markdown(
        "Across all **186,318** reconstructed chains, 9.7% contain a shot. This module clusters "
        "only chains with **at least 3 provider-comparable actions** (109,912 of them), and "
        "longer possessions are likelier to produce a shot — so the base rate for the "
        "population actually being clustered is **12.4%**. Every lift on this page is measured "
        "against 12.4%; using the unfiltered 9.7% would inflate them.\n\n"
        "The 3-action minimum exists because a one- or two-action possession has no sequence "
        "structure to model. The comparable-types filter exists because raw chain length "
        "differs between the two providers by 1.44× (StatsBomb logs carries as dribbles); "
        "filtering brings that to 0.90×, so the clustering is not partly clustering the "
        "provider."
    )

# ============================================================== k sweep
st.subheader("Choosing k, and which representation wins")
sweep = pd.DataFrame(report["k_sweep"])
left, right = st.columns([3, 2])
with left:
    st.dataframe(
        sweep.rename(
            columns={
                "representation": "Representation", "silhouette": "Silhouette",
                "shot_rate_spread_test": "shot-rate spread (test)",
                "shot_rate_spread_train": "spread (train)",
                "max_lift_test": "max lift (test)",
                "clusters_differing_test": "clusters ≠ base",
            }
        ),
        hide_index=True,
        width="stretch",
    )
with right:
    figure, ax = plt.subplots(figsize=(5, 3.6))
    for name, group in sweep.groupby("representation"):
        ax.plot(group["k"], group["max_lift_test"], marker="o", label=name)
    ax.axhline(1.0, ls="--", color="#888", lw=1)
    ax.set_xlabel("k")
    ax.set_ylabel("max shot lift (test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

# ============================================================== clusters
st.divider()
representation = st.radio(
    "Representation", sorted(report["clusterings"].keys()), horizontal=True
)
clustering = report["clusterings"][representation]

st.subheader("Cluster profiles")
profiles = pd.DataFrame(clustering["profiles"])
st.dataframe(
    profiles[["cluster", "label", "n_chains", "share_of_chains", "shot_rate", "n_actions",
              "duration_s", "start_x", "directness", "xt_gain", "set_piece"]]
    .rename(columns={"label": "auto-generated name", "n_actions": "mean actions",
                     "duration_s": "mean seconds", "start_x": "mean start x",
                     "set_piece": "set-piece share"}),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Names are generated from the same profile numbers shown beside them, ranked relative to "
    "the other clusters. They are a reading aid, **not** validation — see the review section."
)

st.subheader("P(shot | cluster) on the test fold, with Wilson intervals")
lift = pd.DataFrame(clustering["shot_lift_test"])

figure, ax = plt.subplots(figsize=(9, 4))
order = lift.sort_values("shot_rate")
positions = np.arange(len(order))
ax.barh(positions, order["shot_rate"], color="#1f6feb", alpha=0.8)
ax.errorbar(
    order["shot_rate"], positions,
    xerr=[order["shot_rate"] - order["ci_low"], order["ci_high"] - order["shot_rate"]],
    fmt="none", ecolor="#22223b", capsize=3, lw=1,
)
ax.axvline(order["base_rate"].iloc[0], ls="--", color="#d62728",
           label=f"base rate {order['base_rate'].iloc[0]:.3f}")
ax.set_yticks(positions)
ax.set_yticklabels([f"cluster {int(c)}" for c in order["cluster"]], fontsize=8)
ax.set_xlabel("P(chain ends in a shot)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="x")
st.pyplot(figure, width="stretch")
plt.close(figure)

st.dataframe(
    lift.rename(columns={"shot_rate": "P(shot)", "ci_low": "CI low", "ci_high": "CI high",
                         "differs_from_base": "differs from base"}),
    hide_index=True,
    width="stretch",
)

# ============================================================== stability
st.subheader("Cross-season stability")
st.markdown(
    "A pattern that only exists in one provider's data is an artefact. Each cluster is refit on "
    "the training split and applied unchanged to the held-out season."
)
stability = pd.DataFrame(clustering["stability"])
stable = int(stability["rate_stable"].sum())
st.dataframe(stability, hide_index=True, width="stretch")

if split_choice == "cross_season":
    st.info(
        f"**{stable} of {len(stability)} clusters keep a statistically indistinguishable shot "
        "rate across the provider change.** On the within-season control it is 8 of 8 for both "
        "representations — so the instability seen here is attributable to the season/provider "
        "shift rather than to the clustering being arbitrary. This is exactly what the control "
        "split is for."
    )
else:
    st.success(
        f"**{stable} of {len(stability)} clusters are stable** within a single season and "
        "provider, which is the expected result and the reference point for reading the "
        "cross-season numbers."
    )

# ============================================================== examples
st.divider()
st.subheader("Browse real possessions from a cluster")

try:
    chains = table("module4_chains_sample.parquet")
except Exception:
    chains = pd.DataFrame()

column = f"cluster_{representation}"
if chains.empty or column not in chains.columns:
    st.caption(f"No sampled chains for {representation} in this bundle.")
else:
    labels = dict(zip(profiles["cluster"], profiles["label"]))
    cluster_pick = st.selectbox(
        "Cluster",
        sorted(chains[column].dropna().unique().astype(int)),
        format_func=lambda c: f"cluster {c} — {labels.get(c, '')}",
    )
    only_shots = st.checkbox("Only possessions that ended in a shot", value=False)

    pool = chains[chains[column] == cluster_pick]
    if only_shots:
        pool = pool[pool["ends_in_shot"]]

    if pool.empty:
        st.caption("No chains match that filter.")
    else:
        games = table("games.parquet").set_index("game_id")
        sample = pool.sample(min(len(pool), 6), random_state=0)

        from mplsoccer import Pitch

        figure, axes = plt.subplots(2, 3, figsize=(13.5, 6))
        for ax, chain in zip(np.atleast_1d(axes).ravel(), sample.itertuples(index=False)):
            pitch = Pitch(**PITCH_KWARGS)
            pitch.draw(ax=ax)
            # Only the chain's summary geometry is in the bundle (start/end), so the arrow is
            # the net displacement rather than the full action path -- the full path needs the
            # action store, which the bundle deliberately excludes.
            pitch.arrows(chain.start_x, chain.start_y, chain.end_x, chain.end_y,
                         width=2.4, headwidth=4.5, headlength=4.5,
                         color="#d62728" if chain.ends_in_shot else "#1f6feb", ax=ax)
            pitch.scatter([chain.start_x], [chain.start_y], s=80, color="#2ca02c",
                          edgecolors="#22223b", zorder=4, ax=ax)
            fixture = ""
            if chain.game_id in games.index:
                row = games.loc[chain.game_id]
                fixture = (
                    f"{club_label(row['provider'], row['home_team_id'])} v "
                    f"{club_label(row['provider'], row['away_team_id'])}"
                )
            ax.set_title(
                f"{fixture}\n{chain.start_minute:.0f}′ · {int(chain.n_actions)} actions · "
                f"{'SHOT' if chain.ends_in_shot else 'no shot'}",
                fontsize=7,
            )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption(
            "Green dot = where the possession began; arrow = net displacement; red = ended in a "
            "shot. Full action-by-action paths are in "
            "`figures/pattern_clusters/` (produced by `scripts/review_patterns.py`)."
        )

# ============================================================== review status
st.divider()
st.subheader("Human validation status")

review_files = sorted(
    glob.glob(str(get_bundle().root / "reports" / "pattern_review_sheet_*.csv"))
)
if not review_files:
    st.warning("No review sheet in this bundle. Run `python scripts/review_patterns.py`.")
else:
    rows = []
    for file in review_files:
        sheet = pd.read_csv(file)
        verdicts = sheet["sensible_y_n"].astype(str).str.strip().str.lower()
        judged = verdicts.isin(["y", "n"])
        rows.append(
            {
                "sheet": Path(file).stem.replace("pattern_review_sheet_", ""),
                "chains sampled": len(sheet),
                "clusters": sheet["cluster"].nunique(),
                "judged": int(judged.sum()),
                "judged sensible": int((verdicts == "y").sum()),
                "proportion sensible": (
                    round((verdicts == "y").sum() / judged.sum(), 3) if judged.any() else None
                ),
            }
        )
    status = pd.DataFrame(rows)
    st.dataframe(status, hide_index=True, width="stretch")

    if status["judged"].sum() == 0:
        st.warning(
            "**The human review step of the validation plan is PENDING.** The project's own "
            "plan calls for sampling patterns and judging whether they are tactically sensible. "
            "That needs a person: the sheets and pitch figures are generated and waiting, with "
            "the `sensible_y_n` column blank. No proportion is reported here because none has "
            "been recorded — this page will not imply a review happened when it did not."
        )
    else:
        st.success(
            f"Human review recorded for {int(status['judged'].sum())} sampled chains; "
            f"{status['proportion sensible'].iloc[0]:.0%} judged sensible."
        )
