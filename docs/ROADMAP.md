# Roadmap — what remains

Modules 1–4 are implemented and reported in the [README](../README.md). What follows is what
is left, recorded so the constraints already established are not re-litigated later.

## Phases 4-5 — Modules 3 and 4: IMPLEMENTED

Result prediction and tactical pattern discovery are built; see the README for results. Two
findings are worth carrying forward:

- **Module 3's GNN+Transformer loses to B0** because 300 independent training matches cannot
  support a sequence model. If this is ever revisited, the fix is more data (more competitions,
  or a provider licence covering more Serie A seasons) rather than a different architecture.
- **Module 4's set-piece rule still over-segments.** Chains restart on every set piece, which
  splits a single phase of play into fragments. Worth revisiting for pattern work specifically:
  a possession that restarts on a throw-in is often the same attack.

## Module 5 — RL pass choice (BLOCKED)

One-step offline contextual bandit rather than a simulator, framed honestly as an
exploratory value estimator.

- **State** — graph over visible players in a 360 freeze-frame.
- **Action** — which visible team-mate to pass to (≤10 discrete).
- **Reward** — xThreat delta of the resulting position, discounted by a learned completion
  probability.
- **Baselines** — most-advanced team-mate, nearest team-mate, and the pass actually played.
- **Evaluation** — off-policy (IPS / doubly-robust) on held-out matches.

Two hard constraints, both established during planning:

1. **Serie A has no 360 data in either season.** Module 5 needs a different competition;
   Euro 2024 (51 matches, 360 for all) is the candidate. This departs from the Serie A
   framing and the README must say so.
2. **360 freeze-frames are anonymous and partially visible** — a sampled frame held 18 of 22
   players, with only `teammate` / `actor` / `keeper` flags and no player ids. The action
   space is therefore position slots, not named players, and the true recipient must be
   matched to the nearest frame object by `end_location`.

## Module 6 — Coach-facing dashboard (partially built)

The Streamlit app in `app/` already covers Modules 1-4: passing networks with centrality and
functional role, the result-prediction ladder with per-match probability timelines, and the
pattern clusters with shot lift. Still missing: video-timestamp deep links, and the real versus
agent-simulated comparison, which depends on Module 5.

## Cross-cutting

- **Weights & Biases** in offline mode unless `WANDB_API_KEY` is set, so no run blocks on an
  account.
- **Kaggle**: cache the processed SPADL store as a Kaggle Dataset so notebooks never
  re-download 1.4 GB, and checkpoint frequently enough to survive session limits.
- Every module reports wall time and peak memory next to its metrics, via
  `eval.resources.ResourceMonitor`.
- Every predictive model uses `eval.splits.temporal_split` and calls
  `reject_random_split`. There is no sanctioned code path for a random split.
