# Roadmap — Phases 4–8

Phases 1–3 are implemented and reported in the [README](../README.md). What follows is
scoped but **not built**. Recorded here so the design decisions already forced by the Phase
1 findings are not re-litigated later.

## Phase 4 — Tabular baseline for result prediction

LightGBM predicting the 3-class final result at each 5-minute checkpoint. The baseline
ladder matters more than the model:

- **B0 — scoreline + minutes remaining only.** In-match result prediction is dominated by
  the current score. Omitting B0 would make every later comparison dishonest, because a
  graph model that beats "possession and shot counts" but not "the scoreline" has
  demonstrated nothing.
- **B1** — B0 plus aggregates (possession, shots, accumulated xT).
- **B2** — full tabular feature set, including rolling pre-match form computed *only* from
  earlier matchweeks.

Metrics: log-loss, Brier score, accuracy and reliability diagrams, reported **per
checkpoint**, not just at full time. Calibration is a first-class result.

Constraint inherited from Phase 1: features must be built from
`schema.PROVIDER_COMPARABLE_TYPES`. Counting `dribble` or `bad_touch` would encode the
provider (8.7× and 296× rate gaps respectively) and collapse on the 2017/18 test season.

**Label availability — verified, with one gap to close first.** The 2015/16 match index
carries `home_score` / `away_score` directly, but they are **null for all 380 Wyscout
matches**. They are recoverable 380/380 from `raw/wyscout/matches_Italy.json`
(`teamsData[*].score`, cross-checkable against the `label` and `winner` fields), so
`adapters.load_games` needs a small backfill before this phase can train. The *running*
scoreline needed for B0 is derivable from SPADL — a shot with `result == success` plus
`owngoal` credited to the opposing side — which yields 2.50 (2015/16) and 2.57 (2017/18) goals
per match against a Serie A actual of ≈2.6, so the derivation is sound for both providers.

**Checkpoint grid decision.** Prefer aligning checkpoints to the **16 window ends** (15', 20',
… 90') rather than 18 five-minute marks: every checkpoint then has a complete 15-minute graph
behind it, so the tabular baseline and the graph model are evaluated on the same support.

## Phase 5 — GNN + Transformer over graph sequences

The windowed networks already exist: 24,320 of them, 15-minute window on a 5-minute stride,
16 steps per match, averaging 11.5 nodes and 39.5 edges. The window length was chosen for
exactly this — a 5-minute window would leave ~13 edges per graph, mostly isolated nodes.

Per window, a GraphSAGE encoder produces a team embedding for both sides; the sequence of
embeddings plus scalar match state feeds a small causal Transformer encoder (2 layers,
4 heads, d=64) with a per-timestep 3-class head. Must beat B2 at every checkpoint to be
worth keeping.

Neighbour sampling becomes worthwhile here, unlike in Module 2 where the graphs are ~13
nodes and full-batch is cheaper.

## Phase 6 — Recurring tactical patterns

Cluster possession-chain embeddings into styles of play (build-up from the back, fast
transition, slow possession). Report P(shot | pattern) with `game_id` and timestamp so a
coach can jump to video.

Caveat to carry forward: the possession reconstruction over-segments by ~25% versus
StatsBomb's native counter (ARI 0.83, boundary Jaccard 0.62), because every set-piece is
treated as a hard restart. Chain-level modelling should revisit that rule, since a
possession that restarts on a throw-in is arguably the same phase of play.

## Phase 7 — RL: pass choice in build-up

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

## Phase 8 — Coach-facing dashboard

Streamlit. Passing network with centrality and functional role highlighted; result
prediction timeline; pattern report with video timestamps; real versus agent-simulated
scenario comparison.

## Cross-cutting

- **Weights & Biases** in offline mode unless `WANDB_API_KEY` is set, so no run blocks on an
  account.
- **Kaggle**: cache the processed SPADL store as a Kaggle Dataset so notebooks never
  re-download 1.4 GB, and checkpoint frequently enough to survive session limits.
- Every module reports wall time and peak memory next to its metrics, via
  `eval.resources.ResourceMonitor`.
- Every predictive model uses `eval.splits.temporal_split` and calls
  `reject_random_split`. There is no sanctioned code path for a random split.
