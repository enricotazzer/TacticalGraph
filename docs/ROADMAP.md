# Roadmap — what remains

Modules 1–4 are implemented and reported in the [README](../README.md), now on two corpora.
What follows is what is left, recorded so the constraints already established are not
re-litigated later. Data availability is documented separately in
[DATA_SOURCES.md](DATA_SOURCES.md) — read that before proposing a corpus change.

## Corpus strategy (settled)

- **`premier_league`** (380 matches, StatsBomb, matchweek split) is the primary corpus for
  Modules 1–4. One provider, one complete season, so no provider confound.
- **`serie_a`** (760 matches, StatsBomb + Wyscout) is retained as a cross-*provider*
  generalisation study, which is the only question it can answer cleanly.
- **Tracking is out of scope.** No open tracking exists at league-season scale. If it is ever
  added, PFF FC's World Cup 2022 release (64 matches, 30 fps, identified players) is the
  candidate, because it overlaps StatsBomb's WC 2022 events and 360 exactly.

## Phases 4-5 — Modules 3 and 4: IMPLEMENTED

Result prediction and tactical pattern discovery are built on both corpora; see the README.
Findings worth carrying forward:

- **Module 3's negative result was four bugs, not a finding.** Each fix changed the conclusion; the current one is at the end of this entry.

  The published claim was "the GNN+Transformer is significantly worse than B0 in all 9 runs
  because 300 independent training matches cannot support a sequence model". Both halves were
  wrong, and they were falsified in this order:

  **(a) Data scarcity was not the cause.** `scripts/estimate_ceiling.py` measured the learning
  curve on the pooled 1,140-match corpus: B0 **plateaus** by ~280 training matches (doubling to
  560 moves it −0.002 to −0.004, inside subsample noise) and the total headroom below B0 is only
  **~0.037** log-loss. The deficit was +0.15 to +0.24 — four to six times that. No amount of
  extra data could have closed it.

  **(b) `optimiser.step()` was inside the per-match loop**, so batch size was literally 1:
  ~260-300 updates per epoch, each from one match's 16 heavily-correlated checkpoints. Best
  validation epoch was 0 or 1 in 6 of 8 runs. Batching 16 matches per step, and encoding all
  window graphs in one PyG pass, moved "significantly worse than B0" from **9 of 9 runs to 1 of
  9** — and on Serie A cross-season the point estimate is now *better* than B0 (Δ −0.0168, CI
  spanning zero). Wall time fell 663 s → 145 s for the same 6-config sweep. A controlled arm with
  an identical config budget confirms the gain is the batching, not a wider search.

  A third claim also had to be withdrawn: the sweep's consistent preference for the *smallest*
  capacity was reported as evidence the corpus could not support a larger model. With batching
  capacity becomes roughly **neutral** — the spread between the best 5k, 13k and 77k config falls
  from 0.25–0.28 to a mean of 0.038 log-loss, and across 9 canonical runs the winner is 13k four
  times, 5k three times, 77k twice. The 77k model went from consistently *worst* to competitive,
  so the old preference was measuring gradient noise, not the corpus.

  **A fourth bug found after those fixes changed the answer again: the graph node features leaked
  the future.** `engineer_node_features` aggregated edges without `window_index` while Module 3
  called it on the windowed tables, so **7 of 10 node features were full-match values repeated
  across all 16 windows**. Fixing it moved the graph model from "worse than B1 everywhere" to
  **indistinguishable from B1 on both unconfounded splits with the point estimate in its favour**
  (Δ −0.0119 on PL matchweek, −0.0042 within-season, 0/3 significant each), and made it the best
  model on the primary corpus (0.7795 vs B1's 0.7913). Seed spread fell 31–45× (±0.068 → ±0.002).

  The effect was the *opposite* of the prediction: leakage is meant to flatter a model. These six
  features were constant along the sequence, so a Transformer got no signal from them — only spent
  capacity and a fold-specific offset that did not transfer. Note the existing causality test could
  never have caught it: it perturbs graphs already handed to the model, so it is blind to feature
  construction. `tests/test_outcome_gnn.py` now carries a truncation test that provably fails on
  the old behaviour.

  **The two fixes below were built before that discovery. Neither rescued the model on its own.**

  1. **Residual on a fitted B1 — built, and it is the decisive experiment.** B1 is fitted (reusing
     `fit_ladder`'s existing per-fold predictions, so it is provably the same B1 the report
     compares against), frozen, and its clipped log-probabilities added to the output with a
     zero-initialised head. The model therefore *is* B1 at step 0 and learns only a correction.
     It is the right experimental design and it is what makes Δ vs B1 interpretable, but on its own
     (with the node features still leaking) it produced Δ vs B1 = +0.0278 / +0.0056 / +0.1795 and
     the wrong conclusion. Post-leak-fix the same design gives **−0.0119 / +0.0264 / −0.0042**.
     Keep the design; the numbers above it in this section are the ones to quote.

     A third fix fell out of building it: early stopping only ever scored the model *after* each
     epoch, so the initial state — the one equal to B1 — could never win, and "B1 is the floor" was
     unenforceable. The trainer now scores the untrained model and reports `best_epoch = -1` when
     nothing beats it (3 of 72 configurations).
  2. **Checkpoint weighting — built, measured, and not justified.** Three schemes (`uniform`,
     `linear`, `b0_signal` = inverse of B0's per-checkpoint train log-loss), each selected on
     validation with all reported metrics left unweighted. The control `uniform` won 5 of 9
     selections, margins were 0.004–0.038 (noise), and the two runs where `b0_signal` won selection
     produced the *worst* two test scores of the nine. Kept as a validation-selected option;
     **not** an improvement.
  3. **Seed variance is no longer the problem it was.** ±0.0022 on PL and ±0.0033 on within-season
     after the leak fix, down from ±0.068 and ±0.147; best-epoch is consistent (3–6 on PL). The
     earlier instability was the leak, not the optimiser. The model still overfits after its
     validation minimum.
  4. **A protocol problem worth recording.** The validation and test folds do not rank the
     baselines the same way. On PL, B1 scores 0.8763 on validation (behind B0's 0.8249) and 0.7913
     on test (ahead of B0). Config selection on this module is therefore selecting against a fold
     that misranks the thing being measured — worth remembering before reading much into any single
     sweep winner.

  **Process note, and it is the most transferable thing in this file.** Module 3's headline
  conclusion has now been rewritten four times. Every single revision was caused by a defect in
  this repository, and not one by new data:

  | claim | falsified by |
  |---|---|
  | "significantly worse than B0 in 9/9; 300 labels are too few" | `optimiser.step()` inside the per-match loop |
  | "the corpus cannot support capacity" | that same batch size of 1 |
  | "cannot beat B1 even when handed it; variance is inherent" | 7 of 10 node features were full-match |

  The evaluation methodology was sound the whole time — temporal splits, per-match bootstrap,
  validation-only sweeps, `reject_random_split`, a truncation test on the state table — and it
  caught **none** of them. Each was found by reading or exercising the code, not by a metric looking
  wrong; two of the three were found only because a number looked *surprising* and got chased.

  The specific gap worth generalising: the causality test guarded the *model* (does the mask leak?)
  while the leak sat in the *features handed to it*. A test that starts downstream of the bug cannot
  see it. `tests/test_outcome_gnn.py` went from 0 tests before this work to 34, and the two that
  matter most — the state-table and node-feature truncation tests — are both of the form "rebuild
  from truncated history and assert nothing changed". That shape is worth applying to every new
  feature this project adds.

  Also measured: **pooling corpora is safe and mildly helpful** (B0 −0.005 to −0.011, B1 −0.017
  on the Premier League fold, B2 −0.17 to −0.23), so cross-provider training needs no domain
  adaptation to be worth doing.
- **Module 2: passing *volume* adds almost nothing over position, passing *direction* adds a lot.**
  The ten `TOPOLOGY_FEATURES` are all volume measures and buy +1.00 pp (Premier League) to +1.07 pp
  (Serie A within-season) over pitch position, inside their own seed spread. Adding the four
  `DIRECTION_FEATURES` takes it to **+2.65 pp / +2.53 pp**, five to ten times the seed spread —
  the first non-trivial contribution from graph structure in this project. The original conclusion
  ("graph topology adds almost nothing for role identification") was true of the features that
  existed, not of graphs.
- **Module 2: pass *value* adds nothing, and that is informative.** The follow-up to direction was
  four threat features — `xt_generated`, `shot_involvement`, and in/out strength on xT-weighted
  edges. `all+threat` beats `all` by **+0.08 pp** (Premier League) and **+0.03 pp**
  (within-season), against seed spreads of 0.25–0.65 pp, and is **worse** on cross-season. A null.
  `threat` alone is the weakest feature set in the project (0.6215, below `topology`'s 0.7538).
  The reading: direction helped because it is *geometric* and the 4-class target is a question
  about pitch location; value is orthogonal to that. The features stay because the centrality
  leaderboard needs them, not because the classifier does — see the graph-gaps section below for
  what they did and did not fix there.
- **Module 4's set-piece rule still over-segments.** Chains restart on every set piece, which
  splits one phase of play into fragments. A possession that restarts on a throw-in is often
  the same attack.

## Known gaps in the graph representation

Raised during review. Item 1's four candidate fixes are now all built and measured; items 2-4 are
not. These are real limitations, not speculation — the measurements behind them are in
[DATA_SOURCES.md](DATA_SOURCES.md).

1. **Pass-only edges make centrality a volume proxy — all four fixes built, and the limitation
   stands.** Every candidate has now been tried: pass direction, role-relative z-scoring,
   xT-weighted edges and shot-chain involvement. Direction is a real win for the *classifier*;
   none of the four makes the *leaderboard* measure tactical importance rather than volume.
   Reweighting changes which position is favoured — `pagerank_xt` swaps a 76% midfielder top-50
   for a 90% forward one — and z-scoring evens the mix by construction. The conclusion to carry
   forward is that this is not a weighting problem: a graph whose only relation is "passed to"
   describes ball circulation, and ball circulation is positional. A different *relation* is
   needed, and the two candidates for it (defensive actions, off-ball movement) are bounded by
   what event data records.

   **The cause is now measured rather than inferred.**
   `features/centrality.residualise_against_position` regresses each metric on mean pitch position
   with a quadratic basis (volume peaks in midfield, so a linear fit would leave that arch in the
   residual and understate the very thing being tested). xT weighting makes a metric **more**
   positional in 6 of 7 cases — `pagerank` R² 0.416 → **0.773** — because xThreat is itself a
   spatial surface. Two things did help and are worth keeping: **`strength_out_xt`**, the one
   metric that got *less* positional (0.368 → 0.244) and the best non-mechanical metric produced
   here; and **residualisation itself**, which improves 6 of 7 metrics' role composition.

   The baseline it is all measured against: midfielders take **84% of the top 50 by degree**
   against a 31-33% population share, and goalkeepers take 0% on all ten metrics. It replicates on
   the single-provider Premier League corpus, so it is not a harmonisation artefact. Per fix:
   - **Pass progressiveness — BUILT, and it is the biggest win here.** `DIRECTION_FEATURES` in
     `models/role_gnn.py` derives progression made/received, mean length and progressive share
     from `mean_dx`/`mean_length`, which had been on the edge table since Module 1 and unused.
     Module 2's contribution over pitch position goes from +1.00 pp (`both`) to **+2.65 pp**
     (`all`) on the Premier League, +2.53 pp on the within-season control and +5.74 pp on the
     confounded split — five to ten times the seed spread, where the old volume-only gap sat
     inside it. Direction separates what volume cannot: forwards *receive* +13.7 m passes, keepers
     *send* +30.3 m ones, and both can share a degree with anyone.
     **They do not transfer to Module 3.** The same four features on the window graphs give
     0.8213 ± 0.1078 against the volume arm's 0.7795 ± 0.0022 — but per seed it is 0.9457 / 0.7565
     / 0.7617, i.e. the two *best* individual runs in the project plus one collapse that drags the
     mean. Module 2 averages these over a whole match (~40 passes per player); Module 3 averages
     them per 15-minute window (~5-10 passes), where the same statistic is mostly noise. `volume`
     stays the Module 3 default; `--node-features volume+direction` runs the other arm. Whether
     the instability is fixable (more regularisation for 16 features, lower LR, or shrinking each
     window's estimate toward the match mean) is untested and is the obvious next experiment.
     Caveats: `topology+direction` without position is worse than position alone on every split,
     so it complements location rather than replacing it; and three of the four are in metres
     rather than shares, so they carry provider annotation differences that the share-based
     features do not.
   - **Role-relative reporting — BUILT.** `role_relative_metrics` in `features/centrality.py`
     z-scores within coarse role. Top-50 composition moves from 84% MID to 32% against a 31%
     population share and a goalkeeper becomes rankable. Note this is **largely true by
     construction** — z-scoring within role forces the composition toward the population — so it
     makes "central for a centre-back" expressible rather than proving the metric measures
     tactical importance.
   - **xT-weighted edges — BUILT, and the honest verdict is that they do not fix this.**
     `features/xthreat.xt_edge_weights` sets `weight = Σ positive xT delta`. The cost worry was
     unfounded: nothing has to be rebuilt and no split-dependent artifact is needed, because the
     persisted edge grouping is reproducible from the actions and the result **left-joins** onto
     `full_edges.parquet`. `attach_xt_edge_weights` asserts the recomputed pass count equals the
     persisted `weight` exactly, which is what makes the join safe — a misattached xT value is
     otherwise invisible. Total cost on the Premier League corpus: **6.7 s**, running the whole
     centrality pipeline twice.
     The result: `pagerank_xt` moves the top-50 from 76% MID to **90% FWD** (92% on Serie A).
     That inverts the bias rather than removing it. Two pre-registered targets failed — the
     xT-weighted metrics **agree with each other less** than the same seven metrics on pass-count
     weights (0.65 vs 0.71 on the Premier League, 0.53 vs 0.70 on Serie A), and goalkeepers stay at 0% on every one of them.
     Only `strength_out_xt` lands near the population share. Full tables in
     [DATA_SOURCES.md](DATA_SOURCES.md).
   - **Shot-chain involvement — BUILT, and it is substantially another volume proxy.**
     `features/chains.shot_chain_involvement` gives a player's share of their team's shot-ending
     possessions, via `possession_id`. It is **split-free** — nothing is fitted — unlike every
     other feature in this group. It does rank forwards (38% of the top 50 against 26% of the
     population, where `strength_out` gave them 2%), but it correlates **+0.710** with
     `degree_total` on the Premier League against a pre-registered bar of < 0.70, so it is
     largely measuring involvement again. `xt_generated` is the feature that clears that bar
     comfortably (+0.457).
     **`shot_conversion` fixes the denominator and swaps the confound.** Dividing by the
     possessions the player was actually in (rather than the team's shot count, which is constant
     within a team-match and so never normalises their touch frequency) drops ρ vs `degree_total`
     to **+0.048** on the Premier League and −0.075 on Serie A. What is left is position: its
     positional R² rises from 0.172 to **0.581**, and its top-50 is 80% forwards. As a GNN feature
     it is worse than `all` on both unconfounded splits (+2.52 and +1.74 pp against +2.65 and
     +2.53), which is what the R² predicts — the model already gets position directly. Kept as
     `all+threat+conv` so the published `all+threat` stays comparable at 22 features.
2. **Verticality as a team style feature.** Validated but not yet a model input: mean pass
   verticality ranks Serie A teams sensibly (Napoli and Juventus most patient in both seasons;
   Sassuolo and Crotone most vertical) and survives the provider change at Spearman ρ = 0.509
   (p = 0.044). It carries a **+0.036 provider offset**, so it must be ranked or z-scored
   within season, never compared absolutely.
3. **Category graphs by action type** — a possession layer and a set-piece layer are buildable;
   a goalkeeping layer is nearly edgeless; a **defensive layer is not buildable from event
   data** (median 3–4 out-of-possession events per player). Defensive phase work needs
   tracking, or must be reported as team-level scalars (line height, recovery zones, PPDA).
4. **Phase-split networks.** In-possession shape is well observed (median 63 events per player
   on StatsBomb); out-of-possession is not. Splitting on `possession_team_id` is cheap and
   would at least stop contested touches polluting the possession network.

## Module 5 — RL pass choice (BLOCKED)

One-step offline contextual bandit rather than a simulator, framed honestly as an exploratory
value estimator.

- **State** — graph over visible players in a 360 freeze-frame.
- **Action** — which visible team-mate to pass to (≤10 discrete).
- **Reward** — xThreat delta of the resulting position, discounted by a learned completion
  probability.
- **Baselines** — most-advanced team-mate, nearest team-mate, and the pass actually played.
- **Evaluation** — off-policy (IPS / doubly-robust) on held-out matches.

Two hard constraints:

1. **Neither corpus has 360 data.** Premier League 2015/16 has 0 of 380; Serie A has none in
   either season. Module 5 needs a third corpus — World Cup 2022 (64 matches) or Euro 2024
   (51) — and the README must say so. Note this *helps* Module 5, whose unit is a pass
   (~50k decisions in 64 matches), while it would hurt Module 3, whose unit is a match.
2. **360 freeze-frames are anonymous and partially visible** — mean 14.9 of 22 players, never
   all 22. The action space is therefore position slots, not named players, and the true
   recipient must be matched to the nearest frame object by `end_location`.

## Module 6 — Coach-facing dashboard (partially built)

The Streamlit app in `app/` covers Modules 1–4: passing networks with centrality and functional
role, the result-prediction ladder with per-match probability timelines, and pattern clusters
with shot lift. Still missing: video-timestamp deep links, the real-versus-agent comparison
(depends on Module 5), and a corpus switcher — the app currently reads whichever corpus was
last exported to `demo_data/`, with the competition named in the sidebar so it cannot be
misread.

## Cross-cutting

- **Weights & Biases** in offline mode unless `WANDB_API_KEY` is set, so no run blocks on an
  account.
- **Kaggle**: cache the processed SPADL store as a Kaggle Dataset so notebooks never
  re-download 1.4 GB, and checkpoint frequently enough to survive session limits.
- Every module reports wall time and peak memory next to its metrics, via
  `eval.resources.ResourceMonitor`.
- Every predictive model uses `eval.splits.temporal_split` and calls `reject_random_split`.
  There is no sanctioned code path for a random split. `temporal_split` validates the split
  kind against the corpus, so asking for `cross_season` on a single-season corpus raises
  instead of returning empty folds.
