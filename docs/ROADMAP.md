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

- **Module 3's negative result was mostly an optimiser bug. Two things were fixed; two remain.**

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
- **Module 2's topology contribution is near zero and shrinks on the cleaner corpus**
  (+0.73 pp on the Premier League, 3-seed mean, spanning +0.25 to +1.06 across seeds, vs
  ~1.1–1.5 pp on Serie A). Position features carry the
  signal; graph topology adds almost nothing for role identification.
- **Module 4's set-piece rule still over-segments.** Chains restart on every set piece, which
  splits one phase of play into fragments. A possession that restarts on a throw-in is often
  the same attack.

## Known gaps in the graph representation

Raised during review and not yet built. These are real limitations, not speculation — the
measurements behind them are in [DATA_SOURCES.md](DATA_SOURCES.md).

1. **Pass-only edges make centrality a volume proxy.** Midfielders take 84% of the top 50 by
   degree against a 33% population share; forwards and goalkeepers are unrankable. Candidate
   fixes, in order of expected value:
   - **xT-weighted edges** — replace `weight = pass count` with `weight = Σ xT delta`, so a
     line-breaking pass outweighs a square ball. Machinery already exists in `features/chains`.
   - **Shot-chain involvement** — participation rate in possessions that end in a shot, via
     `possession_id`. The one metric that would rank forwards at all.
   - **Role-relative reporting** — z-score centrality within coarse role, so "central for a
     centre-back" is expressible.
   - **Pass progressiveness** as node features: `mean_dx` and `mean_length` are already on the
     edge table and unused.
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
