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

- **Module 3's GNN+Transformer loses to B0.** 300 (Serie A) / 260 (PL) independent training
  matches cannot support a sequence model. The fix is more data — more competitions, or a
  provider licence — not a different architecture.
- **Module 2's topology contribution is near zero and shrinks on the cleaner corpus**
  (+0.25 pp on the Premier League vs ~1.1–1.5 pp on Serie A). Position features carry the
  signal; graph topology adds almost nothing for role identification.
- **Module 4's set-piece rule still over-segments.** Chains restart on every set piece, which
  splits one phase of play into fragments. A possession that restarts on a throw-in is often
  the same attack.
- **Module 3's batch size is effectively 1** — one optimiser step per match, ~260-300 noisy
  updates per epoch with no gradient accumulation. This is a plausible contributor to the
  validation oscillation that was treated by lowering the learning rate. Worth fixing before
  concluding anything further about the architecture.

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
