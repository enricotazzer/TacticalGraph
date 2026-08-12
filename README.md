# TacticalGraph

Football tactical analysis on event data, combining graph neural networks, sequence models and
reinforcement learning. Every module is benchmarked against an explicit baseline and validated
quantitatively, and everything is sized to train inside Kaggle's free tier.

**Status: Modules 1–4 complete, on two corpora.** Data ingestion, passing network construction,
the classical centrality baseline, GNN functional-role embeddings, in-match result prediction
(baseline ladder vs GNN+Transformer), and recurring tactical pattern discovery. Module 5 (RL
pass choice) is blocked on 360 data neither corpus has; Module 6 exists as the demo app. See
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Two corpora, and why

| Corpus | Matches | Provider(s) | Split | Role |
|---|---|---|---|---|
| **`premier_league`** | 380 | StatsBomb | matchweek 1-26 / 27-33 / 34-38 | **Primary.** One complete season, one provider — no confound. |
| `serie_a` | 760 | StatsBomb 2015/16 + Wyscout 2017/18 | cross_season / within_season | Cross-**provider** generalisation study. |

The project began on Serie A because that is what the brief asked for, and hit a hard limit:
StatsBomb open data contains exactly one usable Serie A season, so a second season had to come
from Wyscout. That makes the season change *and* the provider change the same event — a drop on
the test fold cannot be attributed to either. The Premier League 2015/16 season is complete
(380 matches, 38 matchweeks × 10) from a single provider, so the split is by matchweek and a
drop is the model's fault.

Serie A was not discarded. The harmonisation work is a real result, and it now answers the
question it can answer cleanly: does a model trained on one provider transfer to another?

Both corpora coexist: derived data is namespaced under `DATA_ROOT/corpora/<slug>/`, because the
two share a season key (`2015-2016`) *and* a provider (`statsbomb`) and differ only by
competition id. `raw/` is shared, since StatsBomb match ids are globally unique (verified: zero
overlap). Scripts take `--corpus`; `temporal_split` validates the split kind against the corpus,
so asking for `cross_season` on a single-season corpus raises instead of returning empty folds.

**Tracking data is out of scope**, and that is a limitation rather than an oversight — no open
tracking exists at league-season scale, and 360 freeze-frames are anonymous and show a mean of
14.9 of 22 players. What this makes impossible is stated explicitly in
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md), along with the one dataset worth revisiting
(PFF FC's World Cup 2022 release: 64 matches, 30 fps, identified players).

Two of the four headline findings are negative, and they are reported as prominently as the
positive ones:

| Module | Finding |
|---|---|
| 2 | The GNN embedding beats classical centrality **~10×** on role alignment — but passing topology adds only **+0.73 pp** (Premier League) to **+1.07 pp** (Serie A control) over pitch position, and on the Premier League corpus it *hurts* clustering. Most recoverable signal is spatial. |
| 3 | **B1 (scoreline + aggregates + xT) is the best model.** The GNN+Transformer is *significantly worse than B0* in all 6 Serie A runs: 300 independent training labels cannot support it. |
| 4 | Clustering finds possession patterns reaching a **57.3% shot rate against a 12.37% base** (4.6× above; 81.3% on the within-season control) — and the interpretable baseline beats the learned encoder at every k. |

---

## Demo

```bash
streamlit run app/Home.py
```

Runs from a **~8.5 MB bundle committed to this repo** (`demo_data/`), so it works on a fresh
clone with no external drive and no re-ingestion. The bundle is small enough because
`engineer_node_features()` derives every model input from the network node/edge tables — the
55 MB action store and 1.4 GB of raw provider JSON are not needed for anything the demo shows.

| Page | Module | What it is |
|---|---|---|
| Data & Harmonisation | M1 | **Live demo.** Harmonisation scorecard; per-match action-rate comparison; pick any match and draw both passing networks; same club side by side across providers. |
| Player Roles | M2 | **Live demo.** Centrality leaderboard; centrality on the pitch by metric; the 3-seed ablation; clustering vs baseline; embedding map; **find players with a similar functional role**. |
| Result Prediction | M3 | *Specification.* B0/B1/B2 ladder, plus the real windowed graph sequences and label availability that already exist. |
| Tactical Patterns | M4 | *Specification.* Design, plus the real possession-chain statistics. |
| Pass Choice RL | M5 | *Specification.* Blocked on 360 data Serie A does not have. |
| Limitations | — | Every caveat that applies to the numbers above. |

Unimplemented modules open with an explicit banner stating that the page is a specification
and that nothing on it is a model output. The app computes no headline metrics of its own —
they are read from the pipeline's report JSONs, so the demo cannot drift from the results
reported below.

The similar-player search is the most direct evidence the embedding learned something: asked
for players like **Jorginho (2015/16)**, its top neighbours are Badelj, Vecino, Hamšík and
Pjanić — and **Jorginho's own 2017/18 season, ranked second at 0.979 cosine, across a change
of data provider**. Buffon's six nearest neighbours are all goalkeepers.

To rebuild the bundle after changing the pipeline:

```bash
python scripts/export_demo_bundle.py     # writes demo_data/ (~8.5 MB)
pytest tests/test_demo_bundle.py         # manifest integrity + checkpoint round-trip
```

---

## The dataset problem, and what was actually done about it

The goal was multi-season Serie A. **StatsBomb open data does not contain it**: Serie A
appears exactly twice, as 2015/16 (380 matches) and 1986/87 (a single match). Multi-season
Serie A event data is a paid licence.

The only open route to a second season is the **Wyscout public dataset**
(Pappalardo et al. 2019, figshare collection `4415000`), which includes *Italian first
division 2017/18*, 380 matches. So the corpus is deliberately two-provider:

| Corpus | Season | Provider | Matches | Actions | Actions/match |
|---|---|---|---|---|---|
| `serie_a` | Serie A 2015/16 | StatsBomb | 380 | 761,745 | 2,004.6 |
| `serie_a` | Serie A 2017/18 | Wyscout | 380 | 495,873 | 1,304.9 |
| `premier_league` | Premier League 2015/16 | StatsBomb | 380 | 758,434 | 1,995.9 |
| **Total** | | | **1,140** | **2,016,052** | |

The Serie A pair buys a cross-*season* split — train on one season, test on a *later* one — at
the cost of confounding the season change with a provider change. That trade-off was accepted
deliberately, so the project's job is to **measure the confound rather than hide it**. That is
what `scripts/validate_harmonization.py` exists for, and its output is reproduced below.

The Premier League corpus was added later, once it became clear the confound was limiting every
downstream conclusion. It is a **complete 380-match single-provider season** (38 matchweeks × 10,
`match_week` present on all 380, 0 conversion failures), split by matchweek, so it carries no
provider effect at all. It has **no 360 data on any of its 380 matches**, which is why it cannot
host the phase/formation work. Serie A stays as the cross-provider study.

### Why SPADL rather than a bespoke schema

`socceraction` ships official converters from both StatsBomb and Wyscout into **SPADL**, a
provider-agnostic action representation, plus xThreat/VAEP implementations. Adopting it
removes the largest source of risk in a two-provider project — that a hand-written adapter
quietly treats the two sources differently — and keeps results comparable with published
research. The adapter layer in `src/tacticalgraph/data/adapters.py` is a single code path
parameterised by provider, not two parallel implementations.

**Design rule — intersection, not union.** Model inputs are restricted to what *both*
providers express. StatsBomb-only richness (carries, pressures, true pass recipients,
fine-grained positions, `statsbomb_xg`, 360 freeze-frames) is written to a separate
`enrichment/` table used **only for validation**, and
`schema.assert_no_enrichment_leakage()` enforces this mechanically at every feature-matrix
boundary.

---

## Harmonisation results

Both providers are missing things the project needs, so those are **reconstructed
identically for both** — never "real value where available, estimate elsewhere", which would
make 2015/16 systematically better than 2017/18 and corrupt the cross-season test.

### Pass recipient (neither provider survives SPADL with one)

Rule: *recipient = the player of the next action within the following 3 actions, same team,
different player; never across a period boundary.*

| Measurement | n | Correct | Wrong | Unresolved | Actions/match |
|---|---|---|---|---|---|
| StatsBomb, native density | 289,296 | **99.59%** | 0.23% | 0.18% | 2,004.6 |
| StatsBomb, degraded to Wyscout-like density | 289,296 | **96.12%** | 2.59% | 1.30% | 1,214.6 |

The second row is the important one. Wyscout has no ground truth, so accuracy there cannot
be measured directly. Degrading the StatsBomb stream (dropping carry-derived `dribble`
actions) to comparable density and re-running the inference gives the best available
estimate of Wyscout-side accuracy: **≈96%**. That is corroborated by the observed
*coverage* on the real Wyscout season, 96.13% of completed passes resolved, versus 99.82%
on StatsBomb.

### Possession chains (StatsBomb has them, Wyscout does not)

Scored against StatsBomb's native possession counter over 40 matches:

| Metric | Value |
|---|---|
| Adjusted Rand index | **0.832** |
| Boundary Jaccard | 0.616 |
| Chains per match (ours vs StatsBomb) | 246.2 vs 195.8 |

The partition agrees strongly (ARI 0.83) but our rule over-segments by ~25%, because it
treats every set-piece as a hard restart. Documented rather than tuned away: Module 4 will
revisit it.

### How far apart are the two seasons?

Two-sample KS statistic on the features models actually consume:

| Feature | KS | mean 2015/16 | mean 2017/18 |
|---|---|---|---|
| pass length (m) | 0.102 | 18.33 | 20.86 |
| start_x | 0.070 | 50.97 | 49.74 |
| pass Δx | 0.053 | 6.62 | 6.47 |
| start_y | 0.022 | 34.23 | 34.16 |

Spatial features are close. **Action-type counts are not**, and this was the single most
consequential finding of Phase 1 — per-match rates:

| Action type | 2015/16 | 2017/18 | ratio |
|---|---|---|---|
| `bad_touch` | 29.6 | 0.1 | **296×** |
| `dribble` | 790.0 | 90.4 | **8.7×** |
| `tackle` | 36.7 | 8.7 | 4.2× |
| `interception` | 26.3 | 86.0 | 0.31× |
| `pass` | 847.7 | 874.0 | **0.97** |
| `shot` | 24.7 | 23.2 | 1.07 |

These gaps are annotation convention, not football. A feature that counts actions naively
would encode *which provider this is* and collapse at test time. Hence
`schema.PROVIDER_COMPARABLE_TYPES`, and hence node `touches` counts only comparable types.

**Passing networks survive precisely because passes do not have this problem:** all
pass-like types together run 974.1 vs 1,005.9 per match, a ratio of **0.968**. That is the
quantitative licence for the whole project.

> Note on methodology: an earlier version of this report compared action-type *shares*,
> which was misleading — StatsBomb's dribble inflation mechanically deflates every other
> type's share, making comparable types look divergent. Per-match rates isolate each type
> and are what the report now uses.

### Identity resolution

Team ids are provider-private, so clubs are bridged with a hand-checked table (20 per
season; **16 appear in both**, 4 relegated after 2015/16 and 4 promoted for 2017/18).
Players are matched on normalised names (accent-folded, punctuation-stripped), full name
first then unique-surname fallback, with surname collisions dropped rather than guessed:
**199 players** matched across the two seasons.

### Visual verification

`scripts/visual_qa.py` writes `figures/provider_comparison.png` — the same club's season
network from each provider, side by side. This is the check that catches a mirrored
coordinate flip, which no summary statistic would reveal. Both providers produce the same
structure: goalkeeper isolated at the back, defensive line, midfield band, forwards ahead,
all attacking left-to-right.

Node positions are also correct in absolute terms, which is worth stating because the plots
*look* compressed (no player's *mean* position is in the attacking third — as it should be):

| | GK | Centre back | DEF | MID | FWD | Centre forward |
|---|---|---|---|---|---|---|
| mean_x (105 m pitch) | 8.8 | 33.2 | 43.9 | 56.4 | 68.1 | 68.6 |

`mean_y ≈ 34` on a 68 m pitch for every role, i.e. dead centre.

---

## Module 2 — functional role: GNN embedding vs classical centrality

### Phase 2 baseline: classical centrality

Degree, strength, betweenness, closeness, eigenvector, PageRank and clustering coefficient
per player-match over 1,520 team-match networks. Betweenness and closeness run on
**inverted** edge weights, since a passing network's weight is volume (a short hop), not
cost — getting this backwards silently inverts the interpretation.

The football sanity check passes convincingly. Top of Serie A by PageRank:

| 2015/16 | 2017/18 |
|---|---|
| Jorginho (MID) | F. Viviani (MID) |
| Federico Viviani (MID) | L. Cigarini (MID) |
| Maximiliano Moralez (FWD) | **Jorginho** (MID) |
| Francesco Magnanelli (MID) | F. Magnanelli (MID) |
| Miralem Pjanić (MID) | E. Pulgar (MID) |

Deep-lying playmakers dominate, as they should, and five players recur across both seasons
and both providers. Napoli tops team pass volume in both seasons (612 and 572 per match) —
Sarri's side, exactly as expected.

### Phase 3: GraphSAGE, and the leakage trap

Node classification of the **4-class coarse role** (GK/DEF/MID/FWD) — the only vocabulary
both providers express. Wyscout has 4 static roles; StatsBomb has a 25-position vocabulary
(24 occur in Serie A 2015/16) with per-match spells. The fine-grained positions are held
back as a **validation signal the model never sees**.

A player's mean pitch position nearly determines their coarse role (see the table above), so
a model handed `(x, y)` can score well while learning nothing about passing structure. The
feature set is therefore split into three variants, and **all three are always reported**:

- `position` — mean/spread of pitch location (4 features)
- `topology` — connectivity and volume shares only, no coordinates (10 features)
- `both` — the union (14 features)

Test accuracy, 3 seeds, mean ± std:

| Corpus / split | `position` | `topology` | `both` | `both − position` |
|---|---|---|---|---|
| **Premier League, matchweek (primary)** | 0.8531 ± 0.0029 | 0.7578 ± 0.0071 | **0.8604 ± 0.0014** | **+0.73 pp ± 0.43** |
| Serie A, within-season (control) | 0.8915 ± 0.0062 | 0.7659 ± 0.0045 | **0.9022 ± 0.0007** | **+1.07 pp ± 0.57** |
| Serie A, cross-season (confounded) | 0.7769 ± 0.0102 | 0.6777 ± 0.0095 | **0.7916 ± 0.0026** | **+1.47 pp ± 0.98** |

The headline +1.47 pp is the *confounded* split. Compare the two unconfounded rows instead:
**+0.73 pp** (Premier League) and **+1.07 pp** (Serie A within-season). Part of the apparent
cross-season contribution was the provider change, not passing structure.

### Clustering: the actual comparison against the baseline

K-means over each representation, scored against the coarse role and against the
fine-grained position **that nothing was trained on**:

| Representation | k | ARI (4-class) | NMI (4-class) | Silhouette | ARI (fine) | NMI (fine) |
|---|---|---|---|---|---|---|
| centrality (baseline) | 4 | 0.051 | 0.096 | 0.196 | 0.030 | 0.076 |
| gnn-topology | 4 | 0.302 | 0.351 | 0.270 | 0.113 | 0.286 |
| gnn-position | 4 | 0.320 | 0.440 | 0.266 | 0.133 | 0.395 |
| **gnn-both** | 4 | **0.516** | **0.542** | **0.309** | **0.175** | **0.418** |
| centrality (baseline) | 12 | 0.042 | 0.111 | 0.153 | 0.055 | 0.104 |
| gnn-position | 12 | 0.240 | 0.422 | 0.219 | **0.340** | **0.498** |
| gnn-both | 12 | 0.276 | 0.443 | 0.264 | 0.321 | 0.480 |

On the Premier League corpus, mean of 3 seeds at k=4 — and the ordering changes:

| Representation | ARI (4-class) | NMI (4-class) | ARI (fine) | NMI (fine) |
|---|---|---|---|---|
| centrality (baseline) | 0.052 | 0.100 | 0.037 | 0.082 |
| gnn-topology | 0.295 | 0.354 | 0.094 | 0.255 |
| **gnn-position** | **0.495** | **0.555** | **0.179** | **0.416** |
| gnn-both | 0.463 | 0.543 | 0.155 | 0.389 |

**`gnn-position` beats `gnn-both` here.** On Serie A, adding topology helped clustering
(0.516 vs 0.320); on the unconfounded Premier League corpus it *hurts* it (0.463 vs 0.495)
while still adding +0.73 pp of classification accuracy. So topology's contribution is marginal
under supervision and slightly negative for unsupervised structure — a stronger version of the
same negative result, not a contradiction of it.

Stability diagnostics (cosine lift = same-player minus different-player similarity):

| Representation | within-player lift (Serie A) | cross-season lift (Serie A, n=199) | half-season lift (PL, n=383) |
|---|---|---|---|
| centrality (baseline) | 0.295 | 0.651 | 0.692 |
| gnn-topology | 0.419 | 0.728 | 0.740 |
| gnn-position | 0.581 | 0.762 | 0.810 |
| **gnn-both** | **0.609** | **0.814** | **0.835** |

The Premier League column uses `half_season_stability` (weeks 1-19 vs 20-38), the
single-season replacement for cross-season stability. It is the cleaner measure: one provider
and one competition, so a low score is the representation's fault with no provider change to
blame. Players appearing in only one half are excluded — that is a transfer window, not
instability.

### What this actually shows — and what it does not

**The GNN embedding decisively beats classical centrality.** On every measure, by a wide
margin: 10× the ARI against coarse role at k=4 (0.516 vs 0.051), 4–5× the NMI against
fine-grained position (0.418 vs 0.076), and roughly double the within-player consistency
lift. This is not a close call, and it is not an artefact of a broken baseline — the
centrality matrix has 100% join coverage, no duplicate keys, and sensible spread on all ten
metrics. Classical centrality genuinely does not encode role: it measures *how much* a
player is involved, not *in what capacity*. A centre-back and a centre-forward can both
have degree 8.

**The embedding recovers structure nobody supervised.** Trained on 4 coarse classes, it
separates Right Back from Left Back, and Centre Defensive Midfield from Centre Midfield —
visible in `figures/role_embedding_cross_season.png` and quantified as NMI 0.42–0.50 against
the fine-grained positions.

**But passing topology is the minor contributor, and that is the honest headline.** Three
findings point the same way:

1. `topology` alone is the *weakest* variant everywhere (0.766 / 0.678 accuracy).
2. Adding topology to position buys only **+1.1 to +1.5 pp**. The direction is reliable
   (positive in 6/6 runs) but the magnitude is small and its spread is comparable to the
   effect itself.
3. At k ≥ 6, `position` alone **matches or beats** `both` at recovering fine-grained
   positions (k=12: 0.340 vs 0.321 ARI). Much of the fine structure the embedding recovers
   — left versus right — is available directly from `mean_y`.

So the defensible claim is: *a learned representation of a player's passing network is a far
better description of role than classical centrality metrics, but most of the recoverable
signal at this granularity is spatial rather than structural.* The stronger version of the
thesis — that who you pass to reveals a functional role that position cannot — is only
weakly supported by this evidence.

**One suggestive result in topology's favour:** it degrades least across the
provider boundary (−8.8 pp, versus −11.5 pp for position and −11.1 pp for both), consistent
with within-network normalised shares being more provider-robust than raw coordinates. With
a single provider pair this is indicative, not established.

**The cross-season cost is ~11 pp** (0.902 → 0.792 for `both`). Because season and provider
change together, that figure is an upper bound on the true seasonal effect and cannot be
decomposed with this corpus.

---

## Module 3 — in-match result prediction

Predicts win/draw/loss at each of **16 checkpoints** (the closing minute of each 15-minute
window, so the tabular ladder and the graph model are scored on identical support).

### The rule that governs this module

> Every feature at checkpoint *t* must be computable from actions with `minute <= t`, and
> nothing else.

That rules out full-match aggregates *and* full-match network metrics — B2's structural
features come from the window that has just closed, not from `centrality_teams.parquet`.
A leak here would not raise an error; it would make every number below look excellent and mean
nothing. `tests/test_match_state.py` enforces it by rebuilding each feature row from a
truncated action stream and asserting it is unchanged, at five different checkpoints.

Transformer causality is enforced by an additive `-inf` mask and **verified empirically**:
perturbing windows 12–15 leaves predictions 0–11 bit-identical.

### Results (test fold, mean over 3 seeds)

All figures are means over 3 seeds. The tabular rungs are deterministic, so only the graph
model varies between seeds.

| Model | Features | cross-season log-loss | within-season log-loss | Brier | Accuracy | ECE |
|---|---|---|---|---|---|---|
| prior | class frequencies | 1.0758 | 1.0268 | 0.651 | 0.432 | 0.012 |
| B0 | scoreline + minutes left | 0.7504 | **0.7125** | 0.429 | 0.667 | 0.024 |
| **B1** | + shots, passes, xThreat | **0.7075** | 0.7305 | 0.407 | 0.692 | 0.023 |
| B2 | + rolling form, network (GBM) | 0.7660 | 0.7181 | 0.439 | 0.654 | 0.040 |
| GNN + Transformer | windowed graph sequences | 0.9012 | 0.9551 | 0.514 | 0.627 | 0.128 |

Paired bootstrap against B0, **resampled by match** (16 correlated rows per match; resampling
rows would shrink every interval ~4× and manufacture significance):

| Model | cross-season Δ vs B0 | 95% CI | Verdict | within-season Δ | 95% CI |
|---|---|---|---|---|---|
| B1 | **−0.0429** | [−0.067, −0.018] | significantly better | +0.0180 | [−0.048, +0.088] |
| B2 | +0.0156 | [−0.017, +0.050] | indistinguishable | +0.0056 | [−0.062, +0.069] |
| GNN + Transformer | **+0.1508** | [+0.085, +0.220] | significantly **worse** | **+0.2425** | [+0.079, +0.424] |

Note the within-season column: **B1's advantage disappears entirely** once the provider change
is removed (Δ +0.018, CI spanning zero), while the graph model's deficit gets larger. Only the
GNN result is consistent across both splits.

### What this shows

**B1 wins on the cross-season split**: shots, passes and accumulated xThreat genuinely add
over the scoreline. But on the **unconfounded within-season control, B0 wins** and B1/B2 are
indistinguishable from it — with 60 test matches the intervals overlap heavily, so B1's
advantage may be specific to the cross-season setting rather than general.

**The GNN+Transformer fails, and the cause is measured rather than guessed.** Train loss falls
1.30 → 0.59 while validation loss rises 0.89 → 2.55: with **300 independent training labels**
a 13k–77k parameter sequence model memorises the training set. Three fixes were tried and none
changed the conclusion:

1. Learning rate 1e-3 → 3e-4 with patience 8 → 25, after the val curve showed violent
   oscillation (0.81 → 1.16 → 0.89 → 1.25) and a stop at epoch 9 with the best score at epoch 0.
2. A **residual path from the scalar state to the logits**, so the model can trivially recover
   the tabular baseline. This is the correct experimental design — without it the comparison
   conflates "graphs do not help" with "the network failed to relearn goal difference".
3. A **capacity sweep (5k / 13k / 77k params) selected on validation**, never on test.

Further tuning would be fishing on the test set, so the negative result stands.

**Two label caveats, reported not hidden.** The derived running scoreline reproduces the
recorded final score for **758/760 games (99.7%)**; the two failures are Wyscout matches where
a goal is simply absent from the event stream, and no phantom goal was inserted to hide it.
And the test season's outcome prior differs from the training season's (away wins 28.9% →
35.0%), a real calibration headwind.

`goal_diff` dominates B2's permutation importance (0.563), which is the expected sanity check —
anything outranking it would mean the feature layer is broken.

---

## Module 4 — recurring tactical patterns

Clusters **109,912 possession chains** (those with ≥3 provider-comparable actions, from
186,318 reconstructed) two ways, and measures how often each pattern precedes a shot.

### Base rate: 12.4%, not 9.7%

9.7% of *all* chains contain a shot, but the module clusters only chains with ≥3 comparable
actions, and longer possessions are likelier to shoot — so the population being clustered has a
**12.4%** base rate. Every lift below is measured against 12.4%; using the unfiltered figure
would inflate them all.

The comparable-types filter is not optional: raw chain length differs between providers by
**1.44×** (mean 7.91 vs 5.51 actions, StatsBomb logging carries as dribbles), which falls to
**0.90×** (4.38 vs 4.87) once filtered. Without it the clustering would partly be clustering the
data provider.

### Results (k = 8, cross-season test fold)

Hand-crafted representation, P(shot | cluster) with Wilson intervals against the 12.0% test
base rate:

| Cluster | Auto-generated name | Share | P(shot) | 95% CI | Lift |
|---|---|---|---|---|---|
| 2 | long from middle third, high threat | 3.0% | **0.573** | [0.549, 0.597] | **4.78×** |
| 4 | open-play direct from middle third | 9.9% | 0.241 | [0.229, 0.252] | 2.01× |
| 0 | set-piece from final third | 11.5% | 0.228 | [0.218, 0.239] | 1.90× |
| 5 | long lateral from middle third, high threat | 8.5% | 0.189 | [0.178, 0.200] | 1.58× |
| 1 | long from middle third | 20.3% | 0.118 | [0.112, 0.124] | 0.98× |
| 6 | set-piece brief direct from defensive third | 20.1% | 0.038 | [0.035, 0.042] | 0.32× |
| 3 | lateral from middle third | 13.3% | 0.034 | [0.030, 0.038] | 0.28× |
| 7 | open-play brief direct from defensive third | 13.4% | **0.004** | [0.003, 0.006] | 0.03× |

**7 of 8 clusters differ from the base rate by more than sampling noise.** Shot rates span
**0.4% to 57.3%** against the 12.37% base — 4.6× above to 32× below. (On the within-season
control the extremes are wider still, 0.19% to 81.3%, because the folds are smaller.)

| Representation | max lift (test) | shot-rate spread | Silhouette | clusters ≠ base |
|---|---|---|---|---|
| **hand-crafted (baseline)** | **4.78×** | 0.569 | 0.112 | 7/8 |
| GRU autoencoder | 3.38× | 0.394 | 0.148 | 7/8 |

### What this shows

**The interpretable baseline beats the learned encoder at every k** (4.78× vs 3.38× at k=8;
5.02× vs 2.43× at k=6). Inspecting the latent clusters shows why: three of eight are 93–95%
set-piece-initiated, so the autoencoder is largely clustering *how a possession started* — the
action-type one-hot dominates its reconstruction loss — rather than how it developed. The same
pattern as Module 2, where simple positional features also beat the learned alternative.

**Cross-season stability isolates the provider effect.** Within a single season and provider,
**8 of 8** clusters keep a statistically indistinguishable shot rate. Across the season/provider
boundary that drops to **4 of 8** (hand-crafted) and **5 of 8** (GRU). The control split is what
makes that attribution possible.

### Human validation: pending, by design

The project's validation plan calls for sampling patterns and judging whether they are
tactically sensible. **That requires a person and has not been done.**
`scripts/review_patterns.py` produces everything a reviewer needs — a 48-row sheet per
representation with match, timestamp, zones and a blank `sensible_y_n` column, plus 16
per-cluster pitch figures drawing the sampled possessions action by action. The app reports the
step as pending rather than implying a review happened.

Cluster names are generated from the same profile numbers printed beside them, ranked relative
to sibling clusters. They are a reading aid and cannot corroborate anything.

---

## Resource footprint

Everything runs on a laptop; nothing here needs a Kaggle GPU session, which is the point.

| Stage | Wall time | Peak RSS | Notes |
|---|---|---|---|
| Download (760 matches) | ~8 min | — | 1.4 GB raw, resumable |
| SPADL conversion, both providers | 53 s (Wyscout) + ~4 min (StatsBomb) | — | 0 failures / 760 games |
| Passing networks (full + windowed) | 128 s | — | 1,520 + 24,320 networks |
| Centrality baseline | 6.9 s | 419 MB | 19,335 player-match rows |
| GNN `position` | 6.6 s | 672 MB | 4 features |
| GNN `topology` | 11.5 s | 703 MB | 10 features |
| GNN `both` | 7.8 s | 719 MB | 14 features |
| M3 state table (12,160 rows × 16 checkpoints) | 10.2 s | 1,987 MB | leakage-safe, per checkpoint |
| M3 baseline ladder (prior/B0/B1/B2) | 1.5 s | 1,754 MB | incl. val-based capacity selection |
| M3 GNN+Transformer (3-config sweep) | 165 s | 948 MB | all three capacities trained |
| M4 chain table (109,912 chains) | 1.0 s | 1,840 MB | from 1.26 M actions |
| M4 GRU autoencoder | 43 s | 362 MB | self-supervised, 30 epochs |

Device: Apple M-series (MPS). GPU allocation is reported as ~0 MB because these graphs
(~13 nodes each, 1,520 graphs) are trained full-batch on CPU tensors — neighbour sampling
would add overhead and approximation for no memory benefit at this scale. It earns its place
in Module 3, where graph sequences get large.

---

## Reproducing

```bash
conda env create -f environment.yml
conda activate tacticalgraph
python -m pip install -e .          # NOT bare `pip` -- see below

cp .env.example .env                # point DATA_ROOT at ~5 GB of free space

python scripts/ingest.py --all               # ~1.4 GB, resumable
python scripts/build_spadl.py --all          # canonical store + enrichment
python scripts/validate_harmonization.py     # Phase 1 gate
python scripts/build_networks.py --all
python scripts/visual_qa.py                  # inspect figures before continuing
python scripts/run_centrality.py             # Module 2 baseline
python scripts/train_roles.py                # Module 2 GNN
python scripts/train_roles.py --split within_season   # unconfounded control

python scripts/train_outcome.py                       # Module 3
python scripts/train_outcome.py --split within_season  # control
python scripts/train_patterns.py --k 8                # Module 4
python scripts/review_patterns.py                     # review sheet + figures

python scripts/export_demo_bundle.py         # refresh demo_data/
pytest                                       # 74 tests
```

### Environment gotchas, all of them load-bearing

- **`socceraction` 1.5.3 pins `python<3.13`, `numpy<2`, `pandas<3`.** Every pre-existing
  conda env on the development machine violated this, hence a dedicated env. Do not
  "upgrade" numpy to satisfy a newer torch.
- **`pandera` 0.17.2 needs `multimethod<2`** — 2.0 removed the `overload` symbol it imports,
  and without the pin the entire `socceraction.spadl` import fails.
- **Use `python -m pip`, never bare `pip`.** On this machine a system Python 3.13 framework
  pip shadows the conda env on `PATH`, which silently resolves `cp313` wheels and produces
  spurious "no matching distribution found for socceraction" errors.
- **`torch==2.6.0`** is the oldest published for macOS arm64 / cp311; the plan's 2.4.1 does
  not exist for this platform.
- **Data lives outside the repo** via `DATA_ROOT`. The development machine's internal disk
  had 1.8 GB free, so bulk data sits on an external drive. `config.data_root()` fails loudly
  if the volume is not mounted rather than silently creating an empty tree.
- **exFAT sidecars**: macOS writes `._`-prefixed AppleDouble files next to every real file on
  exFAT volumes. They are not parquet. All directory listings go through
  `config.clean_glob()`; a naive glob will crash a reader.

---

## Limitations

- **The season/provider confound is real and undecomposable.** 2015/16 is StatsBomb and
  2017/18 is Wyscout, so the ~11 pp cross-season drop mixes a genuine seasonal effect with an
  annotation-convention effect. This is why every headline number is reported next to a
  within-season control. It is a deliberate, documented trade-off, not an oversight.
- **Pass recipients are inferred, not observed** — ~96% accurate on Wyscout-like density.
  Roughly 1 edge in 25 in the 2017/18 networks is wrong or missing.
- **Possession chains over-segment by ~25%** relative to StatsBomb's native counter.
- **Minutes played are estimated from first-to-last action**, symmetrically for both
  providers, because Wyscout exposes no per-match position spells. A player with a single
  action is credited 0 minutes and filtered out of the network.
- **Role labels are 4-class**, forced by Wyscout. The fine-grained validation signal exists
  for one of the two seasons only.
- **Neither corpus has 360 freeze-frame data.** Premier League 2015/16 has 0 of 380 matches;
  Serie A has none in either season. Module 5's pass-choice RL agent needs a third corpus
  (World Cup 2022, 64 matches, or Euro 2024, 51), which departs from the league framing. 360 is
  also anonymous and partially visible — measured on a real match: **mean 14.9 of 22 players,
  0% of frames with all 22**, only `teammate`/`actor`/`keeper` flags and no player ids — so the
  action space is position slots, not named players.
- **No tracking data, and this bounds what the graphs can represent.** No open tracking exists
  at league-season scale. Consequences, stated precisely: **player-level defensive centrality is
  not buildable** (360 has no player identity, and out of possession event data records a median
  of only 3-4 actions per player, with ~5 of 11 players reaching 5); and a **defensive-phase
  graph is not buildable from events at all**, only team-level scalars such as line height and
  recovery zones. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
- **Pass-only edges make centrality a volume proxy.** Midfielders are 33% of the population and
  **84% of the top 50 by degree**; forwards and goalkeepers are effectively unrankable (0-14%
  against a 24% share). The metrics also disagree with each other about who is central
  (`degree_total` says 84% MID, `strength_out` says 56% DEF), which is itself evidence that none
  of them measures tactical importance. Candidate fixes — xT-weighted edges, shot-chain
  involvement, role-relative z-scores — are specified in [`docs/ROADMAP.md`](docs/ROADMAP.md)
  and **not yet built**.
- **Event data is a partial representation of football.** Off-ball movement, verbal
  communication and tactical intent are not observable here, and no amount of modelling
  recovers them.
- **Reduced scale by design.** Three competition-seasons across two corpora, small models,
  short training schedules. These are proof-of-concept results, not state-of-the-art claims.
- **Three seeds** per configuration. Enough to show the ablation ordering is stable; not
  enough for tight confidence intervals on a ~1 pp effect.
- **Module 3 is data-limited, not architecture-limited.** 300 independent training matches is
  the binding constraint. The negative result says nothing about whether graph sequence models
  help at scale — only that they do not help here, and the honest remedy is more data, not a
  bigger model.
- **The Module 3 running scoreline is 99.7% faithful** (758/760 games). Two Wyscout matches are
  missing a goal from the event stream entirely; no substitute goal was invented.
- **Module 4's shot-precursor analysis is associational.** A cluster with a 57% shot rate
  describes possessions that ended in shots; it does not establish that playing that way causes
  them.
- **Module 4's human review has not been performed.** The sheets and figures are generated and
  waiting, so any "proportion judged sensible" is absent rather than estimated.
- **Module 5 is not implemented and cannot be on this corpus**: Serie A has no 360 freeze-frame
  data in either season.

---

## Layout

```
src/tacticalgraph/
  config.py            DATA_ROOT resolution, dataset specs, exFAT-safe globbing
  data/
    download.py        resumable acquisition for both providers
    adapters.py        one conversion path, parameterised by provider
    schema.py          canonical columns, PROVIDER_COMPARABLE_TYPES, leakage guard
    recipient.py       recipient inference + scoring harness
    possession.py      chain reconstruction + scoring harness
    roles.py           25-position -> 4-class collapse, the only place it is defined
    aliases.py         club table + player name matching
    players.py         cross-provider player directory
    enrichment.py      StatsBomb-only fields, quarantined for validation
    spadl_store.py     partitioned parquet store
  graphs/
    passing_network.py full-match and 15-min/5-min-stride windowed networks
  features/
    centrality.py      classical baseline (inverted weights for path metrics)
    match_state.py     M3 labels + leakage-safe per-checkpoint features
    chains.py          M4 possession chains, features and token sequences
  models/
    role_gnn.py            GraphSAGE + the three feature sets
    outcome_baselines.py   the B0/B1/B2 ladder
    outcome_gnn_transformer.py  GraphSAGE per window + causal Transformer
    chain_encoder.py       GRU autoencoder over possession sequences
  eval/
    splits.py          temporal splits; rejects random splits
    clustering.py      representation comparison + stability diagnostics
    outcome_metrics.py log-loss/Brier/ECE, bootstrap resampled BY MATCH
    patterns.py        Wilson intervals, shot lift, cross-season stability
    resources.py       wall time and peak memory for every run
  viz/
    pitch.py           mplsoccer networks, provider comparison plot
  demo/
    bundle.py          loads demo_data/ (or DATA_ROOT); no streamlit import
app/
  Home.py              phase status, corpus, the confound stated up front
  _shared.py           cached loaders, status banner, forces the Agg backend
  pages/1..6_*.py      one page per macro phase
scripts/               one entry point per phase, plus export_demo_bundle.py
demo_data/             ~8.5 MB committed bundle -- the demo's input
tests/                 43 tests
```

## References

- Decroos et al. (2019), *Actions Speak Louder than Goals* — SPADL / VAEP.
- Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer
  competitions*, Sci Data 6:236 — the Wyscout open dataset.
- Hamilton et al. (2017), *Inductive Representation Learning on Large Graphs* — GraphSAGE.
- StatsBomb Open Data — <https://github.com/statsbomb/open-data>.
