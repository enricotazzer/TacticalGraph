# Data sources: what exists, what was chosen, and what is out of reach

This file records the data-availability work so the constraints are not re-litigated. Every
count here was verified against the actual source, not taken from documentation.

## Corpora in this project

| Corpus | Matches | Provider(s) | Split | 360 | Purpose |
|---|---|---|---|---|---|
| `premier_league` | 380 | StatsBomb | matchweek | none | Modules 1-4. Complete single-provider season, **no provider confound**. |
| `serie_a` | 760 | StatsBomb + Wyscout | cross_season / within_season | none | Retained as a cross-*provider* generalisation study. |

The two share a season key (`2015-2016`) *and* a provider (`statsbomb`) and differ only by
StatsBomb competition id (2 vs 12). That is why derived data is namespaced per corpus under
`DATA_ROOT/corpora/<slug>/` — partitioning by season and provider alone would silently merge
two competitions into one table. `raw/` is deliberately shared, because StatsBomb event files
are keyed by globally unique match id (verified: **zero id overlap** between the two).

### Why the Premier League became the primary corpus

Serie A's two seasons come from two providers, so the season change *is* the provider change:
a drop on the test fold cannot be attributed to either. The Premier League 2015/16 season is
complete (380 matches, 38 matchweeks × 10) from one provider, so train/val/test are matchweeks
1-26 / 27-33 / 34-38 and a drop is the model's fault. Serie A was not discarded — the
harmonisation work stands on its own, and it now answers a different question.

## 360 freeze-frame data

`match_available_360` is set on **12 of 80** competition-seasons in StatsBomb open data.
Verified match counts:

| Competition | 360 matches | Complete? |
|---|---|---|
| FIFA World Cup 2022 | 64 | yes |
| AFCON 2023 | 52 | yes |
| UEFA Euro 2024 | 51 | yes |
| La Liga 2020/21 | 35 | no (of 380) |
| 1. Bundesliga 2023/24 | 34 | no (of 306) |
| Ligue 1 2022/23 | 32 | no (of 380) |
| Ligue 1 2021/22 | 26 | no (of 380) |
| MLS 2023 | 6 | no |
| Women's Euro 2022 / 2025 | 31 each | yes |

**362 matches total** (133 male league, 167 male tournament, 62 women's). No league has 360
for a full season — the league entries are samples. Premier League has **0 of 380** in both
its open-data seasons (2015/16 and 2003/04).

### What 360 is, measured

Verified on Bundesliga 2023/24 match 3895292 (Union Berlin v Bayer Leverkusen), 3,195 frames:

- **Anonymous.** Per-object keys are `teammate`, `actor`, `keeper`, `location`. No `player_id`.
- **Partial.** Mean **14.9 of 22** players visible; **0%** of frames contain all 22; min 3.
- **Event-triggered**, roughly one frame per event — not continuous tracking.

Consequences, stated precisely because they bound what is buildable:

- Team-level defensive shape and offensive/defensive phase asymmetry **become measurable** —
  ~15 players are visible regardless of who holds the ball.
- Player-level defensive centrality **does not**. Without player identity, a category graph
  over defensive actions is a position-slot graph, not a player graph.

## Tracking data

Tracking is the only data that fully answers off-ball questions: continuous, identified, and
independent of who touches the ball. What is actually obtainable:

| Source | Matches | Rate | Players identified | Licence |
|---|---|---|---|---|
| **PFF FC — World Cup 2022** | **64** (all) | **30 fps** | yes | free, form-gated; terms not published |
| SkillCorner open data | 10 (A-League 24/25) | 10 fps | yes (~97% accurate) | **MIT** |
| SkillCorner / Friends-of-Tracking | 9 (top-5 leagues) | 10 fps | yes | open |
| Metrica Sports sample | 3 | 25 fps | anonymised | open |
| 3. Liga 2023/24 (STS/Track 160) | 336 (full season) | 25 fps | yes | research agreement only |
| Frauen-Bundesliga 2023/24 (STS) | 116 | 25 fps | yes | research agreement only |

The PFF World Cup figures come from a published dataset table (64 matches, 29,931 passes,
**11,849,751 frames** at 30 fps ≈ 103 minutes per match), not from marketing copy.

**Scope decision: tracking is out of scope for this project, and the limitation is reported
rather than worked around.** Two unresolved points, both checkable and neither resolvable from
public pages:

1. PFF's own blog describes the dataset as *broadcast* tracking while the academic table
   classifies it as *optical*. This determines whether off-camera players are interpolated or
   simply absent — which matters for any off-ball claim.
2. The licence terms sit behind the download form, so whether they permit a public portfolio
   repository is unconfirmed. `fchelp@pff.com` is the stated contact.

If tracking is ever brought in, **PFF World Cup 2022 is the candidate**, because it covers the
same 64 matches as StatsBomb's WC 2022 events *and* 360 — three data layers on one corpus, and
tracking's player identities would unblock the player-level defensive centrality that 360
cannot support.

No open tracking exists at league-season scale for the Premier League, Bundesliga or Serie A.
Premier League tracking is collected commercially (Second Spectrum, now Genius Sports/Hawk-Eye)
and supplied to clubs and licensees only.

## Known limitation this creates

Pass-only passing networks make centrality a volume proxy, and volume is largely positional.
Measured on 737 Serie A players with ≥10 matches, share of the top 50:

| Metric | GK | DEF | MID | FWD |
|---|---|---|---|---|
| `degree_total` | 0% | 16% | **84%** | 0% |
| `pagerank` | 0% | 18% | 70% | 12% |
| `betweenness` | 0% | 42% | 54% | 4% |
| `strength_out` | 0% | **56%** | 42% | 2% |
| *population* | 7% | 37% | 33% | 24% |

Midfielders are 33% of the population and 84% of the top 50 by degree; forwards and
goalkeepers are effectively unrankable. The metrics also *disagree* about who is central,
which is itself evidence that none of them measures tactical importance.

**This is not a harmonisation artefact — it replicates on the clean corpus.** Measured on
359 Premier League players with ≥10 matches (single provider, no cross-provider mapping),
share of the top 50 across all ten metrics:

| Metric | GK | DEF | MID | FWD |
|---|---|---|---|---|
| `degree_in` | 0% | 4% | 78% | 18% |
| `degree_out` | 0% | 22% | 74% | 4% |
| `degree_total` | 0% | 8% | **84%** | 8% |
| `strength_in` | 0% | 24% | 60% | 16% |
| `strength_out` | 0% | **48%** | 50% | 2% |
| `betweenness` | 0% | 30% | 66% | 4% |
| `closeness` | 0% | 30% | 48% | 22% |
| `eigenvector` | 0% | 10% | 78% | 12% |
| `pagerank` | 0% | 10% | 76% | 14% |
| `clustering` | 0% | 22% | 68% | 10% |
| *population* | 8% | 35% | 31% | 26% |

Same shape as Serie A. Midfielders are 31% of the population and 84% of the top 50 by
`degree_total`, and goalkeepers take **0% on all ten metrics** — not one keeper is rankable by
any of them. The metrics also contradict each other: `degree_total` reads 84% MID, while
`strength_out` reads 48% DEF / 50% MID. They cannot all be measuring tactical importance, and
the simpler reading is that none of them is. The cause is the pass-only graph, not the provider.

The actions that would fix this fail provider comparability on the Serie A corpus
(`tackle` 0.24×, `interception` 3.27×, `clearance` 0.63×, `dribble` 0.11×, `bad_touch` 0.00×
Wyscout-over-StatsBomb per game) — but on the single-provider Premier League corpus **that
objection does not apply**, so defensive and carry actions are usable there. Out of possession,
however, event data observes only 105.8 team actions per game with a **median of 3-4 per
player** and only ~5 of 11 players reaching 5 events, so a defensive *graph* remains
unbuildable from events regardless of provider. That is a tracking problem, not a schema one.

### Partial mitigation: role-relative z-scoring

`role_relative_metrics` in `src/tacticalgraph/features/centrality.py` z-scores each metric
within `coarse_role`. Top-50 composition on the same Premier League corpus, before and after:

| Metric | Ranking | GK | DEF | MID | FWD |
|---|---|---|---|---|---|
| `degree_total` | raw | 0% | 8% | 84% | 8% |
| `degree_total` | role-relative | 6% | 34% | 32% | 28% |
| `pagerank` | raw | 0% | 10% | 76% | 14% |
| `pagerank` | role-relative | 6% | 34% | 34% | 26% |
| `betweenness` | raw | 0% | 30% | 66% | 4% |
| `betweenness` | role-relative | 2% | 34% | 42% | 22% |
| `strength_out` | raw | 0% | 48% | 50% | 2% |
| `strength_out` | role-relative | 6% | 34% | 32% | 28% |
| *population* | — | 7.5% | 35.4% | 31.5% | 25.6% |

A goalkeeper is rankable for the first time: Costel Pantilimon, `pagerank_z` = **+1.86**.

These numbers replaced an earlier hand-run measurement when `role_relative_metrics` was finally
wired into `scripts/run_centrality.py` — it had been implemented and unit-tested but reached no
artifact, so the table here could not be reproduced from any code path and had drifted by a few
points per cell (Pantilimon was recorded at +2.12). They now come from
`module2_volume_proxy_<split>.json`, which the script writes on every run.

**The post-z-score composition tracking the population is largely true by construction, and is
not evidence the graph got better.** Z-scoring within role forces the top-50 mix toward the
population mix; the agreement is mechanical, not a finding. What it actually buys is narrower:
"unusually central *for a centre-back*" becomes expressible, and keepers and forwards get a
ranking at all instead of none. It does **not** show that the graph measures tactical importance
rather than volume. That claim needs the richer edge and threat features, measured below.

### The two remaining fixes, measured: xT-weighted edges and shot-chain involvement

Both are now built and reproducible from `scripts/run_centrality.py`, which writes
`module2_volume_proxy_<split>.json` next to the numbers below. Edge weights become the summed
**positive** xT delta of the passes on that lane (`features/xthreat.xt_edge_weights`), and
`shot_involvement` is a player's share of their team's shot-ending possessions
(`features/chains.shot_chain_involvement`). Four targets were registered before running.

**Target 1 — composition moves. MET, and it overshot.** Premier League top-50 share:

| Metric | GK | DEF | MID | FWD |
|---|---|---|---|---|
| `pagerank` (volume) | 0% | 10% | **76%** | 14% |
| `pagerank_xt` | 0% | 0% | 10% | **90%** |
| `strength_out` (volume) | 0% | **48%** | 50% | 2% |
| `strength_out_xt` | 0% | 44% | 32% | 24% |
| `shot_involvement` | 0% | 14% | 48% | 38% |
| `xt_generated` | 0% | 22% | 34% | 44% |
| *population* | 8% | 35% | 31% | 26% |

The registered bar was FWD ≥ 20% and MID ≤ 60% on `pagerank_xt`; it delivered 90% and 10%.
That is not a correction, it is an inversion — a midfielder leaderboard traded for a forward
one. Serie A replicates it at 92% FWD / 8% MID. Only `strength_out_xt` lands near the
population. **Goalkeepers remain at 0% on every xT-weighted metric**, so the one role that was
completely unrankable still is; role-relative z-scoring is the only thing that ranks them.

**Target 2 — the metrics stop contradicting each other. FAILED.** Mean pairwise Spearman ρ
across each family. The comparison is restricted to the **same seven weight-sensitive metrics**
on both sides: `degree_total` is `degree_in + degree_out` by construction, so including the three
degree metrics inflates whichever family holds them (it lifts the volume figure from +0.711 to
+0.725 on the Premier League). Both are shown so that inflation is visible rather than buried.

| Corpus | volume (all 10) | volume (same 7) | xT-weighted (7) |
|---|---|---|---|
| Premier League | +0.725 | **+0.711** | +0.651 |
| Serie A | +0.711 | **+0.694** | +0.528 |

This was the load-bearing target, because unlike composition it is not moved mechanically by
reweighting. It went the wrong way on both corpora, on the like-for-like comparison: xT-weighted
metrics agree with each other *less* than the same metrics on pass-count weights. Reweighting the
edges did not reveal a shared underlying construct; it added variance.

**Target 3 — shot involvement is not a third volume proxy. FAILED on the primary corpus.**
Spearman ρ against `degree_total` is **+0.710** (Premier League) and +0.653 (Serie A), against
a registered bar of < 0.70. It must therefore be reported as substantially another volume
measure. `xt_generated` is the one that clears it comfortably, at +0.457 and +0.498 — a player's
share of the threat their team created is genuinely not the same thing as how often they touched
the ball.

**Target 4 — the threat features improve Module 2's role accuracy. FAILED (a null).** Added to
the GNN as `all+threat` (22 features) against `all` (18), three seeds per split:

| Corpus / split | `all` − `position` | `all+threat` − `position` | seed σ |
|---|---|---|---|
| Premier League, matchweek | +2.65 pp | +2.73 pp | ±0.25–0.34 |
| Serie A, within-season | +2.53 pp | +2.56 pp | ±0.46–0.65 |
| Serie A, cross-season | +5.74 pp | **+5.57 pp** | ±1.11–1.56 |

The registered bars (> +2.65 and > +2.53) are cleared by **+0.08 pp and +0.03 pp**, against seed
standard deviations three to twenty times larger, and the third split moves the other way. This is
a null, and it is the expected one in hindsight: xT generated and shot involvement measure how much
a player's actions were *worth*, and the 4-class GK/DEF/MID/FWD target is a question about *where
they play*. `threat` on its own is the weakest feature set in the project (0.6215 on the Premier
League, below `topology`'s 0.7538 and `direction`'s 0.7422). Value and role are close to
orthogonal here.

Worth recording as a reproducibility check: re-running `all` after all this work reproduces
**+2.65 pp exactly**, to four decimal places on the mean.

**What this establishes.** Reweighting the edges of a pass-only graph changes *which* position
the leaderboard is biased toward without making it measure tactical importance. The limitation
is not the weighting; it is that a graph whose only relation is "passed to" describes ball
circulation, and ball circulation is positional. Fixing it needs a different relation — off-ball
movement, defensive actions, or space occupied — and the first two are bounded by what event
data records (median 3-4 out-of-possession actions per player) and the third needs tracking.
