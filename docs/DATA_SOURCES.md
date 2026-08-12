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

The actions that would fix this fail provider comparability on the Serie A corpus
(`tackle` 0.24×, `interception` 3.27×, `clearance` 0.63×, `dribble` 0.11×, `bad_touch` 0.00×
Wyscout-over-StatsBomb per game) — but on the single-provider Premier League corpus **that
objection does not apply**, so defensive and carry actions are usable there. Out of possession,
however, event data observes only 105.8 team actions per game with a **median of 3-4 per
player** and only ~5 of 11 players reaching 5 events, so a defensive *graph* remains
unbuildable from events regardless of provider. That is a tracking problem, not a schema one.
