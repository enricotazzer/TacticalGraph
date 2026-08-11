# TacticalGraph

Football tactical analysis on Serie A event data, combining graph neural networks, sequence
models and reinforcement learning. Every module is benchmarked against an explicit baseline
and validated quantitatively, and everything is sized to train inside Kaggle's free tier.

**Status: Phases 1–3 complete** — data ingestion and cross-provider harmonisation, passing
network construction, classical centrality baseline, and GNN functional-role embeddings with
a full ablation. Phases 4–8 (result prediction, pattern detection, RL, dashboard) are
scoped in [`docs/ROADMAP.md`](docs/ROADMAP.md) but not yet implemented.

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

| Season | Provider | Matches | Actions | Actions/match |
|---|---|---|---|---|
| Serie A 2015/16 | StatsBomb | 380 | 761,745 | 2,004.6 |
| Serie A 2017/18 | Wyscout | 380 | 495,873 | 1,304.9 |
| **Total** | | **760** | **1,257,618** | |

This buys the strongest available anti-leakage split — train on one season, test on a
*later* one — at the cost of confounding the season change with a provider change. That
trade-off was accepted deliberately, so the project's job is to **measure the confound
rather than hide it**. That is what `scripts/validate_harmonization.py` exists for, and its
output is reproduced below.

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

| Split | `position` | `topology` | `both` | `both − position` |
|---|---|---|---|---|
| within-season (control, single provider) | 0.8915 ± 0.0062 | 0.7659 ± 0.0045 | **0.9022 ± 0.0007** | **+1.07 pp ± 0.57** |
| cross-season (confounded) | 0.7769 ± 0.0102 | 0.6777 ± 0.0095 | **0.7916 ± 0.0026** | **+1.47 pp ± 0.98** |

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

Stability diagnostics (cosine lift = same-player minus different-player similarity):

| Representation | within-player lift | cross-season lift (n=199) |
|---|---|---|
| centrality (baseline) | 0.295 | 0.651 |
| gnn-topology | 0.419 | 0.728 |
| gnn-position | 0.581 | 0.762 |
| **gnn-both** | **0.609** | **0.814** |

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
python scripts/run_centrality.py             # Phase 2
python scripts/train_roles.py                # Phase 3
python scripts/train_roles.py --split within_season   # unconfounded control
pytest                                       # 32 tests
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
- **Serie A has no 360 freeze-frame data in either season.** Module 5's pass-choice RL agent
  will therefore need a different competition (Euro 2024 is the candidate), which departs
  from the Serie A framing. Note also that 360 freeze-frames are anonymous and only
  partially visible (18 of 22 players in a sampled frame; `teammate`/`actor`/`keeper` flags
  and no player ids), so the action space is position slots, not named players.
- **Event data is a partial representation of football.** Off-ball movement, verbal
  communication and tactical intent are not observable here, and no amount of modelling
  recovers them.
- **Reduced scale by design.** Two seasons of one competition, small models, short training
  schedules. These are proof-of-concept results, not state-of-the-art claims.
- **Three seeds** per configuration. Enough to show the ablation ordering is stable; not
  enough for tight confidence intervals on a ~1 pp effect.

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
  models/
    role_gnn.py        GraphSAGE + the three feature sets
  eval/
    splits.py          temporal splits; rejects random splits
    clustering.py      representation comparison + stability diagnostics
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
