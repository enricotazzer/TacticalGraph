#!/usr/bin/env python
"""Module 4's human-review harness.

The project's validation plan calls for sampling discovered patterns and judging whether they
are tactically sensible. That judgement needs a person -- it is not something this script (or
its author) can do -- so the script prepares everything a reviewer needs and leaves the
verdict column blank:

    reports/pattern_review_sheet.csv     N chains per cluster, with match and timestamp
    figures/pattern_clusters/*.png       the sampled chains drawn on a pitch

Fill in `sensible_y_n` (and optionally `reviewer_note`), and the README reports the resulting
proportion. Until then the README states the step as pending rather than implying it happened.

    python scripts/review_patterns.py --representation hand-crafted --per-cluster 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tacticalgraph.config import Paths  # noqa: E402
from tacticalgraph.data.aliases import CLUB_DISPLAY, team_id_to_club  # noqa: E402
from tacticalgraph.data.spadl_store import read_actions, read_games  # noqa: E402
from tacticalgraph.features.chains import cluster_profiles  # noqa: E402
from tacticalgraph.viz.pitch import PITCH_KWARGS  # noqa: E402

log = logging.getLogger("review_patterns")


def club(provider: str, team_id: int) -> str:
    key = team_id_to_club(provider).get(int(team_id))
    return CLUB_DISPLAY.get(key, f"team {team_id}") if key else f"team {team_id}"


def draw_chain(ax, chain_actions: pd.DataFrame, title: str) -> None:
    """Draw one possession as an arrow path on a pitch."""
    from mplsoccer import Pitch

    pitch = Pitch(**PITCH_KWARGS)
    pitch.draw(ax=ax)
    if chain_actions.empty:
        ax.set_title(title, fontsize=7)
        return

    ordered = chain_actions.sort_values(["period_id", "time_seconds"])
    pitch.arrows(
        ordered["start_x"], ordered["start_y"], ordered["end_x"], ordered["end_y"],
        width=1.6, headwidth=4, headlength=4, color="#1f6feb", alpha=0.75, ax=ax,
    )
    pitch.scatter(
        ordered["start_x"].iloc[:1], ordered["start_y"].iloc[:1],
        s=90, color="#2ca02c", edgecolors="#22223b", zorder=4, ax=ax,
    )
    shots = ordered[ordered["type_name"].str.startswith("shot")]
    if not shots.empty:
        pitch.scatter(
            shots["start_x"], shots["start_y"],
            s=110, marker="*", color="#d62728", edgecolors="#22223b", zorder=5, ax=ax,
        )
    ax.set_title(title, fontsize=7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", default="hand-crafted",
                        help="cluster column suffix, e.g. hand-crafted or gru-autoencoder")
    parser.add_argument("--per-cluster", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    paths = Paths.load().ensure()

    chain_path = paths.models / "module4_chains.parquet"
    if not chain_path.exists():
        raise FileNotFoundError(
            f"{chain_path} not found; run `python scripts/train_patterns.py` first"
        )
    chains = pd.read_parquet(chain_path)

    column = f"cluster_{args.representation}"
    if column not in chains.columns:
        available = [c.replace("cluster_", "") for c in chains.columns if c.startswith("cluster_")]
        raise ValueError(f"no clustering {args.representation!r}; available: {available}")

    profile = cluster_profiles(chains, chains[column].to_numpy())
    labels = dict(zip(profile["cluster"], profile["label"]))
    shot_rates = dict(zip(profile["cluster"], profile["shot_rate"]))

    # Sample from the TEST fold: patterns are only interesting if they hold on unseen data.
    pool = chains[chains["fold"] == "test"]
    if pool.empty:
        pool = chains
    sample = (
        pool.groupby(column, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), args.per_cluster), random_state=args.seed))
        .reset_index(drop=True)
    )

    games = read_games(paths).set_index("game_id")
    rows = []
    for row in sample.itertuples(index=False):
        game = games.loc[row.game_id]
        rows.append(
            {
                "cluster": int(getattr(row, column)),
                "cluster_label": labels.get(int(getattr(row, column)), ""),
                "cluster_shot_rate": shot_rates.get(int(getattr(row, column)), float("nan")),
                "game_id": int(row.game_id),
                "possession_id": int(row.possession_id),
                "season": row.season,
                "fixture": f"{club(row.provider, game['home_team_id'])} vs "
                           f"{club(row.provider, game['away_team_id'])}",
                "team_in_possession": club(row.provider, row.team_id),
                "period": int(row.period_id),
                "minute": round(float(row.start_minute), 1),
                "n_actions": int(row.n_actions),
                "duration_s": round(float(row.duration_seconds), 1),
                "start_zone": row.start_zone,
                "end_zone": row.end_zone,
                "directness": round(float(row.directness), 3),
                "xt_gain": round(float(row.xt_gain), 4),
                "set_piece_start": bool(row.started_with_set_piece),
                "ends_in_shot": bool(row.ends_in_shot),
                # For the reviewer to fill in:
                "sensible_y_n": "",
                "reviewer_note": "",
            }
        )
    sheet = pd.DataFrame(rows).sort_values(["cluster", "minute"]).reset_index(drop=True)

    destination = paths.reports / f"pattern_review_sheet_{args.representation}.csv"
    sheet.to_csv(destination, index=False)
    log.info("wrote %s (%d chains across %d clusters)",
             destination, len(sheet), sheet["cluster"].nunique())

    # ------------------------------------------------------------------ figures
    actions = read_actions(paths)
    wanted = set(map(tuple, sample[["game_id", "possession_id"]].to_numpy()))
    relevant = actions[
        [
            (int(g), int(p)) in wanted
            for g, p in zip(actions["game_id"], actions["possession_id"])
        ]
    ]

    figure_dir = paths.figures / "pattern_clusters"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for cluster, group in sample.groupby(column):
        n = len(group)
        columns = min(3, n)
        rows_n = (n + columns - 1) // columns
        figure, axes = plt.subplots(rows_n, columns, figsize=(4.6 * columns, 3.1 * rows_n))
        axes = [axes] if n == 1 else list(pd.Series(axes).explode().dropna()) if rows_n > 1 else list(axes)

        for ax, chain in zip(axes, group.itertuples(index=False)):
            chain_actions = relevant[
                (relevant["game_id"] == chain.game_id)
                & (relevant["possession_id"] == chain.possession_id)
            ]
            draw_chain(
                ax,
                chain_actions,
                f"{club(chain.provider, chain.team_id)} {chain.start_minute:.0f}' | "
                f"{int(chain.n_actions)} actions | "
                f"{'SHOT' if chain.ends_in_shot else 'no shot'}",
            )
        for ax in axes[n:]:
            ax.axis("off")

        figure.suptitle(
            f"[{args.representation}] cluster {int(cluster)}: {labels.get(int(cluster), '')} "
            f"(shot rate {shot_rates.get(int(cluster), float('nan')):.3f})",
            fontsize=10,
        )
        figure.tight_layout()
        out = figure_dir / f"{args.representation}_cluster_{int(cluster)}.png"
        figure.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(figure)
        log.info("wrote %s", out)

    print()
    print("=" * 78)
    print("HUMAN REVIEW REQUIRED -- this step cannot be automated")
    print("=" * 78)
    print(f"1. Open {destination}")
    print(f"2. Look at the pitch plots in {figure_dir}")
    print("3. For each row, set `sensible_y_n` to y or n:")
    print("   does the chain match the cluster's stated pattern label?")
    print("4. Re-run the README/report step to publish the proportion judged sensible.")
    print()
    print("Cluster summary to review against:")
    print(
        profile[["cluster", "label", "n_chains", "shot_rate", "n_actions", "directness"]]
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
