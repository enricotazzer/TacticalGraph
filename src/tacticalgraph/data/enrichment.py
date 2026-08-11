"""StatsBomb-only fields, extracted for validation purposes only.

These columns have no Wyscout counterpart, so they can never be model inputs (see
`schema.assert_no_enrichment_leakage`). They exist so that the harmonisation choices made
for *both* providers can be scored against ground truth on the season where ground truth
happens to exist:

  * `true_recipient_id`   -> scores the inferred recipient (`recipient.py`)
  * `statsbomb_possession`-> scores the reconstructed chains (`possession.py`)
  * `position_name_24`    -> held-out validation signal for Module 2's role embeddings
  * `statsbomb_xg`        -> sanity reference for any xG/xT we fit ourselves

Keeping them in a physically separate table (and a separate directory) is deliberate:
it makes accidental use in a feature matrix an import away rather than a column away.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from tacticalgraph.config import Paths
from tacticalgraph.data.download import load_json

log = logging.getLogger(__name__)


def statsbomb_event_truth(paths: Paths, game_id: int) -> pd.DataFrame:
    """Per-event ground truth keyed by StatsBomb event uuid (`original_event_id`)."""
    events: list[dict[str, Any]] = load_json(
        paths.raw_statsbomb / "events" / f"{game_id}.json"
    )
    rows = []
    for event in events:
        pass_info = event.get("pass") or {}
        shot_info = event.get("shot") or {}
        rows.append(
            {
                "original_event_id": event["id"],
                "true_recipient_id": (pass_info.get("recipient") or {}).get("id"),
                "statsbomb_possession": event.get("possession"),
                "statsbomb_xg": shot_info.get("statsbomb_xg"),
                "under_pressure": bool(event.get("under_pressure", False)),
                "play_pattern": (event.get("play_pattern") or {}).get("name"),
                # Position at the moment of the action: StatsBomb's 24-class vocabulary.
                "position_name_24": (event.get("position") or {}).get("name"),
                "position_id_24": (event.get("position") or {}).get("id"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["game_id"] = game_id
    return frame


def statsbomb_lineup_positions(paths: Paths, game_id: int) -> pd.DataFrame:
    """Per game-player nominal position and minutes, from the lineups file.

    StatsBomb records a list of position spells per player with `from`/`to` stamps. We
    keep the spell the player spent longest in as their nominal position for the match,
    and sum spell durations for minutes played.
    """
    lineups: list[dict[str, Any]] = load_json(
        paths.raw_statsbomb / "lineups" / f"{game_id}.json"
    )

    def _to_minutes(stamp: str | None) -> float | None:
        if not stamp:
            return None
        parts = stamp.split(":")
        try:
            if len(parts) == 3:
                hours, minutes, seconds = (float(p) for p in parts)
                return hours * 60 + minutes + seconds / 60
            minutes, seconds = (float(p) for p in parts)
            return minutes + seconds / 60
        except ValueError:
            return None

    rows = []
    for team in lineups:
        for player in team["lineup"]:
            spells = player.get("positions") or []
            best_position, best_duration, total = None, -1.0, 0.0
            for spell in spells:
                start = _to_minutes(spell.get("from")) or 0.0
                # `to` is null when the player finished the match; 90' is the convention.
                end = _to_minutes(spell.get("to"))
                end = 90.0 if end is None else end
                duration = max(end - start, 0.0)
                total += duration
                if duration > best_duration:
                    best_duration, best_position = duration, spell.get("position")
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team["team_id"],
                    "team_name": team["team_name"],
                    "player_id": player["player_id"],
                    "player_name": player.get("player_nickname")
                    or player["player_name"],
                    "position_name_24": best_position,
                    "minutes_played": total,
                    "started": bool(
                        spells and spells[0].get("start_reason") == "Starting XI"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_enrichment(paths: Paths, game_ids: list[int]) -> dict[str, Path]:
    """Write the two StatsBomb-only validation tables under `enrichment/`."""
    paths.enrichment.mkdir(parents=True, exist_ok=True)

    event_frames, lineup_frames = [], []
    for index, game_id in enumerate(game_ids, start=1):
        try:
            event_frames.append(statsbomb_event_truth(paths, game_id))
            lineup_frames.append(statsbomb_lineup_positions(paths, game_id))
        except FileNotFoundError:
            log.warning("enrichment: missing raw files for game %s", game_id)
        if index % 100 == 0:
            log.info("enrichment: %d/%d games", index, len(game_ids))

    written: dict[str, Path] = {}
    if event_frames:
        dest = paths.enrichment / "statsbomb_event_truth.parquet"
        pd.concat(event_frames, ignore_index=True).to_parquet(dest, index=False)
        written["events"] = dest
    if lineup_frames:
        dest = paths.enrichment / "statsbomb_lineup_positions.parquet"
        pd.concat(lineup_frames, ignore_index=True).to_parquet(dest, index=False)
        written["lineups"] = dest
    return written


def load_enrichment(paths: Paths, name: str) -> pd.DataFrame:
    path = paths.enrichment / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python scripts/build_spadl.py` first"
        )
    return pd.read_parquet(path)
