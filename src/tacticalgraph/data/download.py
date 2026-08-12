"""Raw data acquisition for both providers.

Two very different sources:

* **StatsBomb** -- one JSON file per match, fetched straight from the open-data GitHub
  repo. We use plain HTTP rather than `statsbombpy` because it is one fewer moving part
  and gives us direct control over retries and resumption across 380 matches.
* **Wyscout** -- a handful of zipped bundles on figshare covering 7 competitions at once;
  we keep only the Italy members (Serie A 2017/18).

Everything is idempotent and resumable: a file that already exists on disk and parses as
JSON is skipped. That matters because the StatsBomb pull is ~1.1 GB over 760 requests and
Kaggle sessions get killed on a timer.
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from tacticalgraph.config import (
    STATSBOMB_BASE_URL,
    STATSBOMB_COMPETITION_ID,
    STATSBOMB_SEASON_ID,
    WYSCOUT_COLLECTION_URL,
    WYSCOUT_COUNTRY,
    Paths,
)

log = logging.getLogger(__name__)

USER_AGENT = "TacticalGraph/0.1 (research; contact via repo)"
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0


# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------


class TruncatedDownload(OSError):
    """Server closed the connection before sending everything it promised."""


def _fetch_bytes(url: str, timeout: int = 120) -> bytes:
    """GET with bounded exponential backoff on transient failures.

    The retry net is deliberately wide. Fetching the 74 MB Wyscout bundle over a flaky
    link raised `http.client.IncompleteRead`, which is neither a URLError nor a
    ConnectionError, so a narrower `except` silently lost the whole download. We also
    verify the payload against Content-Length, because a short read that does *not* raise
    is worse: it yields a corrupt zip that fails much later with a confusing error.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected = response.headers.get("Content-Length")
                payload = response.read()
            if expected is not None and len(payload) != int(expected):
                raise TruncatedDownload(
                    f"expected {int(expected)} bytes, got {len(payload)}"
                )
            return payload
        except (
            urllib.error.URLError,
            http.client.HTTPException,  # includes IncompleteRead
            TimeoutError,
            OSError,  # includes ConnectionError and TruncatedDownload
        ) as exc:
            last_error = exc
            # A 404 is a fact about the data, not a transient glitch: do not retry it.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            sleep_for = BACKOFF_SECONDS * (2**attempt)
            log.warning("fetch failed (%s), retrying in %.0fs: %s", exc, sleep_for, url)
            time.sleep(sleep_for)
    raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts") from last_error


def _is_valid_json_file(path: Path) -> bool:
    """A previous run may have died mid-write, leaving truncated JSON behind."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as handle:
            json.load(handle)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def _write_json_atomic(path: Path, payload: bytes) -> None:
    """Write via temp file so an interrupted run never leaves a half-file that later
    looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _download_json(url: str, dest: Path, force: bool = False) -> bool:
    """Fetch `url` to `dest`. Returns True if a download happened, False if skipped."""
    if not force and _is_valid_json_file(dest):
        return False
    _write_json_atomic(dest, _fetch_bytes(url))
    return True


# --------------------------------------------------------------------------------------
# StatsBomb
# --------------------------------------------------------------------------------------


def download_statsbomb_matches(
    paths: Paths,
    force: bool = False,
    competition_id: int = STATSBOMB_COMPETITION_ID,
    season_id: int = STATSBOMB_SEASON_ID,
) -> list[dict[str, Any]]:
    """Fetch one competition-season's match index plus the competition index.

    The on-disk layout deliberately mirrors what `socceraction`'s `StatsBombLoader`
    expects in local mode (`competitions.json`, `matches/{comp}/{season}.json`,
    `events/{game}.json`, `lineups/{game}.json`), so we can reuse the official loader and
    converter instead of hand-parsing StatsBomb JSON.

    `events/` and `lineups/` are keyed by globally unique match id, so several
    competition-seasons coexist in one raw cache without collision -- which is why `raw/`
    sits outside the per-corpus namespace.
    """
    _download_json(
        f"{STATSBOMB_BASE_URL}/competitions.json",
        paths.raw_statsbomb / "competitions.json",
        force=force,
    )

    dest = paths.raw_statsbomb / "matches" / str(competition_id) / f"{season_id}.json"
    url = f"{STATSBOMB_BASE_URL}/matches/{competition_id}/{season_id}.json"
    _download_json(url, dest, force=force)
    matches: list[dict[str, Any]] = json.loads(dest.read_text())
    log.info(
        "StatsBomb match index (comp %s season %s): %d matches",
        competition_id,
        season_id,
        len(matches),
    )
    return matches


def download_statsbomb_season(
    paths: Paths,
    force: bool = False,
    limit: int | None = None,
    competition_id: int = STATSBOMB_COMPETITION_ID,
    season_id: int = STATSBOMB_SEASON_ID,
) -> dict[str, int]:
    """Fetch events + lineups for every match of one competition-season.

    ~380 matches x (3 MB events + small lineup). Resumable; safe to re-run.
    """
    matches = download_statsbomb_matches(
        paths, force=force, competition_id=competition_id, season_id=season_id
    )
    match_ids = [m["match_id"] for m in matches]
    if limit is not None:
        # Chronological, so a --limit subset is a prefix of the season rather than an
        # arbitrary sample. Keeps small dev runs interpretable.
        ordered = sorted(matches, key=lambda m: (m["match_date"], m.get("kick_off") or ""))
        match_ids = [m["match_id"] for m in ordered[:limit]]

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    for index, match_id in enumerate(match_ids, start=1):
        for kind in ("events", "lineups"):
            dest = paths.raw_statsbomb / kind / f"{match_id}.json"
            url = f"{STATSBOMB_BASE_URL}/{kind}/{match_id}.json"
            try:
                if _download_json(url, dest, force=force):
                    stats["downloaded"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad match must not kill the run
                log.error("match %s %s failed: %s", match_id, kind, exc)
                stats["failed"] += 1
        if index % 50 == 0:
            log.info("StatsBomb progress: %d/%d matches", index, len(match_ids))
    log.info("StatsBomb done: %s", stats)
    return stats


# --------------------------------------------------------------------------------------
# Wyscout
# --------------------------------------------------------------------------------------


def _figshare_articles() -> list[dict[str, Any]]:
    payload = _fetch_bytes(f"{WYSCOUT_COLLECTION_URL}?page_size=50")
    return json.loads(payload)


def _figshare_files(article_url: str) -> list[dict[str, Any]]:
    return json.loads(_fetch_bytes(article_url)).get("files", [])


def _extract_country_members(archive: bytes, dest_dir: Path, country: str) -> list[Path]:
    """Pull only the `*_<country>.json` members out of a multi-competition zip.

    The events bundle holds all 7 competitions (~950 MB unpacked); we want the ~190 MB
    Italy member and nothing else.
    """
    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        members = [n for n in zf.namelist() if country.lower() in n.lower()]
        if not members:
            raise RuntimeError(
                f"no member matching country={country!r} in archive; "
                f"members were {zf.namelist()}"
            )
        for member in members:
            dest = dest_dir / Path(member).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            written.append(dest)
    return written


def download_wyscout(paths: Paths, force: bool = False) -> dict[str, Any]:
    """Fetch the Wyscout Serie A 2017/18 bundle from figshare.

    Zipped articles (Events, Matches) are filtered down to the Italy member. Flat JSON
    articles (Players, Teams) are league-agnostic reference tables and kept whole.
    """
    dest_dir = paths.raw_wyscout
    dest_dir.mkdir(parents=True, exist_ok=True)

    zipped = {"Events", "Matches"}
    flat = {"Players", "Teams", "Competitions"}
    articles = {a["title"]: a for a in _figshare_articles()}

    missing = (zipped | flat) - articles.keys()
    if missing:
        raise RuntimeError(f"figshare collection is missing articles: {sorted(missing)}")

    written: dict[str, Any] = {}
    for title in sorted(zipped | flat):
        files = _figshare_files(articles[title]["url"])
        if not files:
            raise RuntimeError(f"figshare article {title!r} exposes no files")
        descriptor = files[0]
        marker = dest_dir / f".{title.lower()}.done"

        if marker.exists() and not force:
            written[title] = "skipped"
            continue

        log.info("Wyscout: fetching %s (%.1f MB)", title, descriptor["size"] / 1e6)
        payload = _fetch_bytes(descriptor["download_url"], timeout=600)

        if title in zipped:
            members = _extract_country_members(payload, dest_dir, WYSCOUT_COUNTRY)
            written[title] = [p.name for p in members]
        else:
            # socceraction's PublicWyscoutLoader reads these by exact filename
            # (`players.json`, `teams.json`), so normalise rather than trusting whatever
            # figshare happens to call the attachment.
            dest = dest_dir / f"{title.lower()}.json"
            dest.write_bytes(payload)
            written[title] = dest.name

        marker.write_text(str(descriptor["size"]))

    log.info("Wyscout done: %s", written)
    return written


# --------------------------------------------------------------------------------------
# Loading helpers used by the adapters
# --------------------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def statsbomb_match_ids(paths: Paths) -> list[int]:
    """Match ids that actually have a valid events file on disk."""
    events_dir = paths.raw_statsbomb / "events"
    if not events_dir.exists():
        return []
    ids = []
    for path in events_dir.glob("*.json"):
        if path.name.startswith("._"):  # exFAT AppleDouble sidecar
            continue
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def wyscout_file(paths: Paths, prefix: str) -> Path:
    """Locate a Wyscout JSON by prefix, e.g. 'events' -> events_Italy.json."""
    candidates = [
        p
        for p in paths.raw_wyscout.glob(f"{prefix}*.json")
        if not p.name.startswith("._")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no Wyscout file starting with {prefix!r} in {paths.raw_wyscout}; "
            "run `python scripts/ingest.py --wyscout` first"
        )
    return candidates[0]


def iter_chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
