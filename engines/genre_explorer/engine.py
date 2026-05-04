"""
GenreExplorerEngine — highest novelty_bias engine.

Identifies genre buckets with zero or very few rated tracks and actively
explores them by querying Last.fm for top tracks in those genres, then
searching Tidal. The engine deliberately targets the least-heard genres
to prevent the user from getting stuck in familiar territory.

Cold-start: with no rating history, all genres are equally unexplored,
so the engine rotates through all buckets alphabetically.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track
from core.diversity import GENRE_BUCKETS, classify_genre

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

# Canonical set of genre buckets sourced from diversity.py
_ALL_GENRES: list[str] = sorted(set(GENRE_BUCKETS.values()))

# Maps genre bucket names to a good Last.fm tag for searching
_GENRE_TO_LASTFM_TAG: dict[str, str] = {
    "Electronic": "electronic",
    "Rock": "rock",
    "Metal": "metal",
    "Pop": "pop",
    "Hip-Hop/R&B": "hip-hop",
    "Jazz": "jazz",
    "Classical": "classical",
    "Folk/Country": "folk",
    "Blues": "blues",
    "Reggae/World": "reggae",
    "Experimental": "experimental",
}


def _load_settings() -> dict:
    try:
        if not _SETTINGS_PATH.exists():
            return {}
        with _SETTINGS_PATH.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _stable_track_id(tidal_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tidal:{tidal_id}"))


class GenreExplorerEngine(BaseEngine):
    """
    Actively explores under-represented genres from the user's rating history.
    Queries Last.fm for top tracks in the least-explored genres, then searches
    Tidal. Returns tracks that fill the biggest gaps in genre coverage.
    """

    def __init__(self, lastfm=None, tidal=None) -> None:
        """
        Pass pre-built connectors for testing.
        When None, loads config from settings.yaml.
        """
        self._lastfm = lastfm if lastfm is not None else self._try_build_lastfm()
        self._tidal = tidal if tidal is not None else self._try_build_tidal()

    def _try_build_lastfm(self):
        cfg = _load_settings().get("lastfm", {})
        if not cfg.get("api_key"):
            return None
        try:
            from connectors.lastfm import LastFmConnector
            conn = LastFmConnector(cfg)
            conn.connect()
            return conn
        except Exception:
            logger.debug("GenreExplorerEngine: could not build LastFmConnector", exc_info=True)
            return None

    def _try_build_tidal(self):
        cfg = _load_settings().get("tidal", {})
        if not (cfg.get("client_id") or cfg.get("token_path")):
            return None
        try:
            from connectors.tidal import TidalConnector
            conn = TidalConnector(cfg)
            conn.authenticate()
            return conn
        except Exception:
            logger.debug("GenreExplorerEngine: could not build TidalConnector", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def health_check(self) -> EngineHealth:
        if self._lastfm is None or not self._lastfm.is_available():
            return EngineHealth(status="unavailable", message="Last.fm not configured")
        if self._tidal is None or not self._tidal.is_available():
            return EngineHealth(
                status="degraded",
                message="Tidal not authenticated — genre exploration works but no Tidal tracks",
            )
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if self._lastfm is None or not self._lastfm.is_available():
            return []
        try:
            return self._run(n, context)
        except Exception:
            logger.exception("GenreExplorerEngine.suggest() failed")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, n: int, context: SessionContext) -> list[Suggestion]:
        excluded = context.excluded_track_ids

        # Count rated tracks per genre bucket to identify under-represented genres
        genre_counts: dict[str, int] = {g: 0 for g in _ALL_GENRES}
        for rt in context.rated_tracks:
            bucket = classify_genre(rt.track.genre_primary, rt.track.genre_tags)
            if bucket in genre_counts:
                genre_counts[bucket] += 1

        # Prioritize genres with fewest rated tracks
        target_genres = sorted(_ALL_GENRES, key=lambda g: genre_counts[g])

        suggestions: list[Suggestion] = []
        # Distribute slots evenly across target genres; at least 1 track per genre
        slots_per_genre = max(1, (n + len(target_genres) - 1) // len(target_genres))

        for genre in target_genres:
            if len(suggestions) >= n:
                break
            genre_suggs = self._explore_genre(genre, excluded, slots_per_genre)
            suggestions.extend(genre_suggs)

        return suggestions[:n]

    def _explore_genre(
        self,
        genre: str,
        excluded: set[str],
        limit: int,
    ) -> list[Suggestion]:
        """Fetch Last.fm top tracks for a genre bucket, find them on Tidal."""
        lastfm_tag = _GENRE_TO_LASTFM_TAG.get(genre, genre.lower().split("/")[0])

        try:
            lastfm_tracks = self._lastfm.get_tag_top_tracks(lastfm_tag, limit=30)
        except Exception:
            logger.debug(f"GenreExplorerEngine: get_tag_top_tracks({lastfm_tag!r}) failed", exc_info=True)
            return []

        results: list[Suggestion] = []
        for lf_track in lastfm_tracks:
            if len(results) >= limit:
                break
            if not self._tidal or not self._tidal.is_available():
                break
            try:
                query = f"{lf_track.title} {lf_track.artist}"
                tidal_results = self._tidal.search_tracks(query, limit=2)
            except Exception:
                continue

            for t in tidal_results:
                track_id = _stable_track_id(t.tidal_id)
                if track_id in excluded:
                    continue

                track = Track(
                    id=track_id,
                    title=t.title,
                    artist=t.artist,
                    album=t.album,
                    duration_ms=t.duration_ms,
                    genre_primary=genre,   # known — this is what we searched for
                    tidal_id=t.tidal_id,
                )
                results.append(Suggestion(
                    track=track,
                    engine_name=self.name,
                    engine_score=0.7,
                    explanation=f"Exploring {genre} — a genre you haven't heard much of yet",
                ))
                break  # one Tidal result per Last.fm track is enough

        return results
