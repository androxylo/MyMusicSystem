"""
LastFmGraphEngine — traverses the Last.fm artist similarity graph.

Starting from artists the user has rated highly (score >= 7), the engine
traverses up to 2 hops of artist similarity. Candidates are scored by
similarity strength and penalized when the user has already rated tracks
by that artist. Tidal is searched for top tracks by the best candidates.

Cold-start: when rated_tracks is empty, seeds from the user's Last.fm
top artists (requires lastfm.username in settings.yaml).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track
from core.diversity import classify_genre

logger = logging.getLogger(__name__)

_DATA_SETTINGS_PATH = Path(__file__).parent.parent.parent / "data" / "settings.yaml"
_CONFIG_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_SETTINGS_PATH = _DATA_SETTINGS_PATH if _DATA_SETTINGS_PATH.exists() else _CONFIG_SETTINGS_PATH

_SEED_SCORE_THRESHOLD = 7    # min user score to use an artist as a seed
_HIGH_SCORE_THRESHOLD = 8    # artists above this score trigger 2-hop traversal
_SIMILARITY_2ND_HOP = 0.6   # min match score to recurse into second hop


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


class LastFmGraphEngine(BaseEngine):
    """
    Discovers new music by traversing the Last.fm artist similarity graph,
    then searching Tidal for top tracks by discovered artists.
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
            logger.debug("LastFmGraphEngine: could not build LastFmConnector", exc_info=True)
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
            logger.debug("LastFmGraphEngine: could not build TidalConnector", exc_info=True)
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
                message="Tidal not authenticated — graph traversal works but no Tidal tracks",
            )
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if self._lastfm is None or not self._lastfm.is_available():
            return []
        try:
            return self._run(n, context)
        except Exception:
            logger.exception("LastFmGraphEngine.suggest() failed")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, n: int, context: SessionContext) -> list[Suggestion]:
        excluded = context.excluded_track_ids

        # Gather seed artists from highly-rated tracks
        seed_artists: dict[str, float] = {}  # artist_name → max score
        for rt in context.rated_tracks:
            if rt.score >= _SEED_SCORE_THRESHOLD:
                name = rt.track.artist
                seed_artists[name] = max(seed_artists.get(name, 0), float(rt.score))

        # Cold-start: fall back to Last.fm user top artists
        if not seed_artists:
            seed_artists = self._cold_start_seeds()

        if not seed_artists:
            logger.debug("LastFmGraphEngine: no seed artists — returning empty")
            return []

        rated_artists = {rt.track.artist for rt in context.rated_tracks}

        # Traverse graph to build candidate artist pool
        candidate_artists: dict[str, float] = {}  # artist_name → composite_score
        for artist, score in sorted(seed_artists.items(), key=lambda x: -x[1]):
            max_hops = 2 if score >= _HIGH_SCORE_THRESHOLD else 1
            self._traverse(artist, score, max_hops, rated_artists, candidate_artists, _hop=0)
            if len(candidate_artists) >= n * 5:
                break  # enough candidates

        if not candidate_artists:
            return []

        # Sort candidates and search Tidal for top tracks
        sorted_artists = sorted(candidate_artists.items(), key=lambda x: -x[1])
        suggestions: list[Suggestion] = []

        for artist_name, composite_score in sorted_artists:
            if len(suggestions) >= n:
                break
            new = self._search_tidal(artist_name, composite_score, excluded, limit=2)
            suggestions.extend(new)

        return suggestions[:n]

    def _cold_start_seeds(self) -> dict[str, float]:
        """Load top artists from Last.fm user history for cold-start scenarios."""
        try:
            artists = self._lastfm.get_user_top_artists(period="overall", limit=15)
            return {a.name: 7.0 for a in artists}
        except Exception:
            logger.debug("LastFmGraphEngine: cold-start seed fetch failed", exc_info=True)
            return {}

    def _traverse(
        self,
        artist_name: str,
        source_score: float,
        max_hops: int,
        rated_artists: set[str],
        candidates: dict[str, float],
        _hop: int,
    ) -> None:
        if _hop >= max_hops:
            return
        try:
            similar = self._lastfm.get_similar_artists(artist_name, limit=15)
        except Exception:
            logger.debug(f"LastFmGraphEngine: get_similar_artists failed for {artist_name!r}", exc_info=True)
            return

        for item in similar:
            match = item.match_score
            # Penalize artists the user has already rated (but don't exclude them)
            # apply an exponential weight for 8+ source scores
            novelty = 0.5 if item.name in rated_artists else 1.0
            composite = ((source_score - 6) ** 2) * match * novelty

            if candidates.get(item.name, 0) < composite:
                candidates[item.name] = composite

            # Recurse only from strong matches
            if _hop + 1 < max_hops and match >= _SIMILARITY_2ND_HOP:
                self._traverse(
                    item.name,
                    source_score * match,
                    max_hops,
                    rated_artists,
                    candidates,
                    _hop + 1,
                )

    def _search_tidal(
        self,
        artist_name: str,
        composite_score: float,
        excluded: set[str],
        limit: int,
    ) -> list[Suggestion]:
        if self._tidal is None or not self._tidal.is_available():
            return []
        try:
            tidal_tracks = self._tidal.search_tracks(artist_name, limit=limit + 2)
        except Exception:
            logger.debug(f"LastFmGraphEngine: Tidal search failed for {artist_name!r}", exc_info=True)
            return []

        results: list[Suggestion] = []
        for t in tidal_tracks:
            if len(results) >= limit:
                break
            track_id = _stable_track_id(t.tidal_id)
            if track_id in excluded:
                continue

            genre = self._infer_genre(t.artist)
            track = Track(
                id=track_id,
                title=t.title,
                artist=t.artist,
                album=t.album,
                duration_ms=t.duration_ms,
                genre_primary=genre,
                tidal_id=t.tidal_id,
            )
            results.append(Suggestion(
                track=track,
                engine_name=self.name,
                engine_score=min(1.0, composite_score),
                explanation=f"Similar to artists you like — discovered via Last.fm similarity graph",
            ))

        return results

    def _infer_genre(self, artist_name: str) -> str:
        """Classify genre from Last.fm artist tags. Falls back to 'Unknown'."""
        try:
            tags = self._lastfm.get_artist_tags(artist_name, limit=5)
            return classify_genre("", tags)
        except Exception:
            return "Unknown"
