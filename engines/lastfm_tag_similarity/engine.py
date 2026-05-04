"""
LastFmTagSimilarityEngine — cosine similarity over per-track Last.fm tags.

Why this beats genre-level ranking:
  Last.fm's crowd-sourced tags are track-level and granular: "ambient",
  "melancholic", "instrumental", "driving", "late night", "female vocalists",
  etc. These cross genre boundaries and capture the *texture* the user
  actually responds to. Genre-level ranking (Electronic, Jazz, ...) is too
  coarse to distinguish "melodic techno" from "hard trance".

Algorithm:
  1. Fetch per-track tags from Last.fm for every liked track (score >= LIKE_THRESHOLD).
  2. Build a preference vector: dict[tag → weight].
     Each liked track contributes its tags weighted by (score / 10.0).
     Tags are summed across all liked tracks.
  3. Source candidate tracks from Last.fm: call get_tag_top_tracks() for each
     top-preference tag (top N_SOURCE_TAGS), up to CANDIDATE_POOL_SIZE total.
  4. Fetch tags for each candidate track; build its tag vector.
  5. Score each candidate by cosine similarity to the preference vector.
  6. Search Tidal for the best-scoring candidates, return Suggestions.

health_check:
  - "unavailable" if Last.fm not configured
  - "unavailable" if fewer than MIN_LIKED_TRACKS tracks rated >= LIKE_THRESHOLD
  - "ok" otherwise

Tag fetch is slow (~0.2 s/track) so the engine is marked speed: slow.
It runs in the background; the orchestrator oversample buffers for this.
"""
from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track
from core.diversity import classify_genre

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

# Min liked tracks before the engine becomes available
_MIN_LIKED_TRACKS = 5

# Tracks scored >= this contribute to the preference vector
_LIKE_THRESHOLD = 7

# How many top-preference tags to source candidates from
_N_SOURCE_TAGS = 8

# Last.fm candidate pool size per tag
_CANDIDATES_PER_TAG = 20

# Max total candidate tracks to score (cap API calls)
_CANDIDATE_POOL_SIZE = 60


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


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse tag vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return min(1.0, dot / (norm_a * norm_b))


class LastFmTagSimilarityEngine(BaseEngine):
    """
    Finds tracks that match the user's taste at the tag level —
    far more granular than genre buckets.
    """

    def __init__(self, lastfm=None, tidal=None) -> None:
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
            logger.debug("LastFmTagSimilarityEngine: could not build LastFmConnector", exc_info=True)
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
            logger.debug("LastFmTagSimilarityEngine: could not build TidalConnector", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def health_check(self) -> EngineHealth:
        if self._lastfm is None or not self._lastfm.is_available():
            return EngineHealth(status="unavailable", message="Last.fm not configured")
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if self._lastfm is None or not self._lastfm.is_available():
            return []
        try:
            return self._run(n, context)
        except Exception:
            logger.exception("LastFmTagSimilarityEngine.suggest() failed")
            return []

    def on_session_complete(self, ratings) -> None:
        pass  # stateless — preference vector built fresh from context each session

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, n: int, context: SessionContext) -> list[Suggestion]:
        excluded = context.excluded_track_ids

        liked = [rt for rt in context.rated_tracks if rt.score >= _LIKE_THRESHOLD]
        if len(liked) < _MIN_LIKED_TRACKS:
            logger.debug(
                "LastFmTagSimilarityEngine: only %d liked tracks (need %d)",
                len(liked), _MIN_LIKED_TRACKS,
            )
            return []

        # Step 1 — Build preference vector from liked tracks' tags
        pref_vec = self._build_preference_vector(liked)
        if not pref_vec:
            return []

        # Step 2 — Top tags drive candidate sourcing
        top_tags = sorted(pref_vec, key=lambda t: -pref_vec[t])[:_N_SOURCE_TAGS]

        # Step 3 — Gather candidate tracks from Last.fm tag charts
        candidate_tracks: list[tuple[str, str]] = []  # (title, artist)
        seen_candidates: set[tuple[str, str]] = set()

        for tag in top_tags:
            if len(candidate_tracks) >= _CANDIDATE_POOL_SIZE:
                break
            try:
                lf_tracks = self._lastfm.get_tag_top_tracks(tag, limit=_CANDIDATES_PER_TAG)
            except Exception:
                logger.debug(f"LastFmTagSimilarityEngine: get_tag_top_tracks({tag!r}) failed", exc_info=True)
                continue
            for lf in lf_tracks:
                key = (lf.title.lower(), lf.artist.lower())
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidate_tracks.append((lf.title, lf.artist))

        if not candidate_tracks:
            return []

        # Step 4 — Score candidates by cosine similarity
        scored: list[tuple[float, str, str]] = []  # (score, title, artist)
        for title, artist in candidate_tracks[:_CANDIDATE_POOL_SIZE]:
            try:
                tag_pairs = self._lastfm.get_track_tags(title, artist, limit=15)
            except Exception:
                logger.debug(f"LastFmTagSimilarityEngine: get_track_tags failed for {artist!r} - {title!r}", exc_info=True)
                continue
            if not tag_pairs:
                continue
            cand_vec = {tag: weight for tag, weight in tag_pairs}
            sim = _cosine(pref_vec, cand_vec)
            if sim > 0:
                scored.append((sim, title, artist))

        scored.sort(key=lambda x: -x[0])

        # Step 5 — Search Tidal for best candidates
        suggestions: list[Suggestion] = []
        for sim, title, artist in scored:
            if len(suggestions) >= n:
                break
            if self._tidal is None or not self._tidal.is_available():
                break
            try:
                tidal_results = self._tidal.search_tracks(f"{title} {artist}", limit=2)
            except Exception:
                continue
            for t in tidal_results:
                track_id = _stable_track_id(t.tidal_id)
                if track_id in excluded:
                    continue
                genre = self._infer_genre(t.artist, t.title)
                track = Track(
                    id=track_id,
                    title=t.title,
                    artist=t.artist,
                    album=t.album,
                    duration_ms=t.duration_ms,
                    genre_primary=genre,
                    tidal_id=t.tidal_id,
                )
                suggestions.append(Suggestion(
                    track=track,
                    engine_name=self.name,
                    engine_score=sim,
                    explanation=(
                        f"Tag similarity {sim:.0%} to your liked tracks "
                        f"(matched on: {', '.join(top_tags[:3])})"
                    ),
                ))
                break  # one track per candidate artist query

        return suggestions[:n]

    def _build_preference_vector(self, liked: list) -> dict[str, float]:
        """Weighted sum of per-track tag vectors for all liked tracks."""
        pref: dict[str, float] = {}
        for rt in liked:
            weight = rt.score / 10.0
            try:
                tag_pairs = self._lastfm.get_track_tags(rt.track.title, rt.track.artist, limit=15)
            except Exception:
                logger.debug(
                    f"LastFmTagSimilarityEngine: skipping {rt.track.artist!r} - {rt.track.title!r} (tag fetch failed)",
                    exc_info=True,
                )
                continue
            for tag, tag_weight in tag_pairs:
                pref[tag] = pref.get(tag, 0.0) + tag_weight * weight
        return pref

    def _infer_genre(self, artist: str, title: str) -> str:
        try:
            tag_pairs = self._lastfm.get_track_tags(title, artist, limit=5)
            tags = [t for t, _ in tag_pairs]
            return classify_genre("", tags)
        except Exception:
            try:
                artist_tags = self._lastfm.get_artist_tags(artist, limit=5)
                return classify_genre("", artist_tags)
            except Exception:
                return "Unknown"
