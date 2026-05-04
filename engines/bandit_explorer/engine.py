"""
BanditExplorerEngine — Thompson Sampling over genre buckets.

For each genre bucket, maintains Beta(α, β) distributions derived from
the full rating history. α = (count liked) + 1, β = (count disliked) + 1,
where "liked" means score >= LIKE_THRESHOLD (default 6) and the +1 terms
are Laplace smoothing priors.

At each session:
  1. Draw a sample from Beta(α, β) for every genre bucket.
  2. Sort genres by their sampled values (descending).
  3. For the top-ranked genres, fetch top tracks from Last.fm and search Tidal.

This is deliberately probabilistic: even a genre with a high success rate
can still lose the Thompson draw, guaranteeing exploration. This is the
primary anti-echo-chamber mechanism at the genre selection level (the
DiversityEnforcer handles per-session deduplication on top of this).

State derives entirely from context.rated_tracks — no separate storage needed.
The engine can be rebuilt from scratch from rating history at any time.
"""
from __future__ import annotations

import logging
import math
import random
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track
from core.diversity import GENRE_BUCKETS, classify_genre

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

# Score threshold: >= this is a "like" for the Beta update
_LIKE_THRESHOLD = 6

# All canonical genre buckets
_ALL_GENRES: list[str] = sorted(set(GENRE_BUCKETS.values()))

# Maps genre bucket names to Last.fm search tags
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


def _thompson_sample(alpha: int, beta: int, rng: random.Random) -> float:
    """
    Draw a sample from Beta(alpha, beta) using the gamma ratio trick.
    Requires alpha >= 1 and beta >= 1 (ensured by Laplace smoothing).
    """
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    denom = x + y
    return x / denom if denom > 0 else 0.5


class BanditExplorerEngine(BaseEngine):
    """
    Thompson Sampling bandit over genre buckets.

    Uses Last.fm to fetch top tracks for the sampled genre(s), then
    searches Tidal to get streamable tracks.
    """

    def __init__(self, lastfm=None, tidal=None, seed: int | None = None) -> None:
        """
        Pass pre-built connectors for testing.
        seed: optional RNG seed for reproducible testing.
        """
        self._lastfm = lastfm if lastfm is not None else self._try_build_lastfm()
        self._tidal = tidal if tidal is not None else self._try_build_tidal()
        self._rng = random.Random(seed)

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
            logger.debug("BanditExplorerEngine: could not build LastFmConnector", exc_info=True)
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
            logger.debug("BanditExplorerEngine: could not build TidalConnector", exc_info=True)
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
                message="Tidal not authenticated — genre sampling works but no tracks returned",
            )
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if self._lastfm is None or not self._lastfm.is_available():
            return []
        try:
            return self._run(n, context)
        except Exception:
            logger.exception("BanditExplorerEngine.suggest() failed")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_beta_params(
        self, rated_tracks: list
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Compute Beta(α, β) parameters for each genre from rated_tracks.
        Both start at 1 (Laplace prior) so samples are always valid.
        """
        alpha = {g: 1 for g in _ALL_GENRES}
        beta = {g: 1 for g in _ALL_GENRES}

        for rt in rated_tracks:
            bucket = classify_genre(rt.track.genre_primary, rt.track.genre_tags)
            if bucket not in alpha:
                continue
            if rt.score >= _LIKE_THRESHOLD:
                alpha[bucket] += 1
            else:
                beta[bucket] += 1

        return alpha, beta

    def _rank_genres_by_thompson(
        self, alpha: dict[str, int], beta: dict[str, int]
    ) -> list[str]:
        """Draw Thompson samples and return genres sorted by sampled value (desc)."""
        samples = {
            g: _thompson_sample(alpha[g], beta[g], self._rng)
            for g in _ALL_GENRES
        }
        return sorted(_ALL_GENRES, key=lambda g: -samples[g])

    def _run(self, n: int, context: SessionContext) -> list[Suggestion]:
        excluded = context.excluded_track_ids
        alpha, beta = self._compute_beta_params(context.rated_tracks)
        ranked_genres = self._rank_genres_by_thompson(alpha, beta)

        suggestions: list[Suggestion] = []
        # Try top 5 genres; stop once we have enough candidates
        for genre in ranked_genres[:5]:
            if len(suggestions) >= n:
                break
            slots = max(1, (n - len(suggestions) + 1) // 2)
            genre_suggs = self._explore_genre(genre, excluded, slots)
            suggestions.extend(genre_suggs)

        return suggestions[:n]

    def _explore_genre(
        self,
        genre: str,
        excluded: set[str],
        limit: int,
    ) -> list[Suggestion]:
        """Fetch Last.fm top tracks for a genre, search Tidal, return Suggestions."""
        lastfm_tag = _GENRE_TO_LASTFM_TAG.get(genre, genre.lower().split("/")[0])

        try:
            lastfm_tracks = self._lastfm.get_tag_top_tracks(lastfm_tag, limit=30)
        except Exception:
            logger.debug(f"BanditExplorerEngine: get_tag_top_tracks({lastfm_tag!r}) failed", exc_info=True)
            return []

        results: list[Suggestion] = []
        for lf_track in lastfm_tracks:
            if len(results) >= limit:
                break
            if not self._tidal or not self._tidal.is_available():
                break
            try:
                tidal_results = self._tidal.search_tracks(
                    f"{lf_track.title} {lf_track.artist}", limit=2
                )
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
                    genre_primary=genre,
                    tidal_id=t.tidal_id,
                )
                results.append(Suggestion(
                    track=track,
                    engine_name=self.name,
                    engine_score=0.65,
                    explanation=f"Thompson Sampling selected {genre} — exploring based on your rating history",
                ))
                break  # one Tidal result per Last.fm track

        return results

    def on_session_complete(self, ratings: list[tuple[Track, int]]) -> None:
        # State is fully derived from rated_tracks at suggest() time — no update needed.
        pass
