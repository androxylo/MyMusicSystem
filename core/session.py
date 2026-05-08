"""
SessionManager — session lifecycle: start, rate, complete.

This module wires together the Orchestrator, Database, and (optionally)
PlaylistManager. Playlist operations are fire-and-forget: a Tidal failure
never blocks a rating.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from core.base_engine import (
    BaseEngine,
    RatedTrack,
    Session,
    SessionConfig,
    SessionContext,
    Track,
)
from core.db import Database
from core.orchestrator import Orchestrator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_NORM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _track_fingerprint(title: str, artist: str) -> str:
    """Normalized 'title|artist' key for cross-session deduplication."""
    def norm(s: str) -> str:
        s = _NORM_RE.sub("", s.lower())
        return _WS_RE.sub(" ", s).strip()
    return f"{norm(title)}|{norm(artist)}"


# ---------------------------------------------------------------------------
# PlaylistManager protocol — implemented in core/playlist_manager.py (Phase 2)
# ---------------------------------------------------------------------------

class PlaylistManagerProtocol(Protocol):
    def sync_candidate_playlist(self, session_id: str, tracks: list[Track]) -> str: ...
    def remove_from_candidate(self, track_id: str) -> None: ...
    def add_to_curated(self, track: Track, score: int) -> None: ...
    def clear_candidate_remaining(self, session_id: str) -> int: ...


# ---------------------------------------------------------------------------
# Session summary returned by complete_session()
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    session_id: str
    avg_rating: float
    top_rated_track: RatedTrack | None
    engine_breakdown: dict[str, dict]  # engine_name → {suggested, in_final, avg_rating}
    genre_stats: dict[str, float]       # genre → avg_rating across this session
    newly_curated_count: int
    newly_curated: list[dict]           # [{title, artist, genre}]


# ---------------------------------------------------------------------------
# Rate-track result
# ---------------------------------------------------------------------------

@dataclass
class RateResult:
    ok: bool
    ratings_so_far: int
    added_to_curated: bool


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages the full lifecycle of a recommendation session.

    Phase 2 note: pass a PlaylistManager instance to enable Tidal playlist sync.
    Until then (Phase 1), pass None and playlist hooks are silently skipped.
    """

    def __init__(
        self,
        db: Database,
        orchestrator: Orchestrator,
        engines: list[BaseEngine],
        curated_threshold: int = 7,
        playlist_manager: PlaylistManagerProtocol | None = None,
    ) -> None:
        self._db = db
        self._orchestrator = orchestrator
        self._engines = engines
        self._curated_threshold = curated_threshold
        self._pm = playlist_manager

    # ------------------------------------------------------------------
    # Session start
    # ------------------------------------------------------------------

    def start_session(self, config: SessionConfig | None = None) -> dict:
        """
        Create a new session, get suggestions, populate the candidate playlist.

        Returns a dict containing:
          session_id, candidate_playlist_url (may be None in Phase 1),
          suggestions list, and engine_allocation.
        """
        if config is None:
            config = SessionConfig()

        session_id = str(uuid.uuid4())
        now = datetime.now()

        # Build context from full rating history + recent sessions
        context = self._build_context(config)

        # Get suggestions from the orchestrator
        suggestions = self._orchestrator.get_suggestions(context)

        allocation = self._orchestrator.get_allocation()

        # Persist the session record
        session = Session(
            id=session_id,
            started_at=now,
            engine_allocation=allocation,
            diversity_config={
                "diversity_mode": config.diversity_mode,
            },
            notes=config.notes,
        )
        self._db.insert_session(session)

        # Persist each suggestion's track and record the engine suggestion
        for rank, s in enumerate(suggestions, start=1):
            self._db.upsert_track(s.track)
            self._db.insert_engine_suggestion(
                session_id=session_id,
                engine_name=s.engine_name,
                track_id=s.track.id,
                engine_score=s.engine_score,
                was_final=True,
                final_rank=rank,
            )

        # Sync candidate playlist (Phase 2)
        candidate_url: str | None = None
        if self._pm is not None:
            try:
                candidate_url = self._pm.sync_candidate_playlist(
                    session_id, [s.track for s in suggestions]
                )
            except Exception:
                logger.exception("PlaylistManager.sync_candidate_playlist failed — continuing")

        return {
            "session_id": session_id,
            "candidate_playlist_url": candidate_url,
            "engine_allocation": allocation,
            "suggestions": [
                {
                    "track_id": s.track.id,
                    "title": s.track.title,
                    "artist": s.track.artist,
                    "album": s.track.album,
                    "genre": s.track.genre_primary,
                    "engine": s.engine_name,
                    "engine_score": round(s.engine_score, 4),
                    "explanation": s.explanation,
                    "tidal_id": s.track.tidal_id,
                }
                for s in suggestions
            ],
        }

    # ------------------------------------------------------------------
    # Rating a track
    # ------------------------------------------------------------------

    def rate_track(self, session_id: str, track_id: str, score: int) -> RateResult:
        """
        Record a rating. Score must be in [1, 10].

        Side effects:
          - Rating persisted to DB
          - Track removed from candidate playlist (if PlaylistManager available)
          - Track added to curated playlists if score >= threshold
        """
        if score < 1 or score > 10:
            raise ValueError(f"Score must be between 1 and 10, got {score}")

        # Verify session exists
        session = self._db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id!r} not found")

        # Verify track exists
        track = self._db.get_track(track_id)
        if track is None:
            raise ValueError(f"Track {track_id!r} not found")

        # Check for duplicate
        existing = self._db.get_rating(session_id, track_id)
        if existing is not None:
            raise ValueError(
                f"Track {track_id!r} already rated in session {session_id!r}"
            )

        # Persist rating (always first — never block on playlist ops)
        self._db.insert_rating(track_id, session_id, score)

        # Count ratings so far this session
        session_ratings = self._db.get_session_ratings(session_id)
        ratings_so_far = len(session_ratings)

        added_to_curated = score >= self._curated_threshold

        # Playlist ops — fire-and-forget
        if self._pm is not None:
            try:
                self._pm.remove_from_candidate(track_id)
            except Exception:
                logger.exception(f"remove_from_candidate({track_id}) failed — ignored")

            if added_to_curated:
                try:
                    self._pm.add_to_curated(track, score)
                except Exception:
                    logger.exception(f"add_to_curated({track_id}) failed — ignored")

        return RateResult(
            ok=True,
            ratings_so_far=ratings_so_far,
            added_to_curated=added_to_curated,
        )

    # ------------------------------------------------------------------
    # Session completion
    # ------------------------------------------------------------------

    def complete_session(self, session_id: str) -> SessionSummary:
        """
        Mark session complete, notify engines, compute and return summary.
        """
        session = self._db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id!r} not found")

        now = datetime.now()
        self._db.complete_session(session_id, now)

        # Load all ratings for this session
        rated = self._db.get_session_ratings(session_id)

        # Notify engines
        pairs: list[tuple[Track, int]] = [(rt.track, rt.score) for rt in rated]
        for engine in self._engines:
            try:
                engine.on_session_complete(pairs)
            except Exception:
                logger.exception(f"Engine {engine.name!r} raised in on_session_complete — ignored")

        # Record engine performance
        suggestions = self._db.get_session_engine_suggestions(session_id)
        self._record_engine_performance(session_id, rated, suggestions)

        # Update genre session history
        self._update_genre_history(session_id, rated)

        # Compile summary
        summary = self._build_summary(session_id, rated, suggestions)

        # Clear candidate playlist (Phase 2)
        if self._pm is not None:
            try:
                self._pm.clear_candidate_remaining(session_id)
            except Exception:
                logger.exception("clear_candidate_remaining failed — ignored")

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, config: SessionConfig) -> SessionContext:
        all_rated = self._db.get_all_rated_tracks()
        recent_sessions = self._db.get_recent_sessions(n=10)
        excluded = self._db.get_all_rated_track_ids()
        fingerprints = {
            _track_fingerprint(rt.track.title, rt.track.artist)
            for rt in all_rated
        }
        return SessionContext(
            rated_tracks=all_rated,
            recent_sessions=recent_sessions,
            session_config=config,
            excluded_track_ids=excluded,
            excluded_track_fingerprints=fingerprints,
        )

    def _record_engine_performance(
        self,
        session_id: str,
        rated: list[RatedTrack],
        suggestions: list[dict],
    ) -> None:
        # Build a map of track_id → score for this session
        score_map = {rt.track.id: rt.score for rt in rated}

        # Group suggestions by engine
        by_engine: dict[str, list[dict]] = {}
        for s in suggestions:
            by_engine.setdefault(s["engine_name"], []).append(s)

        for engine_name, engine_suggs in by_engine.items():
            tracks_suggested = len(engine_suggs)
            final_suggs = [s for s in engine_suggs if s.get("was_final")]
            final_track_ids = [s["track_id"] for s in final_suggs]

            rated_scores = [score_map[tid] for tid in final_track_ids if tid in score_map]
            avg_rating = sum(rated_scores) / len(rated_scores) if rated_scores else 0.0

            self._db.insert_engine_performance(
                engine_name=engine_name,
                session_id=session_id,
                tracks_suggested=tracks_suggested,
                tracks_in_final=len(final_suggs),
                avg_rating_received=avg_rating,
            )

    def _update_genre_history(self, session_id: str, rated: list[RatedTrack]) -> None:
        genre_data: dict[str, list[int]] = {}
        for rt in rated:
            genre = rt.track.genre_primary
            genre_data.setdefault(genre, []).append(rt.score)

        for genre, scores in genre_data.items():
            self._db.upsert_genre_session_history(
                genre=genre,
                session_id=session_id,
                track_count=len(scores),
                avg_rating=sum(scores) / len(scores),
            )

    def _build_summary(
        self,
        session_id: str,
        rated: list[RatedTrack],
        suggestions: list[dict],
    ) -> SessionSummary:
        scores = [rt.score for rt in rated]
        avg_rating = sum(scores) / len(scores) if scores else 0.0
        top_rated = max(rated, key=lambda rt: rt.score) if rated else None

        # Genre stats
        genre_data: dict[str, list[int]] = {}
        for rt in rated:
            genre_data.setdefault(rt.track.genre_primary, []).append(rt.score)
        genre_stats = {g: sum(s) / len(s) for g, s in genre_data.items()}

        # Newly curated
        newly_curated = [
            {
                "title": rt.track.title,
                "artist": rt.track.artist,
                "genre": rt.track.genre_primary,
            }
            for rt in rated
            if rt.score >= self._curated_threshold
        ]

        # Engine breakdown
        score_map = {rt.track.id: rt.score for rt in rated}
        by_engine: dict[str, list[dict]] = {}
        for s in suggestions:
            by_engine.setdefault(s["engine_name"], []).append(s)

        engine_breakdown: dict[str, dict] = {}
        for engine_name, engine_suggs in by_engine.items():
            final = [s for s in engine_suggs if s.get("was_final")]
            rated_scores = [
                score_map[s["track_id"]]
                for s in final
                if s["track_id"] in score_map
            ]
            engine_breakdown[engine_name] = {
                "suggested": len(engine_suggs),
                "in_final": len(final),
                "avg_rating": round(sum(rated_scores) / len(rated_scores), 2)
                if rated_scores
                else None,
            }

        return SessionSummary(
            session_id=session_id,
            avg_rating=round(avg_rating, 2),
            top_rated_track=top_rated,
            engine_breakdown=engine_breakdown,
            genre_stats=genre_stats,
            newly_curated_count=len(newly_curated),
            newly_curated=newly_curated,
        )
