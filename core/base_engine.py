"""
Shared types and the BaseEngine abstract class.
This module is the contract between every layer of the system.
All engines, the orchestrator, the diversity enforcer, and the session manager
import exclusively from here — never from each other.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class Track:
    id: str                          # internal UUID
    title: str
    artist: str
    album: str
    duration_ms: int
    genre_primary: str               # broad genre bucket for diversity enforcement
    genre_tags: list[str] = field(default_factory=list)
    mood_tags: list[str] = field(default_factory=list)
    tidal_id: str | None = None
    spotify_id: str | None = None
    mbid: str | None = None
    bpm: float | None = None
    key: str | None = None
    mode: str | None = None
    audio_features: dict | None = None


@dataclass
class RatedTrack:
    track: Track
    score: int                       # 1–10
    rated_at: datetime
    session_id: str


@dataclass
class Session:
    id: str
    started_at: datetime
    engine_allocation: dict[str, int]  # engine_name → n_slots
    diversity_config: dict
    completed_at: datetime | None = None
    notes: str | None = None


@dataclass
class SessionConfig:
    diversity_mode: str = "strict"   # "strict" | "relaxed"
    notes: str | None = None


@dataclass
class SessionContext:
    rated_tracks: list[RatedTrack]
    recent_sessions: list[Session]
    session_config: SessionConfig
    excluded_track_ids: set[str] = field(default_factory=set)
    # Normalized "title|artist" fingerprints for all previously-rated tracks.
    # Catches the same song reappearing on a different album (different tidal_id).
    excluded_track_fingerprints: set[str] = field(default_factory=set)


@dataclass
class Suggestion:
    track: Track
    engine_name: str
    engine_score: float              # engine's internal confidence [0.0, 1.0]
    explanation: str                 # human-readable; surfaced by the Claude agent
    genre_tags: list[str] = field(default_factory=list)


@dataclass
class EngineCapabilities:
    novelty_bias: float              # 0.0 = pure exploitation, 1.0 = pure exploration
    genre_coverage: str              # 'broad' | 'narrow' | 'targeted'
    cold_start_friendly: bool        # works well with fewer than ~20 ratings
    data_requirements: list[str]     # e.g. ['ratings', 'audio_features', 'lastfm_history']
    speed: str                       # 'fast' | 'slow'
    description: str


@dataclass
class EngineHealth:
    status: str                      # 'ok' | 'degraded' | 'unavailable'
    message: str | None = None


# ---------------------------------------------------------------------------
# Base engine
# ---------------------------------------------------------------------------

class BaseEngine(ABC):
    """
    Abstract base class for all recommendation engines.

    Engines are self-contained: they read from the DB (via SessionContext)
    and return Suggestion objects. They never communicate with each other
    or with the orchestrator directly.

    Contract:
    - suggest() must never raise; return an empty list on failure.
    - on_session_complete() is optional; default is a no-op.
    - health_check() must not make external API calls if status would be 'unavailable'.
    """

    # Set by EngineRegistry from manifest.yaml — do not hardcode in subclasses.
    name: str = ""
    capabilities: EngineCapabilities = field(default_factory=lambda: EngineCapabilities(
        novelty_bias=0.5,
        genre_coverage="broad",
        cold_start_friendly=False,
        data_requirements=[],
        speed="fast",
        description="",
    ))

    @abstractmethod
    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        """
        Return up to n track suggestions given the session context.

        The orchestrator over-samples (asks for more than n), so returning
        slightly more than n is encouraged. Must respect context.excluded_track_ids.
        Must not raise — return [] and log the error on failure.
        """

    def on_session_complete(self, ratings: list[tuple[Track, int]]) -> None:
        """
        Called after a session completes with all (track, score) pairs.
        Override to update internal model state. Default: no-op.
        Must not raise.
        """

    def health_check(self) -> EngineHealth:
        """
        Called by the registry before each session.
        Return 'ok', 'degraded', or 'unavailable'.
        Default: always ok (suitable for engines with no external dependencies).
        """
        return EngineHealth(status="ok")
