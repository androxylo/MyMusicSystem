"""
Shared fixtures for all tests.

Uses an in-memory SQLite database and mock engines/connectors
so tests never touch the filesystem or external APIs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.base_engine import (
    BaseEngine,
    EngineCapabilities,
    EngineHealth,
    RatedTrack,
    Session,
    SessionConfig,
    SessionContext,
    Suggestion,
    Track,
)
from core.db import Database
from core.diversity import DiversityEnforcer
from core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(
    *,
    track_id: str | None = None,
    title: str = "Test Track",
    artist: str = "Test Artist",
    album: str = "Test Album",
    genre_primary: str = "Electronic",
    genre_tags: list[str] | None = None,
    tidal_id: str | None = None,
    bpm: float | None = 120.0,
) -> Track:
    return Track(
        id=track_id or str(uuid.uuid4()),
        title=title,
        artist=artist,
        album=album,
        duration_ms=210_000,
        genre_primary=genre_primary,
        genre_tags=genre_tags or [genre_primary.lower()],
        mood_tags=[],
        tidal_id=tidal_id,
        bpm=bpm,
    )


def make_rated_track(
    track: Track | None = None,
    score: int = 7,
    session_id: str | None = None,
    rated_at: datetime | None = None,
) -> RatedTrack:
    return RatedTrack(
        track=track or make_track(),
        score=score,
        rated_at=rated_at or datetime.now(),
        session_id=session_id or str(uuid.uuid4()),
    )


def make_suggestion(
    track: Track | None = None,
    engine_name: str = "test_engine",
    engine_score: float = 0.8,
) -> Suggestion:
    t = track or make_track()
    return Suggestion(
        track=t,
        engine_name=engine_name,
        engine_score=engine_score,
        explanation="Test explanation",
        genre_tags=t.genre_tags,
    )


def make_session(
    session_id: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> Session:
    return Session(
        id=session_id or str(uuid.uuid4()),
        started_at=started_at or datetime.now() - timedelta(hours=1),
        completed_at=completed_at,
        engine_allocation={"test_engine": 3},
        diversity_config={"diversity_mode": "strict"},
    )


# ---------------------------------------------------------------------------
# In-memory Database fixture
# ---------------------------------------------------------------------------

MIGRATIONS_DIR = Path(__file__).parent.parent / "database" / "migrations"


@pytest.fixture()
def db() -> Database:
    database = Database(":memory:", MIGRATIONS_DIR)
    database.connect()
    yield database
    database.close()


# ---------------------------------------------------------------------------
# Mock engine factory
# ---------------------------------------------------------------------------

def make_mock_engine(
    name: str = "test_engine",
    suggestions: list[Suggestion] | None = None,
    health_status: str = "ok",
    cold_start_friendly: bool = True,
) -> MagicMock:
    engine = MagicMock(spec=BaseEngine)
    engine.name = name
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.5,
        genre_coverage="broad",
        cold_start_friendly=cold_start_friendly,
        data_requirements=[],
        speed="fast",
        description=f"Mock engine: {name}",
    )
    engine.health_check.return_value = EngineHealth(status=health_status)
    engine.suggest.return_value = suggestions or []
    engine.on_session_complete.return_value = None
    return engine


# ---------------------------------------------------------------------------
# DiversityEnforcer fixture (default config)
# ---------------------------------------------------------------------------

@pytest.fixture()
def enforcer() -> DiversityEnforcer:
    return DiversityEnforcer(
        recently_heard_days=30,
        saturation_sessions=3,
        saturation_min_avg_rating=7.0,
        min_engines_represented=2,
    )


# ---------------------------------------------------------------------------
# Orchestrator fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_engines():
    """Two mock engines that together return diverse genre suggestions."""
    genres_a = ["Electronic", "Rock", "Jazz", "Pop", "Metal"]
    genres_b = ["Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World"]

    suggestions_a = [
        make_suggestion(make_track(genre_primary=g, track_id=str(uuid.uuid4())), "engine_a", 0.9 - i * 0.05)
        for i, g in enumerate(genres_a)
    ]
    suggestions_b = [
        make_suggestion(make_track(genre_primary=g, track_id=str(uuid.uuid4())), "engine_b", 0.85 - i * 0.05)
        for i, g in enumerate(genres_b)
    ]

    engine_a = make_mock_engine("engine_a", suggestions_a)
    engine_b = make_mock_engine("engine_b", suggestions_b)
    return engine_a, engine_b


@pytest.fixture()
def session_context():
    return SessionContext(
        rated_tracks=[],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=set(),
        excluded_track_fingerprints=set(),
    )
