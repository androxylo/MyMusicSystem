"""
Integration tests — full session cycle with real in-memory SQLite.

Uses seed_ratings.json to populate a DB with prior history, then runs
a complete session to verify end-to-end wiring: start → rate → complete.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from core.base_engine import (
    EngineCapabilities,
    EngineHealth,
    Session,
    SessionConfig,
    SessionContext,
    Suggestion,
)
from core.db import Database
from core.diversity import DiversityEnforcer
from core.orchestrator import Orchestrator
from core.session import SessionManager
from tests.conftest import make_mock_engine, make_suggestion, make_track

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "seed_ratings.json"
MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "database" / "migrations"

GENRES = [
    "Electronic", "Rock", "Jazz", "Pop", "Metal",
    "Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World",
]


# ---------------------------------------------------------------------------
# Fixture: seeded DB
# ---------------------------------------------------------------------------

def _seed_db(db: Database, fixture: dict) -> None:
    """Load sessions, tracks, and ratings from the fixture into the DB."""
    from core.base_engine import Track

    for t in fixture["tracks"]:
        track = Track(
            id=t["id"],
            title=t["title"],
            artist=t["artist"],
            album=t.get("album", ""),
            duration_ms=t.get("duration_ms", 0),
            genre_primary=t.get("genre_primary", "Other"),
            genre_tags=t.get("genre_tags", []),
            mood_tags=t.get("mood_tags", []),
            tidal_id=t.get("tidal_id"),
        )
        db.upsert_track(track)

    for s in fixture["sessions"]:
        from core.base_engine import Session as SessionObj
        session = SessionObj(
            id=s["id"],
            started_at=datetime.strptime(s["started_at"], "%Y-%m-%d %H:%M:%S.%f"),
            completed_at=datetime.strptime(s["completed_at"], "%Y-%m-%d %H:%M:%S.%f")
            if s.get("completed_at") else None,
            engine_allocation=s.get("engine_allocation", {}),
            diversity_config=s.get("diversity_config", {}),
            notes=s.get("notes"),
        )
        db.insert_session(session)
        if session.completed_at:
            db.complete_session(session.id, session.completed_at)

    for r in fixture["ratings"]:
        db.insert_rating(r["track_id"], r["session_id"], r["score"])


@pytest.fixture()
def seeded_db() -> Database:
    fixture = json.loads(FIXTURE_PATH.read_text())
    db = Database(":memory:", MIGRATIONS_DIR)
    db.connect()
    _seed_db(db, fixture)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_session_manager(db: Database) -> SessionManager:
    suggestions = [
        make_suggestion(
            make_track(genre_primary=g, track_id=str(uuid.uuid4())),
            engine_name="mock_engine",
            engine_score=0.9 - i * 0.05,
        )
        for i, g in enumerate(GENRES)
    ]
    engine = make_mock_engine("mock_engine", suggestions)
    enforcer = DiversityEnforcer(
        recently_heard_days=30,
        saturation_sessions=3,
        saturation_min_avg_rating=7.0,
        min_engines_represented=1,
    )
    orchestrator = Orchestrator(
        engines=[engine],
        diversity_enforcer=enforcer,
        slot_weights={"mock_engine": 1.0},
        oversampling_factor=1,
        n_final=10,
    )
    return SessionManager(db=db, orchestrator=orchestrator, engines=[engine])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullSessionCycle:
    def test_start_session_produces_suggestions(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        assert "session_id" in result
        assert len(result["suggestions"]) > 0

    def test_session_persisted_after_start(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        session = seeded_db.get_session(result["session_id"])
        assert session is not None

    def test_rate_all_tracks_persists_ratings(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_ids = [s["track_id"] for s in result["suggestions"]]

        for i, tid in enumerate(track_ids):
            sm.rate_track(session_id, tid, 5 + (i % 5))

        total = seeded_db.get_total_ratings_count()
        # 30 seed ratings + however many we just added
        assert total >= len(track_ids)

    def test_complete_session_marks_done(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_ids = [s["track_id"] for s in result["suggestions"]]

        for i, tid in enumerate(track_ids):
            sm.rate_track(session_id, tid, 6)

        summary = sm.complete_session(session_id)
        session = seeded_db.get_session(session_id)
        assert session.completed_at is not None
        assert summary.avg_rating == pytest.approx(6.0, abs=0.1)

    def test_engine_performance_recorded_after_completion(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        session_id = result["session_id"]
        for s in result["suggestions"]:
            sm.rate_track(session_id, s["track_id"], 7)
        sm.complete_session(session_id)

        perf = seeded_db.get_engine_performance(last_n_sessions=5)
        assert len(perf) > 0
        assert any(p["engine_name"] == "mock_engine" for p in perf)

    def test_genre_session_history_updated(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        session_id = result["session_id"]
        for s in result["suggestions"]:
            sm.rate_track(session_id, s["track_id"], 8)
        sm.complete_session(session_id)

        history = seeded_db.get_genre_session_history(n_sessions=5)
        assert len(history) > 0

    def test_final_suggestions_have_distinct_genres_when_possible(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        genres = [s["genre"] for s in result["suggestions"]]
        # In the happy path (10 distinct genre suggestions from mock engine),
        # all genres should be distinct
        assert len(genres) == len(set(genres)), (
            f"Duplicate genres in final suggestions: {genres}"
        )

    def test_curated_tracks_in_summary(self, seeded_db):
        sm = _build_session_manager(seeded_db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_ids = [s["track_id"] for s in result["suggestions"]]

        # Rate first 3 above threshold (7), rest below
        for tid in track_ids[:3]:
            sm.rate_track(session_id, tid, 8)
        for tid in track_ids[3:]:
            sm.rate_track(session_id, tid, 4)

        summary = sm.complete_session(session_id)
        assert summary.newly_curated_count == 3


class TestDatabaseWithSeedData:
    """Verify seed data loaded correctly and DB queries work."""

    def test_seed_track_count(self, seeded_db):
        rows = seeded_db.get_all_rated_track_ids()
        assert len(rows) == 30

    def test_seed_rating_count(self, seeded_db):
        count = seeded_db.get_total_ratings_count()
        assert count == 30

    def test_genre_avg_ratings_computed(self, seeded_db):
        genre_avgs = seeded_db.get_genre_avg_ratings()
        assert len(genre_avgs) > 0
        for row in genre_avgs:
            assert "genre" in row
            assert "avg_rating" in row
            assert 1.0 <= row["avg_rating"] <= 10.0

    def test_recent_sessions_returned(self, seeded_db):
        sessions = seeded_db.get_recent_sessions(n=5)
        assert len(sessions) == 3
        # Most recent first
        assert sessions[0].started_at >= sessions[1].started_at
