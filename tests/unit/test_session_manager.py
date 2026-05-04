"""Tests for SessionManager — session lifecycle, rating, completion."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.base_engine import SessionConfig, Suggestion
from core.diversity import DiversityEnforcer
from core.orchestrator import Orchestrator
from core.session import SessionManager
from tests.conftest import (
    Database,
    make_mock_engine,
    make_suggestion,
    make_track,
)

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "database" / "migrations"

GENRES = [
    "Electronic", "Rock", "Jazz", "Pop", "Metal",
    "Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World",
]


def _make_db() -> Database:
    db = Database(":memory:", MIGRATIONS_DIR)
    db.connect()
    return db


def _make_10_suggestions(engine_name: str = "test_engine") -> list[Suggestion]:
    return [
        make_suggestion(
            make_track(genre_primary=g, track_id=str(uuid.uuid4())),
            engine_name=engine_name,
            engine_score=0.9 - i * 0.05,
        )
        for i, g in enumerate(GENRES)
    ]


def _make_session_manager(db: Database, suggestions: list[Suggestion] | None = None) -> SessionManager:
    suggs = suggestions or _make_10_suggestions()
    engine = make_mock_engine("test_engine", suggs)
    enforcer = DiversityEnforcer(min_engines_represented=1)
    orchestrator = Orchestrator(
        engines=[engine],
        diversity_enforcer=enforcer,
        slot_weights={"test_engine": 1.0},
        oversampling_factor=1,
        n_final=10,
    )
    return SessionManager(db=db, orchestrator=orchestrator, engines=[engine])


class TestStartSession:
    def test_returns_session_id_and_suggestions(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        assert "session_id" in result
        assert len(result["suggestions"]) > 0

    def test_session_persisted_in_db(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        session = db.get_session(result["session_id"])
        assert session is not None
        assert session.id == result["session_id"]

    def test_suggestion_tracks_persisted_in_db(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        track_id = result["suggestions"][0]["track_id"]
        track = db.get_track(track_id)
        assert track is not None

    def test_engine_allocation_in_result(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        assert "engine_allocation" in result
        assert "test_engine" in result["engine_allocation"]

    def test_candidate_playlist_url_none_without_playlist_manager(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        assert result["candidate_playlist_url"] is None

    def test_playlist_manager_called_on_start(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        pm = MagicMock()
        pm.sync_candidate_playlist.return_value = "https://tidal.com/browse/playlist/abc"
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine], playlist_manager=pm)
        result = sm.start_session()
        pm.sync_candidate_playlist.assert_called_once()
        assert result["candidate_playlist_url"] == "https://tidal.com/browse/playlist/abc"

    def test_playlist_manager_failure_does_not_crash_session(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        pm = MagicMock()
        pm.sync_candidate_playlist.side_effect = RuntimeError("Tidal down")
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine], playlist_manager=pm)
        result = sm.start_session()  # Must not raise
        assert "session_id" in result


class TestRateTrack:
    def _start(self, db: Database) -> tuple[SessionManager, str, str]:
        sm = _make_session_manager(db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_id = result["suggestions"][0]["track_id"]
        return sm, session_id, track_id

    def test_valid_rating_persisted(self):
        db = _make_db()
        sm, session_id, track_id = self._start(db)
        sm.rate_track(session_id, track_id, 7)
        rating = db.get_rating(session_id, track_id)
        assert rating is not None
        assert rating["score"] == 7

    def test_score_below_1_raises(self):
        db = _make_db()
        sm, session_id, track_id = self._start(db)
        with pytest.raises(ValueError, match="between 1 and 10"):
            sm.rate_track(session_id, track_id, 0)

    def test_score_above_10_raises(self):
        db = _make_db()
        sm, session_id, track_id = self._start(db)
        with pytest.raises(ValueError, match="between 1 and 10"):
            sm.rate_track(session_id, track_id, 11)

    def test_duplicate_rating_raises(self):
        db = _make_db()
        sm, session_id, track_id = self._start(db)
        sm.rate_track(session_id, track_id, 7)
        with pytest.raises(ValueError, match="already rated"):
            sm.rate_track(session_id, track_id, 8)

    def test_unknown_session_raises(self):
        db = _make_db()
        sm, session_id, track_id = self._start(db)
        with pytest.raises(ValueError, match="not found"):
            sm.rate_track("bad-session-id", track_id, 7)

    def test_unknown_track_raises(self):
        db = _make_db()
        sm, session_id, track_id = self._start(db)
        with pytest.raises(ValueError, match="not found"):
            sm.rate_track(session_id, "bad-track-id", 7)

    def test_ratings_so_far_increments(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        session_id = result["session_id"]

        track_ids = [s["track_id"] for s in result["suggestions"]]
        r1 = sm.rate_track(session_id, track_ids[0], 5)
        assert r1.ratings_so_far == 1
        r2 = sm.rate_track(session_id, track_ids[1], 6)
        assert r2.ratings_so_far == 2

    def test_above_threshold_returns_added_to_curated_true(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_id = result["suggestions"][0]["track_id"]
        r = sm.rate_track(session_id, track_id, 7)  # default threshold = 7
        assert r.added_to_curated is True

    def test_below_threshold_returns_added_to_curated_false(self):
        db = _make_db()
        sm = _make_session_manager(db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_id = result["suggestions"][0]["track_id"]
        r = sm.rate_track(session_id, track_id, 6)
        assert r.added_to_curated is False

    def test_playlist_manager_remove_called(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        pm = MagicMock()
        pm.sync_candidate_playlist.return_value = "https://tidal.com/test"
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine], playlist_manager=pm)
        result = sm.start_session()
        session_id = result["session_id"]
        track_id = result["suggestions"][0]["track_id"]
        sm.rate_track(session_id, track_id, 5)
        pm.remove_from_candidate.assert_called_once_with(track_id)

    def test_playlist_manager_add_curated_called_above_threshold(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        pm = MagicMock()
        pm.sync_candidate_playlist.return_value = "https://tidal.com/test"
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine], playlist_manager=pm)
        result = sm.start_session()
        session_id = result["session_id"]
        track_id = result["suggestions"][0]["track_id"]
        sm.rate_track(session_id, track_id, 8)
        pm.add_to_curated.assert_called_once()

    def test_playlist_manager_failure_does_not_fail_rating(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        pm = MagicMock()
        pm.sync_candidate_playlist.return_value = "https://tidal.com/test"
        pm.remove_from_candidate.side_effect = RuntimeError("Tidal down")
        pm.add_to_curated.side_effect = RuntimeError("Tidal down")
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine], playlist_manager=pm)
        result = sm.start_session()
        session_id = result["session_id"]
        track_id = result["suggestions"][0]["track_id"]
        r = sm.rate_track(session_id, track_id, 9)  # Must not raise
        assert r.ok is True
        # Rating must be in DB
        rating = db.get_rating(session_id, track_id)
        assert rating is not None


class TestCompleteSession:
    def _full_session(self, db: Database) -> tuple[SessionManager, str, list[str]]:
        sm = _make_session_manager(db)
        result = sm.start_session()
        session_id = result["session_id"]
        track_ids = [s["track_id"] for s in result["suggestions"]]
        for i, tid in enumerate(track_ids):
            sm.rate_track(session_id, tid, 5 + (i % 5))
        return sm, session_id, track_ids

    def test_session_marked_complete_in_db(self):
        db = _make_db()
        sm, session_id, _ = self._full_session(db)
        sm.complete_session(session_id)
        session = db.get_session(session_id)
        assert session.completed_at is not None

    def test_summary_has_expected_fields(self):
        db = _make_db()
        sm, session_id, _ = self._full_session(db)
        summary = sm.complete_session(session_id)
        assert summary.session_id == session_id
        assert isinstance(summary.avg_rating, float)
        assert isinstance(summary.genre_stats, dict)
        assert isinstance(summary.engine_breakdown, dict)
        assert isinstance(summary.newly_curated, list)

    def test_on_session_complete_called_on_all_engines(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine])
        result = sm.start_session()
        session_id = result["session_id"]
        for s in result["suggestions"]:
            sm.rate_track(session_id, s["track_id"], 6)
        sm.complete_session(session_id)
        engine.on_session_complete.assert_called_once()

    def test_unknown_session_raises(self):
        db = _make_db()
        sm = _make_session_manager(db)
        with pytest.raises(ValueError, match="not found"):
            sm.complete_session("does-not-exist")

    def test_engine_performance_recorded(self):
        db = _make_db()
        sm, session_id, _ = self._full_session(db)
        sm.complete_session(session_id)
        perf = db.get_engine_performance(last_n_sessions=5)
        assert len(perf) > 0

    def test_newly_curated_tracks_in_summary(self):
        db = _make_db()
        suggs = _make_10_suggestions()
        engine = make_mock_engine("test_engine", suggs)
        enforcer = DiversityEnforcer(min_engines_represented=1)
        orchestrator = Orchestrator(
            engines=[engine],
            diversity_enforcer=enforcer,
            slot_weights={"test_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        sm = SessionManager(db=db, orchestrator=orchestrator, engines=[engine], curated_threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]
        track_ids = [s["track_id"] for s in result["suggestions"]]
        # Rate some above threshold
        for tid in track_ids[:3]:
            sm.rate_track(session_id, tid, 8)
        for tid in track_ids[3:]:
            sm.rate_track(session_id, tid, 4)
        summary = sm.complete_session(session_id)
        assert summary.newly_curated_count == 3
        assert len(summary.newly_curated) == 3
