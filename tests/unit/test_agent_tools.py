"""
Unit tests for agent_tools/tools.py.

The full system context is replaced with mocks so no real DB, connectors,
or engines are used. Tests verify the tool functions produce correct shapes
and delegate to the right components.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import agent_tools.tools as tools
from core.base_engine import (
    EngineCapabilities,
    EngineHealth,
    RatedTrack,
    SessionConfig,
    Track,
)
from core.session import RateResult, SessionSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(genre: str = "Electronic") -> Track:
    return Track(
        id=str(uuid.uuid4()),
        title="Test Track",
        artist="Test Artist",
        album="Album",
        duration_ms=200_000,
        genre_primary=genre,
        tidal_id=str(uuid.uuid4()),
    )


def _make_rated_track(score: int = 8, genre: str = "Electronic") -> RatedTrack:
    return RatedTrack(
        track=_make_track(genre),
        score=score,
        rated_at=datetime.now(),
        session_id="sess-1",
    )


def _make_start_result(session_id: str = "sess-1") -> dict:
    return {
        "session_id": session_id,
        "candidate_playlist_url": "https://tidal.com/browse/playlist/abc",
        "engine_allocation": {"mock_engine": 10},
        "suggestions": [
            {
                "track_id": "track-1",
                "title": "Track One",
                "artist": "Artist",
                "album": "Album",
                "genre": "Electronic",
                "engine": "mock_engine",
                "engine_score": 0.9,
                "explanation": "Test",
                "tidal_id": "tid-1",
            }
        ],
    }


def _make_summary(session_id: str = "sess-1") -> SessionSummary:
    rt = _make_rated_track(score=9)
    return SessionSummary(
        session_id=session_id,
        avg_rating=8.0,
        top_rated_track=rt,
        engine_breakdown={"mock_engine": {"suggested": 10, "in_final": 10, "avg_rating": 8.0}},
        genre_stats={"Electronic": 8.0},
        newly_curated_count=1,
        newly_curated=[{"title": "Track One", "artist": "Artist", "genre": "Electronic"}],
    )


def _make_mock_ctx(session_manager=None, playlist_manager=None, registry=None, db=None):
    ctx = MagicMock(spec=tools._SystemContext)
    ctx.session_manager = session_manager or MagicMock()
    ctx.playlist_manager = playlist_manager
    ctx.registry = registry or MagicMock()
    ctx.db = db or MagicMock()
    ctx.settings = {"diversity": {"saturation_sessions": 3, "saturation_min_avg_rating": 7.0}}
    return ctx


# ---------------------------------------------------------------------------
# Fixture: reset the singleton before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ctx():
    """Ensure _ctx is reset before and after each test."""
    tools._ctx = None
    yield
    tools._ctx = None


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------

class TestStartSession:
    def test_returns_session_id_and_suggestions(self):
        sm = MagicMock()
        sm.start_session.return_value = _make_start_result()
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.start_session()
        assert result["session_id"] == "sess-1"
        assert "suggestions" in result
        assert len(result["suggestions"]) == 1

    def test_passes_diversity_mode(self):
        sm = MagicMock()
        sm.start_session.return_value = _make_start_result()
        tools._ctx = _make_mock_ctx(session_manager=sm)

        tools.start_session(diversity_mode="relaxed", notes="test note")
        call_args = sm.start_session.call_args[0][0]
        assert call_args.diversity_mode == "relaxed"
        assert call_args.notes == "test note"

    def test_returns_candidate_playlist_url(self):
        sm = MagicMock()
        sm.start_session.return_value = _make_start_result()
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.start_session()
        assert "tidal.com" in result["candidate_playlist_url"]


# ---------------------------------------------------------------------------
# rate_track
# ---------------------------------------------------------------------------

class TestRateTrack:
    def test_returns_ok_true_on_success(self):
        sm = MagicMock()
        sm.rate_track.return_value = RateResult(ok=True, ratings_so_far=3, added_to_curated=True)
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.rate_track("sess-1", "track-1", 8)
        assert result["ok"] is True
        assert result["ratings_so_far"] == 3
        assert result["added_to_curated"] is True

    def test_added_to_curated_false_below_threshold(self):
        sm = MagicMock()
        sm.rate_track.return_value = RateResult(ok=True, ratings_so_far=1, added_to_curated=False)
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.rate_track("sess-1", "track-1", 5)
        assert result["added_to_curated"] is False

    def test_delegates_to_session_manager(self):
        sm = MagicMock()
        sm.rate_track.return_value = RateResult(ok=True, ratings_so_far=1, added_to_curated=False)
        tools._ctx = _make_mock_ctx(session_manager=sm)

        tools.rate_track("sess-abc", "track-xyz", 7)
        sm.rate_track.assert_called_once_with("sess-abc", "track-xyz", 7)


# ---------------------------------------------------------------------------
# complete_session
# ---------------------------------------------------------------------------

class TestCompleteSession:
    def test_returns_summary_structure(self):
        sm = MagicMock()
        sm.complete_session.return_value = _make_summary()
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.complete_session("sess-1")
        assert "session_summary" in result
        assert "genre_stats" in result
        assert "newly_curated" in result
        assert "next_session_preview" in result

    def test_summary_contains_avg_rating(self):
        sm = MagicMock()
        sm.complete_session.return_value = _make_summary()
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.complete_session("sess-1")
        assert result["session_summary"]["avg_rating"] == 8.0

    def test_top_rated_track_included(self):
        sm = MagicMock()
        sm.complete_session.return_value = _make_summary()
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.complete_session("sess-1")
        top = result["session_summary"]["top_rated_track"]
        assert top is not None
        assert "title" in top
        assert "score" in top

    def test_no_top_rated_when_no_ratings(self):
        sm = MagicMock()
        summary = SessionSummary(
            session_id="s",
            avg_rating=0.0,
            top_rated_track=None,
            engine_breakdown={},
            genre_stats={},
            newly_curated_count=0,
            newly_curated=[],
        )
        sm.complete_session.return_value = summary
        tools._ctx = _make_mock_ctx(session_manager=sm)

        result = tools.complete_session("s")
        assert result["session_summary"]["top_rated_track"] is None


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_required_keys(self):
        db = MagicMock()
        db.get_total_ratings_count.return_value = 42
        db.get_genre_avg_ratings.return_value = [
            {"genre": "Electronic", "avg_rating": 8.0, "count": 20}
        ]
        db.get_engine_performance.return_value = [
            {"engine_name": "mock", "session_count": 5, "avg_tracks_in_final": 2.0, "avg_rating": 7.5}
        ]
        db.get_genre_session_history.return_value = []
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_stats()
        assert "total_ratings" in result
        assert "favorite_genres" in result
        assert "engine_performance" in result
        assert "genre_saturation_warnings" in result

    def test_total_ratings_correct(self):
        db = MagicMock()
        db.get_total_ratings_count.return_value = 99
        db.get_genre_avg_ratings.return_value = []
        db.get_engine_performance.return_value = []
        db.get_genre_session_history.return_value = []
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_stats()
        assert result["total_ratings"] == 99

    def test_saturation_warning_triggered(self):
        db = MagicMock()
        db.get_total_ratings_count.return_value = 0
        db.get_genre_avg_ratings.return_value = []
        db.get_engine_performance.return_value = []
        # 3 sessions with Electronic at avg 8.0 each
        db.get_genre_session_history.return_value = [
            {"genre": "Electronic", "avg_rating": 8.0},
            {"genre": "Electronic", "avg_rating": 8.0},
            {"genre": "Electronic", "avg_rating": 8.0},
        ]
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_stats()
        warnings = result["genre_saturation_warnings"]
        assert any(w["genre"] == "Electronic" for w in warnings)

    def test_no_saturation_below_threshold(self):
        db = MagicMock()
        db.get_total_ratings_count.return_value = 0
        db.get_genre_avg_ratings.return_value = []
        db.get_engine_performance.return_value = []
        # Only 2 sessions (threshold is 3)
        db.get_genre_session_history.return_value = [
            {"genre": "Electronic", "avg_rating": 9.0},
            {"genre": "Electronic", "avg_rating": 9.0},
        ]
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_stats()
        assert result["genre_saturation_warnings"] == []


# ---------------------------------------------------------------------------
# list_engines
# ---------------------------------------------------------------------------

class TestListEngines:
    def test_returns_engines_list(self):
        registry = MagicMock()
        mock_engine = MagicMock()
        mock_engine.name = "test_engine"
        mock_engine.capabilities = EngineCapabilities(
            novelty_bias=0.5, genre_coverage="broad",
            cold_start_friendly=True, data_requirements=[],
            speed="fast", description="A test engine",
        )
        registry.engines = [mock_engine]
        registry.health = {"test_engine": EngineHealth(status="ok")}
        registry.get_slot_weights.return_value = {"test_engine": 0.5}
        tools._ctx = _make_mock_ctx(registry=registry)

        result = tools.list_engines()
        assert "engines" in result
        assert any(e["name"] == "test_engine" for e in result["engines"])

    def test_shows_unavailable_engines(self):
        registry = MagicMock()
        registry.engines = []
        registry.health = {
            "broken_engine": EngineHealth(status="unavailable", message="No credentials")
        }
        tools._ctx = _make_mock_ctx(registry=registry)

        result = tools.list_engines()
        names = [e["name"] for e in result["engines"]]
        assert "broken_engine" in names


# ---------------------------------------------------------------------------
# get_track_info
# ---------------------------------------------------------------------------

class TestGetTrackInfo:
    def test_returns_track_metadata(self):
        track = _make_track()
        db = MagicMock()
        db.get_track.return_value = track
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_track_info(track.id)
        assert result["track_id"] == track.id
        assert result["title"] == "Test Track"
        assert result["artist"] == "Test Artist"
        assert "tidal_url" in result

    def test_returns_error_for_unknown_track(self):
        db = MagicMock()
        db.get_track.return_value = None
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_track_info("unknown-id")
        assert "error" in result

    def test_tidal_url_format(self):
        track = _make_track()
        track.tidal_id = "12345"
        db = MagicMock()
        db.get_track.return_value = track
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_track_info(track.id)
        assert result["tidal_url"] == "https://tidal.com/browse/track/12345"

    def test_tidal_url_none_when_no_tidal_id(self):
        track = _make_track()
        track.tidal_id = None
        db = MagicMock()
        db.get_track.return_value = track
        tools._ctx = _make_mock_ctx(db=db)

        result = tools.get_track_info(track.id)
        assert result["tidal_url"] is None


# ---------------------------------------------------------------------------
# get_playlists
# ---------------------------------------------------------------------------

class TestGetPlaylists:
    def test_returns_error_when_no_playlist_manager(self):
        tools._ctx = _make_mock_ctx(playlist_manager=None)
        result = tools.get_playlists()
        assert "error" in result

    def test_returns_candidate_and_curated(self):
        from core.playlist_manager import PlaylistInfo
        pm = MagicMock()
        pm.get_all_playlists.return_value = [
            PlaylistInfo(
                playlist_type="candidate", genre=None,
                name="Now Rating", tidal_playlist_id="pl-1",
                tidal_url="https://tidal.com/browse/playlist/pl-1",
                track_count=5,
            ),
            PlaylistInfo(
                playlist_type="curated_master", genre=None,
                name="Liked — All", tidal_playlist_id="pl-2",
                tidal_url="https://tidal.com/browse/playlist/pl-2",
                track_count=20,
            ),
        ]
        tools._ctx = _make_mock_ctx(playlist_manager=pm)

        result = tools.get_playlists()
        assert result["candidate"]["name"] == "Now Rating"
        assert any(p["name"] == "Liked — All" for p in result["curated"])


# ---------------------------------------------------------------------------
# set_curated_threshold
# ---------------------------------------------------------------------------

class TestSetCuratedThreshold:
    def test_valid_threshold_accepted(self):
        sm = MagicMock()
        sm._curated_threshold = 7
        pm = MagicMock()
        pm._cfg = MagicMock()
        tools._ctx = _make_mock_ctx(session_manager=sm, playlist_manager=pm)

        with patch("builtins.open", MagicMock()), patch("yaml.dump"):
            result = tools.set_curated_threshold(8)

        assert result["ok"] is True
        assert result["new_threshold"] == 8
        assert sm._curated_threshold == 8

    def test_out_of_range_rejected(self):
        tools._ctx = _make_mock_ctx()
        result = tools.set_curated_threshold(0)
        assert result["ok"] is False

        result = tools.set_curated_threshold(11)
        assert result["ok"] is False

    def test_boundary_values_accepted(self):
        sm = MagicMock()
        sm._curated_threshold = 7
        tools._ctx = _make_mock_ctx(session_manager=sm, playlist_manager=None)

        with patch("builtins.open", MagicMock()), patch("yaml.dump"):
            assert tools.set_curated_threshold(1)["ok"] is True
            assert tools.set_curated_threshold(10)["ok"] is True


# ---------------------------------------------------------------------------
# reconcile_playlists
# ---------------------------------------------------------------------------

class TestReconcilePlaylists:
    def test_returns_error_when_no_playlist_manager(self):
        tools._ctx = _make_mock_ctx(playlist_manager=None)
        result = tools.reconcile_playlists()
        assert "error" in result

    def test_reconciles_specific_session(self):
        pm = MagicMock()
        pm.reconcile_session.return_value = {"candidate_fixes": 2, "curated_fixes": 1}
        tools._ctx = _make_mock_ctx(playlist_manager=pm)

        result = tools.reconcile_playlists(session_id="sess-1")
        assert result["sessions_reconciled"] == 1
        assert result["candidate_fixes"] == 2
        assert result["curated_fixes"] == 1
        pm.reconcile_session.assert_called_once_with("sess-1")

    def test_reconciles_all_recent_when_no_session_id(self):
        from datetime import datetime, timedelta
        from core.base_engine import Session

        pm = MagicMock()
        pm.reconcile_session.return_value = {"candidate_fixes": 1, "curated_fixes": 0}

        db = MagicMock()
        # Two completed sessions within 7 days
        now = datetime.now()
        db.get_recent_sessions.return_value = [
            Session(id="s1", started_at=now - timedelta(days=1), completed_at=now - timedelta(hours=1),
                    engine_allocation={}, diversity_config={}),
            Session(id="s2", started_at=now - timedelta(days=2), completed_at=now - timedelta(hours=2),
                    engine_allocation={}, diversity_config={}),
        ]
        tools._ctx = _make_mock_ctx(playlist_manager=pm, db=db)

        result = tools.reconcile_playlists()
        assert result["sessions_reconciled"] == 2
        assert result["candidate_fixes"] == 2
