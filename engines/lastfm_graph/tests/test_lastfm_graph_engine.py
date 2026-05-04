"""
Unit tests for LastFmGraphEngine.
All external connectors are mocked — no real API calls.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from connectors.lastfm import LastFmArtist, LastFmTrack
from connectors.tidal import TidalTrack
from core.base_engine import (
    EngineCapabilities,
    EngineHealth,
    RatedTrack,
    SessionConfig,
    SessionContext,
    Track,
)
from engines.lastfm_graph.engine import LastFmGraphEngine, _stable_track_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lastfm_mock(similar_artists: list[LastFmArtist] | None = None) -> MagicMock:
    lastfm = MagicMock()
    lastfm.is_available.return_value = True
    lastfm.get_similar_artists.return_value = similar_artists or [
        LastFmArtist("Similar Artist A", match_score=0.9),
        LastFmArtist("Similar Artist B", match_score=0.7),
    ]
    lastfm.get_user_top_artists.return_value = [
        LastFmArtist("Aphex Twin", match_score=100),
        LastFmArtist("Boards of Canada", match_score=80),
    ]
    lastfm.get_artist_tags.return_value = ["electronic", "ambient"]
    return lastfm


def _make_tidal_mock() -> MagicMock:
    tidal = MagicMock()
    tidal.is_available.return_value = True

    def search_tracks(query, limit=10):
        return [
            TidalTrack(
                tidal_id=f"tid-{uuid.uuid4().hex[:6]}",
                title=f"Track for {query}",
                artist=query.split()[0] if query else "Unknown",
                album="Album",
                duration_ms=200_000,
            )
        ]

    tidal.search_tracks.side_effect = search_tracks
    return tidal


def _make_engine(lastfm=None, tidal=None) -> LastFmGraphEngine:
    engine = LastFmGraphEngine(
        lastfm=lastfm if lastfm is not None else _make_lastfm_mock(),
        tidal=tidal if tidal is not None else _make_tidal_mock(),
    )
    engine.name = "lastfm_graph"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.4,
        genre_coverage="broad",
        cold_start_friendly=True,
        data_requirements=["ratings", "lastfm_history"],
        speed="fast",
        description="",
    )
    return engine


def _make_rated_track(artist: str = "Test Artist", score: int = 8) -> RatedTrack:
    track = Track(
        id=str(uuid.uuid4()),
        title="Test Track",
        artist=artist,
        album="Test Album",
        duration_ms=200_000,
        genre_primary="Electronic",
    )
    return RatedTrack(track=track, score=score, rated_at=datetime.now(), session_id="s1")


def _context(rated: list[RatedTrack] | None = None, excluded: set[str] | None = None) -> SessionContext:
    return SessionContext(
        rated_tracks=rated or [],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=excluded or set(),
    )


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_ok_when_both_connectors_available(self):
        engine = _make_engine()
        assert engine.health_check().status == "ok"

    def test_unavailable_when_lastfm_missing(self):
        engine = LastFmGraphEngine(lastfm=None, tidal=_make_tidal_mock())
        engine._lastfm = None
        assert engine.health_check().status == "unavailable"

    def test_degraded_when_tidal_missing(self):
        lastfm = _make_lastfm_mock()
        engine = LastFmGraphEngine(lastfm=lastfm, tidal=None)
        engine._tidal = None
        assert engine.health_check().status == "degraded"

    def test_unavailable_when_lastfm_not_connected(self):
        lastfm = MagicMock()
        lastfm.is_available.return_value = False
        engine = LastFmGraphEngine(lastfm=lastfm, tidal=_make_tidal_mock())
        assert engine.health_check().status == "unavailable"


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_list(self):
        engine = _make_engine()
        result = engine.suggest(5, _context())
        assert isinstance(result, list)

    def test_returns_empty_when_lastfm_unavailable(self):
        lastfm = MagicMock()
        lastfm.is_available.return_value = False
        engine = _make_engine(lastfm=lastfm)
        assert engine.suggest(5, _context()) == []

    def test_cold_start_uses_user_top_artists(self):
        """With no rated tracks, engine should seed from Last.fm user top artists."""
        lastfm = _make_lastfm_mock()
        engine = _make_engine(lastfm=lastfm)
        results = engine.suggest(3, _context(rated=[]))
        assert isinstance(results, list)
        # get_user_top_artists should have been called (cold start path)
        lastfm.get_user_top_artists.assert_called()

    def test_seeded_from_high_rated_artists(self):
        """High-rated tracks' artists should be used as seeds."""
        lastfm = _make_lastfm_mock()
        rated = [_make_rated_track("Aphex Twin", score=9)]
        engine = _make_engine(lastfm=lastfm)
        engine.suggest(3, _context(rated=rated))
        # get_similar_artists should be called with the seed artist
        calls = [str(c) for c in lastfm.get_similar_artists.call_args_list]
        assert any("Aphex Twin" in c for c in calls)

    def test_low_rated_artists_not_seeded(self):
        """Tracks rated below threshold should not trigger graph traversal."""
        lastfm = _make_lastfm_mock()
        rated = [_make_rated_track("Low Rated Artist", score=5)]
        engine = _make_engine(lastfm=lastfm)
        engine.suggest(3, _context(rated=rated))
        # Since only a low-rated track is present, cold_start should kick in
        lastfm.get_user_top_artists.assert_called()

    def test_respects_excluded_track_ids(self):
        engine = _make_engine()
        ctx = _context()
        first_results = engine.suggest(5, ctx)
        if not first_results:
            pytest.skip("Engine returned no results")
        excluded = {s.track.id for s in first_results}
        second_results = engine.suggest(5, _context(excluded=excluded))
        returned_ids = {s.track.id for s in second_results}
        assert not (returned_ids & excluded)

    def test_suggestion_fields_populated(self):
        engine = _make_engine()
        results = engine.suggest(3, _context())
        for s in results:
            assert s.track.id
            assert s.track.title
            assert s.track.artist
            assert s.engine_name == "lastfm_graph"
            assert 0.0 <= s.engine_score <= 1.0
            assert isinstance(s.explanation, str)

    def test_does_not_raise_when_lastfm_fails(self):
        lastfm = _make_lastfm_mock()
        lastfm.get_similar_artists.side_effect = Exception("Last.fm down")
        lastfm.get_user_top_artists.side_effect = Exception("Last.fm down")
        engine = _make_engine(lastfm=lastfm)
        result = engine.suggest(5, _context())
        assert isinstance(result, list)

    def test_does_not_raise_when_tidal_fails(self):
        tidal = MagicMock()
        tidal.is_available.return_value = True
        tidal.search_tracks.side_effect = Exception("Tidal down")
        engine = _make_engine(tidal=tidal)
        result = engine.suggest(5, _context())
        assert isinstance(result, list)

    def test_genre_inferred_from_lastfm_tags(self):
        """Tags ['electronic', 'ambient'] → should classify as 'Electronic'."""
        lastfm = _make_lastfm_mock()
        lastfm.get_artist_tags.return_value = ["electronic"]
        engine = _make_engine(lastfm=lastfm)
        results = engine.suggest(3, _context())
        for s in results:
            # Electronic tags should map to Electronic bucket
            assert s.track.genre_primary in ("Electronic", "Unknown")

    def test_genre_falls_back_to_unknown_on_tag_error(self):
        lastfm = _make_lastfm_mock()
        lastfm.get_artist_tags.side_effect = Exception("Tag lookup failed")
        engine = _make_engine(lastfm=lastfm)
        results = engine.suggest(3, _context())
        for s in results:
            assert s.track.genre_primary == "Unknown"

    def test_results_respect_n_limit(self):
        engine = _make_engine()
        results = engine.suggest(2, _context())
        assert len(results) <= 2
