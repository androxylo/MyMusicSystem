"""
Unit tests for GenreExplorerEngine.
All external connectors are mocked — no real API calls.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, call

import pytest

from connectors.lastfm import LastFmTrack
from connectors.tidal import TidalTrack
from core.base_engine import (
    EngineCapabilities,
    EngineHealth,
    RatedTrack,
    SessionConfig,
    SessionContext,
    Track,
)
from engines.genre_explorer.engine import GenreExplorerEngine, _ALL_GENRES, _stable_track_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lastfm_mock() -> MagicMock:
    lastfm = MagicMock()
    lastfm.is_available.return_value = True

    def get_tag_top_tracks(tag, limit=30):
        return [
            LastFmTrack(
                title=f"{tag.title()} Track {i}",
                artist=f"{tag.title()} Artist {i}",
                mbid="",
                url="",
            )
            for i in range(min(limit, 5))
        ]

    lastfm.get_tag_top_tracks.side_effect = get_tag_top_tracks
    return lastfm


def _make_tidal_mock() -> MagicMock:
    tidal = MagicMock()
    tidal.is_available.return_value = True

    def search_tracks(query, limit=2):
        return [
            TidalTrack(
                tidal_id=f"tid-{uuid.uuid4().hex[:8]}",
                title=f"Tidal: {query[:20]}",
                artist="Tidal Artist",
                album="Tidal Album",
                duration_ms=210_000,
            )
        ]

    tidal.search_tracks.side_effect = search_tracks
    return tidal


def _make_engine(lastfm=None, tidal=None) -> GenreExplorerEngine:
    engine = GenreExplorerEngine(
        lastfm=lastfm if lastfm is not None else _make_lastfm_mock(),
        tidal=tidal if tidal is not None else _make_tidal_mock(),
    )
    engine.name = "genre_explorer"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.8,
        genre_coverage="broad",
        cold_start_friendly=True,
        data_requirements=["ratings"],
        speed="fast",
        description="",
    )
    return engine


def _make_rated_track(genre_primary: str = "Electronic", score: int = 8) -> RatedTrack:
    track = Track(
        id=str(uuid.uuid4()),
        title="Track",
        artist="Artist",
        album="Album",
        duration_ms=200_000,
        genre_primary=genre_primary,
        genre_tags=[genre_primary.lower()],
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
    def test_ok_when_both_available(self):
        assert _make_engine().health_check().status == "ok"

    def test_unavailable_when_lastfm_missing(self):
        engine = GenreExplorerEngine(lastfm=None, tidal=_make_tidal_mock())
        engine._lastfm = None
        assert engine.health_check().status == "unavailable"

    def test_degraded_when_tidal_missing(self):
        engine = GenreExplorerEngine(lastfm=_make_lastfm_mock(), tidal=None)
        engine._tidal = None
        assert engine.health_check().status == "degraded"


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_list(self):
        assert isinstance(_make_engine().suggest(5, _context()), list)

    def test_returns_empty_when_lastfm_unavailable(self):
        lastfm = MagicMock()
        lastfm.is_available.return_value = False
        engine = _make_engine(lastfm=lastfm)
        assert engine.suggest(5, _context()) == []

    def test_cold_start_returns_results_across_genres(self):
        """With no rating history, should return tracks from multiple different genres."""
        engine = _make_engine()
        results = engine.suggest(5, _context(rated=[]))
        assert len(results) > 0
        genres = {s.track.genre_primary for s in results}
        # Should come from at least 2 different genre buckets
        assert len(genres) >= 2

    def test_prioritizes_unexplored_genres(self):
        """Genres with no rated tracks should be targeted before saturated ones."""
        # Many Electronic tracks rated
        many_electronic = [_make_rated_track("Electronic") for _ in range(10)]
        engine = _make_engine()
        results = engine.suggest(3, _context(rated=many_electronic))
        genres = {s.track.genre_primary for s in results}
        # Electronic should not dominate — other genres should appear
        assert "Electronic" not in genres or len(genres) > 1

    def test_genre_primary_matches_target_genre(self):
        """Tracks returned by the engine should have genre_primary = the genre being explored."""
        engine = _make_engine()
        results = engine.suggest(5, _context())
        for s in results:
            assert s.track.genre_primary in _ALL_GENRES

    def test_respects_excluded_track_ids(self):
        engine = _make_engine()
        first = engine.suggest(5, _context())
        if not first:
            pytest.skip("Engine returned no results")
        excluded = {s.track.id for s in first}
        second = engine.suggest(5, _context(excluded=excluded))
        returned = {s.track.id for s in second}
        assert not (returned & excluded)

    def test_suggestion_fields_populated(self):
        engine = _make_engine()
        results = engine.suggest(3, _context())
        for s in results:
            assert s.track.id
            assert s.track.title
            assert s.track.artist
            assert s.engine_name == "genre_explorer"
            assert 0.0 <= s.engine_score <= 1.0
            assert isinstance(s.explanation, str)

    def test_does_not_raise_when_lastfm_fails(self):
        lastfm = _make_lastfm_mock()
        lastfm.get_tag_top_tracks.side_effect = Exception("Last.fm down")
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

    def test_results_within_n_limit(self):
        engine = _make_engine()
        results = engine.suggest(3, _context())
        assert len(results) <= 3

    def test_tidal_track_id_stable(self):
        """Same track always gets the same internal ID."""
        fixed_tidal_id = "fixed-id-123"
        tidal = MagicMock()
        tidal.is_available.return_value = True
        tidal.search_tracks.return_value = [
            TidalTrack(tidal_id=fixed_tidal_id, title="Fixed", artist="A", album="B", duration_ms=100)
        ]
        engine = _make_engine(tidal=tidal)
        r1 = engine.suggest(5, _context())
        r2 = engine.suggest(5, _context())
        ids1 = {s.track.id for s in r1}
        ids2 = {s.track.id for s in r2}
        assert ids1 == ids2

    def test_explanation_mentions_genre(self):
        """Explanation strings should reference the genre being explored."""
        engine = _make_engine()
        results = engine.suggest(3, _context())
        for s in results:
            assert s.track.genre_primary in s.explanation or "genre" in s.explanation.lower()
