"""
Unit tests for TidalMixEngine.
Tidal connector is fully mocked — no real API calls.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from connectors.tidal import TidalTrack
from core.base_engine import EngineCapabilities, EngineHealth, SessionConfig, SessionContext
from engines.tidal_mix.engine import TidalMixEngine, _stable_track_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tidal_mock(mixes: list[dict] | None = None, tracks_per_mix: int = 5) -> MagicMock:
    """Build a mock TidalConnector that returns synthetic mixes and tracks."""
    tidal = MagicMock()
    tidal.is_available.return_value = True

    default_mixes = mixes or [
        {"id": "mix-daily", "title": "My Daily Discovery"},
        {"id": "mix-genre", "title": "Electronic Mix"},
    ]
    tidal.get_user_mixes.return_value = default_mixes

    def get_mix_tracks(mix_id):
        return [
            TidalTrack(
                tidal_id=f"{mix_id}-track-{i}",
                title=f"Track {i}",
                artist=f"Artist {i}",
                album="Album",
                duration_ms=180_000,
            )
            for i in range(tracks_per_mix)
        ]

    tidal.get_mix_tracks.side_effect = get_mix_tracks
    return tidal


def _make_engine(tidal=None) -> TidalMixEngine:
    engine = TidalMixEngine(tidal=tidal or _make_tidal_mock())
    engine.name = "tidal_mix"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.3,
        genre_coverage="broad",
        cold_start_friendly=True,
        data_requirements=[],
        speed="fast",
        description="",
    )
    return engine


def _minimal_context(excluded: set[str] | None = None) -> SessionContext:
    return SessionContext(
        rated_tracks=[],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=excluded or set(),
    )


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_ok_when_tidal_available(self):
        engine = _make_engine()
        assert engine.health_check().status == "ok"

    def test_unavailable_when_tidal_is_none(self):
        engine = TidalMixEngine(tidal=None)
        # Override _try_build_tidal result (settings.yaml has empty credentials)
        engine._tidal = None
        health = engine.health_check()
        assert health.status == "unavailable"

    def test_unavailable_when_tidal_not_authenticated(self):
        tidal = MagicMock()
        tidal.is_available.return_value = False
        engine = TidalMixEngine(tidal=tidal)
        assert engine.health_check().status == "unavailable"


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_list(self):
        engine = _make_engine()
        result = engine.suggest(5, _minimal_context())
        assert isinstance(result, list)

    def test_returns_suggestions_up_to_n(self):
        engine = _make_engine(_make_tidal_mock(tracks_per_mix=8))
        result = engine.suggest(5, _minimal_context())
        assert len(result) <= 5

    def test_returns_empty_when_tidal_unavailable(self):
        tidal = MagicMock()
        tidal.is_available.return_value = False
        engine = _make_engine(tidal)
        result = engine.suggest(5, _minimal_context())
        assert result == []

    def test_returns_empty_when_no_mixes(self):
        tidal = _make_tidal_mock()
        tidal.get_user_mixes.return_value = []
        engine = _make_engine(tidal)
        assert engine.suggest(5, _minimal_context()) == []

    def test_does_not_raise_when_mix_fetch_fails(self):
        tidal = _make_tidal_mock()
        tidal.get_mix_tracks.side_effect = Exception("Tidal down")
        engine = _make_engine(tidal)
        result = engine.suggest(5, _minimal_context())  # must not raise
        assert isinstance(result, list)

    def test_suggestion_fields_populated(self):
        engine = _make_engine()
        results = engine.suggest(3, _minimal_context())
        assert len(results) > 0
        for s in results:
            assert s.track.id
            assert s.track.title
            assert s.track.artist
            assert s.engine_name == "tidal_mix"
            assert 0.0 <= s.engine_score <= 1.0
            assert isinstance(s.explanation, str)

    def test_track_ids_are_stable(self):
        tidal = _make_tidal_mock(mixes=[{"id": "m", "title": "Mix"}], tracks_per_mix=2)
        engine = _make_engine(tidal)
        results1 = engine.suggest(5, _minimal_context())
        results2 = engine.suggest(5, _minimal_context())
        ids1 = {s.track.id for s in results1}
        ids2 = {s.track.id for s in results2}
        assert ids1 == ids2

    def test_respects_excluded_track_ids(self):
        tidal = _make_tidal_mock(mixes=[{"id": "m", "title": "Mix"}], tracks_per_mix=5)
        engine = _make_engine(tidal)

        # First pass: get all track IDs
        all_results = engine.suggest(10, _minimal_context())
        assert all_results

        # Exclude all track IDs
        excluded_ids = {s.track.id for s in all_results}
        ctx_excluded = _minimal_context(excluded=excluded_ids)
        results2 = engine.suggest(10, ctx_excluded)
        returned_ids = {s.track.id for s in results2}
        assert not (returned_ids & excluded_ids), "Excluded tracks appeared in results"

    def test_deduplicates_across_mixes(self):
        """Same track appearing in two mixes should only appear once in output."""
        shared_tidal_id = "shared-track"
        tidal = MagicMock()
        tidal.is_available.return_value = True
        tidal.get_user_mixes.return_value = [
            {"id": "mix-a", "title": "Mix A"},
            {"id": "mix-b", "title": "Mix B"},
        ]
        track = TidalTrack(tidal_id=shared_tidal_id, title="Shared", artist="X", album="", duration_ms=0)

        def get_tracks(mix_id):
            return [track]

        tidal.get_mix_tracks.side_effect = get_tracks
        engine = _make_engine(tidal)
        results = engine.suggest(10, _minimal_context())
        ids = [s.track.id for s in results]
        assert len(ids) == len(set(ids)), "Duplicate track IDs in results"

    def test_genre_primary_is_unknown(self):
        """TidalMix cannot determine genre — should always be 'Unknown'."""
        engine = _make_engine()
        results = engine.suggest(5, _minimal_context())
        for s in results:
            assert s.track.genre_primary == "Unknown"

    def test_tidal_id_preserved_on_track(self):
        tidal = _make_tidal_mock(mixes=[{"id": "m", "title": "M"}], tracks_per_mix=1)
        engine = _make_engine(tidal)
        results = engine.suggest(5, _minimal_context())
        assert results[0].track.tidal_id == "m-track-0"

    def test_does_not_raise_on_suggest_exception(self):
        tidal = _make_tidal_mock()
        tidal.get_user_mixes.side_effect = Exception("Unexpected error")
        engine = _make_engine(tidal)
        result = engine.suggest(5, _minimal_context())
        assert result == []
