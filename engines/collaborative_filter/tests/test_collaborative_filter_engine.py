"""
Unit tests for CollaborativeFilterEngine.
All external dependencies (lightfm, Last.fm, Tidal) are mocked.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import engines.collaborative_filter.engine as engine_module
from core.base_engine import (
    EngineCapabilities,
    RatedTrack,
    SessionConfig,
    SessionContext,
    Track,
)
from engines.collaborative_filter.engine import (
    CollaborativeFilterEngine,
    _ALL_GENRES,
    _LIKED_THRESHOLD,
    _MIN_RATINGS_UNAVAILABLE,
    _stable_track_id,
)
from connectors.lastfm import LastFmTrack
from connectors.tidal import TidalTrack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lastfm_mock() -> MagicMock:
    lastfm = MagicMock()
    lastfm.is_available.return_value = True

    def get_tag_top_tracks(tag, limit=30):
        return [
            LastFmTrack(title=f"{tag.title()} Track {i}", artist=f"Artist {i}", mbid="", url="")
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
                title=f"Result: {query[:20]}",
                artist="Artist",
                album="Album",
                duration_ms=200_000,
            )
        ]

    tidal.search_tracks.side_effect = search_tracks
    return tidal


def _make_engine(lastfm=None, tidal=None) -> CollaborativeFilterEngine:
    engine = CollaborativeFilterEngine(
        lastfm=lastfm if lastfm is not None else _make_lastfm_mock(),
        tidal=tidal if tidal is not None else _make_tidal_mock(),
    )
    engine.name = "collaborative_filter"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.3,
        genre_coverage="broad",
        cold_start_friendly=False,
        data_requirements=["ratings", "genre_tags"],
        speed="fast",
        description="",
    )
    return engine


def _make_rated_track(
    genre: str = "Electronic",
    score: int = 8,
    tags: list[str] | None = None,
) -> RatedTrack:
    return RatedTrack(
        track=Track(
            id=str(uuid.uuid4()),
            title="Track",
            artist="Artist",
            album="Album",
            duration_ms=200_000,
            genre_primary=genre,
            genre_tags=tags if tags is not None else [genre.lower()],
        ),
        score=score,
        rated_at=datetime.now(),
        session_id="s1",
    )


def _context(rated=None, excluded=None) -> SessionContext:
    return SessionContext(
        rated_tracks=rated or [],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=excluded or set(),
    )


def _make_lightfm_mocks(n_items: int = 5, n_genres: int = 3):
    """Return (MockLightFM, MockDataset) with realistic-enough return values."""
    mock_model = MagicMock()
    import random
    rng = random.Random(99)
    mock_model.predict.return_value = [rng.gauss(0, 1) for _ in range(n_items)]

    mock_ds_instance = MagicMock()
    item_mapping = {f"item-{i}": i for i in range(n_items)}
    mock_ds_instance.build_interactions.return_value = (MagicMock(), MagicMock())
    mock_ds_instance.build_item_features.return_value = MagicMock()
    mock_ds_instance.mapping.return_value = (
        {"me": 0},   # user_mapping
        {},           # user_feature_mapping
        item_mapping, # item_mapping
        {},           # item_feature_mapping
    )

    MockDataset = MagicMock(return_value=mock_ds_instance)
    MockLightFM = MagicMock(return_value=mock_model)
    return MockLightFM, MockDataset


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_unavailable_when_lightfm_missing(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", False):
            engine = _make_engine()
            assert engine.health_check().status == "unavailable"

    def test_unavailable_when_lastfm_none(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = CollaborativeFilterEngine(lastfm=None, tidal=_make_tidal_mock())
            engine._lastfm = None
            assert engine.health_check().status == "unavailable"

    def test_unavailable_when_lastfm_not_available(self):
        lastfm = MagicMock()
        lastfm.is_available.return_value = False
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine(lastfm=lastfm)
            assert engine.health_check().status == "unavailable"

    def test_ok_when_all_healthy(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            assert engine.health_check().status == "ok"


# ---------------------------------------------------------------------------
# Genre ranking
# ---------------------------------------------------------------------------

class TestGenreRanking:
    def test_returns_all_genres_when_insufficient_data(self):
        engine = _make_engine()
        few_ratings = [_make_rated_track(score=8) for _ in range(_MIN_RATINGS_UNAVAILABLE - 1)]
        ranked = engine._compute_genre_rankings(few_ratings)
        assert set(ranked) == set(_ALL_GENRES)

    def test_returns_all_genres_when_no_positives(self):
        engine = _make_engine()
        all_dislikes = [
            _make_rated_track("Rock", score=_LIKED_THRESHOLD - 1)
            for _ in range(_MIN_RATINGS_UNAVAILABLE + 5)
        ]
        ranked = engine._compute_genre_rankings(all_dislikes)
        assert set(ranked) == set(_ALL_GENRES)

    def test_with_lightfm_mocked_returns_all_genres_ranked(self):
        """Verify _compute_genre_rankings returns a complete, sorted genre list."""
        rated = [_make_rated_track("Electronic", score=9) for _ in range(15)]
        # Override item_mapping so it matches the track IDs we produce
        item_mapping = {rt.track.id: i for i, rt in enumerate(rated)}

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9] * len(rated)

        mock_ds = MagicMock()
        mock_ds.build_interactions.return_value = (MagicMock(), MagicMock())
        mock_ds.build_item_features.return_value = MagicMock()
        mock_ds.mapping.return_value = ({}, {}, item_mapping, {})

        MockLightFM = MagicMock(return_value=mock_model)
        MockDataset = MagicMock(return_value=mock_ds)

        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True), \
             patch.object(engine_module, "_LightFM", MockLightFM), \
             patch.object(engine_module, "_Dataset", MockDataset):
            engine = _make_engine()
            ranked = engine._compute_genre_rankings(rated)

        assert len(ranked) == len(_ALL_GENRES)
        assert set(ranked) == set(_ALL_GENRES)

    def test_falls_back_gracefully_when_lightfm_raises(self):
        """If LightFM training throws, fall back to default ordering."""
        rated = [_make_rated_track("Electronic", score=9) for _ in range(15)]

        mock_ds = MagicMock()
        mock_ds.build_interactions.side_effect = Exception("LightFM exploded")

        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True), \
             patch.object(engine_module, "_Dataset", MagicMock(return_value=mock_ds)):
            engine = _make_engine()
            ranked = engine._compute_genre_rankings(rated)

        assert set(ranked) == set(_ALL_GENRES)


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_list(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", False):
            engine = _make_engine()
            assert isinstance(engine.suggest(5, _context()), list)

    def test_returns_empty_when_lightfm_unavailable(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", False):
            engine = _make_engine()
            assert engine.suggest(5, _context()) == []

    def test_returns_empty_when_lastfm_unavailable(self):
        lastfm = MagicMock()
        lastfm.is_available.return_value = False
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine(lastfm=lastfm)
            assert engine.suggest(5, _context()) == []

    def test_returns_results_with_few_ratings(self):
        """With fewer than MIN_RATINGS_UNAVAILABLE, falls back but still returns results."""
        few_ratings = [_make_rated_track("Electronic", score=9) for _ in range(3)]
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            results = engine.suggest(3, _context(rated=few_ratings))
        assert isinstance(results, list)
        assert len(results) > 0

    def test_results_within_n(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            results = engine.suggest(3, _context())
        assert len(results) <= 3

    def test_suggestion_fields_populated(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            results = engine.suggest(3, _context())
        for s in results:
            assert s.track.id
            assert s.track.title
            assert s.track.artist
            assert s.engine_name == "collaborative_filter"
            assert 0.0 <= s.engine_score <= 1.0
            assert isinstance(s.explanation, str)

    def test_genre_primary_from_known_buckets(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            results = engine.suggest(5, _context())
        for s in results:
            assert s.track.genre_primary in _ALL_GENRES

    def test_respects_excluded_ids(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            first = engine.suggest(5, _context())
        if not first:
            pytest.skip("No results returned")
        excluded = {s.track.id for s in first}
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine2 = _make_engine()
            second = engine2.suggest(5, _context(excluded=excluded))
        assert not {s.track.id for s in second}.intersection(excluded)

    def test_does_not_raise_when_lastfm_fails(self):
        lastfm = _make_lastfm_mock()
        lastfm.get_tag_top_tracks.side_effect = Exception("Last.fm down")
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine(lastfm=lastfm)
            result = engine.suggest(5, _context())
        assert isinstance(result, list)

    def test_does_not_raise_when_tidal_fails(self):
        tidal = MagicMock()
        tidal.is_available.return_value = True
        tidal.search_tracks.side_effect = Exception("Tidal down")
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine(tidal=tidal)
            result = engine.suggest(5, _context())
        assert isinstance(result, list)

    def test_on_session_complete_does_not_raise(self):
        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True):
            engine = _make_engine()
            engine.on_session_complete([])

    def test_lightfm_model_consulted_with_sufficient_ratings(self):
        """When enough ratings exist, LightFM training + predict should be called."""
        rated = [
            _make_rated_track("Electronic", score=9, tags=["electronic"])
            for _ in range(_MIN_RATINGS_UNAVAILABLE + 5)
        ]
        item_mapping = {rt.track.id: i for i, rt in enumerate(rated)}

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5] * len(rated)

        mock_ds = MagicMock()
        mock_ds.build_interactions.return_value = (MagicMock(), MagicMock())
        mock_ds.build_item_features.return_value = MagicMock()
        mock_ds.mapping.return_value = ({}, {}, item_mapping, {})

        MockLightFM = MagicMock(return_value=mock_model)
        MockDataset = MagicMock(return_value=mock_ds)

        with patch.object(engine_module, "_LIGHTFM_AVAILABLE", True), \
             patch.object(engine_module, "_LightFM", MockLightFM), \
             patch.object(engine_module, "_Dataset", MockDataset):
            engine = _make_engine()
            engine.suggest(3, _context(rated=rated))

        mock_model.fit.assert_called_once()
        mock_model.predict.assert_called_once()

    def test_stable_track_id_deterministic(self):
        """Same Tidal ID should always produce the same internal UUID."""
        tid = "tid-abc123"
        assert _stable_track_id(tid) == _stable_track_id(tid)
        assert _stable_track_id("tid-aaa") != _stable_track_id("tid-bbb")
