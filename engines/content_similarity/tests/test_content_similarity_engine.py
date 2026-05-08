"""
Unit tests for ContentSimilarityEngine.
External dependencies (pynndescent, DB) are mocked — no real index built.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import engines.content_similarity.engine as engine_module
from core.base_engine import (
    EngineCapabilities,
    EngineHealth,
    RatedTrack,
    SessionConfig,
    SessionContext,
    Track,
)
from engines.content_similarity.engine import (
    ContentSimilarityEngine,
    _FEATURE_DIM,
    _LIKED_THRESHOLD,
    _MIN_CORPUS_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(seed: int = 0) -> list[float]:
    """Return a deterministic 8-dim feature vector."""
    import random
    rng = random.Random(seed)
    return [rng.random() for _ in range(_FEATURE_DIM)]


def _make_track(
    genre: str = "Electronic",
    with_features: bool = True,
    seed: int = 0,
) -> Track:
    return Track(
        id=str(uuid.uuid4()),
        title=f"Track {seed}",
        artist="Artist",
        album="Album",
        duration_ms=200_000,
        genre_primary=genre,
        audio_features={"features": _make_features(seed)} if with_features else None,
    )


def _make_rated(track: Track, score: int = 8) -> RatedTrack:
    return RatedTrack(track=track, score=score, rated_at=datetime.now(), session_id="s1")


def _make_db_mock(tracks: list[Track] | None = None) -> MagicMock:
    db = MagicMock()
    db.get_tracks_with_audio_features.return_value = tracks if tracks is not None else []
    return db


def _context(rated=None, excluded=None) -> SessionContext:
    return SessionContext(
        rated_tracks=rated or [],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=excluded or set(),
        excluded_track_fingerprints=set(),
    )


def _make_engine(db=None, seed: int = 42) -> ContentSimilarityEngine:
    engine = ContentSimilarityEngine(db=db or _make_db_mock(), seed=seed)
    engine.name = "content_similarity"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.2,
        genre_coverage="narrow",
        cold_start_friendly=False,
        data_requirements=["ratings", "audio_features"],
        speed="fast",
        description="",
    )
    return engine


def _make_nndescent_mock(
    corpus_size: int = 5,
    distances: list[float] | None = None,
) -> MagicMock:
    """
    Return a mock NNDescent class that simulates pynndescent's query() API.

    pynndescent returns (indices, distances) as 2-D arrays where
    indices[0] and distances[0] are the results for the first query vector.
    """
    import numpy as np

    if distances is None:
        distances = [0.1 * i for i in range(corpus_size)]
    n = min(corpus_size, len(distances))
    idx_arr = np.array([list(range(n))], dtype=np.int32)
    dist_arr = np.array([distances[:n]], dtype=np.float32)

    mock_idx = MagicMock()
    mock_idx.prepare.return_value = None
    mock_idx.query.return_value = (idx_arr, dist_arr)

    mock_cls = MagicMock(return_value=mock_idx)
    return mock_cls


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_unavailable_when_pynndescent_missing(self):
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", False):
            engine = _make_engine()
            assert engine.health_check().status == "unavailable"

    def test_unavailable_when_db_none(self):
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = ContentSimilarityEngine(db=None)
            engine._db = None
            assert engine.health_check().status == "unavailable"

    def test_degraded_when_corpus_too_small(self):
        db = _make_db_mock(tracks=[_make_track(seed=i) for i in range(5)])
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            health = engine.health_check()
        assert health.status == "degraded"
        assert str(5) in health.message

    def test_ok_when_corpus_large_enough(self):
        tracks = [_make_track(seed=i) for i in range(_MIN_CORPUS_SIZE + 1)]
        db = _make_db_mock(tracks=tracks)
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            assert engine.health_check().status == "ok"

    def test_unavailable_when_db_query_raises(self):
        db = _make_db_mock()
        db.get_tracks_with_audio_features.side_effect = Exception("DB down")
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            assert engine.health_check().status == "unavailable"


# ---------------------------------------------------------------------------
# Preference vector
# ---------------------------------------------------------------------------

class TestPreferenceVector:
    def test_returns_none_when_no_liked_tracks(self):
        engine = _make_engine()
        rated = [_make_rated(_make_track(), score=3)]
        assert engine._compute_preference_vector(rated) is None

    def test_returns_none_when_liked_tracks_have_no_features(self):
        engine = _make_engine()
        track = _make_track(with_features=False)
        rated = [_make_rated(track, score=9)]
        assert engine._compute_preference_vector(rated) is None

    def test_single_liked_track_returns_its_features(self):
        engine = _make_engine()
        features = _make_features(seed=7)
        track = _make_track(seed=7, with_features=True)
        track.audio_features = {"features": features}
        rated = [_make_rated(track, score=10)]
        pref = engine._compute_preference_vector(rated)
        # Weight = 10/10 = 1.0 → pref == features
        assert pref is not None
        assert len(pref) == _FEATURE_DIM
        for a, b in zip(pref, features):
            assert abs(a - b) < 1e-9

    def test_higher_scored_tracks_pull_stronger(self):
        """Track with score=10 should count more than track with score=7."""
        engine = _make_engine()
        t1 = _make_track(seed=1); t1.audio_features = {"features": [1.0] * _FEATURE_DIM}
        t2 = _make_track(seed=2); t2.audio_features = {"features": [0.0] * _FEATURE_DIM}
        rated = [_make_rated(t1, score=10), _make_rated(t2, score=7)]
        pref = engine._compute_preference_vector(rated)
        # t1 weight=1.0, t2 weight=0.7 → pref[0] = 1.0/(1.0+0.7) ≈ 0.588
        assert pref is not None
        assert pref[0] > 0.5

    def test_length_equals_feature_dim(self):
        engine = _make_engine()
        track = _make_track(seed=5)
        pref = engine._compute_preference_vector([_make_rated(track, score=8)])
        assert pref is not None
        assert len(pref) == _FEATURE_DIM

    def test_ignores_disliked_tracks(self):
        engine = _make_engine()
        liked = _make_track(seed=1); liked.audio_features = {"features": [1.0] * _FEATURE_DIM}
        disliked = _make_track(seed=2); disliked.audio_features = {"features": [0.0] * _FEATURE_DIM}
        rated = [
            _make_rated(liked, score=_LIKED_THRESHOLD),
            _make_rated(disliked, score=_LIKED_THRESHOLD - 1),
        ]
        pref = engine._compute_preference_vector(rated)
        assert pref is not None
        # Only liked track contributes → pref[0] == 1.0
        assert abs(pref[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_empty_when_pynndescent_unavailable(self):
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", False):
            engine = _make_engine()
            assert engine.suggest(5, _context()) == []

    def test_returns_empty_when_db_none(self):
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = ContentSimilarityEngine(db=None)
            engine._db = None
            assert engine.suggest(5, _context()) == []

    def test_returns_empty_when_corpus_is_empty(self):
        db = _make_db_mock(tracks=[])
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            assert engine.suggest(5, _context()) == []

    def test_cold_start_returns_results_from_corpus(self):
        """No liked tracks → random shuffle from corpus."""
        tracks = [_make_track(seed=i) for i in range(5)]
        db = _make_db_mock(tracks=tracks)
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            results = engine.suggest(3, _context(rated=[]))
        assert len(results) == 3

    def test_results_within_n(self):
        tracks = [_make_track(seed=i) for i in range(10)]
        db = _make_db_mock(tracks=tracks)
        mock_cls = _make_nndescent_mock(corpus_size=10)
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True), \
             patch.object(engine_module, "_NNDescent", mock_cls):
            engine = _make_engine(db=db)
            liked = _make_track(seed=99); liked.audio_features = {"features": _make_features(99)}
            rated = [_make_rated(liked, score=9)]
            results = engine.suggest(3, _context(rated=rated))
        assert len(results) <= 3

    def test_suggestion_fields_populated(self):
        tracks = [_make_track(seed=i) for i in range(5)]
        db = _make_db_mock(tracks=tracks)
        mock_cls = _make_nndescent_mock(corpus_size=5)
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True), \
             patch.object(engine_module, "_NNDescent", mock_cls):
            engine = _make_engine(db=db)
            liked = _make_track(seed=99); liked.audio_features = {"features": _make_features(99)}
            results = engine.suggest(3, _context(rated=[_make_rated(liked, score=9)]))
        for s in results:
            assert s.track.id
            assert s.track.title
            assert s.engine_name == "content_similarity"
            assert 0.0 <= s.engine_score <= 1.0
            assert isinstance(s.explanation, str)

    def test_respects_excluded_ids(self):
        tracks = [_make_track(seed=i) for i in range(10)]
        excluded_ids = {tracks[0].id, tracks[1].id}
        db = _make_db_mock(tracks=tracks)
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            # Cold start so we can test exclusion without pynndescent
            results = engine.suggest(5, _context(excluded=excluded_ids))
        result_ids = {s.track.id for s in results}
        assert not result_ids.intersection(excluded_ids)

    def test_excludes_already_rated_tracks(self):
        tracks = [_make_track(seed=i) for i in range(10)]
        rated_track = tracks[0]
        rated = [_make_rated(rated_track, score=3)]
        db = _make_db_mock(tracks=tracks)
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            results = engine.suggest(5, _context(rated=rated))
        result_ids = {s.track.id for s in results}
        assert rated_track.id not in result_ids

    def test_does_not_raise_when_db_fails(self):
        db = _make_db_mock()
        db.get_tracks_with_audio_features.side_effect = Exception("DB exploded")
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True):
            engine = _make_engine(db=db)
            result = engine.suggest(5, _context())
        assert isinstance(result, list)

    def test_engine_score_in_range(self):
        tracks = [_make_track(seed=i) for i in range(5)]
        db = _make_db_mock(tracks=tracks)
        # Distances from 0.0 (identical) to 2.0 (opposite)
        distances = [0.0, 0.5, 1.0, 1.5, 2.0]
        mock_cls = _make_nndescent_mock(corpus_size=5, distances=distances)
        liked = _make_track(seed=99); liked.audio_features = {"features": _make_features(99)}
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True), \
             patch.object(engine_module, "_NNDescent", mock_cls):
            engine = _make_engine(db=db)
            results = engine.suggest(5, _context(rated=[_make_rated(liked, score=9)]))
        for s in results:
            assert 0.0 <= s.engine_score <= 1.0

    def test_on_session_complete_does_not_raise(self):
        engine = _make_engine()
        engine.on_session_complete([])

    def test_similar_tracks_score_higher_than_dissimilar(self):
        """Track at distance 0.1 should score higher than track at distance 1.8."""
        tracks = [_make_track(seed=i) for i in range(2)]
        db = _make_db_mock(tracks=tracks)
        # indices=[0,1], distances=[0.1, 1.8]
        mock_cls = _make_nndescent_mock(corpus_size=2, distances=[0.1, 1.8])
        liked = _make_track(seed=99); liked.audio_features = {"features": _make_features(99)}
        with patch.object(engine_module, "_PYNNDESCENT_AVAILABLE", True), \
             patch.object(engine_module, "_NNDescent", mock_cls):
            engine = _make_engine(db=db)
            results = engine.suggest(2, _context(rated=[_make_rated(liked, score=9)]))
        assert len(results) == 2
        assert results[0].engine_score > results[1].engine_score
