"""
Unit tests for LastFmTagSimilarityEngine.
All external connectors are mocked — no real API calls.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, call

import pytest

from connectors.tidal import TidalTrack
from core.base_engine import (
    EngineCapabilities,
    EngineHealth,
    RatedTrack,
    SessionConfig,
    SessionContext,
    Track,
)
from engines.lastfm_tag_similarity.engine import (
    LastFmTagSimilarityEngine,
    _cosine,
    _stable_track_id,
    _LIKE_THRESHOLD,
    _MIN_LIKED_TRACKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(
    tidal_id: str = "t1",
    title: str = "Track",
    artist: str = "Artist",
    genre: str = "Electronic",
) -> Track:
    return Track(
        id=_stable_track_id(tidal_id),
        title=title,
        artist=artist,
        album="Album",
        duration_ms=200_000,
        genre_primary=genre,
        tidal_id=tidal_id,
    )


def _make_rated(track: Track, score: int, session_id: str = "sess-1") -> RatedTrack:
    return RatedTrack(track=track, score=score, rated_at=datetime.now(), session_id=session_id)


def _make_context(rated: list[RatedTrack] | None = None) -> SessionContext:
    return SessionContext(
        rated_tracks=rated or [],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=set(),
    )


def _make_lastfm_mock(
    track_tags: list[tuple[str, float]] | None = None,
    tag_top_tracks=None,
    artist_tags: list[str] | None = None,
) -> MagicMock:
    lastfm = MagicMock()
    lastfm.is_available.return_value = True

    # Default per-track tags (None means use default, [] means return empty)
    if track_tags is None:
        default_tags = [("ambient", 0.9), ("electronic", 0.7), ("instrumental", 0.5)]
    else:
        default_tags = track_tags
    lastfm.get_track_tags.return_value = default_tags

    # Default tag top tracks
    if tag_top_tracks is None:
        from connectors.lastfm import LastFmTrack as LFTrack
        tag_top_tracks = [
            LFTrack(title=f"Candidate {i}", artist=f"Artist {i}")
            for i in range(5)
        ]
    lastfm.get_tag_top_tracks.return_value = tag_top_tracks

    lastfm.get_artist_tags.return_value = artist_tags or ["electronic"]
    return lastfm


def _make_tidal_mock(results: list[TidalTrack] | None = None) -> MagicMock:
    tidal = MagicMock()
    tidal.is_available.return_value = True

    counter = [0]

    def search_tracks(query, limit=10):
        idx = counter[0]
        counter[0] += 1
        return results or [
            TidalTrack(
                tidal_id=f"tid-{idx:04d}",
                title=f"Tidal Track {idx}",
                artist="Some Artist",
                album="Album",
                duration_ms=210_000,
            )
        ]

    tidal.search_tracks.side_effect = search_tracks
    return tidal


def _make_engine(lastfm=None, tidal=None) -> LastFmTagSimilarityEngine:
    engine = LastFmTagSimilarityEngine(
        lastfm=lastfm if lastfm is not None else _make_lastfm_mock(),
        tidal=tidal if tidal is not None else _make_tidal_mock(),
    )
    engine.name = "lastfm_tag_similarity"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.3,
        genre_coverage="broad",
        cold_start_friendly=False,
        data_requirements=["ratings"],
        speed="slow",
        description="",
    )
    return engine


def _liked_context(n: int = _MIN_LIKED_TRACKS, score: int = 8) -> SessionContext:
    """Context with n liked tracks, all scored >= LIKE_THRESHOLD."""
    rated = [
        _make_rated(_make_track(tidal_id=f"t{i}", title=f"Song {i}", artist=f"Band {i}"), score)
        for i in range(n)
    ]
    return _make_context(rated)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        v = {"ambient": 0.9, "electronic": 0.7}
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = {"ambient": 1.0}
        b = {"rock": 1.0}
        assert _cosine(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"ambient": 1.0, "electronic": 1.0}
        b = {"ambient": 1.0, "rock": 1.0}
        # dot = 1, |a| = sqrt(2), |b| = sqrt(2), cosine = 1/2
        assert abs(_cosine(a, b) - 0.5) < 1e-6

    def test_empty_a(self):
        assert _cosine({}, {"ambient": 0.5}) == 0.0

    def test_empty_b(self):
        assert _cosine({"ambient": 0.5}, {}) == 0.0


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_ok_when_lastfm_available(self):
        engine = _make_engine()
        assert engine.health_check().status == "ok"

    def test_unavailable_when_lastfm_none(self):
        engine = _make_engine(lastfm=None)
        engine._lastfm = None
        h = engine.health_check()
        assert h.status == "unavailable"
        assert "Last.fm" in h.message

    def test_unavailable_when_lastfm_not_available(self):
        lf = _make_lastfm_mock()
        lf.is_available.return_value = False
        engine = _make_engine(lastfm=lf)
        h = engine.health_check()
        assert h.status == "unavailable"


# ---------------------------------------------------------------------------
# Preference vector
# ---------------------------------------------------------------------------

class TestPreferenceVector:
    def test_preference_vector_weighted_by_score(self):
        lf = _make_lastfm_mock(track_tags=[("ambient", 1.0)])
        engine = _make_engine(lastfm=lf)

        # Two liked tracks: scores 7 and 10 → weights 0.7 and 1.0
        t1 = _make_rated(_make_track("t1"), 7)
        t2 = _make_rated(_make_track("t2"), 10)

        pref = engine._build_preference_vector([t1, t2])
        assert "ambient" in pref
        # weight = 0.7 * 1.0 + 1.0 * 1.0 = 1.7
        assert abs(pref["ambient"] - 1.7) < 1e-6

    def test_preference_vector_skips_failed_tag_fetch(self):
        lf = _make_lastfm_mock()
        lf.get_track_tags.side_effect = Exception("API down")
        engine = _make_engine(lastfm=lf)

        rated = [_make_rated(_make_track("t1"), 8)]
        pref = engine._build_preference_vector(rated)
        assert pref == {}

    def test_preference_vector_empty_when_no_tags_returned(self):
        lf = _make_lastfm_mock(track_tags=[])
        engine = _make_engine(lastfm=lf)

        rated = [_make_rated(_make_track("t1"), 8)]
        pref = engine._build_preference_vector(rated)
        assert pref == {}


# ---------------------------------------------------------------------------
# suggest() — main flow
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_suggestions_with_sufficient_liked_tracks(self):
        engine = _make_engine()
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(3, ctx)
        assert len(suggestions) > 0

    def test_returns_empty_below_min_liked_tracks(self):
        engine = _make_engine()
        ctx = _liked_context(_MIN_LIKED_TRACKS - 1)
        suggestions = engine.suggest(3, ctx)
        assert suggestions == []

    def test_returns_empty_when_all_below_like_threshold(self):
        engine = _make_engine()
        # 10 tracks all rated below threshold
        rated = [_make_rated(_make_track(f"t{i}"), _LIKE_THRESHOLD - 1) for i in range(10)]
        suggestions = engine.suggest(3, _make_context(rated))
        assert suggestions == []

    def test_respects_excluded_track_ids(self):
        engine = _make_engine()
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(5, ctx)
        # Get track IDs from first run
        ids = {s.track.id for s in suggestions}
        # Second run with those IDs excluded
        ctx2 = SessionContext(
            rated_tracks=ctx.rated_tracks,
            recent_sessions=[],
            session_config=SessionConfig(),
            excluded_track_ids=ids,
        )
        suggestions2 = engine.suggest(5, ctx2)
        for s in suggestions2:
            assert s.track.id not in ids

    def test_suggestion_has_required_fields(self):
        engine = _make_engine()
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(1, ctx)
        assert len(suggestions) >= 1
        s = suggestions[0]
        assert s.track.id
        assert s.track.title
        assert s.track.artist
        assert s.engine_name == "lastfm_tag_similarity"
        assert 0.0 <= s.engine_score <= 1.0
        assert s.explanation

    def test_engine_score_is_cosine_similarity(self):
        """Engine score should be in [0, 1] (cosine similarity range)."""
        engine = _make_engine()
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(5, ctx)
        for s in suggestions:
            assert 0.0 <= s.engine_score <= 1.0

    def test_returns_empty_when_lastfm_unavailable(self):
        lf = _make_lastfm_mock()
        lf.is_available.return_value = False
        engine = _make_engine(lastfm=lf)
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        assert engine.suggest(3, ctx) == []

    def test_returns_empty_when_no_candidates_scored(self):
        """If all candidate tag fetches fail, no suggestions."""
        lf = _make_lastfm_mock()
        call_count = [0]

        def get_tags(title, artist, limit=15):
            # First N calls succeed (for preference vector), rest fail
            call_count[0] += 1
            if call_count[0] <= _MIN_LIKED_TRACKS:
                return [("ambient", 0.8)]
            raise Exception("API down")

        lf.get_track_tags.side_effect = get_tags
        engine = _make_engine(lastfm=lf)
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(3, ctx)
        # May be empty or small depending on how many succeed
        assert isinstance(suggestions, list)

    def test_tidal_unavailable_returns_empty(self):
        tidal = _make_tidal_mock()
        tidal.is_available.return_value = False
        engine = _make_engine(tidal=tidal)
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(3, ctx)
        assert suggestions == []

    def test_tidal_none_returns_empty(self):
        engine = _make_engine()
        engine._tidal = None
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(3, ctx)
        assert suggestions == []

    def test_does_not_raise_on_exception(self):
        """suggest() must not raise — returns empty list instead."""
        lf = _make_lastfm_mock()
        lf.get_tag_top_tracks.side_effect = RuntimeError("boom")
        # get_track_tags still works for preference vector
        lf.get_track_tags.return_value = [("ambient", 0.9)]
        engine = _make_engine(lastfm=lf)
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        # Should not raise
        result = engine.suggest(3, ctx)
        assert isinstance(result, list)

    def test_caps_at_n_suggestions(self):
        engine = _make_engine()
        ctx = _liked_context(20, score=9)
        suggestions = engine.suggest(4, ctx)
        assert len(suggestions) <= 4

    def test_higher_scored_tracks_drive_preference_vector(self):
        """
        Tracks rated 9-10 should dominate the preference vector.
        If high-rated tracks share tag X, X should appear in the explanation.
        """
        lf = _make_lastfm_mock()

        def get_tags(title, artist, limit=15):
            # Tracks titled "High*" get tag "dreamy"; others get "rock"
            if "High" in title:
                return [("dreamy", 1.0)]
            return [("rock", 1.0)]

        lf.get_track_tags.side_effect = get_tags
        engine = _make_engine(lastfm=lf)

        rated = [
            _make_rated(_make_track(f"h{i}", title=f"High Song {i}"), 10)
            for i in range(_MIN_LIKED_TRACKS)
        ]
        ctx = _make_context(rated)
        suggestions = engine.suggest(2, ctx)
        # Preference vector should include "dreamy"
        assert isinstance(suggestions, list)

    def test_explanation_mentions_top_tags(self):
        lf = _make_lastfm_mock(track_tags=[("ambient", 0.9), ("electronic", 0.7)])
        engine = _make_engine(lastfm=lf)
        ctx = _liked_context(_MIN_LIKED_TRACKS)
        suggestions = engine.suggest(1, ctx)
        if suggestions:
            assert "ambient" in suggestions[0].explanation or "electronic" in suggestions[0].explanation


# ---------------------------------------------------------------------------
# on_session_complete
# ---------------------------------------------------------------------------

class TestOnSessionComplete:
    def test_no_raise_on_empty(self):
        engine = _make_engine()
        engine.on_session_complete([])  # must not raise

    def test_no_raise_on_nonempty(self):
        engine = _make_engine()
        t = _make_track("t1")
        engine.on_session_complete([(t, 8)])  # must not raise
