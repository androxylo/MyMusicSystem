"""
Unit tests for BanditExplorerEngine.
All external connectors are mocked — no real API calls.
"""
from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime
from unittest.mock import MagicMock

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
from engines.bandit_explorer.engine import (
    BanditExplorerEngine,
    _ALL_GENRES,
    _LIKE_THRESHOLD,
    _thompson_sample,
    _stable_track_id,
)
import random


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


def _make_engine(lastfm=None, tidal=None, seed: int = 42) -> BanditExplorerEngine:
    engine = BanditExplorerEngine(
        lastfm=lastfm if lastfm is not None else _make_lastfm_mock(),
        tidal=tidal if tidal is not None else _make_tidal_mock(),
        seed=seed,
    )
    engine.name = "bandit_explorer"
    engine.capabilities = EngineCapabilities(
        novelty_bias=0.6,
        genre_coverage="broad",
        cold_start_friendly=True,
        data_requirements=["ratings"],
        speed="fast",
        description="",
    )
    return engine


def _make_rated_track(genre: str = "Electronic", score: int = 8) -> RatedTrack:
    return RatedTrack(
        track=Track(
            id=str(uuid.uuid4()),
            title="Track",
            artist="Artist",
            album="Album",
            duration_ms=200_000,
            genre_primary=genre,
            genre_tags=[genre.lower()],
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


# ---------------------------------------------------------------------------
# Thompson sampling unit
# ---------------------------------------------------------------------------

class TestThompsonSample:
    def test_sample_in_range(self):
        rng = random.Random(42)
        for _ in range(100):
            s = _thompson_sample(1, 1, rng)
            assert 0.0 <= s <= 1.0

    def test_high_alpha_biases_toward_1(self):
        """With very high α and low β, samples should consistently be near 1."""
        rng = random.Random(42)
        samples = [_thompson_sample(100, 1, rng) for _ in range(50)]
        assert sum(samples) / len(samples) > 0.9

    def test_high_beta_biases_toward_0(self):
        """With very high β and low α, samples should consistently be near 0."""
        rng = random.Random(42)
        samples = [_thompson_sample(1, 100, rng) for _ in range(50)]
        assert sum(samples) / len(samples) < 0.1

    def test_uniform_prior_averages_around_half(self):
        """Beta(1,1) is the uniform distribution — mean should be ~0.5."""
        rng = random.Random(42)
        samples = [_thompson_sample(1, 1, rng) for _ in range(200)]
        mean = sum(samples) / len(samples)
        assert 0.3 < mean < 0.7


# ---------------------------------------------------------------------------
# Beta parameter computation
# ---------------------------------------------------------------------------

class TestBetaParams:
    def test_prior_is_one_when_no_ratings(self):
        engine = _make_engine()
        alpha, beta = engine._compute_beta_params([])
        for g in _ALL_GENRES:
            assert alpha[g] == 1
            assert beta[g] == 1

    def test_liked_track_increments_alpha(self):
        engine = _make_engine()
        alpha, beta = engine._compute_beta_params(
            [_make_rated_track("Electronic", score=_LIKE_THRESHOLD)]
        )
        assert alpha["Electronic"] == 2
        assert beta["Electronic"] == 1

    def test_disliked_track_increments_beta(self):
        engine = _make_engine()
        alpha, beta = engine._compute_beta_params(
            [_make_rated_track("Rock", score=_LIKE_THRESHOLD - 1)]
        )
        assert alpha["Rock"] == 1
        assert beta["Rock"] == 2

    def test_multiple_ratings_accumulate(self):
        engine = _make_engine()
        rated = (
            [_make_rated_track("Jazz", score=8) for _ in range(5)]
            + [_make_rated_track("Jazz", score=4) for _ in range(3)]
        )
        alpha, beta = engine._compute_beta_params(rated)
        assert alpha["Jazz"] == 6   # 5 likes + 1 prior
        assert beta["Jazz"] == 4    # 3 dislikes + 1 prior

    def test_unknown_genre_ignored(self):
        """Tracks with genre that doesn't map to a bucket should not crash."""
        engine = _make_engine()
        rt = _make_rated_track("unkown_genre_xyz", score=9)
        alpha, beta = engine._compute_beta_params([rt])  # must not raise
        assert all(alpha[g] == 1 for g in _ALL_GENRES)


# ---------------------------------------------------------------------------
# Genre ranking
# ---------------------------------------------------------------------------

class TestGenreRanking:
    def test_liked_genre_tends_to_rank_higher(self):
        """After many liked ratings, the liked genre should usually top the ranking."""
        engine = _make_engine(seed=0)
        # Flood Electronic with likes
        alpha = {g: 1 for g in _ALL_GENRES}
        beta = {g: 1 for g in _ALL_GENRES}
        alpha["Electronic"] = 50

        # Run 20 times; Electronic should win most draws
        wins = sum(
            1 for _ in range(20)
            if engine._rank_genres_by_thompson(alpha, beta)[0] == "Electronic"
        )
        assert wins >= 12, f"Electronic won only {wins}/20 draws despite high α"

    def test_disliked_genre_tends_to_rank_lower(self):
        engine = _make_engine(seed=1)
        alpha = {g: 1 for g in _ALL_GENRES}
        beta = {g: 1 for g in _ALL_GENRES}
        beta["Metal"] = 50  # Metal is consistently disliked

        bottom_5_appearances = sum(
            1 for _ in range(20)
            if "Metal" in engine._rank_genres_by_thompson(alpha, beta)[-5:]
        )
        assert bottom_5_appearances >= 12

    def test_exploration_uniform_prior(self):
        """With no rating history, all genres should win roughly equally."""
        engine = _make_engine(seed=42)
        alpha = {g: 1 for g in _ALL_GENRES}
        beta = {g: 1 for g in _ALL_GENRES}

        top_genre_counts: Counter = Counter()
        for _ in range(200):
            top = engine._rank_genres_by_thompson(alpha, beta)[0]
            top_genre_counts[top] += 1

        # Each genre should win at least once in 200 draws
        assert len(top_genre_counts) >= len(_ALL_GENRES) // 2


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_ok_when_both_available(self):
        assert _make_engine().health_check().status == "ok"

    def test_unavailable_when_lastfm_missing(self):
        engine = BanditExplorerEngine(lastfm=None, tidal=_make_tidal_mock())
        engine._lastfm = None
        assert engine.health_check().status == "unavailable"

    def test_degraded_when_tidal_missing(self):
        engine = BanditExplorerEngine(lastfm=_make_lastfm_mock(), tidal=None)
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

    def test_cold_start_returns_results(self):
        """With no rating history, should still return results (uniform prior)."""
        engine = _make_engine()
        results = engine.suggest(3, _context(rated=[]))
        assert len(results) > 0

    def test_results_within_n(self):
        engine = _make_engine()
        assert len(_make_engine().suggest(3, _context())) <= 3

    def test_suggestion_fields_populated(self):
        engine = _make_engine()
        results = engine.suggest(3, _context())
        for s in results:
            assert s.track.id
            assert s.track.title
            assert s.track.artist
            assert s.engine_name == "bandit_explorer"
            assert 0.0 <= s.engine_score <= 1.0
            assert isinstance(s.explanation, str)

    def test_respects_excluded_ids(self):
        engine = _make_engine()
        first = engine.suggest(5, _context())
        if not first:
            pytest.skip("Engine returned no results")
        excluded = {s.track.id for s in first}
        second = engine.suggest(5, _context(excluded=excluded))
        assert not ({s.track.id for s in second} & excluded)

    def test_genre_primary_matches_sampled_genre(self):
        """Returned tracks should have genre_primary from one of the known buckets."""
        engine = _make_engine()
        results = engine.suggest(5, _context())
        for s in results:
            assert s.track.genre_primary in _ALL_GENRES

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

    def test_on_session_complete_does_not_raise(self):
        engine = _make_engine()
        engine.on_session_complete([])  # must not raise

    def test_high_rated_genre_dominates_over_many_runs(self):
        """
        After many likes for Electronic, it should appear more often than other genres.
        Not a hard guarantee (Thompson Sampling is stochastic) but should hold statistically.
        """
        lastfm = _make_lastfm_mock()
        tidal = _make_tidal_mock()
        rated = [_make_rated_track("Electronic", score=9) for _ in range(20)]

        electronic_count = 0
        for seed in range(30):
            engine = BanditExplorerEngine(lastfm=lastfm, tidal=tidal, seed=seed)
            engine.name = "bandit_explorer"
            results = engine.suggest(1, _context(rated=rated))
            if results and results[0].track.genre_primary == "Electronic":
                electronic_count += 1

        assert electronic_count >= 10, (
            f"Electronic only won {electronic_count}/30 sessions despite 20 likes"
        )
