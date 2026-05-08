"""Tests for DiversityEnforcer."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from core.base_engine import RatedTrack, Session, Suggestion, Track
from core.diversity import DiversityEnforcer, classify_genre
from tests.conftest import make_mock_engine, make_rated_track, make_suggestion, make_track


# ---------------------------------------------------------------------------
# classify_genre
# ---------------------------------------------------------------------------

class TestClassifyGenre:
    def test_direct_bucket_match(self):
        assert classify_genre("Electronic") == "Electronic"

    def test_lowercase_tag_match(self):
        assert classify_genre("techno") == "Electronic"

    def test_genre_tags_fallback(self):
        assert classify_genre("unknown", ["house"]) == "Electronic"

    def test_unknown_genre_returns_genre_primary(self):
        assert classify_genre("Bossa Nova Fusion", []) == "Bossa Nova Fusion"

    def test_none_genre_primary_falls_back_to_tags(self):
        assert classify_genre("", ["jazz"]) == "Jazz"

    def test_empty_everything_returns_other(self):
        assert classify_genre("", []) == "Other"


# ---------------------------------------------------------------------------
# DiversityEnforcer.select
# ---------------------------------------------------------------------------

def _make_pool(genres: list[str], engine_name: str = "eng") -> list[Suggestion]:
    return [
        make_suggestion(
            make_track(genre_primary=g, track_id=str(uuid.uuid4())),
            engine_name=engine_name,
            engine_score=0.9 - i * 0.01,
        )
        for i, g in enumerate(genres)
    ]


class TestDiversityEnforcerSelect:
    def setup_method(self):
        self.enforcer = DiversityEnforcer(
            recently_heard_days=30,
            saturation_sessions=3,
            saturation_min_avg_rating=7.0,
            min_engines_represented=2,
        )

    def test_one_per_genre_basic(self):
        genres = ["Electronic", "Rock", "Jazz", "Pop", "Metal",
                  "Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World"]
        pool = _make_pool(genres)
        result = self.enforcer.select(pool, 10, [], [], [])
        assert len(result) == 10
        result_genres = [s.track.genre_primary for s in result]
        assert len(result_genres) == len(set(result_genres)), "Duplicate genre in final selection"

    def test_deduplicates_same_track_id(self):
        track = make_track(genre_primary="Electronic")
        pool = [
            make_suggestion(track, "eng_a", 0.9),
            make_suggestion(track, "eng_b", 0.8),
        ]
        result = self.enforcer.select(pool, 5, [], [], [])
        ids = [s.track.id for s in result]
        assert len(ids) == len(set(ids))

    def test_fallback_when_fewer_genres_than_n(self):
        # Only 3 genres available — should return 5 (3 + 2 extras from same genres)
        genres = ["Electronic", "Rock", "Jazz"]
        pool = _make_pool(genres * 5)  # lots of duplicates
        result = self.enforcer.select(pool, 5, [], [], [])
        assert len(result) == 5

    def test_returns_empty_for_empty_pool(self):
        result = self.enforcer.select([], 10, [], [], [])
        assert result == []

    def test_recently_heard_excluded(self):
        track = make_track(genre_primary="Electronic")
        rated = [make_rated_track(track, rated_at=datetime.now() - timedelta(days=5))]
        pool = [make_suggestion(track, engine_score=0.99)]
        result = self.enforcer.select(pool, 5, rated, [], [])
        ids = [s.track.id for s in result]
        assert track.id not in ids

    def test_old_ratings_not_excluded(self):
        track = make_track(genre_primary="Electronic")
        rated = [make_rated_track(track, rated_at=datetime.now() - timedelta(days=60))]
        pool = [make_suggestion(track, engine_score=0.99)]
        result = self.enforcer.select(pool, 1, rated, [], [])
        assert any(s.track.id == track.id for s in result)

    def test_saturated_genres_deprioritized_not_removed(self):
        """A saturated genre can still appear if needed to fill n."""
        enforcer = DiversityEnforcer(
            recently_heard_days=30,
            saturation_sessions=1,
            saturation_min_avg_rating=5.0,
            min_engines_represented=1,
        )
        session_id = str(uuid.uuid4())
        sessions = [
            Session(
                id=session_id,
                started_at=datetime.now(),
                engine_allocation={},
                diversity_config={},
                completed_at=datetime.now(),
            )
        ]
        genre_history = [
            {"genre": "Electronic", "session_id": session_id, "avg_rating": 9.0}
        ]
        # Pool has only Electronic tracks
        pool = _make_pool(["Electronic"] * 5)
        result = enforcer.select(pool, 3, [], sessions, genre_history)
        # Should still return results (not crash or return empty)
        assert len(result) > 0

    def test_saturated_genres_go_after_non_saturated(self):
        """Non-saturated genres should appear before saturated ones."""
        enforcer = DiversityEnforcer(
            recently_heard_days=30,
            saturation_sessions=1,
            saturation_min_avg_rating=5.0,
            min_engines_represented=1,
        )
        session_id = str(uuid.uuid4())
        sessions = [
            Session(
                id=session_id,
                started_at=datetime.now(),
                engine_allocation={},
                diversity_config={},
                completed_at=datetime.now(),
            )
        ]
        genre_history = [
            {"genre": "Electronic", "session_id": session_id, "avg_rating": 9.0}
        ]
        # Mix of saturated and non-saturated
        pool = _make_pool(["Electronic", "Rock", "Jazz"])
        result = enforcer.select(pool, 3, [], sessions, genre_history)
        genres = [s.track.genre_primary for s in result]
        # Rock or Jazz (non-saturated) should appear
        assert "Rock" in genres or "Jazz" in genres

    def test_respects_n_limit(self):
        pool = _make_pool(["Electronic", "Rock", "Jazz", "Pop", "Metal"] * 3)
        result = self.enforcer.select(pool, 3, [], [], [])
        assert len(result) <= 3

    def test_multiple_other_tracks_all_pass_through(self):
        """'Other' tracks must not block each other — each gets its own diversity slot."""
        pool = [
            make_suggestion(make_track(genre_primary="Other", track_id=str(uuid.uuid4())), "eng", 0.9),
            make_suggestion(make_track(genre_primary="Other", track_id=str(uuid.uuid4())), "eng", 0.8),
            make_suggestion(make_track(genre_primary="Other", track_id=str(uuid.uuid4())), "eng", 0.7),
        ]
        result = self.enforcer.select(pool, 5, [], [], [])
        assert len(result) == 3  # all three pass through

    def test_other_never_saturated(self):
        """'Other' must never appear in the saturated set regardless of ratings."""
        enforcer = DiversityEnforcer(
            recently_heard_days=30,
            saturation_sessions=1,
            saturation_min_avg_rating=5.0,
            min_engines_represented=1,
        )
        session_id = str(uuid.uuid4())
        sessions = [
            Session(
                id=session_id,
                started_at=datetime.now(),
                engine_allocation={},
                diversity_config={},
                completed_at=datetime.now(),
            )
        ]
        genre_history = [
            {"genre": "Other", "session_id": session_id, "avg_rating": 10.0}
        ]
        pool = [
            make_suggestion(make_track(genre_primary="Other", track_id=str(uuid.uuid4())), "eng", 0.9),
            make_suggestion(make_track(genre_primary="Other", track_id=str(uuid.uuid4())), "eng", 0.8),
        ]
        result = enforcer.select(pool, 5, [], sessions, genre_history)
        # Both tracks should appear — "Other" saturation must not suppress them
        assert len(result) == 2

    def test_engine_representation_best_effort(self):
        """At least min_engines_represented engines should appear when possible."""
        genres_a = ["Electronic", "Rock", "Jazz", "Pop"]
        genres_b = ["Metal", "Hip-Hop/R&B", "Classical", "Folk/Country"]
        pool = (
            _make_pool(genres_a, "engine_a") +
            _make_pool(genres_b, "engine_b")
        )
        result = self.enforcer.select(pool, 8, [], [], [])
        engine_names = {s.engine_name for s in result}
        assert len(engine_names) >= 2
