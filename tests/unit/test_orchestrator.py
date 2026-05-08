"""Tests for Orchestrator slot allocation and suggestion collection."""
from __future__ import annotations

import uuid

import pytest

from core.base_engine import SessionConfig, SessionContext
from core.diversity import DiversityEnforcer
from core.orchestrator import Orchestrator
from tests.conftest import make_mock_engine, make_suggestion, make_track, session_context


def make_orchestrator(
    engines,
    slot_weights: dict | None = None,
    oversampling_factor: int = 2,
    n_final: int = 10,
) -> Orchestrator:
    enforcer = DiversityEnforcer(min_engines_represented=1)
    weights = slot_weights or {e.name: 1.0 for e in engines}
    return Orchestrator(
        engines=engines,
        diversity_enforcer=enforcer,
        slot_weights=weights,
        oversampling_factor=oversampling_factor,
        n_final=n_final,
    )


class TestSlotAllocation:
    def test_equal_weights_all_engines_get_slots(self):
        engines = [make_mock_engine(f"e{i}") for i in range(3)]
        orc = make_orchestrator(engines, {e.name: 1.0 for e in engines})
        allocation = orc.get_allocation()
        assert set(allocation) == {"e0", "e1", "e2"}
        assert all(v >= 1 for v in allocation.values())

    def test_single_engine_gets_all_slots(self):
        engine = make_mock_engine("solo")
        orc = make_orchestrator([engine], {"solo": 1.0}, n_final=10)
        allocation = orc.get_allocation()
        assert "solo" in allocation
        assert allocation["solo"] >= 1

    def test_unequal_weights_proportional_allocation(self):
        e1 = make_mock_engine("heavy")
        e2 = make_mock_engine("light")
        orc = make_orchestrator([e1, e2], {"heavy": 0.8, "light": 0.2}, n_final=10)
        allocation = orc.get_allocation()
        assert allocation["heavy"] > allocation["light"]

    def test_no_engine_gets_zero_slots(self):
        engines = [make_mock_engine(f"e{i}") for i in range(5)]
        weights = {e.name: 1.0 for e in engines}
        orc = make_orchestrator(engines, weights, n_final=5)
        allocation = orc.get_allocation()
        assert all(v >= 1 for v in allocation.values())

    def test_empty_engine_list_returns_empty_allocation(self):
        orc = make_orchestrator([], {})
        allocation = orc.get_allocation()
        assert allocation == {}

    def test_unavailable_engine_not_in_allocation(self):
        # Unavailable engines are removed before reaching Orchestrator
        # (EngineRegistry handles this); this test verifies the orchestrator
        # only sees what it's given.
        e1 = make_mock_engine("ok_engine")
        orc = make_orchestrator([e1], {"ok_engine": 1.0})
        allocation = orc.get_allocation()
        assert "ok_engine" in allocation


class TestSuggestionCollection:
    def _ctx(self) -> SessionContext:
        return SessionContext(
            rated_tracks=[],
            recent_sessions=[],
            session_config=SessionConfig(),
            excluded_track_ids=set(),
        )

    def test_oversampling_factor_applied(self):
        suggestions = [
            make_suggestion(make_track(genre_primary="Electronic", track_id=str(uuid.uuid4())), "eng", 0.9 - i * 0.01)
            for i in range(20)
        ]
        engine = make_mock_engine("eng", suggestions)
        orc = make_orchestrator([engine], {"eng": 1.0}, oversampling_factor=3, n_final=5)
        orc.get_suggestions(self._ctx())
        # Engine should have been asked for at least n_final * oversampling_factor candidates
        call_args = engine.suggest.call_args
        asked_n = call_args[0][0]
        assert asked_n >= 3  # at least oversampling_factor × 1 slot

    def test_engine_exception_does_not_crash_session(self):
        bad_engine = make_mock_engine("bad")
        bad_engine.suggest.side_effect = RuntimeError("API down")
        good_suggestions = [
            make_suggestion(make_track(genre_primary=g, track_id=str(uuid.uuid4())), "good")
            for g in ["Electronic", "Rock", "Jazz", "Pop", "Metal",
                       "Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World"]
        ]
        good_engine = make_mock_engine("good", good_suggestions)
        orc = make_orchestrator([bad_engine, good_engine], {"bad": 0.5, "good": 0.5})
        result = orc.get_suggestions(self._ctx())
        assert len(result) > 0

    def test_no_engines_returns_empty(self):
        orc = make_orchestrator([], {})
        result = orc.get_suggestions(self._ctx())
        assert result == []

    def test_returns_at_most_n_final(self):
        suggestions = [
            make_suggestion(make_track(genre_primary=f"Genre{i}", track_id=str(uuid.uuid4())), "eng", 0.9)
            for i in range(50)
        ]
        engine = make_mock_engine("eng", suggestions)
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx())
        assert len(result) <= 5

    def test_multiple_engines_both_queried(self):
        genres_a = ["Electronic", "Rock", "Jazz"]
        genres_b = ["Pop", "Metal", "Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World"]
        ea = make_mock_engine("ea", [make_suggestion(make_track(genre_primary=g, track_id=str(uuid.uuid4())), "ea") for g in genres_a])
        eb = make_mock_engine("eb", [make_suggestion(make_track(genre_primary=g, track_id=str(uuid.uuid4())), "eb") for g in genres_b])
        orc = make_orchestrator([ea, eb], {"ea": 0.5, "eb": 0.5})
        orc.get_suggestions(self._ctx())
        assert ea.suggest.called
        assert eb.suggest.called


class TestPoolFilter:
    def _ctx(self, fingerprints: set[str] | None = None) -> SessionContext:
        return SessionContext(
            rated_tracks=[],
            recent_sessions=[],
            session_config=SessionConfig(),
            excluded_track_ids=set(),
            excluded_track_fingerprints=fingerprints or set(),
        )

    def test_karaoke_title_dropped(self):
        junk = make_suggestion(make_track(title="My Song [Karaoke Version]", genre_primary="Rock", track_id=str(uuid.uuid4())), "eng")
        clean = make_suggestion(make_track(title="My Song", genre_primary="Jazz", track_id=str(uuid.uuid4())), "eng")
        engine = make_mock_engine("eng", [junk, clean])
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx())
        titles = [s.track.title for s in result]
        assert "My Song [Karaoke Version]" not in titles
        assert "My Song" in titles

    def test_in_the_style_of_dropped(self):
        junk = make_suggestion(make_track(title="Song in the Style of Elvis", genre_primary="Rock", track_id=str(uuid.uuid4())), "eng")
        engine = make_mock_engine("eng", [junk])
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx())
        assert result == []

    def test_tribute_to_dropped(self):
        junk = make_suggestion(make_track(title="Tribute to Freddie King", genre_primary="Blues", track_id=str(uuid.uuid4())), "eng")
        engine = make_mock_engine("eng", [junk])
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx())
        assert result == []

    def test_fingerprint_dedup_drops_same_title_artist(self):
        # Simulate a track rated in a prior session under a different tidal_id
        import re
        norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()
        fp = f"{norm('Going Down')}|{norm('Freddie King')}"

        same_song_new_album = make_suggestion(
            make_track(title="Going Down", artist="Freddie King", genre_primary="Blues", track_id=str(uuid.uuid4())),
            "eng",
        )
        engine = make_mock_engine("eng", [same_song_new_album])
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx(fingerprints={fp}))
        assert result == []

    def test_fingerprint_dedup_normalizes_punctuation(self):
        norm = lambda s: __import__("re").sub(r"\s+", " ", __import__("re").sub(r"[^\w\s]", "", s.lower())).strip()
        fp = f"{norm('Gymnopédie No. 1')}|{norm('Erik Satie')}"
        # Same track, punctuation stripped differently in title
        track = make_suggestion(
            make_track(title="Gymnopédie No 1", artist="Erik Satie", genre_primary="Classical", track_id=str(uuid.uuid4())),
            "eng",
        )
        engine = make_mock_engine("eng", [track])
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx(fingerprints={fp}))
        assert result == []

    def test_clean_tracks_pass_through(self):
        tracks = [
            make_suggestion(make_track(title="Going Down", artist="Freddie King", genre_primary="Blues", track_id=str(uuid.uuid4())), "eng"),
            make_suggestion(make_track(title="Clair de Lune", artist="Debussy", genre_primary="Classical", track_id=str(uuid.uuid4())), "eng"),
        ]
        engine = make_mock_engine("eng", tracks)
        orc = make_orchestrator([engine], {"eng": 1.0}, n_final=5)
        result = orc.get_suggestions(self._ctx())
        assert len(result) == 2
