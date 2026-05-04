"""
Integration tests: full session cycle → Tidal playlist state assertions.

Tidal connector is mocked; DB is real in-memory SQLite.
Validates the interplay between SessionManager, PlaylistManager, and Database.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from connectors.tidal import TidalPlaylist
from core.base_engine import SessionConfig, Suggestion
from core.db import Database
from core.diversity import DiversityEnforcer
from core.orchestrator import Orchestrator
from core.playlist_manager import PlaylistConfig, PlaylistManager
from core.session import SessionManager
from tests.conftest import make_mock_engine, make_suggestion, make_track

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "database" / "migrations"

GENRES = [
    "Electronic", "Rock", "Jazz", "Pop", "Metal",
    "Hip-Hop/R&B", "Classical", "Folk/Country", "Blues", "Reggae/World",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> Database:
    db = Database(":memory:", MIGRATIONS_DIR)
    db.connect()
    return db


_pl_counter = 0


def _make_tidal_mock() -> MagicMock:
    """
    Tidal mock that tracks which tracks are in each playlist (by tidal_id).
    Each create_playlist call returns a unique playlist ID.
    """
    tidal = MagicMock()
    # Map of playlist_id → set of tidal_track_ids
    playlist_contents: dict[str, set[str]] = {}

    def create_playlist(name, description=""):
        pl_id = f"pl-{uuid.uuid4().hex[:8]}"
        playlist_contents[pl_id] = set()
        return TidalPlaylist(
            playlist_id=pl_id,
            name=name,
            url=f"https://tidal.com/browse/playlist/{pl_id}",
            track_count=0,
        )

    def set_playlist_tracks(pl_id, track_ids):
        playlist_contents[pl_id] = set(track_ids)

    def add_tracks_to_playlist(pl_id, track_ids):
        playlist_contents.setdefault(pl_id, set()).update(track_ids)

    def remove_track_from_playlist(pl_id, tidal_id):
        playlist_contents.get(pl_id, set()).discard(tidal_id)

    def is_track_in_playlist(pl_id, tidal_id):
        return tidal_id in playlist_contents.get(pl_id, set())

    def get_playlist_track_ids(pl_id):
        return list(playlist_contents.get(pl_id, set()))

    tidal.create_playlist.side_effect = create_playlist
    tidal.set_playlist_tracks.side_effect = set_playlist_tracks
    tidal.add_tracks_to_playlist.side_effect = add_tracks_to_playlist
    tidal.remove_track_from_playlist.side_effect = remove_track_from_playlist
    tidal.is_track_in_playlist.side_effect = is_track_in_playlist
    tidal.get_playlist_track_ids.side_effect = get_playlist_track_ids

    # Expose playlist_contents for assertions
    tidal._contents = playlist_contents
    return tidal


def _make_suggestions(genres: list[str] | None = None) -> list[Suggestion]:
    return [
        make_suggestion(
            make_track(genre_primary=g, track_id=str(uuid.uuid4()), tidal_id=str(uuid.uuid4())),
            engine_name="mock_engine",
            engine_score=0.9 - i * 0.05,
        )
        for i, g in enumerate(genres or GENRES)
    ]


def _make_session_manager(
    db: Database,
    tidal: MagicMock,
    suggestions: list[Suggestion] | None = None,
    threshold: int = 7,
) -> tuple[SessionManager, list[Suggestion]]:
    suggs = suggestions or _make_suggestions()
    engine = make_mock_engine("mock_engine", suggs)
    enforcer = DiversityEnforcer(min_engines_represented=1)
    orchestrator = Orchestrator(
        engines=[engine],
        diversity_enforcer=enforcer,
        slot_weights={"mock_engine": 1.0},
        oversampling_factor=1,
        n_final=10,
    )
    pm_cfg = PlaylistConfig(
        curated_threshold=threshold,
        candidate_name="Now Rating",
        curated_master_name="Liked \u2014 All",
        curated_genre_prefix="Liked \u2014 ",
    )
    pm = PlaylistManager(tidal=tidal, db=db, config=pm_cfg)
    sm = SessionManager(
        db=db,
        orchestrator=orchestrator,
        engines=[engine],
        curated_threshold=threshold,
        playlist_manager=pm,
    )
    return sm, suggs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCandidatePlaylistSync:
    def test_now_rating_created_on_start(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal)
        sm.start_session()
        assert tidal.create_playlist.call_count == 1
        call_name = tidal.create_playlist.call_args[0][0]
        assert call_name == "Now Rating"

    def test_now_rating_has_10_tracks_after_start(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, suggs = _make_session_manager(db, tidal)
        sm.start_session()
        pl_row = db.get_tidal_playlist("candidate")
        pl_id = pl_row["tidal_playlist_id"]
        assert len(tidal._contents[pl_id]) == 10

    def test_track_removed_from_now_rating_after_rating(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]
        pl_row = db.get_tidal_playlist("candidate")
        pl_id = pl_row["tidal_playlist_id"]

        sm.rate_track(session_id, first["track_id"], 5)
        assert first["tidal_id"] not in tidal._contents[pl_id]

    def test_now_rating_cleared_on_complete_session(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal)
        result = sm.start_session()
        session_id = result["session_id"]

        # Rate only half the tracks
        for s in result["suggestions"][:5]:
            sm.rate_track(session_id, s["track_id"], 6)

        sm.complete_session(session_id)

        pl_row = db.get_tidal_playlist("candidate")
        pl_id = pl_row["tidal_playlist_id"]
        # All remaining tracks should be removed from Tidal
        assert len(tidal._contents[pl_id]) == 0

    def test_now_rating_replaced_for_new_session(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal)
        result1 = sm.start_session()
        session_id1 = result1["session_id"]
        for s in result1["suggestions"]:
            sm.rate_track(session_id1, s["track_id"], 5)
        sm.complete_session(session_id1)

        # Start a new session with different tracks
        new_suggs = _make_suggestions()
        engine2 = make_mock_engine("mock_engine", new_suggs)
        enforcer2 = DiversityEnforcer(min_engines_represented=1)
        orchestrator2 = Orchestrator(
            engines=[engine2],
            diversity_enforcer=enforcer2,
            slot_weights={"mock_engine": 1.0},
            oversampling_factor=1,
            n_final=10,
        )
        from core.playlist_manager import PlaylistManager, PlaylistConfig
        pm2 = PlaylistManager(
            tidal=tidal,
            db=db,
            config=PlaylistConfig(curated_threshold=7),
        )
        sm2 = SessionManager(
            db=db,
            orchestrator=orchestrator2,
            engines=[engine2],
            curated_threshold=7,
            playlist_manager=pm2,
        )
        result2 = sm2.start_session()
        # create_playlist should still only be called once (reuses existing)
        assert tidal.create_playlist.call_count == 1
        # set_playlist_tracks called twice (once per session)
        assert tidal.set_playlist_tracks.call_count == 2


class TestCuratedPlaylistSync:
    def test_track_added_to_liked_all_above_threshold(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal, threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]

        sm.rate_track(session_id, first["track_id"], 8)

        master_row = db.get_tidal_playlist("curated_master")
        assert master_row is not None
        master_id = master_row["tidal_playlist_id"]
        assert first["tidal_id"] in tidal._contents[master_id]

    def test_track_added_to_genre_playlist_above_threshold(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, suggs = _make_session_manager(db, tidal, threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]
        first_genre = first["genre"]

        sm.rate_track(session_id, first["track_id"], 8)

        genre_row = db.get_tidal_playlist("curated_genre", genre=first_genre)
        assert genre_row is not None
        genre_pl_id = genre_row["tidal_playlist_id"]
        assert first["tidal_id"] in tidal._contents[genre_pl_id]

    def test_track_not_added_below_threshold(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal, threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]

        sm.rate_track(session_id, first["track_id"], 6)

        master_row = db.get_tidal_playlist("curated_master")
        assert master_row is None  # never created

    def test_same_genre_rated_twice_uses_one_playlist(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        # Two Electronic tracks
        suggs = [
            make_suggestion(
                make_track(genre_primary="Electronic", track_id=str(uuid.uuid4()), tidal_id=str(uuid.uuid4())),
                "mock_engine", 0.9 - i * 0.1,
            )
            for i in range(10)
        ]
        sm, _ = _make_session_manager(db, tidal, suggestions=suggs, threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]

        # Rate two Electronic tracks above threshold
        for s in result["suggestions"][:2]:
            sm.rate_track(session_id, s["track_id"], 8)

        # Genre playlist for Electronic should be created exactly once
        genre_creates = [
            c for c in tidal.create_playlist.call_args_list
            if "Electronic" in str(c)
        ]
        assert len(genre_creates) == 1

    def test_different_genres_create_separate_playlists(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal, threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]

        # Rate all tracks above threshold (each has a different genre)
        for s in result["suggestions"]:
            sm.rate_track(session_id, s["track_id"], 8)

        genre_playlists = db.get_all_tidal_playlists()
        genre_types = [r for r in genre_playlists if r["playlist_type"] == "curated_genre"]
        # Should have one per distinct genre
        genres_rated = {s["genre"] for s in result["suggestions"]}
        assert len(genre_types) == len(genres_rated)


class TestTidalUnavailableResiience:
    def test_rating_saved_to_db_when_tidal_down(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        tidal.remove_track_from_playlist.side_effect = Exception("Tidal down")
        tidal.add_tracks_to_playlist.side_effect = Exception("Tidal down")
        sm, _ = _make_session_manager(db, tidal)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]

        r = sm.rate_track(session_id, first["track_id"], 8)  # Must not raise
        assert r.ok is True

        # DB rating should be there
        rating = db.get_rating(session_id, first["track_id"])
        assert rating is not None
        assert rating["score"] == 8

    def test_reconcile_fixes_candidate_gaps(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]

        # Rating saved but Tidal remove failed
        tidal.remove_track_from_playlist.side_effect = Exception("Tidal down")
        sm.rate_track(session_id, first["track_id"], 5)

        # Restore Tidal
        tidal.remove_track_from_playlist.side_effect = None
        pl_row = db.get_tidal_playlist("candidate")
        pl_id = pl_row["tidal_playlist_id"]
        # Manually add track back to mock playlist to simulate stale state
        tidal._contents[pl_id].add(first["tidal_id"])
        # Make is_track_in_playlist return True for it
        tidal.is_track_in_playlist.side_effect = lambda pid, tid: tid in tidal._contents.get(pid, set())

        from core.playlist_manager import PlaylistManager, PlaylistConfig
        pm = PlaylistManager(tidal=tidal, db=db, config=PlaylistConfig())
        stats = pm.reconcile_session(session_id)
        assert stats["candidate_fixes"] >= 1

    def test_reconcile_fixes_curated_gaps(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        sm, _ = _make_session_manager(db, tidal, threshold=7)
        result = sm.start_session()
        session_id = result["session_id"]
        first = result["suggestions"][0]

        # Rating saved but curated add failed
        tidal.add_tracks_to_playlist.side_effect = Exception("Tidal down")
        sm.rate_track(session_id, first["track_id"], 9)

        # Restore Tidal
        tidal.add_tracks_to_playlist.side_effect = None
        tidal.is_track_in_playlist.return_value = False  # track not yet in curated

        from core.playlist_manager import PlaylistManager, PlaylistConfig
        pm = PlaylistManager(tidal=tidal, db=db, config=PlaylistConfig(curated_threshold=7))
        stats = pm.reconcile_session(session_id)
        assert stats["curated_fixes"] >= 1
