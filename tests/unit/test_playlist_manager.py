"""
Unit tests for PlaylistManager — Tidal connector is fully mocked.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from connectors.tidal import TidalPlaylist
from core.base_engine import Session, Track
from core.db import Database
from core.playlist_manager import PlaylistConfig, PlaylistInfo, PlaylistManager
from tests.conftest import make_track

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "database" / "migrations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> Database:
    db = Database(":memory:", MIGRATIONS_DIR)
    db.connect()
    return db


def _insert_session(db: Database, session_id: str = "sess-1") -> None:
    """Insert a minimal session so FK constraints on candidate_playlist_tracks pass."""
    db.insert_session(Session(
        id=session_id,
        started_at=datetime.now(),
        engine_allocation={},
        diversity_config={},
    ))


def _make_tidal_mock() -> MagicMock:
    """
    Tidal mock where each create_playlist returns a unique playlist ID.
    Tracks per-playlist membership so is_track_in_playlist is realistic.
    """
    tidal = MagicMock()
    playlist_contents: dict[str, set[str]] = {}

    def create_playlist(name, description=""):
        pl_id = f"tidal-pl-{uuid.uuid4().hex[:8]}"
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

    tidal.create_playlist.side_effect = create_playlist
    tidal.set_playlist_tracks.side_effect = set_playlist_tracks
    tidal.add_tracks_to_playlist.side_effect = add_tracks_to_playlist
    tidal.remove_track_from_playlist.side_effect = remove_track_from_playlist
    tidal.is_track_in_playlist.side_effect = is_track_in_playlist
    tidal._contents = playlist_contents
    return tidal


def _make_pm(db: Database, tidal: MagicMock | None = None, threshold: int = 7) -> PlaylistManager:
    if tidal is None:
        tidal = _make_tidal_mock()
    cfg = PlaylistConfig(
        curated_threshold=threshold,
        candidate_name="Now Rating",
        curated_master_name="Liked \u2014 All",
        curated_genre_prefix="Liked \u2014 ",
    )
    return PlaylistManager(tidal=tidal, db=db, config=cfg)


def _track_with_tidal(genre: str = "Electronic", tidal_id: str | None = None) -> Track:
    return make_track(genre_primary=genre, tidal_id=tidal_id or str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# sync_candidate_playlist
# ---------------------------------------------------------------------------

class TestSyncCandidatePlaylist:
    def test_creates_playlist_on_first_call(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        _insert_session(db, "sess-1")
        tracks = [_track_with_tidal() for _ in range(3)]
        pm.sync_candidate_playlist("sess-1", tracks)
        tidal.create_playlist.assert_called_once()

    def test_stores_playlist_id_in_db(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [_track_with_tidal()])
        row = db.get_tidal_playlist("candidate")
        assert row is not None
        assert "tidal-pl-" in row["tidal_playlist_id"]

    def test_reuses_existing_playlist_on_second_call(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        _insert_session(db, "sess-1")
        _insert_session(db, "sess-2")
        tracks = [_track_with_tidal()]
        pm.sync_candidate_playlist("sess-1", tracks)
        pm.sync_candidate_playlist("sess-2", tracks)
        assert tidal.create_playlist.call_count == 1

    def test_replaces_tracks_not_appends(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        _insert_session(db, "sess-1")
        _insert_session(db, "sess-2")
        tracks_a = [_track_with_tidal()]
        tracks_b = [_track_with_tidal(), _track_with_tidal()]
        pm.sync_candidate_playlist("sess-1", tracks_a)
        pm.sync_candidate_playlist("sess-2", tracks_b)
        assert tidal.set_playlist_tracks.call_count == 2
        _, second_call_ids = tidal.set_playlist_tracks.call_args_list[1][0]
        assert set(second_call_ids) == {t.tidal_id for t in tracks_b}

    def test_returns_tidal_url(self):
        db = _make_db()
        pm = _make_pm(db)
        _insert_session(db, "sess-1")
        url = pm.sync_candidate_playlist("sess-1", [_track_with_tidal()])
        assert "tidal.com" in url

    def test_records_candidate_tracks_in_db(self):
        db = _make_db()
        pm = _make_pm(db)
        track = _track_with_tidal()
        db.upsert_track(track)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [track])
        rows = db.get_unremoved_candidate_tracks("sess-1")
        assert any(r["track_id"] == track.id for r in rows)

    def test_skips_tidal_set_for_tracks_without_tidal_id(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        _insert_session(db, "sess-1")
        no_tidal_track = make_track(tidal_id=None)
        pm.sync_candidate_playlist("sess-1", [no_tidal_track])
        # set_playlist_tracks should be called with an empty list
        call_args = tidal.set_playlist_tracks.call_args
        assert call_args is None or call_args[0][1] == []


# ---------------------------------------------------------------------------
# remove_from_candidate
# ---------------------------------------------------------------------------

class TestRemoveFromCandidate:
    def _setup(self) -> tuple[Database, MagicMock, PlaylistManager, Track]:
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        track = _track_with_tidal()
        db.upsert_track(track)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [track])
        return db, tidal, pm, track

    def test_calls_tidal_remove(self):
        db, tidal, pm, track = self._setup()
        pl_id = db.get_tidal_playlist("candidate")["tidal_playlist_id"]
        pm.remove_from_candidate(track.id)
        tidal.remove_track_from_playlist.assert_called_once_with(pl_id, track.tidal_id)

    def test_marks_removed_in_db(self):
        db, tidal, pm, track = self._setup()
        pm.remove_from_candidate(track.id)
        rows = db.get_unremoved_candidate_tracks("sess-1")
        assert not any(r["track_id"] == track.id for r in rows)

    def test_does_not_raise_when_tidal_unavailable(self):
        db, tidal, pm, track = self._setup()
        tidal.remove_track_from_playlist.side_effect = Exception("Tidal down")
        pm.remove_from_candidate(track.id)  # Must not raise

    def test_does_not_raise_when_no_candidate_playlist(self):
        db = _make_db()
        pm = _make_pm(db)
        pm.remove_from_candidate("nonexistent-track")  # Must not raise


# ---------------------------------------------------------------------------
# add_to_curated
# ---------------------------------------------------------------------------

class TestAddToCurated:
    def _setup(self, threshold: int = 7) -> tuple[Database, MagicMock, PlaylistManager, Track]:
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal, threshold=threshold)
        track = _track_with_tidal(genre="Electronic")
        db.upsert_track(track)
        return db, tidal, pm, track

    def test_above_threshold_creates_master_and_genre_playlists(self):
        db, tidal, pm, track = self._setup()
        pm.add_to_curated(track, score=8)
        assert tidal.create_playlist.call_count == 2

    def test_above_threshold_adds_to_master_and_genre(self):
        db, tidal, pm, track = self._setup()
        pm.add_to_curated(track, score=8)
        assert tidal.add_tracks_to_playlist.call_count == 2

    def test_above_threshold_genre_playlist_stored_in_db(self):
        db, tidal, pm, track = self._setup()
        pm.add_to_curated(track, score=8)
        row = db.get_tidal_playlist("curated_genre", genre="Electronic")
        assert row is not None
        assert "Electronic" in row["name"]

    def test_below_threshold_does_nothing(self):
        db, tidal, pm, track = self._setup(threshold=7)
        pm.add_to_curated(track, score=6)
        tidal.create_playlist.assert_not_called()
        tidal.add_tracks_to_playlist.assert_not_called()

    def test_at_threshold_is_included(self):
        db, tidal, pm, track = self._setup(threshold=7)
        pm.add_to_curated(track, score=7)
        assert tidal.add_tracks_to_playlist.call_count == 2

    def test_idempotent_second_add_skipped(self):
        db, tidal, pm, track = self._setup()
        pm.add_to_curated(track, score=8)
        # Track is now in both playlists in the mock
        pm.add_to_curated(track, score=9)
        # add_tracks_to_playlist should not be called again (idempotent)
        assert tidal.add_tracks_to_playlist.call_count == 2  # only from first call

    def test_genre_playlist_created_once_per_genre(self):
        db, tidal, pm, _ = self._setup()
        track1 = _track_with_tidal(genre="Electronic")
        track2 = _track_with_tidal(genre="Electronic")
        db.upsert_track(track1)
        db.upsert_track(track2)
        pm.add_to_curated(track1, score=8)
        pm.add_to_curated(track2, score=9)
        genre_creates = [c for c in tidal.create_playlist.call_args_list if "Electronic" in str(c)]
        assert len(genre_creates) == 1

    def test_different_genres_create_separate_playlists(self):
        db, tidal, pm, _ = self._setup()
        rock_track = _track_with_tidal(genre="Rock")
        jazz_track = _track_with_tidal(genre="Jazz")
        db.upsert_track(rock_track)
        db.upsert_track(jazz_track)
        pm.add_to_curated(rock_track, score=8)
        pm.add_to_curated(jazz_track, score=8)
        # 1 master + Rock genre + Jazz genre = 3 creates
        assert tidal.create_playlist.call_count == 3

    def test_does_not_raise_when_tidal_unavailable(self):
        db, tidal, pm, track = self._setup()
        tidal.create_playlist.side_effect = Exception("Tidal down")
        pm.add_to_curated(track, score=8)  # Must not raise

    def test_track_without_tidal_id_skipped(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        no_tidal = make_track(tidal_id=None, genre_primary="Electronic")
        db.upsert_track(no_tidal)
        pm.add_to_curated(no_tidal, score=9)
        tidal.add_tracks_to_playlist.assert_not_called()


# ---------------------------------------------------------------------------
# clear_candidate_remaining
# ---------------------------------------------------------------------------

class TestClearCandidateRemaining:
    def test_removes_unrated_tracks_from_tidal(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        _insert_session(db, "sess-1")
        tracks = [_track_with_tidal() for _ in range(3)]
        for t in tracks:
            db.upsert_track(t)
        pm.sync_candidate_playlist("sess-1", tracks)
        count = pm.clear_candidate_remaining("sess-1")
        assert count == 3
        assert tidal.remove_track_from_playlist.call_count == 3

    def test_marks_all_removed_in_db(self):
        db = _make_db()
        pm = _make_pm(db)
        track = _track_with_tidal()
        db.upsert_track(track)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [track])
        pm.clear_candidate_remaining("sess-1")
        rows = db.get_unremoved_candidate_tracks("sess-1")
        assert rows == []

    def test_returns_zero_when_all_already_removed(self):
        db = _make_db()
        pm = _make_pm(db)
        track = _track_with_tidal()
        db.upsert_track(track)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [track])
        pm.remove_from_candidate(track.id)
        count = pm.clear_candidate_remaining("sess-1")
        assert count == 0

    def test_tidal_failure_does_not_raise(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        tidal.remove_track_from_playlist.side_effect = Exception("Tidal down")
        pm = _make_pm(db, tidal)
        track = _track_with_tidal()
        db.upsert_track(track)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [track])
        pm.clear_candidate_remaining("sess-1")  # Must not raise
        rows = db.get_unremoved_candidate_tracks("sess-1")
        assert rows == []


# ---------------------------------------------------------------------------
# get_all_playlists
# ---------------------------------------------------------------------------

class TestGetAllPlaylists:
    def test_returns_empty_before_any_sync(self):
        db = _make_db()
        pm = _make_pm(db)
        assert pm.get_all_playlists() == []

    def test_returns_candidate_after_sync(self):
        db = _make_db()
        pm = _make_pm(db)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [_track_with_tidal()])
        types = [p.playlist_type for p in pm.get_all_playlists()]
        assert "candidate" in types

    def test_returns_curated_playlists_after_add(self):
        db = _make_db()
        tidal = _make_tidal_mock()
        pm = _make_pm(db, tidal)
        track = _track_with_tidal(genre="Electronic")
        db.upsert_track(track)
        pm.add_to_curated(track, score=9)
        types = {p.playlist_type for p in pm.get_all_playlists()}
        assert "curated_master" in types
        assert "curated_genre" in types

    def test_playlist_info_has_tidal_url(self):
        db = _make_db()
        pm = _make_pm(db)
        _insert_session(db, "sess-1")
        pm.sync_candidate_playlist("sess-1", [_track_with_tidal()])
        for p in pm.get_all_playlists():
            assert p.tidal_url != ""
