"""
Database access layer — raw sqlite3, no ORM.

The Database class is the only place that touches the SQLite file.
All other modules receive a Database instance via dependency injection.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from core.base_engine import RatedTrack, Session, Track

logger = logging.getLogger(__name__)

_DATETIME_FMT = "%Y-%m-%d %H:%M:%S.%f"


def _parse_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in (_DATETIME_FMT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {value!r}")


def _fmt_dt(dt: datetime | None) -> str | None:
    return dt.strftime(_DATETIME_FMT) if dt is not None else None


class Database:
    """
    Wraps a SQLite connection. Call connect() before use, close() when done.
    Supports ':memory:' for in-process testing.
    """

    def __init__(self, path: str | Path, migrations_dir: Path) -> None:
        self._path = str(path)
        self._migrations_dir = Path(migrations_dir)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._run_migrations()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() has not been called")
        return self._conn

    # ------------------------------------------------------------------
    # Migration runner
    # ------------------------------------------------------------------

    def _run_migrations(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        applied = {row[0] for row in self.conn.execute("SELECT version FROM _migrations")}

        if not self._migrations_dir.exists():
            return

        for mf in sorted(self._migrations_dir.glob("*.sql")):
            try:
                version = int(mf.stem.split("_")[0])
            except (ValueError, IndexError):
                logger.warning(f"Skipping migration file with unexpected name: {mf.name}")
                continue

            if version in applied:
                continue

            logger.info(f"Applying migration {mf.name}")
            self.conn.executescript(mf.read_text())
            self.conn.execute("INSERT INTO _migrations (version) VALUES (?)", (version,))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Track CRUD
    # ------------------------------------------------------------------

    def upsert_track(self, track: Track) -> None:
        self.conn.execute(
            """
            INSERT INTO tracks
                (id, tidal_id, spotify_id, mbid, title, artist, album,
                 duration_ms, genre_primary, genre_tags, mood_tags,
                 bpm, key, mode, audio_features, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tidal_id = excluded.tidal_id,
                spotify_id = excluded.spotify_id,
                mbid = excluded.mbid,
                title = excluded.title,
                artist = excluded.artist,
                album = excluded.album,
                duration_ms = excluded.duration_ms,
                genre_primary = excluded.genre_primary,
                genre_tags = excluded.genre_tags,
                mood_tags = excluded.mood_tags,
                bpm = excluded.bpm,
                key = excluded.key,
                mode = excluded.mode,
                audio_features = excluded.audio_features
            """,
            (
                track.id,
                track.tidal_id,
                track.spotify_id,
                track.mbid,
                track.title,
                track.artist,
                track.album,
                track.duration_ms,
                track.genre_primary,
                json.dumps(track.genre_tags),
                json.dumps(track.mood_tags),
                track.bpm,
                track.key,
                track.mode,
                json.dumps(track.audio_features) if track.audio_features else None,
                None,  # source set separately if needed
            ),
        )
        self.conn.commit()

    def get_track(self, track_id: str) -> Track | None:
        row = self.conn.execute(
            "SELECT * FROM tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return self._row_to_track(row) if row else None

    def get_tracks_by_ids(self, track_ids: list[str]) -> list[Track]:
        if not track_ids:
            return []
        placeholders = ",".join("?" * len(track_ids))
        rows = self.conn.execute(
            f"SELECT * FROM tracks WHERE id IN ({placeholders})", track_ids
        ).fetchall()
        return [self._row_to_track(r) for r in rows]

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def insert_session(self, session: Session) -> None:
        self.conn.execute(
            """
            INSERT INTO sessions (id, started_at, completed_at, engine_allocation,
                                  diversity_config, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                _fmt_dt(session.started_at),
                _fmt_dt(session.completed_at),
                json.dumps(session.engine_allocation),
                json.dumps(session.diversity_config),
                session.notes,
            ),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def complete_session(self, session_id: str, completed_at: datetime) -> None:
        self.conn.execute(
            "UPDATE sessions SET completed_at = ? WHERE id = ?",
            (_fmt_dt(completed_at), session_id),
        )
        self.conn.commit()

    def get_recent_sessions(self, n: int = 10) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE completed_at IS NOT NULL "
            "ORDER BY started_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    # ------------------------------------------------------------------
    # Rating CRUD
    # ------------------------------------------------------------------

    def insert_rating(self, track_id: str, session_id: str, score: int) -> None:
        self.conn.execute(
            "INSERT INTO ratings (track_id, session_id, score) VALUES (?, ?, ?)",
            (track_id, session_id, score),
        )
        self.conn.commit()

    def get_rating(self, session_id: str, track_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM ratings WHERE session_id = ? AND track_id = ?",
            (session_id, track_id),
        ).fetchone()
        return dict(row) if row else None

    def get_session_ratings(self, session_id: str) -> list[RatedTrack]:
        rows = self.conn.execute(
            """
            SELECT r.score, r.rated_at, r.session_id,
                   t.id, t.tidal_id, t.spotify_id, t.mbid,
                   t.title, t.artist, t.album, t.duration_ms,
                   t.genre_primary, t.genre_tags, t.mood_tags,
                   t.bpm, t.key, t.mode, t.audio_features
            FROM ratings r
            JOIN tracks t ON r.track_id = t.id
            WHERE r.session_id = ?
            ORDER BY r.rated_at
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_rated_track(r) for r in rows]

    def get_all_rated_tracks(self) -> list[RatedTrack]:
        """Full rating history — used to build SessionContext."""
        rows = self.conn.execute(
            """
            SELECT r.score, r.rated_at, r.session_id,
                   t.id, t.tidal_id, t.spotify_id, t.mbid,
                   t.title, t.artist, t.album, t.duration_ms,
                   t.genre_primary, t.genre_tags, t.mood_tags,
                   t.bpm, t.key, t.mode, t.audio_features
            FROM ratings r
            JOIN tracks t ON r.track_id = t.id
            ORDER BY r.rated_at DESC
            """
        ).fetchall()
        return [self._row_to_rated_track(r) for r in rows]

    # ------------------------------------------------------------------
    # Engine suggestions
    # ------------------------------------------------------------------

    def insert_engine_suggestion(
        self,
        session_id: str,
        engine_name: str,
        track_id: str,
        engine_score: float,
        was_final: bool,
        final_rank: int | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO engine_suggestions
                (session_id, engine_name, track_id, engine_score, was_final, final_rank)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, engine_name, track_id, engine_score, int(was_final), final_rank),
        )
        self.conn.commit()

    def get_session_engine_suggestions(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM engine_suggestions WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Engine performance
    # ------------------------------------------------------------------

    def insert_engine_performance(
        self,
        engine_name: str,
        session_id: str,
        tracks_suggested: int,
        tracks_in_final: int,
        avg_rating_received: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO engine_performance
                (engine_name, session_id, tracks_suggested, tracks_in_final, avg_rating_received)
            VALUES (?, ?, ?, ?, ?)
            """,
            (engine_name, session_id, tracks_suggested, tracks_in_final, avg_rating_received),
        )
        self.conn.commit()

    def get_engine_performance(self, last_n_sessions: int = 10) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT ep.engine_name,
                   COUNT(*) as session_count,
                   AVG(ep.tracks_in_final) as avg_tracks_in_final,
                   AVG(ep.avg_rating_received) as avg_rating
            FROM engine_performance ep
            JOIN (
                SELECT id FROM sessions ORDER BY started_at DESC LIMIT ?
            ) recent ON ep.session_id = recent.id
            GROUP BY ep.engine_name
            """,
            (last_n_sessions,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Genre session history
    # ------------------------------------------------------------------

    def upsert_genre_session_history(
        self, genre: str, session_id: str, track_count: int, avg_rating: float
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO genre_session_history (genre, session_id, track_count, avg_rating)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(genre, session_id) DO UPDATE SET
                track_count = excluded.track_count,
                avg_rating = excluded.avg_rating
            """,
            (genre, session_id, track_count, avg_rating),
        )
        self.conn.commit()

    def get_genre_session_history(self, n_sessions: int = 5) -> list[dict]:
        """Return genre history for the last n completed sessions."""
        rows = self.conn.execute(
            """
            SELECT gsh.genre, gsh.session_id, gsh.track_count, gsh.avg_rating
            FROM genre_session_history gsh
            JOIN sessions s ON gsh.session_id = s.id
            WHERE s.completed_at IS NOT NULL
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (n_sessions * 20,),  # generous upper bound; diversity enforcer groups by session
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_rated_track_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT DISTINCT track_id FROM ratings").fetchall()
        return {row[0] for row in rows}

    def get_tracks_with_audio_features(self, limit: int = 5000) -> list[Track]:
        """Return tracks that have audio_features populated (for ANN search)."""
        rows = self.conn.execute(
            "SELECT * FROM tracks WHERE audio_features IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_track(r) for r in rows]

    # ------------------------------------------------------------------
    # Playlist management (Phase 2)
    # ------------------------------------------------------------------

    def upsert_tidal_playlist(
        self,
        playlist_type: str,
        name: str,
        tidal_playlist_id: str,
        tidal_playlist_url: str,
        genre: str | None = None,
    ) -> None:
        from datetime import datetime
        self.conn.execute(
            """
            INSERT INTO tidal_playlists
                (playlist_type, genre, tidal_playlist_id, tidal_playlist_url, name, last_synced_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tidal_playlist_id) DO UPDATE SET
                name = excluded.name,
                tidal_playlist_url = excluded.tidal_playlist_url,
                last_synced_at = excluded.last_synced_at
            """,
            (playlist_type, genre, tidal_playlist_id, tidal_playlist_url, name, _fmt_dt(datetime.now())),
        )
        self.conn.commit()

    def get_tidal_playlist(self, playlist_type: str, genre: str | None = None) -> dict | None:
        if genre is not None:
            row = self.conn.execute(
                "SELECT * FROM tidal_playlists WHERE playlist_type = ? AND genre = ?",
                (playlist_type, genre),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM tidal_playlists WHERE playlist_type = ? AND genre IS NULL",
                (playlist_type,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_tidal_playlists(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tidal_playlists ORDER BY playlist_type, genre"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_playlist_track_count(self, tidal_playlist_id: str, track_count: int) -> None:
        self.conn.execute(
            "UPDATE tidal_playlists SET track_count = ? WHERE tidal_playlist_id = ?",
            (track_count, tidal_playlist_id),
        )
        self.conn.commit()

    def add_candidate_track(self, tidal_playlist_id: str, track_id: str, session_id: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO candidate_playlist_tracks
                (tidal_playlist_id, track_id, session_id)
            VALUES (?, ?, ?)
            """,
            (tidal_playlist_id, track_id, session_id),
        )
        self.conn.commit()

    def remove_candidate_track(self, tidal_playlist_id: str, track_id: str) -> None:
        from datetime import datetime
        self.conn.execute(
            """
            UPDATE candidate_playlist_tracks
            SET removed_at = ?
            WHERE tidal_playlist_id = ? AND track_id = ? AND removed_at IS NULL
            """,
            (_fmt_dt(datetime.now()), tidal_playlist_id, track_id),
        )
        self.conn.commit()

    def get_unremoved_candidate_tracks(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT cpt.track_id, cpt.tidal_playlist_id, t.tidal_id
            FROM candidate_playlist_tracks cpt
            JOIN tracks t ON cpt.track_id = t.id
            WHERE cpt.session_id = ? AND cpt.removed_at IS NULL
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_candidate_tracks_for_session(self, session_id: str) -> int:
        """Mark all candidate playlist tracks for a session as removed. Returns count."""
        from datetime import datetime
        cursor = self.conn.execute(
            """
            UPDATE candidate_playlist_tracks
            SET removed_at = ?
            WHERE session_id = ? AND removed_at IS NULL
            """,
            (_fmt_dt(datetime.now()), session_id),
        )
        self.conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def get_total_ratings_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]

    def get_genre_avg_ratings(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT t.genre_primary as genre, AVG(r.score) as avg_rating, COUNT(*) as count
            FROM ratings r
            JOIN tracks t ON r.track_id = t.id
            GROUP BY t.genre_primary
            ORDER BY avg_rating DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Row → dataclass converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_track(row: sqlite3.Row) -> Track:
        d = dict(row)
        return Track(
            id=d["id"],
            title=d["title"],
            artist=d["artist"],
            album=d.get("album", ""),
            duration_ms=d.get("duration_ms", 0),
            genre_primary=d.get("genre_primary", "Other"),
            genre_tags=json.loads(d.get("genre_tags") or "[]"),
            mood_tags=json.loads(d.get("mood_tags") or "[]"),
            tidal_id=d.get("tidal_id"),
            spotify_id=d.get("spotify_id"),
            mbid=d.get("mbid"),
            bpm=d.get("bpm"),
            key=d.get("key"),
            mode=d.get("mode"),
            audio_features=json.loads(d["audio_features"]) if d.get("audio_features") else None,
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        d = dict(row)
        return Session(
            id=d["id"],
            started_at=_parse_dt(d["started_at"]),
            completed_at=_parse_dt(d.get("completed_at")),
            engine_allocation=json.loads(d.get("engine_allocation") or "{}"),
            diversity_config=json.loads(d.get("diversity_config") or "{}"),
            notes=d.get("notes"),
        )

    @staticmethod
    def _row_to_rated_track(row: sqlite3.Row) -> RatedTrack:
        d = dict(row)
        track = Track(
            id=d["id"],
            title=d["title"],
            artist=d["artist"],
            album=d.get("album", ""),
            duration_ms=d.get("duration_ms", 0),
            genre_primary=d.get("genre_primary", "Other"),
            genre_tags=json.loads(d.get("genre_tags") or "[]"),
            mood_tags=json.loads(d.get("mood_tags") or "[]"),
            tidal_id=d.get("tidal_id"),
            spotify_id=d.get("spotify_id"),
            mbid=d.get("mbid"),
            bpm=d.get("bpm"),
            key=d.get("key"),
            mode=d.get("mode"),
            audio_features=json.loads(d["audio_features"]) if d.get("audio_features") else None,
        )
        return RatedTrack(
            track=track,
            score=d["score"],
            rated_at=_parse_dt(d["rated_at"]),
            session_id=d["session_id"],
        )
