"""
PlaylistManager — owns all Tidal playlist state.

Two playlist concepts:
  Candidate ("Now Rating"): rotating, always ≤ 10 tracks.
    Created once, replaced at session start, drained as tracks are rated.
  Curated ("Liked — All", "Liked — <Genre>"): permanent, append-only.
    Created lazily (first time a track qualifies). Never removes tracks.

All Tidal operations are wrapped in try/except; errors are logged but not
re-raised. The caller (SessionManager) commits the DB rating first, then
calls playlist operations fire-and-forget.

The DB is the source of truth. reconcile_session() can replay any session's
ratings against the current Tidal state to fix gaps caused by prior failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from connectors.tidal import TidalConnector
from core.base_engine import Track
from core.db import Database

logger = logging.getLogger(__name__)


@dataclass
class PlaylistInfo:
    playlist_type: str      # 'candidate' | 'curated_master' | 'curated_genre'
    genre: str | None
    name: str
    tidal_playlist_id: str
    tidal_url: str
    track_count: int


@dataclass
class PlaylistConfig:
    curated_threshold: int = 7
    candidate_name: str = "Now Rating"
    curated_master_name: str = "Liked \u2014 All"
    curated_genre_prefix: str = "Liked \u2014 "


class PlaylistManager:
    """
    Manages Tidal playlist state: candidate + curated playlists.

    Pass a PlaylistConfig to override naming and threshold defaults.
    All Tidal failures are logged at WARNING level and do not propagate.
    """

    def __init__(
        self,
        tidal: TidalConnector,
        db: Database,
        config: PlaylistConfig | None = None,
    ) -> None:
        self._tidal = tidal
        self._db = db
        self._cfg = config or PlaylistConfig()

    # ------------------------------------------------------------------
    # Candidate playlist ("Now Rating")
    # ------------------------------------------------------------------

    def sync_candidate_playlist(self, session_id: str, tracks: list[Track]) -> str:
        """
        Replace the "Now Rating" Tidal playlist with the given tracks.

        Creates the playlist if it doesn't exist yet.
        Persists playlist ID + URL to DB.
        Records each track in candidate_playlist_tracks for reconciliation.

        Returns the Tidal playlist URL (https://tidal.com/browse/playlist/<uuid>).
        Raises on failure — caller should catch and log.
        """
        tidal_track_ids = [t.tidal_id for t in tracks if t.tidal_id]
        if not tidal_track_ids:
            logger.warning("sync_candidate_playlist: no tracks have tidal_id — cannot sync")

        # Ensure the candidate playlist exists in Tidal
        playlist_row = self._db.get_tidal_playlist("candidate")
        if playlist_row is None:
            pl = self._tidal.create_playlist(
                self._cfg.candidate_name,
                description="Tracks currently being rated — managed by MyMusicSystem",
            )
            self._db.upsert_tidal_playlist(
                playlist_type="candidate",
                name=pl.name,
                tidal_playlist_id=pl.playlist_id,
                tidal_playlist_url=pl.url,
            )
            playlist_row = self._db.get_tidal_playlist("candidate")

        tidal_pl_id = playlist_row["tidal_playlist_id"]
        tidal_url = playlist_row["tidal_playlist_url"]

        # Replace all tracks
        if tidal_track_ids:
            self._tidal.set_playlist_tracks(tidal_pl_id, tidal_track_ids)

        # Record in candidate_playlist_tracks (upsert tracks first to satisfy FK)
        for track in tracks:
            self._db.upsert_track(track)
            self._db.add_candidate_track(tidal_pl_id, track.id, session_id)

        self._db.update_playlist_track_count(tidal_pl_id, len(tidal_track_ids))
        logger.info(f"sync_candidate_playlist: {len(tidal_track_ids)} tracks → '{self._cfg.candidate_name}'")

        return tidal_url

    def remove_from_candidate(self, track_id: str) -> None:
        """
        Remove a rated track from the "Now Rating" Tidal playlist.
        Logs failures; does not raise.
        """
        try:
            playlist_row = self._db.get_tidal_playlist("candidate")
            if playlist_row is None:
                logger.debug("remove_from_candidate: no candidate playlist in DB — skipping")
                return

            tidal_pl_id = playlist_row["tidal_playlist_id"]

            # Look up the track's tidal_id
            from core.db import Database
            track = self._db.get_track(track_id)
            if track is None or not track.tidal_id:
                logger.debug(f"remove_from_candidate: track {track_id} has no tidal_id — skipping Tidal remove")
            else:
                self._tidal.remove_track_from_playlist(tidal_pl_id, track.tidal_id)

            # Update DB tracking
            self._db.remove_candidate_track(tidal_pl_id, track_id)

            # Update track count
            current_count = playlist_row.get("track_count", 0)
            self._db.update_playlist_track_count(tidal_pl_id, max(0, current_count - 1))

        except Exception:
            logger.warning(f"remove_from_candidate({track_id}) failed", exc_info=True)

    # ------------------------------------------------------------------
    # Curated playlists ("Liked — All", "Liked — <Genre>")
    # ------------------------------------------------------------------

    def add_to_curated(self, track: Track, score: int) -> None:
        """
        Add a track to "Liked — All" and "Liked — <genre_primary>" playlists
        if score >= threshold. Idempotent. Logs failures; does not raise.
        """
        if score < self._cfg.curated_threshold:
            return

        try:
            self._add_to_curated_playlist(
                playlist_type="curated_master",
                name=self._cfg.curated_master_name,
                genre=None,
                track=track,
            )
        except Exception:
            logger.warning(f"add_to_curated: master playlist failed for track {track.id}", exc_info=True)

        try:
            genre_name = self._cfg.curated_genre_prefix + track.genre_primary
            self._add_to_curated_playlist(
                playlist_type="curated_genre",
                name=genre_name,
                genre=track.genre_primary,
                track=track,
            )
        except Exception:
            logger.warning(f"add_to_curated: genre playlist failed for track {track.id}", exc_info=True)

    def _add_to_curated_playlist(
        self,
        playlist_type: str,
        name: str,
        genre: str | None,
        track: Track,
    ) -> None:
        """Ensure playlist exists, then add track idempotently."""
        if not track.tidal_id:
            logger.debug(f"add_to_curated: track {track.id} has no tidal_id — cannot add to Tidal")
            return

        # Ensure playlist exists
        playlist_row = self._db.get_tidal_playlist(playlist_type, genre=genre)
        if playlist_row is None:
            pl = self._tidal.create_playlist(
                name,
                description=f"Curated by MyMusicSystem — {name}",
            )
            self._db.upsert_tidal_playlist(
                playlist_type=playlist_type,
                name=pl.name,
                tidal_playlist_id=pl.playlist_id,
                tidal_playlist_url=pl.url,
                genre=genre,
            )
            playlist_row = self._db.get_tidal_playlist(playlist_type, genre=genre)

        tidal_pl_id = playlist_row["tidal_playlist_id"]

        # Idempotent add
        if self._tidal.is_track_in_playlist(tidal_pl_id, track.tidal_id):
            logger.debug(f"Track {track.tidal_id} already in {name!r} — skipping add")
            return

        self._tidal.add_tracks_to_playlist(tidal_pl_id, [track.tidal_id])
        current = playlist_row.get("track_count", 0)
        self._db.update_playlist_track_count(tidal_pl_id, current + 1)
        logger.info(f"Added track '{track.title}' → '{name}'")

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def clear_candidate_remaining(self, session_id: str) -> int:
        """
        Remove all unrated tracks still in "Now Rating" for a completed session.
        Called at session completion as a safety net.
        Returns the count of tracks removed from Tidal.
        """
        try:
            unremoved = self._db.get_unremoved_candidate_tracks(session_id)
            if not unremoved:
                return 0

            playlist_row = self._db.get_tidal_playlist("candidate")
            tidal_pl_id = playlist_row["tidal_playlist_id"] if playlist_row else None

            removed_count = 0
            for row in unremoved:
                tidal_id = row.get("tidal_id")
                if tidal_pl_id and tidal_id:
                    try:
                        self._tidal.remove_track_from_playlist(tidal_pl_id, tidal_id)
                        removed_count += 1
                    except Exception:
                        logger.warning(f"clear_candidate_remaining: could not remove {tidal_id}", exc_info=True)

            # Mark all as removed in DB regardless of Tidal success
            self._db.clear_candidate_tracks_for_session(session_id)

            if tidal_pl_id:
                self._db.update_playlist_track_count(tidal_pl_id, 0)

            logger.info(f"clear_candidate_remaining: removed {removed_count} tracks from '{self._cfg.candidate_name}'")
            return removed_count

        except Exception:
            logger.warning(f"clear_candidate_remaining({session_id}) failed", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile_session(self, session_id: str) -> dict:
        """
        Replay all DB ratings for a session against current Tidal playlist state.
        Fixes any inconsistency from prior API failures.

        Returns: {candidate_fixes: N, curated_fixes: N}
        """
        candidate_fixes = 0
        curated_fixes = 0

        try:
            rated = self._db.get_session_ratings(session_id)
            playlist_row = self._db.get_tidal_playlist("candidate")

            for rt in rated:
                track = rt.track

                # Candidate: should be removed since it was rated
                if playlist_row and track.tidal_id:
                    tidal_pl_id = playlist_row["tidal_playlist_id"]
                    try:
                        if self._tidal.is_track_in_playlist(tidal_pl_id, track.tidal_id):
                            self._tidal.remove_track_from_playlist(tidal_pl_id, track.tidal_id)
                            self._db.remove_candidate_track(tidal_pl_id, track.id)
                            candidate_fixes += 1
                    except Exception:
                        logger.warning(f"reconcile: candidate remove failed for {track.id}", exc_info=True)

                # Curated: should be added if score >= threshold
                if rt.score >= self._cfg.curated_threshold:
                    try:
                        self._add_to_curated_playlist("curated_master", self._cfg.curated_master_name, None, track)
                    except Exception:
                        logger.warning(f"reconcile: curated_master add failed for {track.id}", exc_info=True)
                    try:
                        genre_name = self._cfg.curated_genre_prefix + track.genre_primary
                        self._add_to_curated_playlist("curated_genre", genre_name, track.genre_primary, track)
                        curated_fixes += 1
                    except Exception:
                        logger.warning(f"reconcile: curated_genre add failed for {track.id}", exc_info=True)

        except Exception:
            logger.exception(f"reconcile_session({session_id}) failed unexpectedly")

        return {"candidate_fixes": candidate_fixes, "curated_fixes": curated_fixes}

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_all_playlists(self) -> list[PlaylistInfo]:
        """Return all system-managed playlists with their Tidal URLs and track counts."""
        rows = self._db.get_all_tidal_playlists()
        return [
            PlaylistInfo(
                playlist_type=r["playlist_type"],
                genre=r.get("genre"),
                name=r["name"],
                tidal_playlist_id=r["tidal_playlist_id"],
                tidal_url=r.get("tidal_playlist_url", ""),
                track_count=r.get("track_count", 0),
            )
            for r in rows
        ]
