"""
Tidal connector — wraps tidalapi.

Responsibilities:
- Session authentication (OAuth device flow or token reuse)
- Track search
- Playlist CRUD: create, get, add tracks, remove track, replace all tracks
- User's algorithmic mixes (for TidalMixEngine)

All methods raise ConnectorError on failure.
Call is_available() before use in health checks.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from connectors import ConnectorError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TidalTrack:
    """Lightweight DTO — avoids leaking tidalapi objects into core."""
    __slots__ = ("tidal_id", "title", "artist", "album", "duration_ms", "url")

    def __init__(
        self,
        tidal_id: str,
        title: str,
        artist: str,
        album: str,
        duration_ms: int,
        url: str = "",
    ) -> None:
        self.tidal_id = tidal_id
        self.title = title
        self.artist = artist
        self.album = album
        self.duration_ms = duration_ms
        self.url = url


class TidalPlaylist:
    """Lightweight DTO for a Tidal playlist."""
    __slots__ = ("playlist_id", "name", "url", "track_count")

    def __init__(self, playlist_id: str, name: str, url: str, track_count: int) -> None:
        self.playlist_id = playlist_id
        self.name = name
        self.url = url
        self.track_count = track_count


class TidalConnector:
    """
    Wraps tidalapi.Session.

    Usage:
        connector = TidalConnector(config)
        connector.authenticate()   # once at startup
        tracks = connector.search_tracks("Aphex Twin", limit=5)
    """

    def __init__(self, config: dict) -> None:
        """
        config keys:
          client_id, client_secret
          Optional: token_path (path to persist OAuth tokens)
        """
        self._config = config
        self._session = None
        self._authenticated = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """
        Authenticate with Tidal via OAuth device flow.
        Persists tokens to token_path if configured.
        Raises ConnectorError on failure.
        """
        try:
            import tidalapi
        except ImportError:
            raise ConnectorError("tidalapi is not installed — run: pip install tidalapi")

        try:
            client_id = self._config.get("client_id", "")
            client_secret = self._config.get("client_secret", "")
            if client_id and client_secret:
                config = tidalapi.Config(client_id=client_id, client_secret=client_secret)
            else:
                config = tidalapi.Config()   # use tidalapi's built-in default credentials
            self._session = tidalapi.Session(config)

            token_path = self._config.get("token_path")
            if token_path:
                import json
                from pathlib import Path
                p = Path(token_path)
                if p.exists():
                    tokens = json.loads(p.read_text())
                    loaded = self._session.load_oauth_session(
                        token_type=tokens["token_type"],
                        access_token=tokens["access_token"],
                        refresh_token=tokens.get("refresh_token"),
                        expiry_time=tokens.get("expiry_time"),
                    )
                    if loaded and self._session.check_login():
                        self._authenticated = True
                        logger.info("Tidal: reused existing OAuth tokens")
                        return

            # Device flow login
            login_url, future = self._session.login_oauth()
            logger.info(f"Tidal OAuth: open this URL to authenticate: {login_url}")
            print(f"\nTidal authentication required.\nOpen this URL: {login_url}\n")
            future.result()  # blocks until user completes auth

            if not self._session.check_login():
                raise ConnectorError("Tidal authentication failed after OAuth flow")

            self._authenticated = True
            logger.info("Tidal: authenticated via OAuth device flow")

            if token_path:
                self._persist_tokens(token_path)

        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal authentication error: {exc}") from exc

    def _persist_tokens(self, token_path: str) -> None:
        import json
        from pathlib import Path
        try:
            tokens = {
                "token_type": self._session.token_type,
                "access_token": self._session.access_token,
                "refresh_token": self._session.refresh_token,
                "expiry_time": str(self._session.expiry_time) if self._session.expiry_time else None,
            }
            Path(token_path).parent.mkdir(parents=True, exist_ok=True)
            Path(token_path).write_text(json.dumps(tokens, indent=2))
        except Exception as exc:
            logger.warning(f"Tidal: could not persist tokens: {exc}")

    def is_available(self) -> bool:
        try:
            return self._authenticated and self._session is not None and self._session.check_login()
        except Exception:
            return False

    @property
    def _s(self):
        """Return the authenticated session or raise ConnectorError."""
        if not self._authenticated or self._session is None:
            raise ConnectorError("TidalConnector is not authenticated — call authenticate() first")
        return self._session

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_tracks(self, query: str, limit: int = 10) -> list[TidalTrack]:
        """Search Tidal for tracks matching the query string."""
        try:
            import tidalapi
            results = self._s.search(query, models=[tidalapi.Track], limit=limit)
            tracks_raw = results.get("tracks", []) if isinstance(results, dict) else getattr(results, "tracks", [])
            return [self._to_tidal_track(t) for t in (tracks_raw or [])]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal search_tracks({query!r}) failed: {exc}") from exc

    def get_artist_top_tracks(self, artist_id: str, limit: int = 10) -> list[TidalTrack]:
        """Return top tracks for a given Tidal artist ID."""
        try:
            import tidalapi
            artist = tidalapi.Artist(self._s, artist_id)
            tracks = artist.get_top_tracks(limit=limit)
            return [self._to_tidal_track(t) for t in tracks]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal get_artist_top_tracks({artist_id!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # User mixes (for TidalMixEngine)
    # ------------------------------------------------------------------

    def get_mix_tracks(self, mix_id: str) -> list[TidalTrack]:
        """Return all tracks from a Tidal mix (e.g. My Daily Discovery)."""
        try:
            import tidalapi
            mix = tidalapi.Mix(self._s, mix_id)
            tracks = mix.items()
            return [self._to_tidal_track(t) for t in tracks]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal get_mix_tracks({mix_id!r}) failed: {exc}") from exc

    def get_user_mixes(self) -> list[dict]:
        """Return the user's available mixes with id and title."""
        try:
            page = self._s.mixes()
            mixes = []
            for category in page.categories:
                for item in getattr(category, "items", []):
                    mixes.append({"id": str(item.id), "title": item.title})
            return mixes
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal get_user_mixes() failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Playlist CRUD
    # ------------------------------------------------------------------

    def create_playlist(self, name: str, description: str = "") -> TidalPlaylist:
        """Create a new user playlist. Returns the created playlist."""
        try:
            pl = self._s.user.create_playlist(name, description)
            return TidalPlaylist(
                playlist_id=str(pl.id),
                name=pl.name,
                url=f"https://tidal.com/browse/playlist/{pl.id}",
                track_count=0,
            )
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal create_playlist({name!r}) failed: {exc}") from exc

    def get_playlist(self, playlist_id: str) -> TidalPlaylist:
        """Fetch playlist metadata by Tidal playlist ID."""
        try:
            import tidalapi
            pl = tidalapi.Playlist(self._s, playlist_id)
            tracks = pl.tracks()
            return TidalPlaylist(
                playlist_id=str(pl.id),
                name=pl.name,
                url=f"https://tidal.com/browse/playlist/{pl.id}",
                track_count=len(tracks),
            )
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal get_playlist({playlist_id!r}) failed: {exc}") from exc

    def get_playlist_track_ids(self, playlist_id: str) -> list[str]:
        """Return the Tidal track IDs currently in a playlist."""
        try:
            import tidalapi
            pl = tidalapi.Playlist(self._s, playlist_id)
            return [str(t.id) for t in pl.tracks()]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal get_playlist_track_ids({playlist_id!r}) failed: {exc}") from exc

    def set_playlist_tracks(self, playlist_id: str, tidal_track_ids: list[str]) -> None:
        """
        Replace the entire contents of a playlist with the given track IDs.
        Clears first, then adds in order.
        """
        try:
            import tidalapi
            pl = tidalapi.UserPlaylist(self._s, playlist_id)
            # Clear existing tracks
            existing = [str(t.id) for t in pl.tracks()]
            if existing:
                pl.remove_by_indices(list(range(len(existing))))
            # Add new tracks
            if tidal_track_ids:
                pl.add(tidal_track_ids)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal set_playlist_tracks({playlist_id!r}) failed: {exc}") from exc

    def add_tracks_to_playlist(self, playlist_id: str, tidal_track_ids: list[str]) -> None:
        """Append tracks to a playlist. Does not deduplicate."""
        try:
            import tidalapi
            pl = tidalapi.UserPlaylist(self._s, playlist_id)
            pl.add(tidal_track_ids)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Tidal add_tracks_to_playlist({playlist_id!r}) failed: {exc}") from exc

    def remove_track_from_playlist(self, playlist_id: str, tidal_track_id: str) -> None:
        """Remove the first occurrence of a track from a playlist by its Tidal track ID."""
        try:
            import tidalapi
            pl = tidalapi.UserPlaylist(self._s, playlist_id)
            tracks = pl.tracks()
            indices = [i for i, t in enumerate(tracks) if str(t.id) == tidal_track_id]
            if not indices:
                logger.debug(f"Track {tidal_track_id} not found in playlist {playlist_id} — nothing to remove")
                return
            pl.remove_by_indices([indices[0]])
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(
                f"Tidal remove_track_from_playlist({playlist_id!r}, {tidal_track_id!r}) failed: {exc}"
            ) from exc

    def is_track_in_playlist(self, playlist_id: str, tidal_track_id: str) -> bool:
        """Return True if the track is already in the playlist (for idempotent adds)."""
        try:
            existing = self.get_playlist_track_ids(playlist_id)
            return tidal_track_id in existing
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(
                f"Tidal is_track_in_playlist({playlist_id!r}, {tidal_track_id!r}) failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tidal_track(t) -> TidalTrack:
        try:
            artist = t.artist.name if t.artist else "Unknown"
        except Exception:
            artist = "Unknown"
        try:
            album = t.album.name if t.album else ""
        except Exception:
            album = ""
        return TidalTrack(
            tidal_id=str(t.id),
            title=t.name or "",
            artist=artist,
            album=album,
            duration_ms=int((t.duration or 0) * 1000),
            url=f"https://tidal.com/browse/track/{t.id}",
        )
