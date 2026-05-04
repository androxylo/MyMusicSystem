"""
Spotify connector — wraps spotipy.

Note (Nov 2024): Spotify removed the recommendations endpoint and restricted
audio-features to apps approved before Nov 2024. This connector is kept for
artist/track search and metadata only. Audio features come from Essentia.

Provides:
- Artist search
- Track search
- Artist top tracks
- Get track metadata (including spotify_id linkage)
"""
from __future__ import annotations

import logging

from connectors import ConnectorError

logger = logging.getLogger(__name__)


class SpotifyTrack:
    __slots__ = ("spotify_id", "title", "artist", "album", "duration_ms", "url", "popularity")

    def __init__(
        self,
        spotify_id: str,
        title: str,
        artist: str,
        album: str,
        duration_ms: int,
        url: str = "",
        popularity: int = 0,
    ) -> None:
        self.spotify_id = spotify_id
        self.title = title
        self.artist = artist
        self.album = album
        self.duration_ms = duration_ms
        self.url = url
        self.popularity = popularity


class SpotifyConnector:
    """
    Wraps spotipy.Spotify (client credentials flow — no user auth needed for search).

    Usage:
        connector = SpotifyConnector(config)
        connector.connect()
        tracks = connector.search_tracks("Aphex Twin", limit=5)
    """

    def __init__(self, config: dict) -> None:
        """config keys: client_id, client_secret"""
        self._config = config
        self._sp = None

    def connect(self) -> None:
        """Initialize spotipy with client credentials. Raises ConnectorError on failure."""
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
        except ImportError:
            raise ConnectorError("spotipy is not installed — run: pip install spotipy")

        try:
            auth_manager = SpotifyClientCredentials(
                client_id=self._config.get("client_id", ""),
                client_secret=self._config.get("client_secret", ""),
            )
            self._sp = spotipy.Spotify(auth_manager=auth_manager)
            # Quick connectivity check
            self._sp.search(q="test", type="track", limit=1)
            logger.info("Spotify: connected via client credentials")
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Spotify connection failed: {exc}") from exc

    def is_available(self) -> bool:
        try:
            return self._sp is not None
        except Exception:
            return False

    @property
    def _client(self):
        if self._sp is None:
            raise ConnectorError("SpotifyConnector.connect() has not been called")
        return self._sp

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_tracks(self, query: str, limit: int = 10) -> list[SpotifyTrack]:
        """Search Spotify for tracks. Returns lightweight DTOs."""
        try:
            results = self._client.search(q=query, type="track", limit=limit)
            items = results.get("tracks", {}).get("items", [])
            return [self._to_spotify_track(t) for t in items]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Spotify search_tracks({query!r}) failed: {exc}") from exc

    def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> list[SpotifyTrack]:
        """Return top tracks for a Spotify artist ID."""
        try:
            results = self._client.artist_top_tracks(artist_id, country=market)
            return [self._to_spotify_track(t) for t in results.get("tracks", [])]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Spotify get_artist_top_tracks({artist_id!r}) failed: {exc}") from exc

    def get_track(self, spotify_id: str) -> SpotifyTrack | None:
        """Fetch a single track by Spotify ID. Returns None if not found."""
        try:
            t = self._client.track(spotify_id)
            return self._to_spotify_track(t) if t else None
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Spotify get_track({spotify_id!r}) failed: {exc}") from exc

    def search_artists(self, query: str, limit: int = 10) -> list[dict]:
        """Search for artists, returning id/name/genres/popularity dicts."""
        try:
            results = self._client.search(q=query, type="artist", limit=limit)
            items = results.get("artists", {}).get("items", [])
            return [
                {
                    "spotify_id": a["id"],
                    "name": a["name"],
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity", 0),
                }
                for a in items
            ]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Spotify search_artists({query!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_spotify_track(t: dict) -> SpotifyTrack:
        artists = t.get("artists", [])
        artist_name = artists[0]["name"] if artists else "Unknown"
        album = t.get("album", {})
        return SpotifyTrack(
            spotify_id=t["id"],
            title=t["name"],
            artist=artist_name,
            album=album.get("name", ""),
            duration_ms=t.get("duration_ms", 0),
            url=t.get("external_urls", {}).get("spotify", ""),
            popularity=t.get("popularity", 0),
        )
