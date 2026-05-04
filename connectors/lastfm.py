"""
Last.fm connector — wraps pylast.

Provides:
- Artist similarity graph traversal (for LastFmGraphEngine)
- Tag-based top track lookup (for GenreExplorerEngine)
- User listening history (scrobbles)
- Artist/track search
"""
from __future__ import annotations

import logging

from connectors import ConnectorError

logger = logging.getLogger(__name__)


class LastFmArtist:
    __slots__ = ("name", "mbid", "match_score", "url")

    def __init__(self, name: str, mbid: str = "", match_score: float = 0.0, url: str = "") -> None:
        self.name = name
        self.mbid = mbid
        self.match_score = match_score
        self.url = url


class LastFmTrack:
    __slots__ = ("title", "artist", "mbid", "url", "playcount")

    def __init__(self, title: str, artist: str, mbid: str = "", url: str = "", playcount: int = 0) -> None:
        self.title = title
        self.artist = artist
        self.mbid = mbid
        self.url = url
        self.playcount = playcount


class LastFmConnector:
    """
    Wraps pylast.LastFMNetwork.

    Usage:
        connector = LastFmConnector(config)
        connector.connect()
        similar = connector.get_similar_artists("Aphex Twin", limit=10)
    """

    def __init__(self, config: dict) -> None:
        """
        config keys: api_key, api_secret, username (optional for user history)
        """
        self._config = config
        self._network = None

    def connect(self) -> None:
        """Initialize the pylast network connection. Raises ConnectorError on failure."""
        try:
            import pylast
        except ImportError:
            raise ConnectorError("pylast is not installed — run: pip install pylast")

        try:
            self._network = pylast.LastFMNetwork(
                api_key=self._config.get("api_key", ""),
                api_secret=self._config.get("api_secret", ""),
                username=self._config.get("username", ""),
            )
            # Verify credentials with a lightweight call
            if self._config.get("username"):
                self._network.get_user(self._config["username"])
            logger.info("Last.fm: connected")
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm connection failed: {exc}") from exc

    def is_available(self) -> bool:
        try:
            return self._network is not None
        except Exception:
            return False

    @property
    def _net(self):
        if self._network is None:
            raise ConnectorError("LastFmConnector.connect() has not been called")
        return self._network

    # ------------------------------------------------------------------
    # Artist similarity
    # ------------------------------------------------------------------

    def get_similar_artists(self, artist_name: str, limit: int = 20) -> list[LastFmArtist]:
        """Return artists similar to the given artist, sorted by match score."""
        try:
            artist = self._net.get_artist(artist_name)
            similar = artist.get_similar(limit=limit)
            result = []
            for item in similar:
                a = item.item
                result.append(LastFmArtist(
                    name=a.name,
                    mbid=a.get_mbid() or "",
                    match_score=float(item.match),
                    url=a.get_url() or "",
                ))
            return result
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_similar_artists({artist_name!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Tag-based discovery
    # ------------------------------------------------------------------

    def get_tag_top_tracks(self, tag: str, limit: int = 50) -> list[LastFmTrack]:
        """Return top tracks for a Last.fm tag (genre/mood tag)."""
        try:
            t = self._net.get_tag(tag)
            top = t.get_top_tracks(limit=limit)
            result = []
            for item in top:
                track = item.item
                result.append(LastFmTrack(
                    title=track.title,
                    artist=track.artist.name,
                    mbid=track.get_mbid() or "",
                    url=track.get_url() or "",
                ))
            return result
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_tag_top_tracks({tag!r}) failed: {exc}") from exc

    def get_tag_top_artists(self, tag: str, limit: int = 30) -> list[LastFmArtist]:
        """Return top artists for a Last.fm tag."""
        try:
            t = self._net.get_tag(tag)
            top = t.get_top_artists(limit=limit)
            result = []
            for item in top:
                a = item.item
                result.append(LastFmArtist(
                    name=a.name,
                    mbid=a.get_mbid() or "",
                    url=a.get_url() or "",
                ))
            return result
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_tag_top_artists({tag!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Artist top tracks
    # ------------------------------------------------------------------

    def get_artist_top_tracks(self, artist_name: str, limit: int = 10) -> list[LastFmTrack]:
        """Return top tracks for a given artist name."""
        try:
            artist = self._net.get_artist(artist_name)
            top = artist.get_top_tracks(limit=limit)
            result = []
            for item in top:
                track = item.item
                result.append(LastFmTrack(
                    title=track.title,
                    artist=artist_name,
                    mbid=track.get_mbid() or "",
                    url=track.get_url() or "",
                    playcount=int(item.weight) if item.weight else 0,
                ))
            return result
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_artist_top_tracks({artist_name!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Per-track tags
    # ------------------------------------------------------------------

    def get_track_tags(self, title: str, artist: str, limit: int = 15) -> list[tuple[str, float]]:
        """
        Return top tags for a specific track as (tag_name, weight) pairs.
        Weight is normalized to [0, 1] from Last.fm's 0-100 scale.
        Tags are lowercased.
        """
        try:
            track = self._net.get_track(artist, title)
            tags = track.get_top_tags(limit=limit)
            return [
                (t.item.name.lower(), min(1.0, int(t.weight) / 100.0))
                for t in tags
                if t.weight
            ]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_track_tags({artist!r}, {title!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # User history
    # ------------------------------------------------------------------

    def get_artist_tags(self, artist_name: str, limit: int = 5) -> list[str]:
        """Return top tag names for an artist (used for genre classification)."""
        try:
            artist = self._net.get_artist(artist_name)
            tags = artist.get_top_tags(limit=limit)
            return [tag.item.name for tag in tags]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_artist_tags({artist_name!r}) failed: {exc}") from exc

    def get_user_top_artists(self, username: str | None = None, period: str = "overall", limit: int = 50) -> list[LastFmArtist]:
        """
        Return top artists for the user.
        period: 'overall' | '7day' | '1month' | '3month' | '6month' | '12month'
        """
        try:
            import pylast
            uname = username or self._config.get("username", "")
            user = self._net.get_user(uname)
            period_map = {
                "overall": pylast.PERIOD_OVERALL,
                "7day": pylast.PERIOD_7DAYS,
                "1month": pylast.PERIOD_1MONTH,
                "3month": pylast.PERIOD_3MONTHS,
                "6month": pylast.PERIOD_6MONTHS,
                "12month": pylast.PERIOD_12MONTHS,
            }
            top = user.get_top_artists(period=period_map.get(period, pylast.PERIOD_OVERALL), limit=limit)
            return [
                LastFmArtist(name=item.item.name, match_score=float(item.weight or 0))
                for item in top
            ]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_user_top_artists() failed: {exc}") from exc

    def get_user_recent_tracks(self, username: str | None = None, limit: int = 200) -> list[LastFmTrack]:
        """Return the user's recent scrobbles."""
        try:
            uname = username or self._config.get("username", "")
            user = self._net.get_user(uname)
            recent = user.get_recent_tracks(limit=limit)
            result = []
            for played in recent:
                t = played.track
                result.append(LastFmTrack(
                    title=t.title,
                    artist=t.artist.name,
                    mbid=t.get_mbid() or "",
                    url=t.get_url() or "",
                ))
            return result
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Last.fm get_user_recent_tracks() failed: {exc}") from exc
