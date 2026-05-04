"""
MusicBrainz connector — wraps musicbrainzngs.

Provides:
- Track/recording lookup by MBID
- Artist lookup and search
- ISRC → MBID resolution
- AcoustID fingerprint lookup (for deduplication / identity resolution)

Used primarily for identity resolution: given a title+artist from Last.fm or Tidal,
look up the canonical MBID so the same recording is never rated twice under
different IDs.
"""
from __future__ import annotations

import logging

from connectors import ConnectorError

logger = logging.getLogger(__name__)

_APP_NAME = "MyMusicSystem"
_APP_VERSION = "0.1"
_CONTACT = "noreply@example.com"  # MusicBrainz requires a contact URL or email


class MusicBrainzConnector:
    """
    Wraps musicbrainzngs.

    Usage:
        connector = MusicBrainzConnector(config)
        connector.connect()
        mbid = connector.search_recording_mbid("Windowlicker", "Aphex Twin")
    """

    def __init__(self, config: dict | None = None) -> None:
        """config is currently unused; kept for consistency."""
        self._config = config or {}
        self._connected = False

    def connect(self) -> None:
        """Set the user agent (required by MusicBrainz API policy)."""
        try:
            import musicbrainzngs
        except ImportError:
            raise ConnectorError("musicbrainzngs is not installed — run: pip install musicbrainzngs")

        try:
            musicbrainzngs.set_useragent(_APP_NAME, _APP_VERSION, _CONTACT)
            self._connected = True
            logger.info("MusicBrainz: connected (user agent set)")
        except Exception as exc:
            raise ConnectorError(f"MusicBrainz setup failed: {exc}") from exc

    def is_available(self) -> bool:
        return self._connected

    def _check(self):
        if not self._connected:
            raise ConnectorError("MusicBrainzConnector.connect() has not been called")

    # ------------------------------------------------------------------
    # Recording lookup
    # ------------------------------------------------------------------

    def search_recording_mbid(self, title: str, artist: str) -> str | None:
        """
        Search for a recording and return the best-match MBID, or None.
        Uses strict=False for fuzzy matching.
        """
        self._check()
        try:
            import musicbrainzngs as mb
            result = mb.search_recordings(
                recording=title,
                artist=artist,
                limit=5,
                strict=False,
            )
            recordings = result.get("recording-list", [])
            if not recordings:
                return None
            # Return the MBID of the first result (highest score)
            return recordings[0]["id"]
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(
                f"MusicBrainz search_recording_mbid({title!r}, {artist!r}) failed: {exc}"
            ) from exc

    def get_recording(self, mbid: str) -> dict | None:
        """
        Fetch full recording metadata by MBID.
        Returns a dict with: mbid, title, artist, length_ms, isrcs, tags.
        """
        self._check()
        try:
            import musicbrainzngs as mb
            result = mb.get_recording_by_id(
                mbid,
                includes=["artists", "releases", "tags", "isrcs"],
            )
            rec = result.get("recording")
            if not rec:
                return None

            artist_credits = rec.get("artist-credit", [])
            artist_name = ""
            for credit in artist_credits:
                if isinstance(credit, dict) and "artist" in credit:
                    artist_name = credit["artist"].get("name", "")
                    break

            length_ms = int(rec.get("length", 0) or 0)

            tags = [t["name"] for t in rec.get("tag-list", [])]
            isrcs = rec.get("isrc-list", [])

            return {
                "mbid": rec["id"],
                "title": rec.get("title", ""),
                "artist": artist_name,
                "length_ms": length_ms,
                "isrcs": isrcs,
                "tags": tags,
            }
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"MusicBrainz get_recording({mbid!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Artist lookup
    # ------------------------------------------------------------------

    def search_artist_mbid(self, artist_name: str) -> str | None:
        """Return the MBID of the best-matching artist, or None."""
        self._check()
        try:
            import musicbrainzngs as mb
            result = mb.search_artists(artist=artist_name, limit=3)
            artists = result.get("artist-list", [])
            return artists[0]["id"] if artists else None
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(
                f"MusicBrainz search_artist_mbid({artist_name!r}) failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # ISRC resolution
    # ------------------------------------------------------------------

    def mbid_from_isrc(self, isrc: str) -> str | None:
        """
        Resolve an ISRC code to a MusicBrainz recording MBID.
        Returns the first match or None.
        """
        self._check()
        try:
            import musicbrainzngs as mb
            result = mb.get_recordings_by_isrc(isrc)
            recordings = result.get("isrc", {}).get("recording-list", [])
            return recordings[0]["id"] if recordings else None
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"MusicBrainz mbid_from_isrc({isrc!r}) failed: {exc}") from exc
