"""
TidalMixEngine — surfaces tracks from the user's Tidal algorithmic mixes.

Low-effort, high-signal source: Tidal's own ML has already personalized
these mixes for the user. The engine reads the mix contents and filters
tracks that have already been rated.

No data requirements: works from the first session.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


def _load_tidal_config() -> dict | None:
    try:
        if not _SETTINGS_PATH.exists():
            return None
        with _SETTINGS_PATH.open() as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("tidal", {})
        if not cfg.get("client_id"):
            return None
        return cfg
    except Exception:
        return None


def _stable_track_id(tidal_id: str) -> str:
    """Generate a deterministic internal UUID from a Tidal track ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tidal:{tidal_id}"))


class TidalMixEngine(BaseEngine):
    """
    Reads the user's Tidal algorithmic mixes and returns unrated tracks.

    Genre classification is not available from Tidal's mix API; genre_primary
    is set to "Unknown" and will be enriched downstream if needed.
    """

    def __init__(self, tidal=None) -> None:
        """
        Pass a pre-built TidalConnector for testing.
        When None, loads config from settings.yaml and builds the connector.
        Authentication must have already happened (tokens stored at token_path).
        """
        self._tidal = tidal if tidal is not None else self._try_build_tidal()

    def _try_build_tidal(self):
        cfg = _load_tidal_config()
        if cfg is None:
            return None
        try:
            from connectors.tidal import TidalConnector
            return TidalConnector(cfg)
        except Exception:
            logger.debug("TidalMixEngine: could not build TidalConnector", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def health_check(self) -> EngineHealth:
        if self._tidal is None:
            return EngineHealth(status="unavailable", message="Tidal not configured in settings.yaml")
        if not self._tidal.is_available():
            return EngineHealth(status="unavailable", message="Tidal not authenticated — call authenticate() first")
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if self._tidal is None or not self._tidal.is_available():
            return []
        try:
            return self._collect_from_mixes(n, context)
        except Exception:
            logger.exception("TidalMixEngine.suggest() failed")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_from_mixes(self, n: int, context: SessionContext) -> list[Suggestion]:
        excluded = context.excluded_track_ids

        mixes = self._tidal.get_user_mixes()
        if not mixes:
            logger.debug("TidalMixEngine: no mixes returned from Tidal")
            return []

        candidates: list[Suggestion] = []
        seen_tidal_ids: set[str] = set()

        for mix in mixes:
            if len(candidates) >= n * 2:
                break  # oversample up to 2× requested, then stop
            try:
                tracks = self._tidal.get_mix_tracks(mix["id"])
            except Exception:
                logger.warning(
                    f"TidalMixEngine: failed to fetch tracks for mix {mix['id']!r}",
                    exc_info=True,
                )
                continue

            for t in tracks:
                if t.tidal_id in seen_tidal_ids:
                    continue
                seen_tidal_ids.add(t.tidal_id)

                track_id = _stable_track_id(t.tidal_id)
                if track_id in excluded:
                    continue

                track = Track(
                    id=track_id,
                    title=t.title,
                    artist=t.artist,
                    album=t.album,
                    duration_ms=t.duration_ms,
                    genre_primary="Unknown",
                    tidal_id=t.tidal_id,
                )
                candidates.append(Suggestion(
                    track=track,
                    engine_name=self.name,
                    engine_score=0.6,
                    explanation=f"From your Tidal mix: {mix['title']}",
                ))

        return candidates[:n]
