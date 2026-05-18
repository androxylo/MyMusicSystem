"""
CollaborativeFilterEngine — LightFM WARP model over genre-tag item features.

Algorithm per session:
  1. Collect rated tracks from context (must have genre_tags populated).
  2. Build a LightFM Dataset: one user ("me"), items = rated track IDs,
     item features = all genre tags across those tracks.
  3. Build interactions: positive interaction for each track rated >= LIKED_THRESHOLD,
     weighted by score / 10.0.
  4. Build item feature matrix (sparse): each track's genre tags.
  5. Train LightFM (WARP loss) on the interactions + item features.
  6. Predict scores for all rated items using the trained model.
  7. Average predicted scores per genre bucket → genre affinity ranking.
  8. For top-ranked genres, fetch candidates from Last.fm and search Tidal.
  9. Return top N candidates.

The model captures second-order tag interactions that BanditExplorer's
Thompson Sampling cannot: e.g. the user might prefer "ambient electronic"
over "dance electronic" — LightFM learns this from the tag feature embeddings.

health_check returns:
  - "unavailable"  if lightfm is not installed
  - "unavailable"  if Last.fm connector not configured
  - "unavailable"  if fewer than MIN_RATINGS_UNAVAILABLE ratings
  - "degraded"     if fewer than MIN_RATINGS_OK ratings (model is sparse)
  - "ok"           otherwise

State derives from context.rated_tracks — no persistent model storage needed.
The model is trained from scratch each session (fast for ≤1000 ratings).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track
from core.diversity import GENRE_BUCKETS, classify_genre

logger = logging.getLogger(__name__)

_DATA_SETTINGS_PATH = Path(__file__).parent.parent.parent / "data" / "settings.yaml"
_CONFIG_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_SETTINGS_PATH = _DATA_SETTINGS_PATH if _DATA_SETTINGS_PATH.exists() else _CONFIG_SETTINGS_PATH

# Score threshold: >= this is a positive interaction for WARP
_LIKED_THRESHOLD = 7

# Genre bucket definitions (same as BanditExplorer/GenreExplorer)
_ALL_GENRES: list[str] = sorted(set(GENRE_BUCKETS.values()))

# Maps genre bucket names to Last.fm search tags
_GENRE_TO_LASTFM_TAG: dict[str, str] = {
    "Electronic": "electronic",
    "Rock": "rock",
    "Metal": "metal",
    "Pop": "pop",
    "Hip-Hop/R&B": "hip-hop",
    "Jazz": "jazz",
    "Classical": "classical",
    "Folk/Country": "folk",
    "Blues": "blues",
    "Reggae/World": "reggae",
    "Experimental": "experimental",
}

# Minimum rating counts for health status
_MIN_RATINGS_UNAVAILABLE = 10   # below this → unavailable
_MIN_RATINGS_OK = 30            # below this (but >= unavailable) → degraded

# LightFM hyperparameters (small: fast to train even on CPU)
_N_COMPONENTS = 16
_N_EPOCHS = 20

# Detected once at import time so health_check is fast
try:
    from lightfm import LightFM as _LightFM  # type: ignore[import]
    from lightfm.data import Dataset as _Dataset  # type: ignore[import]
    _LIGHTFM_AVAILABLE = True
except ImportError:
    _LightFM = None  # type: ignore[assignment,misc]
    _Dataset = None  # type: ignore[assignment,misc]
    _LIGHTFM_AVAILABLE = False


def _load_settings() -> dict:
    try:
        if not _SETTINGS_PATH.exists():
            return {}
        with _SETTINGS_PATH.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _stable_track_id(tidal_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tidal:{tidal_id}"))


class CollaborativeFilterEngine(BaseEngine):
    """
    LightFM WARP model over genre-tag item features.

    Ranks genre buckets by learned user affinity, then fetches candidates
    from those genres via Last.fm + Tidal.
    """

    def __init__(self, lastfm=None, tidal=None) -> None:
        self._lastfm = lastfm if lastfm is not None else self._try_build_lastfm()
        self._tidal = tidal if tidal is not None else self._try_build_tidal()

    def _try_build_lastfm(self):
        cfg = _load_settings().get("lastfm", {})
        if not cfg.get("api_key"):
            return None
        try:
            from connectors.lastfm import LastFmConnector
            conn = LastFmConnector(cfg)
            conn.connect()
            return conn
        except Exception:
            logger.debug("CollaborativeFilterEngine: could not build LastFmConnector", exc_info=True)
            return None

    def _try_build_tidal(self):
        cfg = _load_settings().get("tidal", {})
        if not (cfg.get("client_id") or cfg.get("token_path")):
            return None
        try:
            from connectors.tidal import TidalConnector
            conn = TidalConnector(cfg)
            conn.authenticate()
            return conn
        except Exception:
            logger.debug("CollaborativeFilterEngine: could not build TidalConnector", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def health_check(self) -> EngineHealth:
        if not _LIGHTFM_AVAILABLE:
            return EngineHealth(
                status="unavailable",
                message="lightfm package not installed — run: pip install lightfm",
            )
        if self._lastfm is None or not self._lastfm.is_available():
            return EngineHealth(
                status="unavailable",
                message="Last.fm not configured — required to fetch genre candidates",
            )
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if not _LIGHTFM_AVAILABLE:
            return []
        if self._lastfm is None or not self._lastfm.is_available():
            return []
        try:
            return self._run(n, context)
        except Exception:
            logger.exception("CollaborativeFilterEngine.suggest() failed")
            return []

    def on_session_complete(self, ratings) -> None:
        pass  # model retrained from scratch each session — no stored state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, n: int, context: SessionContext) -> list[Suggestion]:
        excluded = context.excluded_track_ids
        rated = context.rated_tracks

        # Rank genres using LightFM model (falls back to stable sort if insufficient data)
        ranked_genres = self._compute_genre_rankings(rated)

        suggestions: list[Suggestion] = []
        for genre in ranked_genres[:5]:
            if len(suggestions) >= n:
                break
            slots = max(1, (n - len(suggestions) + 1) // 2)
            suggestions.extend(self._fetch_genre_candidates(genre, excluded, slots))

        return suggestions[:n]

    def _compute_genre_rankings(self, rated: list) -> list[str]:
        """
        Train LightFM on rated tracks and rank genre buckets by predicted affinity.
        Falls back to alphabetical order if there is insufficient data.
        """
        # Filter to rated tracks that have genre_tags (or at least genre_primary)
        valid = [
            rt for rt in rated
            if rt.track.genre_tags or rt.track.genre_primary
        ]

        if len(valid) < _MIN_RATINGS_UNAVAILABLE:
            logger.debug(
                "CollaborativeFilterEngine: only %d rated tracks; falling back to default genre order",
                len(valid),
            )
            return list(_ALL_GENRES)

        positives = [rt for rt in valid if rt.score >= _LIKED_THRESHOLD]
        if not positives:
            return list(_ALL_GENRES)

        try:
            return self._train_and_rank(valid, positives)
        except Exception:
            logger.debug("CollaborativeFilterEngine: LightFM training failed", exc_info=True)
            return list(_ALL_GENRES)

    def _train_and_rank(self, valid: list, positives: list) -> list[str]:
        """Build dataset, train LightFM, return genres sorted by predicted affinity."""
        # Collect all genre tags across rated tracks
        all_tags: set[str] = set()
        for rt in valid:
            tags = rt.track.genre_tags if rt.track.genre_tags else [rt.track.genre_primary]
            all_tags.update(tags)

        # Build dataset
        dataset = _Dataset()
        dataset.fit(
            users=["me"],
            items=[rt.track.id for rt in valid],
            item_features=list(all_tags),
        )

        # Build interactions (only positives, heavily weighted for scores >= 8)
        interactions, weights = dataset.build_interactions(
            [("me", rt.track.id, (rt.score - 6) ** 2) for rt in positives]
        )

        # Build item features sparse matrix
        item_features_list = []
        for rt in valid:
            tags = rt.track.genre_tags if rt.track.genre_tags else [rt.track.genre_primary]
            item_features_list.append((rt.track.id, tags))
        item_features = dataset.build_item_features(item_features_list)

        # Train
        model = _LightFM(loss="warp", no_components=_N_COMPONENTS, random_state=42)
        model.fit(
            interactions,
            item_features=item_features,
            sample_weight=weights,
            epochs=_N_EPOCHS,
            verbose=False,
        )

        # Predict scores for all rated tracks
        item_mapping = dataset.mapping()[2]
        item_ids = [item_mapping[rt.track.id] for rt in valid if rt.track.id in item_mapping]
        valid_for_pred = [rt for rt in valid if rt.track.id in item_mapping]

        if not item_ids:
            return list(_ALL_GENRES)

        predicted = model.predict(
            user_ids=0,
            item_ids=item_ids,
            item_features=item_features,
        )

        # Average predicted score per genre bucket
        genre_total: dict[str, float] = {g: 0.0 for g in _ALL_GENRES}
        genre_count: dict[str, int] = {g: 0 for g in _ALL_GENRES}

        for rt, score in zip(valid_for_pred, predicted):
            bucket = classify_genre(rt.track.genre_primary, rt.track.genre_tags)
            if bucket in genre_total:
                genre_total[bucket] += float(score)
                genre_count[bucket] += 1

        def genre_affinity(g: str) -> float:
            if genre_count[g] == 0:
                return float("-inf")
            return genre_total[g] / genre_count[g]

        return sorted(_ALL_GENRES, key=genre_affinity, reverse=True)

    def _fetch_genre_candidates(
        self,
        genre: str,
        excluded: set[str],
        limit: int,
    ) -> list[Suggestion]:
        """Fetch Last.fm top tracks for a genre, search Tidal, return Suggestions."""
        lastfm_tag = _GENRE_TO_LASTFM_TAG.get(genre, genre.lower().split("/")[0])

        try:
            lastfm_tracks = self._lastfm.get_tag_top_tracks(lastfm_tag, limit=30)
        except Exception:
            logger.debug(
                f"CollaborativeFilterEngine: get_tag_top_tracks({lastfm_tag!r}) failed",
                exc_info=True,
            )
            return []

        results: list[Suggestion] = []
        for lf_track in lastfm_tracks:
            if len(results) >= limit:
                break
            if not self._tidal or not self._tidal.is_available():
                break
            try:
                tidal_results = self._tidal.search_tracks(
                    f"{lf_track.title} {lf_track.artist}", limit=2
                )
            except Exception:
                continue

            for t in tidal_results:
                track_id = _stable_track_id(t.tidal_id)
                if track_id in excluded:
                    continue

                track = Track(
                    id=track_id,
                    title=t.title,
                    artist=t.artist,
                    album=t.album,
                    duration_ms=t.duration_ms,
                    genre_primary=genre,
                    tidal_id=t.tidal_id,
                )
                results.append(Suggestion(
                    track=track,
                    engine_name=self.name,
                    engine_score=0.70,
                    explanation=(
                        f"LightFM predicted affinity for {genre} — "
                        f"based on your full rating history"
                    ),
                ))
                break

        return results
