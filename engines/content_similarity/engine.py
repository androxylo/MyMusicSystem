"""
ContentSimilarityEngine — ANN search over Essentia audio feature vectors.

Workflow per session:
  1. Load all tracks with audio_features from the DB (the corpus).
  2. Filter out already-rated and excluded tracks.
  3. Build a weighted preference vector from liked tracks in rating history
     (score >= LIKED_THRESHOLD) that have audio_features.
  4. Build a pynndescent NNDescent index over the filtered corpus.
  5. Query the index for n nearest neighbours by cosine distance.
  6. Return as Suggestion objects.

Cold-start behaviour (no liked tracks with audio features):
  Returns random tracks from the corpus rather than failing the session.

The engine requires:
  - pynndescent package installed (pip install pynndescent)
  - Tracks in the DB with audio_features populated by EssentiaExtractor

health_check returns:
  - "unavailable"  if pynndescent is not importable
  - "unavailable"  if DB is not configured
  - "degraded"     if corpus has < MIN_CORPUS_SIZE tracks with audio_features
  - "ok"           otherwise
"""
from __future__ import annotations

import logging
import math
import random
import uuid
from pathlib import Path

import yaml

from core.base_engine import BaseEngine, EngineHealth, SessionContext, Suggestion, Track

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

# Score threshold: >= this counts as "liked" for the preference vector
_LIKED_THRESHOLD = 7

# Warn (degraded) when corpus is smaller than this
_MIN_CORPUS_SIZE = 20

# Feature vector dimensionality (must match EssentiaExtractor.FEATURE_DIM)
_FEATURE_DIM = 8

# pynndescent: number of neighbours used when building the index graph.
# Higher = better recall, slower build. 30 is a good default for small corpora.
_N_NEIGHBORS = 30

# Detected once at import time so health_check is fast
try:
    from pynndescent import NNDescent as _NNDescent  # type: ignore[import]
    import numpy as _np
    _PYNNDESCENT_AVAILABLE = True
except ImportError:
    _NNDescent = None  # type: ignore[assignment,misc]
    _np = None  # type: ignore[assignment]
    _PYNNDESCENT_AVAILABLE = False


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


class ContentSimilarityEngine(BaseEngine):
    """
    Suggests tracks that sound like the user's liked tracks.

    Requires pynndescent (pip install pynndescent) and audio_features in the DB.
    """

    def __init__(self, db=None, seed: int | None = None) -> None:
        """
        Parameters
        ----------
        db:
            Pre-built Database instance (for testing). If None, self-initializes
            from settings.yaml.
        seed:
            Optional RNG seed for reproducible cold-start shuffling.
        """
        self._db = db if db is not None else self._try_build_db()
        self._rng = random.Random(seed)

    def _try_build_db(self):
        cfg = _load_settings()
        db_path = cfg.get("database_path")
        if not db_path:
            return None
        try:
            from core.db import Database
            migrations_dir = Path(__file__).parent.parent.parent / "database" / "migrations"
            db = Database(db_path, migrations_dir)
            db.connect()
            return db
        except Exception:
            logger.debug("ContentSimilarityEngine: could not connect to DB", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # BaseEngine interface
    # ------------------------------------------------------------------

    def health_check(self) -> EngineHealth:
        if not _PYNNDESCENT_AVAILABLE:
            return EngineHealth(
                status="unavailable",
                message="pynndescent package not installed — run: pip install pynndescent",
            )
        if self._db is None:
            return EngineHealth(
                status="unavailable",
                message="Database not configured in settings.yaml",
            )
        try:
            corpus = self._db.get_tracks_with_audio_features(limit=_MIN_CORPUS_SIZE + 1)
        except Exception:
            return EngineHealth(status="unavailable", message="DB query failed")

        if len(corpus) < _MIN_CORPUS_SIZE:
            return EngineHealth(
                status="degraded",
                message=f"Only {len(corpus)} tracks have audio features; "
                        f"need {_MIN_CORPUS_SIZE} for reliable similarity search",
            )
        return EngineHealth(status="ok")

    def suggest(self, n: int, context: SessionContext) -> list[Suggestion]:
        if not _PYNNDESCENT_AVAILABLE or self._db is None:
            return []
        try:
            return self._run(n, context)
        except Exception:
            logger.exception("ContentSimilarityEngine.suggest() failed")
            return []

    def on_session_complete(self, ratings) -> None:
        pass  # index is rebuilt from DB each session — no separate state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, n: int, context: SessionContext) -> list[Suggestion]:
        rated_ids = {rt.track.id for rt in context.rated_tracks}
        excluded = context.excluded_track_ids | rated_ids

        # Load corpus: tracks with audio features, excluding already-rated
        all_tracks = self._db.get_tracks_with_audio_features(limit=5000)
        corpus = [t for t in all_tracks if t.id not in excluded]

        if not corpus:
            return []

        pref_vector = self._compute_preference_vector(context.rated_tracks)

        if pref_vector is None:
            # Cold start: return random sample from corpus
            pool = list(corpus)
            self._rng.shuffle(pool)
            return [
                self._make_suggestion(t, 0.5, "Chosen from tracks with audio analysis (exploring)")
                for t in pool[:n]
            ]

        return self._ann_search(corpus, pref_vector, n)

    def _compute_preference_vector(
        self, rated_tracks: list
    ) -> list[float] | None:
        """
        Weighted average of feature vectors from liked tracks.
        Weight = score / 10.0 so higher-rated tracks pull harder.
        Returns None if no liked tracks have audio features.
        """
        liked = [
            rt for rt in rated_tracks
            if rt.score >= _LIKED_THRESHOLD
            and rt.track.audio_features
            and rt.track.audio_features.get("features")
        ]
        if not liked:
            return None

        pref = [0.0] * _FEATURE_DIM
        total_weight = 0.0
        for rt in liked:
            w = rt.score / 10.0
            for i, v in enumerate(rt.track.audio_features["features"][:_FEATURE_DIM]):
                pref[i] += v * w
            total_weight += w

        if total_weight == 0.0:
            return None
        return [v / total_weight for v in pref]

    def _ann_search(
        self, corpus: list[Track], pref_vector: list[float], n: int
    ) -> list[Suggestion]:
        """Build pynndescent index and query for nearest neighbours."""
        valid_corpus: list[Track] = []
        vectors: list[list[float]] = []

        for track in corpus:
            features = (track.audio_features or {}).get("features", [])
            if len(features) < _FEATURE_DIM:
                continue
            vectors.append(features[:_FEATURE_DIM])
            valid_corpus.append(track)

        if not valid_corpus:
            return []

        data = _np.array(vectors, dtype=_np.float32)
        query = _np.array([pref_vector], dtype=_np.float32)

        # pynndescent requires n_neighbors < n_samples
        n_neighbors = min(_N_NEIGHBORS, len(valid_corpus) - 1)
        if n_neighbors < 1:
            return []

        n_query = min(n * 3, len(valid_corpus))

        index = _NNDescent(data, metric="cosine", n_neighbors=n_neighbors, random_state=42)
        index.prepare()
        indices, distances = index.query(query, k=n_query)

        results: list[Suggestion] = []
        for i, dist in zip(indices[0], distances[0]):
            if len(results) >= n:
                break
            track = valid_corpus[i]
            # Cosine distance ∈ [0, 2] → similarity ∈ [0, 1]
            similarity = max(0.0, 1.0 - dist / 2.0)
            results.append(
                self._make_suggestion(
                    track,
                    similarity,
                    f"Sounds similar to your liked tracks "
                    f"(audio similarity: {similarity:.0%})",
                )
            )

        return results

    def _make_suggestion(self, track: Track, score: float, explanation: str) -> Suggestion:
        return Suggestion(
            track=track,
            engine_name=self.name,
            engine_score=score,
            explanation=explanation,
        )
