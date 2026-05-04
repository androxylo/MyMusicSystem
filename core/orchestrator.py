"""
Orchestrator — slot allocation and suggestion collection.

Coordinates all loaded engines: assigns each a quota (oversampled),
collects their candidates, then delegates to DiversityEnforcer for
the final selection.
"""
from __future__ import annotations

import logging
import math

from core.base_engine import BaseEngine, SessionContext, Suggestion
from core.diversity import DiversityEnforcer

logger = logging.getLogger(__name__)

_DEFAULT_OVERSAMPLING = 3
_DEFAULT_N_FINAL = 10


class Orchestrator:
    """
    Drives the suggestion-collection and blending pipeline.

    Requires:
      - A list of loaded, healthy engines (from EngineRegistry)
      - A DiversityEnforcer instance
      - Slot weights per engine (from EngineRegistry.get_slot_weights())
    """

    def __init__(
        self,
        engines: list[BaseEngine],
        diversity_enforcer: DiversityEnforcer,
        slot_weights: dict[str, float],
        oversampling_factor: int = _DEFAULT_OVERSAMPLING,
        n_final: int = _DEFAULT_N_FINAL,
    ) -> None:
        self._engines = engines
        self._diversity = diversity_enforcer
        self._slot_weights = slot_weights
        self._oversampling_factor = oversampling_factor
        self._n_final = n_final

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_suggestions(self, context: SessionContext) -> list[Suggestion]:
        """
        Run the full pipeline: allocate → oversample → diversity pass.

        Returns up to self._n_final Suggestion objects.
        """
        if not self._engines:
            logger.error("No engines available — cannot generate suggestions")
            return []

        allocation = self._allocate_slots()
        pool = self._collect_suggestions(allocation, context)

        logger.info(
            f"Pool size after collection: {len(pool)} candidates from "
            f"{len({s.engine_name for s in pool})} engines"
        )

        final = self._diversity.select(
            pool=pool,
            n=self._n_final,
            rated_tracks=context.rated_tracks,
            recent_sessions=context.recent_sessions,
            genre_session_history=self._genre_session_history(context),
        )

        logger.info(f"Final selection: {len(final)} tracks")
        return final

    def get_allocation(self) -> dict[str, int]:
        """Return the slot allocation map without running suggestions. Used for session metadata."""
        return self._allocate_slots()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_slots(self) -> dict[str, int]:
        """
        Normalize slot weights across loaded engines and assign integer slot counts.

        Each engine gets at least 1 slot. Total may exceed n_final (intentional —
        the diversity pass selects the best n_final).
        """
        if not self._engines:
            return {}

        total_weight = sum(
            self._slot_weights.get(e.name, 1.0) for e in self._engines
        )
        if total_weight == 0:
            total_weight = len(self._engines)

        allocation: dict[str, int] = {}
        for engine in self._engines:
            weight = self._slot_weights.get(engine.name, 1.0)
            slots = max(1, math.ceil(weight / total_weight * self._n_final))
            allocation[engine.name] = slots

        return allocation

    def _collect_suggestions(
        self,
        allocation: dict[str, int],
        context: SessionContext,
    ) -> list[Suggestion]:
        """
        Ask each engine for oversampled suggestions and return the merged pool.
        An engine that raises is skipped; it must not propagate exceptions.
        """
        pool: list[Suggestion] = []

        for engine in self._engines:
            n_slots = allocation.get(engine.name, 1)
            n_ask = n_slots * self._oversampling_factor

            try:
                suggestions = engine.suggest(n_ask, context)
            except Exception:
                logger.exception(
                    f"Engine {engine.name!r} raised unexpectedly in suggest() — skipping"
                )
                suggestions = []

            if suggestions:
                logger.debug(
                    f"Engine {engine.name!r}: got {len(suggestions)} suggestions "
                    f"(asked {n_ask})"
                )
            else:
                logger.warning(f"Engine {engine.name!r}: returned no suggestions")

            pool.extend(suggestions)

        return pool

    @staticmethod
    def _genre_session_history(context: SessionContext) -> list[dict]:
        """
        Build a lightweight genre session history from context.rated_tracks.

        In a full system this comes from db.get_genre_session_history(); the
        orchestrator builds it here from the already-loaded context so it
        doesn't need a DB reference.
        """
        # Group rated tracks by session_id and genre_primary
        session_genre: dict[tuple[str, str], list[int]] = {}
        for rt in context.rated_tracks:
            key = (rt.session_id, rt.track.genre_primary)
            session_genre.setdefault(key, []).append(rt.score)

        history: list[dict] = []
        for (session_id, genre), scores in session_genre.items():
            history.append(
                {
                    "genre": genre,
                    "session_id": session_id,
                    "track_count": len(scores),
                    "avg_rating": sum(scores) / len(scores),
                }
            )
        return history
