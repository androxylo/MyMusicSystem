"""
Contract tests — every engine must satisfy these invariants.

These tests are parametric: adding a new engine directory automatically
adds it to the test suite. External APIs are mocked via conftest fixtures.
"""
from __future__ import annotations

import textwrap
import time
import uuid
from pathlib import Path

import pytest
import yaml

from core.base_engine import (
    BaseEngine,
    EngineCapabilities,
    EngineHealth,
    SessionConfig,
    SessionContext,
    Suggestion,
    Track,
)
from core.engine_registry import EngineRegistry

ENGINES_DIR = Path(__file__).parent.parent.parent / "engines"
ENGINES_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "engines.yaml"

# How long a single suggest() call may take in tests (seconds)
SUGGEST_TIMEOUT_S = 5.0


def _load_engines_config() -> dict:
    if ENGINES_CONFIG_PATH.exists():
        import yaml
        with ENGINES_CONFIG_PATH.open() as f:
            data = yaml.safe_load(f) or {}
        return data.get("engines", {})
    return {}


def _discover_engine_names() -> list[str]:
    if not ENGINES_DIR.exists():
        return []
    return [
        d.name
        for d in ENGINES_DIR.iterdir()
        if d.is_dir() and (d / "manifest.yaml").exists() and (d / "engine.py").exists()
    ]


def _load_engine(name: str) -> BaseEngine | None:
    """Load a single engine by name, returning None if it can't be loaded."""
    config = _load_engines_config()
    # Override health to avoid hitting external APIs during contract tests
    registry = EngineRegistry(ENGINES_DIR, config)
    registry.load()
    for engine in registry.engines:
        if engine.name == name:
            return engine
    return None


def _minimal_context() -> SessionContext:
    return SessionContext(
        rated_tracks=[],
        recent_sessions=[],
        session_config=SessionConfig(),
        excluded_track_ids=set(),
    )


# ---------------------------------------------------------------------------
# Parametric engine discovery
# ---------------------------------------------------------------------------

engine_names = _discover_engine_names()

# Skip contract tests when no engines exist yet (Phase 1)
contract_skip = pytest.mark.skipif(
    len(engine_names) == 0,
    reason="No engines found in engines/ directory — contract tests will run in Phase 3+",
)


@contract_skip
@pytest.mark.parametrize("engine_name", engine_names)
class TestEngineContract:
    """All engines must pass every test in this class."""

    @pytest.fixture(autouse=True)
    def load_engine(self, engine_name):
        engine = _load_engine(engine_name)
        if engine is None:
            pytest.skip(f"Engine {engine_name!r} could not be loaded (unavailable?)")
        self.engine = engine
        self.ctx = _minimal_context()

    def test_suggest_returns_list(self):
        result = self.engine.suggest(5, self.ctx)
        assert isinstance(result, list), "suggest() must return a list"

    def test_suggest_does_not_raise(self):
        # Any exception from suggest() is a contract violation
        try:
            self.engine.suggest(5, self.ctx)
        except Exception as exc:
            pytest.fail(f"suggest() raised an exception: {exc!r}")

    def test_suggest_results_are_suggestion_objects(self):
        results = self.engine.suggest(5, self.ctx)
        for r in results:
            assert isinstance(r, Suggestion), f"Got {type(r)!r} instead of Suggestion"

    def test_suggestion_fields_populated(self):
        results = self.engine.suggest(3, self.ctx)
        for s in results:
            assert s.track is not None
            assert isinstance(s.track, Track)
            assert s.track.id, "track.id must be non-empty"
            assert s.track.title, "track.title must be non-empty"
            assert s.track.artist, "track.artist must be non-empty"
            assert s.engine_name, "engine_name must be non-empty"
            assert 0.0 <= s.engine_score <= 1.0, f"engine_score {s.engine_score} out of [0, 1]"
            assert isinstance(s.explanation, str), "explanation must be a string"

    def test_suggest_respects_excluded_ids(self):
        # First call to get some track IDs to exclude
        results = self.engine.suggest(5, self.ctx)
        if not results:
            pytest.skip("Engine returned no results — cannot test exclusion")

        exclude = {results[0].track.id}
        ctx2 = SessionContext(
            rated_tracks=self.ctx.rated_tracks,
            recent_sessions=self.ctx.recent_sessions,
            session_config=self.ctx.session_config,
            excluded_track_ids=exclude,
        )
        results2 = self.engine.suggest(5, ctx2)
        returned_ids = {s.track.id for s in results2}
        overlap = returned_ids & exclude
        assert not overlap, f"Engine returned excluded track IDs: {overlap}"

    def test_suggest_completes_within_timeout(self):
        start = time.monotonic()
        self.engine.suggest(5, self.ctx)
        elapsed = time.monotonic() - start
        assert elapsed < SUGGEST_TIMEOUT_S, (
            f"suggest() took {elapsed:.1f}s — exceeds {SUGGEST_TIMEOUT_S}s timeout"
        )

    def test_health_check_returns_valid_status(self):
        health = self.engine.health_check()
        assert isinstance(health, EngineHealth)
        assert health.status in ("ok", "degraded", "unavailable"), (
            f"health_check() returned invalid status: {health.status!r}"
        )

    def test_on_session_complete_does_not_raise_with_empty_list(self):
        try:
            self.engine.on_session_complete([])
        except Exception as exc:
            pytest.fail(f"on_session_complete([]) raised: {exc!r}")

    def test_capabilities_structure(self):
        cap = self.engine.capabilities
        assert isinstance(cap, EngineCapabilities)
        assert 0.0 <= cap.novelty_bias <= 1.0
        assert cap.genre_coverage in ("broad", "narrow", "targeted")
        assert isinstance(cap.cold_start_friendly, bool)
        assert isinstance(cap.data_requirements, list)
        assert cap.speed in ("fast", "slow")
        assert isinstance(cap.description, str)
