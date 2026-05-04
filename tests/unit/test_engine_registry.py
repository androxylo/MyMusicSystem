"""Tests for EngineRegistry — discovery, filtering, health checks."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core.base_engine import BaseEngine, EngineHealth
from core.engine_registry import EngineRegistry


# ---------------------------------------------------------------------------
# Helpers to write a temporary engine directory
# ---------------------------------------------------------------------------

def write_engine(
    engines_dir: Path,
    name: str,
    enabled: bool = True,
    health_status: str = "ok",
    class_name: str | None = None,
) -> Path:
    """Create a minimal engine directory with manifest.yaml and engine.py."""
    eng_dir = engines_dir / name
    eng_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "enabled": enabled,
        "description": f"Test engine: {name}",
        "capabilities": {
            "novelty_bias": 0.5,
            "genre_coverage": "broad",
            "cold_start_friendly": True,
            "data_requirements": [],
            "speed": "fast",
        },
    }
    (eng_dir / "manifest.yaml").write_text(yaml.dump(manifest))

    cls = class_name or f"{name.capitalize()}Engine"
    engine_code = textwrap.dedent(f"""\
        from core.base_engine import BaseEngine, EngineHealth, Suggestion

        class {cls}(BaseEngine):
            def suggest(self, n, context):
                return []

            def health_check(self):
                return EngineHealth(status="{health_status}")
    """)
    (eng_dir / "engine.py").write_text(engine_code)
    return eng_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineRegistryDiscovery:
    def test_discovers_single_enabled_engine(self, tmp_path):
        write_engine(tmp_path, "alpha")
        registry = EngineRegistry(tmp_path, {"alpha": {"enabled": True, "slot_weight": 1.0}})
        registry.load()
        assert len(registry.engines) == 1
        assert registry.engines[0].name == "alpha"

    def test_manifest_disabled_engine_excluded(self, tmp_path):
        write_engine(tmp_path, "disabled_eng", enabled=False)
        registry = EngineRegistry(tmp_path, {"disabled_eng": {"enabled": True, "slot_weight": 1.0}})
        registry.load()
        assert registry.engines == []

    def test_config_disabled_engine_excluded(self, tmp_path):
        write_engine(tmp_path, "cfg_disabled")
        registry = EngineRegistry(tmp_path, {"cfg_disabled": {"enabled": False}})
        registry.load()
        assert registry.engines == []

    def test_unavailable_engine_excluded(self, tmp_path):
        write_engine(tmp_path, "unavail", health_status="unavailable")
        registry = EngineRegistry(tmp_path, {"unavail": {"enabled": True, "slot_weight": 1.0}})
        registry.load()
        assert registry.engines == []

    def test_degraded_engine_included(self, tmp_path):
        write_engine(tmp_path, "degraded", health_status="degraded")
        registry = EngineRegistry(tmp_path, {"degraded": {"enabled": True, "slot_weight": 1.0}})
        registry.load()
        assert len(registry.engines) == 1

    def test_multiple_engines_discovered(self, tmp_path):
        write_engine(tmp_path, "alpha")
        write_engine(tmp_path, "beta")
        write_engine(tmp_path, "gamma")
        config = {e: {"enabled": True, "slot_weight": 1.0} for e in ["alpha", "beta", "gamma"]}
        registry = EngineRegistry(tmp_path, config)
        registry.load()
        assert len(registry.engines) == 3

    def test_missing_engines_dir_returns_empty(self, tmp_path):
        registry = EngineRegistry(tmp_path / "nonexistent", {})
        registry.load()
        assert registry.engines == []

    def test_directory_without_manifest_skipped(self, tmp_path):
        no_manifest = tmp_path / "no_manifest"
        no_manifest.mkdir()
        (no_manifest / "engine.py").write_text("# nothing useful")
        registry = EngineRegistry(tmp_path, {})
        registry.load()
        assert registry.engines == []

    def test_engine_name_set_from_manifest(self, tmp_path):
        write_engine(tmp_path, "myengine")
        registry = EngineRegistry(tmp_path, {"myengine": {"enabled": True}})
        registry.load()
        assert registry.engines[0].name == "myengine"

    def test_capabilities_populated_from_manifest(self, tmp_path):
        write_engine(tmp_path, "cap_test")
        registry = EngineRegistry(tmp_path, {"cap_test": {"enabled": True}})
        registry.load()
        cap = registry.engines[0].capabilities
        assert cap.novelty_bias == 0.5
        assert cap.genre_coverage == "broad"
        assert cap.cold_start_friendly is True
        assert cap.speed == "fast"

    def test_slot_weights_returned(self, tmp_path):
        write_engine(tmp_path, "weighted")
        registry = EngineRegistry(
            tmp_path, {"weighted": {"enabled": True, "slot_weight": 0.75}}
        )
        registry.load()
        weights = registry.get_slot_weights()
        assert weights["weighted"] == pytest.approx(0.75)

    def test_engine_class_load_error_skipped(self, tmp_path):
        eng_dir = tmp_path / "broken"
        eng_dir.mkdir()
        (eng_dir / "manifest.yaml").write_text(
            yaml.dump({"name": "broken", "enabled": True, "capabilities": {}})
        )
        (eng_dir / "engine.py").write_text("raise SyntaxError('bad code'")
        registry = EngineRegistry(tmp_path, {"broken": {"enabled": True}})
        registry.load()  # Must not raise
        assert registry.engines == []

    def test_health_dict_includes_all_checked_engines(self, tmp_path):
        write_engine(tmp_path, "ok_eng", health_status="ok")
        write_engine(tmp_path, "dead_eng", health_status="unavailable")
        registry = EngineRegistry(
            tmp_path,
            {
                "ok_eng": {"enabled": True},
                "dead_eng": {"enabled": True},
            },
        )
        registry.load()
        assert "ok_eng" in registry.health
        assert "dead_eng" in registry.health
        assert registry.health["dead_eng"].status == "unavailable"
