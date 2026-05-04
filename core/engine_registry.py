"""
Engine registry — discovers, loads, and health-checks all engines.

Engines live in engines/<name>/ directories. Each must have:
  - manifest.yaml   (metadata + capabilities)
  - engine.py       (a class that extends BaseEngine; auto-discovered)

The registry filters by:
  1. manifest enabled: true
  2. config/engines.yaml enabled: true (per-engine override)
  3. health_check() returns status != 'unavailable'

Engines with status 'degraded' are included but logged.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from core.base_engine import BaseEngine, EngineCapabilities, EngineHealth

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open() as f:
        return yaml.safe_load(f) or {}


def _capabilities_from_manifest(cap_dict: dict) -> EngineCapabilities:
    return EngineCapabilities(
        novelty_bias=float(cap_dict.get("novelty_bias", 0.5)),
        genre_coverage=cap_dict.get("genre_coverage", "broad"),
        cold_start_friendly=bool(cap_dict.get("cold_start_friendly", False)),
        data_requirements=list(cap_dict.get("data_requirements", [])),
        speed=cap_dict.get("speed", "fast"),
        description=cap_dict.get("description", ""),
    )


def _load_engine_class(engine_dir: Path) -> type[BaseEngine] | None:
    """Import engine.py from the given directory and return the first BaseEngine subclass."""
    engine_file = engine_dir / "engine.py"
    if not engine_file.exists():
        logger.warning(f"No engine.py found in {engine_dir}")
        return None

    spec = importlib.util.spec_from_file_location(
        f"engines.{engine_dir.name}.engine", engine_file
    )
    if spec is None or spec.loader is None:
        logger.warning(f"Cannot create module spec for {engine_file}")
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        logger.exception(f"Error loading engine module from {engine_file}")
        return None

    # Find the first concrete BaseEngine subclass defined in this module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        try:
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseEngine)
                and attr is not BaseEngine
                and not getattr(attr, "__abstractmethods__", None)
            ):
                return attr
        except TypeError:
            continue

    logger.warning(f"No concrete BaseEngine subclass found in {engine_file}")
    return None


class EngineRegistry:
    """
    Discovers and instantiates all enabled, healthy engines.

    Usage:
        registry = EngineRegistry(engines_dir, engines_config)
        registry.load()
        engines = registry.engines  # list[BaseEngine]
    """

    def __init__(
        self,
        engines_dir: Path,
        engines_config: dict,  # content of config/engines.yaml 'engines' key
    ) -> None:
        self._engines_dir = Path(engines_dir)
        self._engines_config = engines_config  # {name: {enabled, slot_weight, ...}}
        self._engines: list[BaseEngine] = []
        self._health: dict[str, EngineHealth] = {}

    def load(self) -> None:
        """Discover, instantiate, and health-check all engines. Populates self.engines."""
        self._engines = []
        self._health = {}

        if not self._engines_dir.exists():
            logger.warning(f"Engines directory not found: {self._engines_dir}")
            return

        for engine_dir in sorted(self._engines_dir.iterdir()):
            if not engine_dir.is_dir():
                continue

            manifest_path = engine_dir / "manifest.yaml"
            if not manifest_path.exists():
                continue

            manifest = _load_manifest(manifest_path)
            name = manifest.get("name", engine_dir.name)

            # Filter 1: manifest-level enabled flag
            if not manifest.get("enabled", True):
                logger.debug(f"Engine {name!r} disabled in manifest")
                continue

            # Filter 2: engines.yaml config enabled flag
            cfg = self._engines_config.get(name, {})
            if not cfg.get("enabled", True):
                logger.debug(f"Engine {name!r} disabled in engines.yaml")
                continue

            # Load the class
            engine_cls = _load_engine_class(engine_dir)
            if engine_cls is None:
                continue

            # Instantiate and inject metadata
            try:
                engine = engine_cls()
            except Exception:
                logger.exception(f"Error instantiating engine {name!r}")
                continue

            engine.name = name
            engine.capabilities = _capabilities_from_manifest(
                manifest.get("capabilities", {})
            )

            # Health check
            try:
                health = engine.health_check()
            except Exception as exc:
                health = EngineHealth(status="unavailable", message=str(exc))

            self._health[name] = health

            if health.status == "unavailable":
                logger.warning(
                    f"Engine {name!r} is unavailable: {health.message} — excluded from session"
                )
                continue

            if health.status == "degraded":
                logger.warning(
                    f"Engine {name!r} is degraded: {health.message} — included with warning"
                )

            self._engines.append(engine)
            logger.info(f"Engine {name!r} loaded (status={health.status})")

    @property
    def engines(self) -> list[BaseEngine]:
        return list(self._engines)

    @property
    def health(self) -> dict[str, EngineHealth]:
        return dict(self._health)

    def get_slot_weights(self) -> dict[str, float]:
        """
        Return {engine_name: slot_weight} for all loaded engines.
        Falls back to equal weight if an engine has no config entry.
        """
        weights: dict[str, float] = {}
        for engine in self._engines:
            cfg = self._engines_config.get(engine.name, {})
            weights[engine.name] = float(cfg.get("slot_weight", 1.0))
        return weights
