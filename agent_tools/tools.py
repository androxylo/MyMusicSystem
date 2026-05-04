"""
Agent tool interface — the functions Claude calls to drive music sessions.

All public functions in this module are designed to be exposed as MCP tools.
They share a single initialized system context (lazy singleton) so connectors
and engines are not rebuilt on every call.

Typical session flow:
    1. url = start_session()["candidate_playlist_url"]
       → Open this in Tidal. Listen to all 10 tracks.
    2. For each track you've listened to:
           rate_track(session_id, track_id, score)   # score 1–10
    3. complete_session(session_id)
       → Returns summary + genre stats + newly curated tracks.
    4. get_playlists()
       → Shows URLs for "Now Rating", "Liked — All", and each genre playlist.

Other tools:
    get_stats()             — Overall stats, engine performance, saturation warnings
    list_engines()          — Which engines are active and healthy
    get_track_info()        — Full metadata for a specific track
    set_curated_threshold() — Change the minimum score for curated playlists
    reconcile_playlists()   — Fix Tidal playlist state from DB ratings
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
_ENGINES_CONFIG_PATH = Path(__file__).parent.parent / "config" / "engines.yaml"
_ENGINES_DIR = Path(__file__).parent.parent / "engines"
_MIGRATIONS_DIR = Path(__file__).parent.parent / "database" / "migrations"


# ---------------------------------------------------------------------------
# System context (lazy singleton)
# ---------------------------------------------------------------------------

class _SystemContext:
    __slots__ = (
        "db",
        "session_manager",
        "playlist_manager",
        "registry",
        "settings",
    )

    def __init__(self, db, session_manager, playlist_manager, registry, settings):
        self.db = db
        self.session_manager = session_manager
        self.playlist_manager = playlist_manager
        self.registry = registry
        self.settings = settings


_ctx: _SystemContext | None = None


def _load_settings() -> dict:
    try:
        if not _SETTINGS_PATH.exists():
            logger.warning(f"settings.yaml not found at {_SETTINGS_PATH}")
            return {}
        with _SETTINGS_PATH.open() as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error(f"Failed to load settings.yaml: {exc}")
        return {}


def _load_engines_config() -> dict:
    try:
        if not _ENGINES_CONFIG_PATH.exists():
            return {}
        with _ENGINES_CONFIG_PATH.open() as f:
            data = yaml.safe_load(f) or {}
        return data.get("engines", {})
    except Exception as exc:
        logger.error(f"Failed to load engines.yaml: {exc}")
        return {}


def _init_system() -> _SystemContext:
    """
    Initialize all system components from config files.
    Called once on first tool invocation.
    """
    settings = _load_settings()
    engines_cfg = _load_engines_config()

    # Database
    from core.db import Database
    db_path = settings.get("database_path", "./data/music.db")
    db = Database(db_path, _MIGRATIONS_DIR)
    db.connect()
    logger.info(f"Database connected: {db_path}")

    # Connectors
    tidal_connector = _build_tidal(settings.get("tidal", {}))
    lastfm_connector = _build_lastfm(settings.get("lastfm", {}))

    # Engine registry (engines self-initialize from settings.yaml)
    from core.engine_registry import EngineRegistry
    registry = EngineRegistry(_ENGINES_DIR, engines_cfg)
    registry.load()
    engines = registry.engines
    if not engines:
        logger.warning("No engines loaded — sessions will produce no suggestions")

    # Orchestrator
    from core.orchestrator import Orchestrator
    from core.diversity import DiversityEnforcer
    div_cfg = settings.get("diversity", {})
    enforcer = DiversityEnforcer(
        recently_heard_days=div_cfg.get("recently_heard_days", 30),
        saturation_sessions=div_cfg.get("saturation_sessions", 3),
        saturation_min_avg_rating=div_cfg.get("saturation_min_avg_rating", 7.0),
        min_engines_represented=div_cfg.get("min_engines_represented", 3),
    )
    slot_weights = registry.get_slot_weights() if engines else {}
    orchestrator = Orchestrator(
        engines=engines,
        diversity_enforcer=enforcer,
        slot_weights=slot_weights,
        oversampling_factor=div_cfg.get("oversampling_factor", 3),
        n_final=10,
    )

    # Playlist manager
    playlist_manager = None
    if tidal_connector is not None:
        from core.playlist_manager import PlaylistManager, PlaylistConfig
        pl_cfg = settings.get("playlists", {})
        pm_config = PlaylistConfig(
            curated_threshold=pl_cfg.get("curated_threshold", 7),
            candidate_name=pl_cfg.get("candidate_name", "Now Rating"),
            curated_master_name=pl_cfg.get("curated_master_name", "Liked \u2014 All"),
            curated_genre_prefix=pl_cfg.get("curated_genre_prefix", "Liked \u2014 "),
        )
        playlist_manager = PlaylistManager(tidal=tidal_connector, db=db, config=pm_config)

    # Session manager
    from core.session import SessionManager
    pl_settings = settings.get("playlists", {})
    session_manager = SessionManager(
        db=db,
        orchestrator=orchestrator,
        engines=engines,
        curated_threshold=pl_settings.get("curated_threshold", 7),
        playlist_manager=playlist_manager,
    )

    logger.info(
        f"System initialized: {len(engines)} engine(s), "
        f"Tidal={'ok' if tidal_connector else 'unavailable'}, "
        f"Last.fm={'ok' if lastfm_connector else 'unavailable'}"
    )
    return _SystemContext(
        db=db,
        session_manager=session_manager,
        playlist_manager=playlist_manager,
        registry=registry,
        settings=settings,
    )


def _build_tidal(cfg: dict):
    if not (cfg.get("client_id") or cfg.get("token_path")):
        return None
    try:
        from connectors.tidal import TidalConnector
        connector = TidalConnector(cfg)
        connector.authenticate()
        return connector
    except Exception as exc:
        logger.warning(f"Tidal connector unavailable: {exc}")
        return None


def _build_lastfm(cfg: dict):
    if not cfg.get("api_key"):
        return None
    try:
        from connectors.lastfm import LastFmConnector
        connector = LastFmConnector(cfg)
        connector.connect()
        return connector
    except Exception as exc:
        logger.warning(f"Last.fm connector unavailable: {exc}")
        return None


def _get_ctx() -> _SystemContext:
    global _ctx
    if _ctx is None:
        _ctx = _init_system()
    return _ctx


def reset_context() -> None:
    """Force re-initialization of the system context (useful after config changes)."""
    global _ctx
    if _ctx is not None:
        try:
            _ctx.db.close()
        except Exception:
            pass
    _ctx = None


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def start_session(diversity_mode: str = "strict", notes: str | None = None) -> dict:
    """
    Start a new recommendation session.

    Creates the session, gets 10 suggestions from the engines, and populates
    the "Now Rating" Tidal playlist. Returns a dict with:
      - session_id: use this in subsequent rate_track() and complete_session() calls
      - candidate_playlist_url: open in Tidal to start listening
      - suggestions: list of {track_id, title, artist, album, genre, engine, explanation}
      - engine_allocation: how many slots each engine received

    Args:
        diversity_mode: "strict" (one track per genre) or "relaxed" (fallback allowed)
        notes: optional text attached to the session record
    """
    from core.base_engine import SessionConfig
    ctx = _get_ctx()
    config = SessionConfig(diversity_mode=diversity_mode, notes=notes)
    return ctx.session_manager.start_session(config)


def rate_track(session_id: str, track_id: str, score: int) -> dict:
    """
    Rate a track in the current session.

    Score must be between 1 and 10. Tracks scored >= the curated threshold
    (default: 7) are automatically added to "Liked — All" and the genre playlist.

    The track is removed from the "Now Rating" Tidal playlist immediately.
    DB is always updated first; Tidal failures are logged and do not raise.

    Returns:
        ok: True on success
        ratings_so_far: number of tracks rated in this session so far
        added_to_curated: True if the track was added to curated playlists
    """
    ctx = _get_ctx()
    result = ctx.session_manager.rate_track(session_id, track_id, score)
    return {
        "ok": result.ok,
        "ratings_so_far": result.ratings_so_far,
        "added_to_curated": result.added_to_curated,
    }


def complete_session(session_id: str) -> dict:
    """
    Complete a session and return a summary.

    Clears any remaining unrated tracks from "Now Rating". Notifies engines
    (so bandit/learning engines can update their models). Records engine
    performance metrics.

    Returns:
        session_summary: {avg_rating, top_rated_track, engine_breakdown}
        genre_stats: {genre: avg_rating} for this session
        newly_curated: list of {title, artist, genre} for tracks added to curated playlists
        next_session_preview: brief readiness message
    """
    ctx = _get_ctx()
    summary = ctx.session_manager.complete_session(session_id)

    top = None
    if summary.top_rated_track:
        t = summary.top_rated_track
        top = {
            "title": t.track.title,
            "artist": t.track.artist,
            "score": t.score,
            "genre": t.track.genre_primary,
        }

    return {
        "session_summary": {
            "session_id": summary.session_id,
            "avg_rating": summary.avg_rating,
            "top_rated_track": top,
            "engine_breakdown": summary.engine_breakdown,
        },
        "genre_stats": summary.genre_stats,
        "newly_curated": summary.newly_curated,
        "newly_curated_count": summary.newly_curated_count,
        "next_session_preview": "All engines ready for the next session.",
    }


def get_stats(last_n_sessions: int = 10) -> dict:
    """
    Return overall recommendation system statistics.

    Returns:
        total_ratings: all-time count of rated tracks
        recent_avg_rating: average score across the last N sessions
        favorite_genres: list of {genre, avg_rating, count} sorted by avg_rating
        engine_performance: per-engine {session_count, avg_tracks_in_final, avg_rating}
        genre_saturation_warnings: genres appearing in many recent sessions with high ratings
    """
    ctx = _get_ctx()
    db = ctx.db

    total = db.get_total_ratings_count()
    genre_avgs = db.get_genre_avg_ratings()
    engine_perf = db.get_engine_performance(last_n_sessions)
    genre_history = db.get_genre_session_history(n_sessions=last_n_sessions)

    # Compute recent avg rating from genre history
    all_scores_recent = [g["avg_rating"] for g in genre_history if g.get("avg_rating")]
    recent_avg = round(sum(all_scores_recent) / len(all_scores_recent), 2) if all_scores_recent else None

    # Saturation warnings: genres in multiple recent sessions with high avg rating
    from collections import defaultdict
    genre_session_counts: dict[str, list[float]] = defaultdict(list)
    for row in genre_history:
        genre_session_counts[row["genre"]].append(row["avg_rating"])

    settings = ctx.settings.get("diversity", {})
    sat_sessions = settings.get("saturation_sessions", 3)
    sat_threshold = settings.get("saturation_min_avg_rating", 7.0)

    saturation_warnings = []
    for genre, avgs in genre_session_counts.items():
        if len(avgs) >= sat_sessions and (sum(avgs) / len(avgs)) >= sat_threshold:
            saturation_warnings.append({
                "genre": genre,
                "sessions_appeared": len(avgs),
                "avg_rating": round(sum(avgs) / len(avgs), 2),
                "note": "High-rated genre appearing frequently — anti-echo enforcement active",
            })

    return {
        "total_ratings": total,
        "recent_avg_rating": recent_avg,
        "favorite_genres": [
            {"genre": r["genre"], "avg_rating": round(r["avg_rating"], 2), "count": r["count"]}
            for r in genre_avgs
        ],
        "engine_performance": [
            {
                "engine": r["engine_name"],
                "sessions": r["session_count"],
                "avg_tracks_in_final": round(r["avg_tracks_in_final"] or 0, 1),
                "avg_rating": round(r["avg_rating"] or 0, 2),
            }
            for r in engine_perf
        ],
        "genre_saturation_warnings": saturation_warnings,
    }


def list_engines() -> dict:
    """
    Return status and capabilities of all registered engines.

    Returns a dict with:
        engines: list of {name, description, capabilities, health, slot_weight}
    """
    ctx = _get_ctx()
    registry = ctx.registry
    engines_cfg = _load_engines_config()

    result = []
    for engine in registry.engines:
        health = registry.health.get(engine.name)
        cap = engine.capabilities
        cfg = engines_cfg.get(engine.name, {})
        result.append({
            "name": engine.name,
            "description": cap.description,
            "capabilities": {
                "novelty_bias": cap.novelty_bias,
                "genre_coverage": cap.genre_coverage,
                "cold_start_friendly": cap.cold_start_friendly,
                "data_requirements": cap.data_requirements,
                "speed": cap.speed,
            },
            "health": {
                "status": health.status if health else "unknown",
                "message": health.message if health else None,
            },
            "slot_weight": cfg.get("slot_weight", 1.0),
        })

    # Also include unavailable engines so Claude can see what failed
    for name, health in registry.health.items():
        if health.status == "unavailable":
            cfg = engines_cfg.get(name, {})
            result.append({
                "name": name,
                "description": "(unavailable)",
                "capabilities": None,
                "health": {"status": "unavailable", "message": health.message},
                "slot_weight": cfg.get("slot_weight", 0),
            })

    return {"engines": result}


def get_track_info(track_id: str) -> dict:
    """
    Return full metadata for a track by its internal track_id.

    Returns: {track_id, title, artist, album, genre, tidal_id, tidal_url, bpm, ...}
    """
    ctx = _get_ctx()
    track = ctx.db.get_track(track_id)
    if track is None:
        return {"error": f"Track {track_id!r} not found"}

    tidal_url = (
        f"https://tidal.com/browse/track/{track.tidal_id}"
        if track.tidal_id
        else None
    )
    return {
        "track_id": track.id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "genre": track.genre_primary,
        "genre_tags": track.genre_tags,
        "mood_tags": track.mood_tags,
        "duration_ms": track.duration_ms,
        "tidal_id": track.tidal_id,
        "tidal_url": tidal_url,
        "bpm": track.bpm,
        "mbid": track.mbid,
    }


def get_playlists() -> dict:
    """
    Return all system-managed Tidal playlists with their URLs and track counts.

    Returns:
        candidate: {name, tidal_url, track_count} for the "Now Rating" playlist
        curated: list of {type, genre, name, tidal_url, track_count}
    """
    ctx = _get_ctx()
    if ctx.playlist_manager is None:
        return {
            "error": "Playlist manager not initialized (Tidal not configured)",
            "candidate": None,
            "curated": [],
        }

    playlists = ctx.playlist_manager.get_all_playlists()
    candidate = None
    curated = []

    for p in playlists:
        info = {
            "name": p.name,
            "tidal_url": p.tidal_url,
            "track_count": p.track_count,
        }
        if p.playlist_type == "candidate":
            candidate = info
        else:
            curated.append({
                "type": p.playlist_type,
                "genre": p.genre,
                **info,
            })

    return {"candidate": candidate, "curated": curated}


def set_curated_threshold(score: int) -> dict:
    """
    Update the minimum rating score for auto-adding tracks to curated playlists.

    Changes take effect for all subsequent rate_track() calls in this session.
    Does not retroactively add or remove tracks from existing playlists.
    Persists the new value to settings.yaml.

    Args:
        score: integer in [1, 10]

    Returns: {ok: true, new_threshold: score}
    """
    if score < 1 or score > 10:
        return {"ok": False, "error": f"Score must be between 1 and 10, got {score}"}

    ctx = _get_ctx()

    # Update the session manager's threshold in memory
    ctx.session_manager._curated_threshold = score

    # Update playlist manager config if present
    if ctx.playlist_manager is not None:
        ctx.playlist_manager._cfg.curated_threshold = score

    # Persist to settings.yaml
    try:
        settings = _load_settings()
        settings.setdefault("playlists", {})["curated_threshold"] = score
        with _SETTINGS_PATH.open("w") as f:
            yaml.dump(settings, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"set_curated_threshold: updated settings.yaml → {score}")
    except Exception as exc:
        logger.warning(f"set_curated_threshold: could not persist to settings.yaml: {exc}")

    return {"ok": True, "new_threshold": score}


def reconcile_playlists(session_id: str | None = None) -> dict:
    """
    Fix Tidal playlist state by replaying ratings from the DB.

    Use this if a Tidal API failure left the playlists out of sync.
    If session_id is provided, reconciles only that session.
    Otherwise, reconciles all completed sessions from the last 7 days.

    Returns: {sessions_reconciled: N, candidate_fixes: N, curated_fixes: N}
    """
    ctx = _get_ctx()
    if ctx.playlist_manager is None:
        return {"error": "Playlist manager not initialized (Tidal not configured)"}

    total_candidate_fixes = 0
    total_curated_fixes = 0
    sessions_reconciled = 0

    if session_id is not None:
        stats = ctx.playlist_manager.reconcile_session(session_id)
        return {
            "sessions_reconciled": 1,
            "candidate_fixes": stats["candidate_fixes"],
            "curated_fixes": stats["curated_fixes"],
        }

    # Reconcile all recent completed sessions
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=7)
    recent = ctx.db.get_recent_sessions(n=50)
    for session in recent:
        if session.completed_at is None:
            continue
        if session.completed_at < cutoff:
            continue
        try:
            stats = ctx.playlist_manager.reconcile_session(session.id)
            total_candidate_fixes += stats["candidate_fixes"]
            total_curated_fixes += stats["curated_fixes"]
            sessions_reconciled += 1
        except Exception:
            logger.exception(f"reconcile_playlists: failed for session {session.id}")

    return {
        "sessions_reconciled": sessions_reconciled,
        "candidate_fixes": total_candidate_fixes,
        "curated_fixes": total_curated_fixes,
    }
