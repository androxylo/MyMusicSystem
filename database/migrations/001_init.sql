-- Migration 001: initial schema

CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    tidal_id TEXT,
    spotify_id TEXT,
    mbid TEXT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    duration_ms INTEGER,
    genre_primary TEXT NOT NULL DEFAULT 'Other',
    genre_tags TEXT NOT NULL DEFAULT '[]',   -- JSON array
    mood_tags TEXT NOT NULL DEFAULT '[]',    -- JSON array
    bpm REAL,
    key TEXT,
    mode TEXT,
    audio_features TEXT,                     -- JSON blob (Essentia output)
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT                              -- which engine/connector first added this track
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    engine_allocation TEXT NOT NULL DEFAULT '{}',  -- JSON: {engine_name: n_slots}
    diversity_config TEXT NOT NULL DEFAULT '{}',   -- JSON snapshot of config used
    notes TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL REFERENCES tracks(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 10),
    rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, track_id)
);

CREATE TABLE IF NOT EXISTS engine_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    engine_name TEXT NOT NULL,
    track_id TEXT NOT NULL REFERENCES tracks(id),
    engine_score REAL,
    was_final INTEGER NOT NULL DEFAULT 0,    -- 1 if selected into final 10
    final_rank INTEGER                       -- 1-10 if was_final=1
);

CREATE TABLE IF NOT EXISTS engine_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_name TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    tracks_suggested INTEGER NOT NULL DEFAULT 0,
    tracks_in_final INTEGER NOT NULL DEFAULT 0,
    avg_rating_received REAL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS genre_session_history (
    genre TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    track_count INTEGER NOT NULL DEFAULT 0,
    avg_rating REAL,
    PRIMARY KEY (genre, session_id)
);

-- Playlist management (used from Phase 2 onward)
CREATE TABLE IF NOT EXISTS tidal_playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_type TEXT NOT NULL,      -- 'candidate' | 'curated_genre' | 'curated_master'
    genre TEXT,                       -- NULL for candidate and master; genre name for genre lists
    tidal_playlist_id TEXT UNIQUE NOT NULL,
    tidal_playlist_url TEXT,
    name TEXT NOT NULL,
    track_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_synced_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidate_playlist_tracks (
    tidal_playlist_id TEXT NOT NULL REFERENCES tidal_playlists(tidal_playlist_id),
    track_id TEXT NOT NULL REFERENCES tracks(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    removed_at TIMESTAMP,             -- NULL if still in playlist
    PRIMARY KEY (tidal_playlist_id, track_id, session_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ratings_track ON ratings(track_id);
CREATE INDEX IF NOT EXISTS idx_ratings_session ON ratings(session_id);
CREATE INDEX IF NOT EXISTS idx_ratings_rated_at ON ratings(rated_at);
CREATE INDEX IF NOT EXISTS idx_engine_suggestions_session ON engine_suggestions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_genre_session_history_session ON genre_session_history(session_id);
