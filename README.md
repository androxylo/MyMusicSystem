# MyMusicSystem

A personal, AI-powered music recommendation system that learns your taste, enforces listening diversity, and manages your Tidal playlists — all driven by a local engine pipeline exposed to Claude via MCP.

---

## What it does

Each session, the system:

1. **Generates 10 candidate tracks** by polling multiple recommendation engines in parallel, oversampling, then running a diversity pass.
2. **Populates a Tidal playlist** ("Now Rating") with those tracks.
3. **Accepts 1–10 ratings** per track as you listen.
4. **Learns from ratings** — engines update their models after each session.
5. **Auto-curates** tracks rated ≥ 7 into "Liked — All" and per-genre playlists.

You interact with the system through Claude (via MCP tools) or directly from the CLI (`generate_list.py`).

---

## Architecture

```
Claude (MCP client)
    │
    ▼
mcp_server.py          ← MCP server, wraps agent_tools/tools.py
    │
    ▼
agent_tools/tools.py   ← Public API: start_session, rate_track, complete_session, …
    │
    ├── core/session.py          ← Session lifecycle (start → rate → complete)
    ├── core/orchestrator.py     ← Slot allocation, oversampling, junk/dedup filter
    ├── core/diversity.py        ← Genre diversity enforcement (strict / relaxed)
    ├── core/engine_registry.py  ← Engine discovery, instantiation, health checks
    ├── core/playlist_manager.py ← Tidal playlist CRUD ("Now Rating", curated)
    └── core/db.py               ← SQLite: tracks, sessions, ratings, engine metrics

Connectors (thin wrappers, raise ConnectorError on failure):
    connectors/tidal.py     ← tidalapi: search, playlist CRUD, OAuth tokens
    connectors/lastfm.py    ← pylast: similar artists, tag top tracks, loved tracks
    connectors/spotify.py   ← (future)
    connectors/musicbrainz.py ← (future)

Engines (engines/<name>/engine.py + manifest.yaml):
    lastfm_graph          ← Similar-artist graph walk via Last.fm; searches Tidal
    lastfm_tag_similarity ← Tag-based Last.fm discovery; searches Tidal
    genre_explorer        ← Targets under-explored genre buckets via Last.fm + Tidal
    bandit_explorer       ← Thompson Sampling over genre buckets (anti-echo-chamber)
    tidal_mix             ← Pulls from Tidal's algorithmic mixes
    content_similarity    ← ANN search over Essentia audio feature vectors (local)
    collaborative_filter  ← LightFM matrix factorization on rating history
```

---

## Engines

| Engine | Slot weight | Novelty | Cold-start | Data needs |
|---|---|---|---|---|
| `lastfm_graph` | 0.25 | medium | no | ratings |
| `content_similarity` | 0.20 | low | no | audio features |
| `genre_explorer` | 0.20 | high | yes | ratings |
| `bandit_explorer` | 0.20 | medium | yes | ratings |
| `lastfm_tag_similarity` | 0.20 | medium | no | ratings |
| `tidal_mix` | 0.15 | low | yes | — |

The orchestrator normalises weights, assigns integer slots, oversamples by 3×, then runs the diversity pass to select the final 10.

---

## Diversity enforcement

`core/diversity.py` maps every track to one of 11 genre buckets:

> Electronic · Rock · Metal · Pop · Hip-Hop/R&B · Jazz · Classical · Folk/Country · Blues · Reggae/World · Experimental

In **strict** mode, each of the 10 final tracks must come from a different bucket. In **relaxed** mode, the constraint falls back gracefully when not enough candidates exist.

Additional guards:
- Tracks heard in the last 30 days are suppressed.
- Genres that average ≥ 7.0 across 3+ recent sessions are throttled (saturation protection).
- At least 3 different engines must be represented in the final 10.
- Karaoke/tribute/cover tracks are filtered by title regex before the diversity pass.
- Same song under a different Tidal ID is deduplicated by normalised `title|artist` fingerprint.

---

## Session flow

```
start_session()
  └─ Orchestrator.get_suggestions()
       ├─ allocate slots (weighted, ≥1 per engine)
       ├─ collect candidates (3× oversample per engine)
       ├─ filter pool (junk titles + fingerprint dedup)
       └─ DiversityEnforcer.select() → 10 tracks
  └─ PlaylistManager.sync_candidate_playlist() → Tidal "Now Rating"

rate_track(session_id, track_id, score)   # 1–10, repeat for each track
  └─ DB write → if score ≥ threshold: add to curated playlists

complete_session(session_id)
  └─ engine.on_session_complete() → bandit / CF model updates
  └─ returns summary: avg_rating, top track, engine breakdown, genre stats
```

---

## Setup

### Requirements

- Python 3.11+
- Tidal account (HiFi or higher)
- Last.fm account + API key

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml]"          # includes LightFM and Annoy for CF + ANN engines
pip install -e ".[audio]"       # optional: Essentia for content_similarity engine
```

### Authenticate Tidal

```bash
python3 tools/tidal_auth.py
# Opens a browser URL — log in, tokens saved to data/tidal_tokens.json
```

### Configure

Edit `config/settings.yaml`:

```yaml
tidal:
  token_path: ./data/tidal_tokens.json

lastfm:
  api_key: YOUR_KEY
  username: YOUR_USERNAME

playlists:
  curated_threshold: 7        # min score to add to curated playlists
```

Enable/disable engines and adjust slot weights in `config/engines.yaml`.

### Wire up MCP (Claude Code)

Add to `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "music": {
      "command": ".venv/bin/python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/MyMusicSystem"
    }
  }
}
```

Then restart Claude Code. Claude gains tools: `start_session`, `rate_track`, `complete_session`, `get_playlists`, `get_stats`, `list_engines`, `get_track_info`, `set_curated_threshold`, `reconcile_playlists`.

---

## CLI (no MCP)

```bash
python3 generate_list.py              # strict diversity mode
python3 generate_list.py --mode relaxed
python3 generate_list.py --notes "in the mood for something dark"
```

Prints session ID, Tidal playlist URL, and a numbered track table. Use `rate_track` / `complete_session` via MCP or extend the CLI to close the loop.

---

## Running tests

```bash
pytest                    # all unit + contract tests
pytest tests/unit/        # unit tests only
```

---

## Data

All persistent state lives in `data/`:

| File | Contents |
|---|---|
| `music.db` | SQLite: tracks, sessions, ratings, engine_metrics |
| `tidal_tokens.json` | OAuth tokens (git-ignored) |

The database is migrated automatically on startup from `database/migrations/`.
