# Spotify Web API

**Base URL:** https://api.spotify.com/v1/
**Docs:** https://developer.spotify.com/documentation/web-api
**Changelog:** https://developer.spotify.com/documentation/web-api/changelog

## Critical Breaking Changes — November 2024

Two major endpoints were restricted/removed, breaking most existing recommendation systems:

1. **`GET /recommendations` — permanently removed.** Previously accepted seed artists/tracks/genres
   plus target audio feature values (target_energy, target_valence, etc.) and returned up to 100
   track recommendations. This endpoint no longer exists.

2. **`GET /audio-features` and `GET /audio-analysis` — restricted.** New app registrations
   cannot access these without applying for Extended Access (quota mode upgrade). Existing apps
   in development mode may still work. Verify current status at the Spotify developer dashboard.

**Implication:** Spotify can no longer serve as the recommendation engine itself. It is still
valuable for user context, library data, and catalog search.

## Authentication (OAuth 2.0)

| Flow | Use Case |
|---|---|
| Authorization Code | Server-side apps needing user data |
| Authorization Code + PKCE | Client-side / mobile apps |
| Client Credentials | App-only access (no user context) |

**Required scopes for a recommendation system:**

```
user-read-recently-played       # Recently played tracks
user-top-read                   # Top tracks and artists by time range
user-read-playback-state        # Current playback
user-library-read               # Saved/liked tracks and albums
playlist-read-private           # Private playlists
playlist-modify-public          # Create/edit public playlists
playlist-modify-private         # Create/edit private playlists
user-follow-read                # Followed artists
```

## Useful Endpoints (Still Working)

### Listening History & Taste Profile

```
GET /me/player/recently-played
```
- Up to 50 recently played tracks
- Includes `played_at` timestamps (ISO 8601)
- Paginate with `before`/`after` Unix timestamp cursors
- Scope: `user-read-recently-played`

```
GET /me/top/tracks
GET /me/top/artists
```
- `time_range`: `short_term` (4 weeks), `medium_term` (6 months), `long_term` (years)
- `limit` 1–50, `offset` for pagination
- Scope: `user-top-read`

### Library

```
GET /me/tracks                  # Saved/liked tracks (paginated, up to 50/page)
GET /me/following               # Followed artists
GET /me/playlists               # User's playlists
```

### Catalog & Discovery

```
GET /artists/{id}/related-artists       # 20 related artists (still works)
GET /artists/{id}/top-tracks            # Top tracks per market
GET /search?q=...&type=track,artist     # Full catalog search
GET /browse/featured-playlists          # Editorial playlists
GET /browse/categories/{id}/playlists   # Genre/mood editorial playlists
```

### Audio Features (Restricted — Verify Access)

```
GET /audio-features/{id}
GET /audio-features?ids=id1,id2,...     # Batch up to 100
```

Fields (if accessible):
- `acousticness`, `danceability`, `energy`, `instrumentalness`
- `key` (0–11, Pitch Class), `loudness` (dB), `mode` (0=minor, 1=major)
- `speechiness`, `tempo` (BPM), `time_signature`, `valence` (0–1 positiveness)
- `liveness`, `duration_ms`

## Rate Limits

- ~25 requests/second under basic quota
- 429 responses include `Retry-After` header
- Quota upgrades (Extended Access) required for high-volume or restricted endpoints

## Python Library: spotipy

**GitHub:** https://github.com/spotipy-dev/spotipy
**PyPI:** `pip install spotipy`
**Stars:** ~4,800

```python
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope="user-read-recently-played user-top-read user-library-read"
))

# Recently played
recent = sp.current_user_recently_played(limit=50)

# Top tracks
top_tracks = sp.current_user_top_tracks(limit=50, time_range="medium_term")

# Related artists
related = sp.artist_related_artists("artist_id")

# Audio features (if access granted)
features = sp.audio_features(["track_id_1", "track_id_2"])
```

## Assessment for This Project

Spotify is valuable for:
1. **User taste seeding** — top tracks/artists over multiple time ranges
2. **Cross-referencing library** — find overlap between your Spotify and Tidal libraries
3. **Catalog similarity** — `related-artists` still works as a graph signal
4. **Track search / ISRC lookup** — finding the same track across platforms

Spotify is NOT useful for:
- Real-time recommendations (endpoint removed)
- Audio feature vectors (restricted for new apps)
- Full listening history (50-item limit on recently-played)

Supplement with Last.fm for history depth and Essentia for audio features.
