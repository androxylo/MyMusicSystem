# Tidal API

**Official developer portal:** https://developer.tidal.com
**Official API docs:** https://developer.tidal.com/documentation
**Internal API base:** https://api.tidal.com/v1/

## Overview

Tidal has a registered developer program with an OAuth 2.0 REST API, but it is significantly
more restricted than Spotify's. There is no audio features endpoint and no listening history
endpoint — the two most useful signals for a recommendation system.

The most practical access path for a personal project is the **tidalapi Python library**
using your own account credentials via OAuth device flow.

## Authentication

**OAuth 2.0** (Authorization Code Flow or device code flow).

- Register an app at developer.tidal.com to get `client_id` / `client_secret`
- Device code flow: user visits a URL, authorizes, token is issued — no password needed
- Tokens are Bearer tokens sent in `Authorization: Bearer <token>` headers
- Session persistence via saved token files (tidalapi handles this automatically)

## What Is Accessible

### Catalog & Search
- `/search` — tracks, albums, artists, playlists
- `/tracks/{id}` — track metadata (title, artist, album, ISRC, duration, explicit flag)
- `/albums/{id}` and `/albums/{id}/tracks`
- `/artists/{id}`, `/artists/{id}/toptracks`, `/artists/{id}/albums`, `/artists/{id}/similar`
- `/playlists/{uuid}`

### User Library
- `/users/{userId}/favorites/tracks` — your favorited tracks
- `/users/{userId}/favorites/albums`
- `/users/{userId}/favorites/artists`
- `/users/{userId}/playlists` — your playlists (read + write)
- `/me` — current user info

### Recommendations / Mixes
- `/pages/home` — home page content (algorithmic mixes, editorial picks)
- `/mixes/{mixId}/items` — tracks inside a mix (e.g. "My Daily Discovery", "My New Arrivals")

### Playback
- `/tracks/{id}/playbackinfopostpaywall` — stream URLs (requires active subscription)
- Quality tiers: Normal (AAC 96k), High (AAC 320k), HiFi (FLAC 16/44.1), Master (MQA/HiRes FLAC)

## What Is NOT Accessible

| Missing Data | Notes |
|---|---|
| Audio features | No danceability, energy, tempo, valence, key — Tidal has no such API |
| Listening history / play counts | Not exposed via any known endpoint |
| Personalization signals / taste profile | Not exposed |
| Social features | Follower/following lists, public activity — not available |
| Lyrics | Tidal has synced lyrics in-app but not reliably accessible via API |

## Rate Limits

- Not formally documented for the official API
- Heavy usage on the internal API triggers 429 errors and temporary bans
- The internal API requires an active Tidal subscription — tied to your account token

## Python Library: python-tidal

**GitHub:** https://github.com/tamland/python-tidal
**PyPI:** `pip install tidalapi`
**Docs:** https://tidalapi.netlify.app
**Stars:** ~600

Active and maintained. Wraps the internal Tidal API with clean Python objects.
Supports OAuth device-flow login, search, catalog browsing, user favorites, playlists,
mixes, and artist/album/track objects.

```python
import tidalapi

session = tidalapi.Session()
session.login_oauth_simple()  # Opens browser for device auth

# Get user favorites
favorites = session.user.favorites
tracks = favorites.tracks()

# Search
results = session.search("Radiohead", [tidalapi.Artist, tidalapi.Track])

# Get a mix
mixes = session.user.get_mixes()
mix_tracks = mixes[0].items()
```

## Other Tidal Tools

- **Moosync** (https://github.com/Moosync/Moosync, ~800 stars) — cross-platform music player
  with Tidal, Spotify, YouTube, Last.fm integration. Vue.js + Electron.
- Various downloader tools (tidal-dl, tidal-dl-ng) — legal gray area, not recommended.
- Migration tools (tidal-to-spotify and similar) — use MusicBrainz for track matching.

## Assessment for This Project

Tidal serves best as:
1. The **playback layer** (stream tracks the system recommends)
2. A **library source** (sync favorites/playlists as seed data)
3. An **algorithmic mix reader** (pull Tidal's own mixes as discovery input)

Audio features and listening history must come from other sources (Essentia + Last.fm scrobbling).
