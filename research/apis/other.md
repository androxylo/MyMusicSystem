# Other Music APIs

---

## Discogs API

**Base URL:** https://api.discogs.com/
**Docs:** https://www.discogs.com/developers
**Auth:** OAuth 1.0a for user data; simple token (`?token=...`) for read-only database
**Cost:** Free — 60 authenticated requests/minute; 25 unauthenticated/minute

### Best Use: Genre/Style Taxonomy

Discogs has the most granular music genre taxonomy of any service:
- Top-level genres: Electronic, Rock, Jazz, Classical, Hip Hop, Folk/World/Country, etc.
- Styles (subgenres): e.g. `Electronic > Techno > Detroit Techno`, or
  `Electronic > Ambient > Dark Ambient`

This is far more detailed than Spotify's genre list or Last.fm tags.

### Database Endpoints

```
GET /database/search?q=OK+Computer&type=release&artist=Radiohead
GET /releases/{id}           # Full release metadata
GET /masters/{id}            # Master (canonical) release
GET /artists/{id}            # Artist profile, members, aliases
GET /labels/{id}             # Label info and releases
```

**Search filters:** `type`, `genre`, `style`, `format`, `country`, `year`, `barcode`, `catno`

### Release Metadata Includes
- Genre and style tags
- Format (Vinyl/CD/Digital + detailed pressing info)
- Label and catalog number
- Credits: producers, engineers, performers by instrument
- Country of release
- Tracklist with durations
- Videos (YouTube links)
- Barcode, matrix numbers

### User Data (OAuth)
```
GET /users/{username}/collection   # Your Discogs collection
GET /users/{username}/wantlist     # Want list
```

**Assessment:** Use Discogs primarily for genre/style enrichment and credits metadata.
Not a discovery or recommendation source, but excellent for augmenting track profiles.

---

## Apple Music API (MusicKit)

**Docs:** https://developer.apple.com/documentation/applemusicapi
**Auth:** MusicKit JWT tokens (developer key + team ID from Apple Developer account)
**Cost:** Requires Apple Developer Program membership ($99/year)

### What's Available

**Catalog (public, no user auth):**
```
GET /v1/catalog/{storefront}/songs/{id}
GET /v1/catalog/{storefront}/albums/{id}
GET /v1/catalog/{storefront}/artists/{id}
GET /v1/catalog/{storefront}/search?term=...&types=songs,artists,albums
```

Song attributes: albumName, artistName, genreNames, isrc, name, releaseDate,
durationInMillis, previews (30-second preview audio URLs).

**User data (requires user authorization via MusicKit):**
```
GET /v1/me/library/songs
GET /v1/me/library/albums
GET /v1/me/library/artists
GET /v1/me/library/playlists
GET /v1/me/ratings/songs          # User ratings (stars)
GET /v1/me/recommendations        # Personalized recommendations (opaque)
GET /v1/me/history/heavy-rotation # Recently heavily played
GET /v1/me/recent/played          # Recently played
```

**Charts:**
```
GET /v1/catalog/{storefront}/charts?types=songs&genre=...
```

### What Apple Music Does NOT Provide
- No audio features (no tempo, energy, valence, etc.)
- No "similar tracks" endpoint
- No mood-based filtering
- Personalized recommendations exist but are opaque black-box results

**Assessment:** Interesting mainly for the 30-second preview URLs (could feed Essentia
analysis) and cross-platform track matching via ISRC. The $99/year developer cost makes
it a lower priority for a personal project.

---

## YouTube Music (Unofficial)

**Official YouTube Data API v3:** https://developers.google.com/youtube/v3

There is **no official YouTube Music API** with music-specific endpoints.
The general YouTube Data API v3 can search videos and get metadata, but has no
audio features, genre classification, or music-specific discovery.

**Quota:** 10,000 units/day free; search costs 100 units/request = 100 searches/day.

### ytmusicapi (Unofficial)

**GitHub:** https://github.com/sigma67/ytmusicapi
**PyPI:** `pip install ytmusicapi`

A reverse-engineered Python library mimicking YouTube Music's internal API:
- Search: songs, albums, artists, playlists
- Artist pages, album contents
- User library: liked songs, listening history, playlists
- Track recommendations ("Up next", radio mode)
- Browse by mood/genre
- Requires authentication via browser cookie headers (not OAuth)

```python
from ytmusicapi import YTMusic
ytmusic = YTMusic("browser_cookies.json")  # Set up once via browser

results = ytmusic.search("Radiohead", filter="songs")
radio = ytmusic.get_watch_playlist(videoId="...", radio=True)
```

**Caveats:**
- Not officially supported; breaks with YouTube Music UI updates
- Requires browser cookie auth — not a stable OAuth flow
- Technically violates YouTube ToS
- No SLA, no rate limit guarantees

**Assessment:** Useful for prototyping or personal use, but not for anything production.
If YouTube Music is a major source for you, worth experimenting with.

---

## Gracenote (Nielsen)

**Docs:** https://developer.gracenote.com
**Cost:** Enterprise licensing (not realistic for personal projects)

The professional-grade metadata and identification service used by many commercial
streaming services. Provides detailed genre taxonomy, mood, tempo, BPM, playlist
generation API, and music identification. Mentioned for completeness — not actionable
at a personal project scale.

---

## AudioDB

**Docs:** https://theaudiodb.com/api_guide.php
**Cost:** Free (limited) / ~$5/month Patreon for full access

Provides: artist bios, album/track metadata, some mood tags, genre info, YouTube
music video links, artist images. Useful as a lightweight free supplementary metadata
source for enriching artist/album profiles.
