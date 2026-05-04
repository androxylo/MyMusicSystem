# MusicBrainz API

**Base URL:** https://musicbrainz.org/ws/2/
**Docs:** https://musicbrainz.org/doc/MusicBrainz_API
**Auth:** None required for read (but must set User-Agent header); OAuth for edits
**Cost:** Free; data is CC0 (public domain)
**Rate limit:** 1 request/second without account; registered bots get higher limits
**DB download:** https://musicbrainz.org/doc/MusicBrainz_Database/Download

## Overview

MusicBrainz is a structured music encyclopedia — not a streaming or listening API.
Its primary value is **identity resolution**: given a Tidal or Spotify track, its MBID
links to Wikipedia, Wikidata, Discogs IDs, ISRCs, and other platform IDs.

**Required header for all requests:**
```
User-Agent: YourAppName/1.0 (your@email.com)
```

## Core Entities

| Entity | Description |
|---|---|
| `artist` | Musicians, bands, orchestras, composers |
| `recording` | A specific audio performance (maps to a track) |
| `release` | A specific physical/digital release (album edition) |
| `release-group` | The logical album (groups all editions of an album) |
| `work` | The underlying composition/song |
| `label` | Record labels |
| `genre` | Formal genre taxonomy |

Each entity has a UUID called an MBID (MusicBrainz Identifier).

## Key Queries

### Lookup by MBID
```
GET /ws/2/recording/{mbid}?inc=artists+releases+tags+ratings+isrcs&fmt=json
GET /ws/2/artist/{mbid}?inc=releases+tags+ratings+url-rels&fmt=json
GET /ws/2/release-group/{mbid}?inc=artists+releases+tags&fmt=json
```

### Search (Lucene syntax)
```
GET /ws/2/artist?query=Radiohead&fmt=json
GET /ws/2/recording?query=artist:Radiohead+AND+recording:Creep&fmt=json
GET /ws/2/release?query=release:OK+Computer+AND+artist:Radiohead&fmt=json
```

### Include parameters (`inc=`) — chain with `+`
- `artists`, `releases`, `release-groups`, `recordings`, `works`
- `tags`, `ratings`, `user-tags`, `user-ratings`
- `aliases`
- `isrcs` (International Standard Recording Codes)
- `url-rels` (external URL relationships — links to Spotify, Discogs, Wikidata, etc.)
- `artist-credits`
- `annotation`

## URL Relationships — Identity Resolution

The most powerful feature for cross-platform linking. Example:
```
GET /ws/2/artist/{mbid}?inc=url-rels&fmt=json
```
Returns URLs of type:
- `streaming` — Spotify, Apple Music, Deezer links
- `discogs` — Discogs artist URL
- `wikidata` — Wikidata entity
- `wikipedia` — Wikipedia page
- `social network` — official social media
- `youtube` — official YouTube channel
- `soundcloud` — SoundCloud profile

This lets you resolve: Tidal artist → MBID → Spotify artist ID, for cross-platform joins.

## AcoustID Integration

**AcoustID** (https://acoustid.org/webservice) pairs with MusicBrainz:
1. Generate a Chromaprint fingerprint from audio: `fpcalc audio_file.mp3`
2. POST to AcoustID API with the fingerprint → get MBIDs for matching recordings
3. Use the MBID to look up full MusicBrainz metadata

```python
# pip install pyacoustid
import acoustid
results = acoustid.match("YOUR_ACOUSTID_API_KEY", "track.mp3")
for score, recording_id, title, artist in results:
    print(f"{score:.2f} {title} by {artist} (MBID: {recording_id})")
```

AcoustID API key: https://acoustid.org/api-key (free, requires registration)

## Python Client: musicbrainzngs

```python
pip install musicbrainzngs
```

```python
import musicbrainzngs

musicbrainzngs.set_useragent("MyMusicSystem", "0.1", "contact@example.com")

# Search for an artist
result = musicbrainzngs.search_artists(artist="Radiohead", limit=5)
artist = result["artist-list"][0]
mbid = artist["id"]

# Get artist with URL relationships
artist_detail = musicbrainzngs.get_artist_by_id(mbid, includes=["url-rels", "tags"])

# Search for a recording
result = musicbrainzngs.search_recordings(
    recording="Pyramid Song", artist="Radiohead", limit=5
)
```

## Bulk Database Download

For high-volume use, download the full DB dump rather than hitting the API:
- Full PostgreSQL dump + incremental weekly updates
- Contains all entities, relationships, and tags
- Can be loaded into a local PostgreSQL instance
- Good option if you want to do large-scale offline enrichment

## Assessment for This Project

MusicBrainz serves as the **glue layer** between platforms:
1. **ISRC → MBID → Spotify ID / Discogs ID** cross-platform track matching
2. **Artist relationship graph** — "member of", "collaborated with" typed edges
3. **Genre tags** — MusicBrainz has a curated (not community-wild) genre taxonomy
4. **Offline enrichment** — download the DB dump, enrich your local track database

Not a real-time data source — use it for enrichment and identity resolution, not discovery.
