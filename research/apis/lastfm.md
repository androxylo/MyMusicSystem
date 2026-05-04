# Last.fm API

**Base URL:** https://ws.audioscrobbler.com/2.0/
**Docs:** https://www.last.fm/api
**Auth:** API key for read-only; Web Auth (OAuth-like) for scrobbling
**Cost:** Free with API key registration
**Format:** JSON or XML (prefer JSON with `&format=json`)
**Rate limit:** ~5 requests/second; no hard cap documented, abuse leads to bans

## Why This Is the Most Valuable Free API

Last.fm has the richest **open** music graph for this use case:
- Full scrobble history with timestamps (no 50-item cap like Spotify)
- Artist + track similarity scores from collaborative listening data
- Deep community-sourced mood/genre tag system ("melancholic", "rainy day", "driving")
- 20+ years of accumulated listening data across millions of users

## Setup

```python
pip install pylast
```

```python
import pylast

network = pylast.LastFMNetwork(
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    username="your_username",
    password_hash=pylast.md5("your_password")  # for write access
)
```

## User Data Endpoints

### Full Listening History
```
user.getRecentTracks
  ?user=USERNAME&limit=200&page=N&from=UNIX_TS&to=UNIX_TS
```
- Returns every scrobble ever made, paginated (up to 200/page)
- Filter by `from`/`to` Unix timestamps for date ranges
- Each entry: track name, artist, album, timestamp, MusicBrainz ID (if matched)
- **This is the primary source of dense listening history**

### Top Items by Period
```
user.getTopTracks    ?period=overall|7day|1month|3month|6month|12month
user.getTopArtists   (same periods)
user.getTopAlbums    (same periods)
user.getTopTags      # User's most-used tags across their listening
```

### Other User Endpoints
```
user.getLovedTracks              # Explicitly loved tracks
user.getFriends                  # Friends (for collaborative filtering)
user.getWeeklyTrackChart         # Weekly chart, optionally by date range
user.getWeeklyArtistChart
user.getInfo                     # Profile, join date, total scrobbles
```

## Artist Similarity & Metadata

### getSimilar — Key for Recommendations
```
artist.getSimilar ?artist=NAME&limit=100&autocorrect=1
```
- Returns up to 100 similar artists with `match` scores (0.0–1.0)
- Powered by collaborative listening co-occurrence (not editorial)
- **Very high quality signal** for building a similarity graph

```
artist.getTopTracks ?artist=NAME&limit=50
artist.getTopAlbums ?artist=NAME&limit=50
artist.getTopTags   ?artist=NAME&limit=20   # Community-sourced tags
artist.getInfo      ?artist=NAME            # Bio, stats, similar, tags
artist.search       ?artist=NAME
```

## Track Similarity & Tags

```
track.getSimilar  ?artist=NAME&track=NAME&limit=100
track.getTopTags  ?artist=NAME&track=NAME&limit=20
track.getInfo     ?artist=NAME&track=NAME
track.search      ?track=NAME
```

Track tags are extremely useful: community members tag tracks with mood descriptors
like "melancholic", "driving", "chill", "workout", "late night", etc.

## Tag Endpoints — Mood/Genre Discovery

```
tag.getTopTracks   ?tag=TAG_NAME&limit=50   # Top tracks for any tag
tag.getTopArtists  ?tag=TAG_NAME&limit=50
tag.getTopAlbums   ?tag=TAG_NAME
tag.getSimilar     ?tag=TAG_NAME            # Related tags
chart.getTopTags                            # Global trending tags
```

Examples of useful tags: `electronic`, `ambient`, `post-rock`, `melancholic`,
`instrumental`, `female vocalists`, `jazz`, `math rock`, `drone`, `shoegaze`

## Scrobbling (Write Operations — Requires Auth)

```
track.scrobble            # Submit a played track (with timestamp)
track.updateNowPlaying    # Set "now playing" status
track.love / track.unlove # Mark track as loved
```

**Integration pattern for Tidal:** Use a scrobbler (e.g. the "Web Scrobbler" browser
extension or a custom script using python-tidal + pylast) to automatically submit
every Tidal track play to Last.fm. Over time, this builds a full listening history
that can be queried via the API.

## pylast Usage Examples

```python
# Get similar artists
artist = network.get_artist("Radiohead")
similar = artist.get_similar(limit=50)
for item in similar:
    print(item.item.name, item.match)

# Get user's recent tracks
user = network.get_user("your_username")
recent = user.get_recent_tracks(limit=200)

# Get top tags for a track
track = network.get_track("Radiohead", "Pyramid Song")
tags = track.get_top_tags(limit=10)

# Get top tracks for a mood tag
tag = network.get_tag("melancholic")
top_tracks = tag.get_top_tracks(limit=50)

# Scrobble a track
network.scrobble(
    artist="Radiohead",
    title="Pyramid Song",
    timestamp=int(time.time())
)
```

## Assessment for This Project

Last.fm is the backbone of the data layer:
1. **Listening history store** — scrobble all Tidal plays here; query full history for CF
2. **Similarity graph** — `getSimilar` for artists and tracks gives a ready-made graph
3. **Mood/genre tags** — tag data enriches track profiles without needing Essentia
4. **Cold start** — for new tracks, Last.fm tags + similar artists provide initial signal

**GitHub:** https://github.com/pylast/pylast (~500 stars)
