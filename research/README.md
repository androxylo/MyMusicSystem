# Music Recommendation System — Research

Research conducted 2026-05-03. Knowledge cutoff of sources: August 2025.
Live web access was unavailable during research; all URLs should be verified before use.

## Structure

```
research/
├── README.md               # This file — index and high-level summary
├── apis/
│   ├── tidal.md            # Tidal API (primary platform)
│   ├── spotify.md          # Spotify Web API (secondary account)
│   ├── lastfm.md           # Last.fm API — best free history + similarity source
│   ├── listenbrainz.md     # ListenBrainz — open scrobbling + CF pipeline
│   ├── musicbrainz.md      # MusicBrainz — identity resolution + metadata
│   ├── audio_features.md   # Essentia, AcousticBrainz, Cyanite.ai, ACRCloud, AudD
│   └── other.md            # Discogs, Apple Music, YouTube Music, Gracenote, AudioDB
├── projects/
│   ├── libraries.md        # Key Python libraries and their roles
│   └── similar_systems.md  # Existing open-source recommender projects
├── techniques/
│   └── algorithms.md       # ML techniques, patterns, and architecture options
└── summary.md              # Gaps, opportunities, and recommended stack
```

## Quick Reference: What Each API Contributes

| Source | Playback | User History | Audio Features | Similarity | Cost |
|---|---|---|---|---|---|
| Tidal (python-tidal) | Yes | No | No | No | Own subscription |
| Spotify (spotipy) | No | Limited (50 recent) | Restricted | Related artists | Free tier |
| Last.fm (pylast) | No | Full scrobble archive | No | Artist + track | Free |
| ListenBrainz | No | Full, open | No | CF-based recs | Free |
| MusicBrainz | No | No | No | No (graph) | Free (CC0) |
| Essentia (local) | No | No | Yes (full) | No | Free (OSS) |
| Cyanite.ai | No | No | Yes (ML) | Yes (similarity) | Paid |
| AcousticBrainz dump | No | No | Yes (static) | No | Free (archived) |
| Discogs | No | No | No | No | Free |

## Recommended Core Stack (Personal Project)

- **Playback + library sync:** python-tidal
- **Listening history:** Last.fm scrobbling from Tidal + pylast for queries
- **Audio features:** Essentia (self-hosted) or AcousticBrainz dump for known tracks
- **Similarity signal:** Last.fm getSimilar + your own ratings
- **Identity resolution:** MusicBrainz MBIDs + AcoustID
- **Recommendation algorithm:** LightFM (hybrid CF + content) or contextualbandits (bandit loop)
- **Catalog cross-reference:** Spotify API (search, related artists)
