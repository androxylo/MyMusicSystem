# Summary: Gaps, Opportunities & Recommended Stack

## The Core Problem

You want a system that:
1. Suggests new music you haven't heard
2. You try the suggestions and rank/rate them
3. The system learns from your ratings to improve future suggestions
4. Your primary playback platform is Tidal

**The fundamental challenge:** Tidal has no audio features API and no listening history
API. Spotify removed its recommendations endpoint in November 2024 and restricted audio
features. This means you cannot rely on any single streaming platform's native ML
infrastructure — you must assemble your own stack.

---

## Key Gaps in the Ecosystem

### 1. No end-to-end "try and rank" app exists
All the building blocks are there (bandits, CF, audio feature extraction, platform APIs)
but nobody has packaged them into a cohesive interactive application. This is the most
significant gap — and the opportunity for this project.

### 2. No Tidal-native recommender
Every open-source recommender targets Spotify. Tidal's API restrictions (no audio features,
no listening history) make it harder, but python-tidal gives enough access to make it
viable as a playback layer with history tracked externally.

### 3. Spotify's November 2024 changes broke the ecosystem
The removal of `GET /recommendations` and restriction of audio-features made Spotify
useless as a recommendation engine. Many existing projects are now broken or abandoned.
The ecosystem hasn't fully caught up — which creates opportunity.

### 4. Real-time online learning is research-grade
Production systems (ListenBrainz, Spotify) retrain in batch. True per-rating model
updates are only practical with bandit algorithms — not with matrix factorization.
A hybrid (batch CF for the global model, online bandits for session exploration) is
the pragmatic middle ground.

---

## Recommended Stack for This Project

### Data Layer

| Component | Tool | Purpose |
|---|---|---|
| Playback + library | python-tidal | Stream tracks, read favorites/playlists |
| Listening history | Last.fm (scrobble Tidal plays) + pylast | Full history, no 50-item cap |
| Audio features | Essentia (local) | BPM, key, mood, MFCCs — no API dependency |
| Track metadata | MusicBrainz + AcoustID | MBID identity resolution, cross-platform |
| Genre enrichment | Last.fm tags + Discogs | Community mood/genre tags, detailed taxonomy |
| Catalog cross-ref | Spotify API (search, related artists) | Secondary discovery signal |

**Key decision:** Scrobble all Tidal plays to Last.fm immediately. This builds the
listening history corpus that powers collaborative filtering over time.

---

### Recommendation Layer

**Phase 1 — Cold start (first weeks, minimal data):**

Content-based using Essentia features:
- Extract audio features for all tracks in your Tidal favorites
- Build a preference vector (weighted average of liked track features)
- ANN search (annoy) in feature space for similar unrated tracks
- Supplement with Last.fm `artist.getSimilar` for graph-based discovery

**Phase 2 — As rating data accumulates:**

Hybrid CF + content using LightFM:
- Training data: (track, rating) pairs from your try-and-rank sessions + play counts
- Item features: Essentia audio vectors + Last.fm tag embeddings
- LightFM (WARP loss) handles cold start for new tracks via content features,
  and improves for known tracks as CF signal grows

**Phase 3 — Session-level exploration:**

Contextual bandit for deciding *what to present*:
- Arms = genre/mood clusters (derived from Essentia features or Last.fm tags)
- Context = current session state (time of day, recent track mood, user energy)
- Thompson Sampling picks which cluster to draw from
- User ratings update the arm distributions immediately

---

### Experiment Loop Design

```
1. System selects a batch of ~5–10 candidate tracks
   (from content-based retrieval or CF recommendations)

2. User listens to 30–60 second preview for each
   (or full track — Tidal playback via python-tidal)

3. User rates: Skip / Maybe / Like / Love
   (maps to scores: -1, 0, +1, +2)

4. Ratings stored locally with timestamp + track feature vector

5. Bandit model updated immediately per rating

6. CF model retrained in batch (daily/weekly) on accumulated ratings

7. Next batch generated using updated models
```

---

### Technology Choices

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python | All libraries are Python-first |
| Feature extraction | Essentia | Most complete open-source audio ML |
| CF model | LightFM | Best hybrid cold-start + CF |
| Bandit | contextualbandits (BootstrappedTS) | Drop-in online updates per rating |
| ANN search | annoy | Simple, fast, production-ready |
| Vector DB (optional) | chromadb | If adding LLM-based semantic search |
| Local storage | SQLite or DuckDB | Simple, embedded, no server needed |
| History | Last.fm | Full history, free, queryable |

---

## What to Verify Before Building

The research was conducted from training knowledge (cutoff August 2025). Before
committing to any component, verify:

1. **Spotify audio features access** — check if your existing Spotify app registration
   can still call `GET /audio-features`, or if you need to apply for Extended Access.
   URL: https://developer.spotify.com/dashboard

2. **Tidal developer portal status** — the official API at developer.tidal.com has
   evolved; verify current endpoint availability and whether a personal app registration
   is sufficient.
   URL: https://developer.tidal.com

3. **python-tidal maintenance status** — check the repo for recent commits and
   open issues regarding Tidal API changes.
   URL: https://github.com/tamland/python-tidal

4. **Last.fm API key registration** — straightforward but requires account creation.
   URL: https://www.last.fm/api/account/create

5. **Essentia installation** — `essentia-tensorflow` requires compatible TensorFlow version;
   check compatibility matrix for your Python version.
   URL: https://essentia.upf.edu/installing.html

6. **AcousticBrainz dump availability** — confirm the static dataset is still downloadable.
   URL: https://acousticbrainz.org/download
