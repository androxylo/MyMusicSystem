# Similar Open-Source Projects

---

## Music Recommendation Systems

### microsoft/recommenders
**GitHub:** https://github.com/microsoft/recommenders (~20,000 stars)
**Tech:** Python, PySpark, TensorFlow, PyTorch

The most comprehensive open-source recommendation toolkit. Contains:
- Jupyter notebooks for 15+ algorithms (SVD, SAR, NCF, Wide & Deep, LightGCN, etc.)
- Utilities for dataset loading, model evaluation, feature engineering
- Examples specifically for music/media recommendation use cases

**Why it matters:** The notebooks serve as runnable implementations of every major
recommendation paradigm. Good reference for choosing and implementing an approach.

---

### RecSys 2018 Spotify Challenge Implementations

The Spotify Automatic Playlist Continuation challenge (2018) used the Million Playlist
Dataset (1M Spotify playlists, 66M track-playlist pairs) and produced dozens of
high-quality open-source solutions.

**Key approaches used by top teams:**
- word2vec / track2vec on playlist sequences
- Matrix factorization (ALS, BPR) on track-playlist co-occurrence
- Graph-based (tracks as nodes, playlist co-occurrence as edges)
- Ensemble of all of the above

**Notable repos:**
- https://github.com/dbdmg/recsys2018-spotify-challenge — word2vec + CF ensemble,
  one of the top-performing solutions
- Search GitHub for `recsys2018 spotify` for ~30 more implementations

**Why it matters:** Real-world validation that playlist co-occurrence + word2vec embeddings
work extremely well for music recommendation at scale.

---

### your_spotify
**GitHub:** https://github.com/Yooooomi/your_spotify (~3,000 stars)
**Tech:** Node.js, React, MongoDB

Self-hosted Spotify listening history tracker. Captures all Spotify play events via
the API and stores them in a local MongoDB database. Rich dashboards for listening stats.

**Why it matters:** Solves the "Spotify only gives 50 recent plays" problem by
continuously polling and storing. Could be adapted for Tidal via the python-tidal
event stream, or used as-is for the Spotify library cross-reference.

---

### Moosync
**GitHub:** https://github.com/Moosync/Moosync (~800 stars)
**Tech:** Vue.js, Electron, TypeScript

Cross-platform desktop music player with plugin-based streaming service integrations:
Tidal, Spotify, YouTube Music, Last.fm. Shows a working architecture for combining
multiple streaming sources in one application.

---

### beets
**GitHub:** https://github.com/beetbox/beets (~12,500 stars)
**Tech:** Python

Music library manager with MusicBrainz auto-tagging and an extensive plugin system.
Relevant plugins:
- `beets-xtractor` — calls Essentia/AcousticBrainz for audio features, stores results
  in the beets database
- `beets-autobpm` — BPM detection
- `lastgenre` — fetches Last.fm genre tags

**Why it matters:** If you build a local music library, beets gives you a mature
pipeline for tagging, feature extraction, and metadata management.

---

## Collaborative Filtering / Matrix Factorization Projects

### benfred/implicit
**GitHub:** https://github.com/benfred/implicit (~3,500 stars)
**Tech:** Python, C++, CUDA

See `libraries.md` for full details. The de-facto standard for fast ALS on implicit
feedback (play counts, skips). Used by many production recommendation systems.

---

### lyst/lightfm
**GitHub:** https://github.com/lyst/lightfm (~4,700 stars)
**Tech:** Python, C (Cython)

Hybrid CF + content model. See `libraries.md` for details.

---

### NicolasHug/Surprise
**GitHub:** https://github.com/NicolasHug/Surprise (~6,300 stars)
**Tech:** Python, Cython

Easiest library for explicit-rating CF (SVD, KNN, etc.). See `libraries.md`.

---

### PreferredAI/cornac
**GitHub:** https://github.com/PreferredAI/cornac (~900 stars)
**Tech:** Python, TensorFlow

Multi-modal recommender with 40+ algorithms. The **CVAECF** model is directly
applicable: it takes content features (Essentia audio vectors) alongside CF
interaction data, naturally solving the cold-start problem.

---

## Bandit / Explore-Exploit Systems

### spotify/confidence
**GitHub:** https://github.com/spotify/confidence (~500 stars)
**Tech:** Python

Spotify's open-sourced bandit experimentation framework. Uses Thompson Sampling.
Shows how Spotify approaches explore/exploit at production scale — though for
A/B testing rather than individual recommendations.

---

### david-cortes/contextualbandits
**GitHub:** https://github.com/david-cortes/contextualbandits (~1,100 stars)
**Tech:** Python, scikit-learn

The most directly applicable bandit library. The "online" update methods let you
update the model after each user rating — exactly the try-and-rank loop being designed.
See `libraries.md` for full details.

---

## Open Recommendation Pipelines

### metabrainz/listenbrainz-server
**GitHub:** https://github.com/metabrainz/listenbrainz-server (~300 stars)
**Tech:** Python, Flask, PostgreSQL, Redis, Apache Spark

The full ListenBrainz server codebase — readable implementation of a real CF
recommendation pipeline running in production. Spark-based ALS on public listen data.

---

### metabrainz/troi-recommendation-pipeline
**GitHub:** https://github.com/metabrainz/troi-recommendation-pipeline (~100 stars)
**Tech:** Python, Apache Spark

The actual recommendation pipeline code ListenBrainz uses. Generates weekly playlists
using ALS collaborative filtering on the public listen graph. Good reference for
understanding how to structure a batch recommendation pipeline.

---

## Graph-Based Discovery

### Artist similarity graphs using Last.fm getSimilar
No single canonical repo, but a common pattern:
1. Start from a seed artist
2. Call `artist.getSimilar` recursively to N hops
3. Build a weighted graph (nodes = artists, edges = similarity score)
4. Use graph centrality, PageRank, or random walks to rank unexplored artists

This is conceptually similar to how Spotify's "Artist Radio" worked internally.

---

## Nearest Neighbor Projects

### spotify/annoy
**GitHub:** https://github.com/spotify/annoy (~13,000 stars)
**Tech:** C++, Python

Approximate Nearest Neighbors — Spotify's production engine for similarity search.
Used to find similar tracks given an audio feature or learned embedding vector.

### spotify/voyager
**GitHub:** https://github.com/spotify/voyager (~1,300 stars)
**Tech:** C++, Python, Java

HNSW-based successor to annoy. Better recall at same query speed.

---

## LLM-Augmented Recommendation (Emerging)

No well-established open-source project in this space yet, but the pattern is:
1. Generate text descriptions of tracks (genre, mood, instrumentation, lyrical themes)
2. Embed descriptions with a text embedding model (OpenAI, sentence-transformers, etc.)
3. Store in a vector database (Chroma, pgvector, Qdrant)
4. At query time, embed user preference description → semantic nearest-neighbor search

**chroma-core/chroma** (https://github.com/chroma-core/chroma, ~17,000 stars) is the
most popular vector DB for this pattern.

**Advantage:** Handles natural language queries ("something like Radiohead but more upbeat")
and extreme cold start (zero interaction data for new tracks).

---

## Notable Gaps (Opportunities)

1. **No complete interactive ranking + bandit backend as a packaged app.**
   All the pieces exist but nobody has assembled: streaming UI → rate tracks → bandit
   update → new recommendations as a cohesive open-source project.

2. **No Tidal-native recommender.** All open-source work targets Spotify. The absence
   of audio features from Tidal's API and the lack of listening history make this harder,
   but the core gap is simply that nobody has built it.

3. **Real-time online learning is research-grade.** Production systems (ListenBrainz,
   Spotify) retrain in batch (daily/weekly). True per-event online updates are done in
   bandit systems (contextualbandits, VW) but not in matrix factorization models.

4. **Spotify November 2024 API changes broke most existing projects.** Many repos
   that relied on `GET /recommendations` are now unmaintained or broken. This is recent
   enough that replacements haven't fully emerged yet.
