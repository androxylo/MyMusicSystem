# ML Techniques & Architecture Patterns for Music Recommendation

---

## Core Recommendation Paradigms

### 1. Collaborative Filtering (CF)

**Idea:** Users with similar listening histories will like similar music.

**Explicit feedback CF** (user rates tracks 1–5):
- Best library: **Surprise** (SVD, SVD++, KNN)
- SVD factorizes the user-track rating matrix into latent factors
- Each user and track gets an embedding; dot product predicts rating
- SVD++ extends this with implicit signals (which tracks you've heard at all)

**Implicit feedback CF** (play counts, skips, listen completion):
- Best library: **implicit** (ALS, BPR)
- ALS (Alternating Least Squares): fast, parallelizable, handles sparsity
- BPR (Bayesian Personalized Ranking): directly optimizes ranking rather than rating
- Treat "listened 10 times" as higher confidence than "listened once"

**Practical data sources for CF:**
- Last.fm scrobble history → play counts per (user, track)
- Your explicit ratings from the try-and-rank experiment loop
- Tidal favorites (binary: favorited vs. not)

---

### 2. Content-Based Filtering

**Idea:** Recommend tracks similar to ones the user liked, based on audio features.

**Feature vectors per track:**
- Essentia features: BPM, key, loudness, MFCCs (13–40 dims), chroma, mood scores
- Last.fm tags (one-hot or TF-IDF over tag vocabulary)
- Genre embeddings from Discogs style taxonomy

**Similarity computation:**
- Cosine similarity between feature vectors (most common)
- Euclidean distance in normalized feature space
- ANN search (annoy/voyager) for real-time nearest-neighbor lookup at scale

**Cold start advantage:** Works for any track with audio features, even with zero
user interaction history. Critical for discovering tracks you've never heard.

---

### 3. Hybrid Models

**Idea:** Combine CF and content signals. CF improves with data; content handles cold start.

**LightFM** (BPR/WARP loss):
- Learns user and item embeddings that blend CF (interaction matrix) and content (feature matrix)
- As interactions grow, CF signal dominates; for new tracks, content features carry the prediction
- Directly applicable: Essentia audio features as item features, rating history as interactions

**cornac CVAECF:**
- Conditional Variational Autoencoder for CF — uses audio features as conditioning input
- More complex but potentially higher quality than LightFM

---

### 4. Sequence Models (Session-Based Recommendation)

**Idea:** Given the last N tracks played, predict what to play next.

**Approaches:**
- **Markov chains:** Transition probability from track A to track B (simple, interpretable)
- **Word2vec/track2vec:** Train on playlist sequences; similar tracks cluster in embedding space
- **LSTM / GRU:** Recurrent model over listening session
- **Transformer (SASRec, BERT4Rec):** Self-attention over item sequences; current state of the art

**When to use:** "What should play next given this session?" rather than "What are the
user's overall favorites?" Works well for maintaining mood/energy continuity.

---

### 5. Bandit Algorithms (Explore-Exploit)

**Idea:** Each candidate track (or genre/cluster) is an "arm." Balance exploiting
known preferences with exploring unfamiliar territory. Directly maps to the try-and-rank
experiment loop.

**Multi-Armed Bandit (no context):**
- **Thompson Sampling:** Maintain Beta(α, β) per arm; sample and pick highest. Update
  α on "liked", β on "disliked." Bayesian, handles uncertainty naturally.
- **UCB1:** Pick arm with highest (mean + exploration bonus). Deterministic.
- **Epsilon-greedy:** With probability ε pick random, else pick best known.

**Contextual Bandit (with context):**
- **LinUCB:** Linear regression per arm with UCB bonus. Context = current state
  (time of day, recent track features, user energy preference, etc.)
- **Neural Bandit:** Deep network predicts reward given (context, arm) pair
- **BootstrappedTS:** Thompson Sampling via bootstrap resampling; works with any
  base ML model (logistic regression, gradient boosting, etc.)

**Library:** `contextualbandits` (LinUCB, BootstrappedTS) or `vowpalwabbit` (production-grade)

**Key insight:** This paradigm is the most natural fit for the interactive try-and-rank
experiment. Each "experiment session" generates (context, action, reward) tuples that
continuously improve the exploration strategy.

---

### 6. Graph-Based Methods

**Idea:** Model music as a graph. Artists/tracks are nodes; edges are similarity/co-listening.

**Approaches:**
- **Last.fm similarity graph:** Recursive getSimilar calls → weighted artist graph
  → PageRank or random walk to rank unexplored nodes
- **Playlist co-occurrence graph:** Tracks appearing in same playlist are connected
  → community detection reveals genre clusters
- **LightGCN (Graph Convolutional Network):** Deep learning on the bipartite user-item
  interaction graph. State of the art for CF; implemented in microsoft/recommenders

---

### 7. LLM-Augmented Recommendation

**Idea:** Use language models to generate or understand music descriptions, then do
semantic similarity search.

**Pattern 1 — Semantic search:**
1. Generate text descriptions of tracks: "Atmospheric post-rock instrumental, slow
   build, distorted guitars, melancholic mood, ~140 BPM"
2. Embed descriptions with a text model (OpenAI text-embedding-3, sentence-transformers, etc.)
3. Store in vector DB (Chroma, pgvector)
4. Embed user query ("something melancholic and slow with guitars") → nearest neighbors

**Pattern 2 — LLM as recommender:**
1. Describe user taste in natural language (top artists, liked tracks, tags)
2. Prompt an LLM: "The user likes X, Y, Z. Suggest 10 similar artists they haven't
   heard, with reasons."
3. Search Tidal/Spotify for the suggested artists

**Pattern 3 — Explanation + feedback:**
Use an LLM to explain *why* a recommendation was made, then let the user critique the
reasoning. Structured feedback improves the feature weights.

**Current state:** Still emerging in open source; mostly bespoke implementations.
Best for cold start and natural language interaction.

---

## Architecture Patterns Seen in Production

### Pattern A: Rate → Embed → Retrieve

```
User rates tracks
       ↓
Build preference vector (weighted average of liked track feature vectors)
       ↓
ANN search in audio feature space for similar unrated tracks
       ↓
Present candidates → user rates → update preference vector
       ↓
Repeat
```

Simplest to implement. Works well with Essentia features + annoy index.
No complex model to train — the "model" is the user's preference vector.

---

### Pattern B: Bandit Loop

```
Initialize: one arm per genre/cluster/mood
       ↓
Each session: Thompson Sampling picks an arm (genre/mood region)
       ↓
Sample tracks from that arm (nearest neighbors in feature space)
       ↓
User listens and rates
       ↓
Update Beta distribution for the arm (liked=+α, disliked=+β)
       ↓
Repeat
```

Best for explicit exploration of new genres. Quantifies uncertainty naturally.
Converges to user preferences while never fully stopping exploration.

---

### Pattern C: Batch CF + Online Bandit

```
Offline (daily/weekly batch):
  - Pull full scrobble history from Last.fm
  - Train ALS/LightFM model on play counts + ratings
  - Generate candidate set: top-N unheard tracks per user

Online (each session):
  - Bandit (Thompson Sampling) selects which candidates to present
  - User rates → immediate bandit update
  - Feedback stored for next batch CF retrain
```

Combines the statistical power of CF (global patterns) with the responsiveness
of bandits (per-session exploration). This is conceptually how Spotify's recommendation
works at the session level.

---

### Pattern D: Sequence → Next Track

```
Recent listening session: [A, B, C, D, ...]
       ↓
Sequence model (SASRec/BERT4Rec or track2vec)
       ↓
Predicted next track distribution
       ↓
Filter by: must be unheard, must match current mood context
       ↓
Present top candidate
```

Best for "what plays next" rather than "what to discover overall."

---

## Evaluation Metrics

| Metric | Use Case |
|---|---|
| RMSE / MAE | Explicit rating prediction accuracy |
| Precision@K / Recall@K | Top-K recommendation quality |
| NDCG@K | Ranking quality (normalized discounted cumulative gain) |
| MAP (Mean Average Precision) | Overall ranking across users |
| Serendipity | How surprising/novel the recommendations are |
| Coverage | Fraction of catalog the system can recommend |
| Diversity | How varied the recommendations are (avoid filter bubble) |

For an interactive try-and-rank system, the most important metrics in practice are:
- **Like rate:** Fraction of presented tracks that get positive rating
- **Skip rate:** Fraction skipped immediately (strong negative signal)
- **Exploration rate:** How often recommendations are from new genres/artists

---

## Recommended Reading

- "Collaborative Filtering for Implicit Feedback Datasets" — Hu, Koren, Volinsky (2008)
  ALS paper; foundational for implicit feedback CF
- "BPR: Bayesian Personalized Ranking from Implicit Feedback" — Rendle et al. (2009)
- "Neural Collaborative Filtering" — He et al. (2017)
- "Self-Attentive Sequential Recommendation" (SASRec) — Kang & McAuley (2018)
- "A Contextual-Bandit Approach to Personalized News Article Recommendation" — Li et al. (2010)
  LinUCB paper; directly applicable to track recommendation
- Spotify Research Blog: https://research.atspotify.com/publications/
  (reinforcement learning for recommendations, bandits, embeddings)
