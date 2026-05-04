# Key Libraries

All libraries listed here are Python unless noted. Stars are approximate as of mid-2025.

---

## Platform Clients

| Library | Stars | Install | Purpose |
|---|---|---|---|
| spotipy | ~4,800 | `pip install spotipy` | Spotify Web API client |
| python-tidal | ~600 | `pip install tidalapi` | Tidal API client |
| pylast | ~500 | `pip install pylast` | Last.fm API client |
| ytmusicapi | ~3,000 | `pip install ytmusicapi` | YouTube Music (unofficial) |
| musicbrainzngs | ~350 | `pip install musicbrainzngs` | MusicBrainz API client |
| pyacoustid | ~100 | `pip install pyacoustid` | AcoustID fingerprint matching |

---

## Audio Analysis

### librosa
**GitHub:** https://github.com/librosa/librosa (~7,000 stars)
**Install:** `pip install librosa`

General-purpose audio analysis. Good for prototyping and exploration:
- MFCCs, chroma features, spectral features, onset detection
- Beat tracking, tempo estimation
- Pitch shifting, time stretching
- Good visualization support (spectrogram plots)

Best for: quick exploration and prototyping. Less comprehensive than Essentia for
production feature pipelines.

```python
import librosa
y, sr = librosa.load("track.mp3")
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
```

### essentia
**GitHub:** https://github.com/MTG/essentia (~2,800 stars)
**Install:** `pip install essentia essentia-tensorflow`

Production-grade audio analysis with pre-trained ML classifiers. The same library
that powered AcousticBrainz. See `apis/audio_features.md` for full details.

Best for: building a feature extraction pipeline for recommendations.

### Chromaprint / fpcalc
**GitHub:** https://github.com/acoustid/chromaprint
**Install:** system package (`brew install chromaprint`) or `pip install pyacoustid`

Generates acoustic fingerprints. Used with AcoustID for track identification.

---

## Recommendation Algorithms

### Surprise
**GitHub:** https://github.com/NicolasHug/Surprise (~6,300 stars)
**Install:** `pip install scikit-surprise`

Scikit-style library for classical collaborative filtering. Best entry point for
explicit rating data (user rated this track 1–5 stars).

Algorithms:
- **SVD** — Simon Funk's matrix factorization (Netflix Prize winner technique)
- **SVD++** — SVD with implicit feedback integration
- **NMF** — Non-negative Matrix Factorization
- **SlopeOne** — simple, fast, interpretable
- **KNN** — user-based and item-based nearest neighbor
- **BaselineOnly** — bias model baseline

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["user", "track", "rating"]], reader)
algo = SVD(n_factors=100, n_epochs=20)
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5)
```

Best for: the "user rates tracks → get recommendations" loop. Easiest to get started.

### implicit
**GitHub:** https://github.com/benfred/implicit (~3,500 stars)
**Install:** `pip install implicit`

Fast ALS (Alternating Least Squares) and BPR (Bayesian Personalized Ranking) for
implicit feedback — e.g., play counts, skips, listen time.

- GPU support (CUDA)
- Handles sparse matrices efficiently (millions of items)
- ALS finds a latent factor model; BPR optimizes ranking directly

```python
import implicit
import scipy.sparse as sparse

# user_items: sparse matrix of shape (users, items)
# values = play counts (implicit signal)
model = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.1)
model.fit(user_items)

# Recommend for a user
user_id = 0
recommendations = model.recommend(user_id, user_items[user_id])
```

Best for: building a full collaborative filter on top of scrobble history (play counts).

### LightFM
**GitHub:** https://github.com/lyst/lightfm (~4,700 stars)
**Install:** `pip install lightfm`

Hybrid model combining collaborative filtering with content features. Handles cold
start by using item/user features when interaction data is sparse.

Loss functions:
- **WARP** (Weighted Approximate-Rank Pairwise) — optimizes ranking directly, best for recommendations
- **BPR** — Bayesian Personalized Ranking
- **Logistic** — for explicit feedback

```python
from lightfm import LightFM
from lightfm.data import Dataset

# Build dataset with user/item features
dataset = Dataset()
dataset.fit(users, items)
dataset.fit_partial(items=items, item_features=item_feature_names)

interactions, weights = dataset.build_interactions([(user, item, rating)])
item_features = dataset.build_item_features([(item, [feat1, feat2])])

model = LightFM(loss="warp", no_components=64)
model.fit(interactions, item_features=item_features, epochs=30)

# Predict scores for all items for a user
scores = model.predict(user_id, item_ids, item_features=item_features)
```

Best for: **the primary recommendation model** — it naturally combines your ratings
(CF) with Essentia audio features (content) and handles new tracks gracefully.

### spotlight
**GitHub:** https://github.com/maciejkula/spotlight (~3,000 stars)
**Install:** `pip install spotlight`

PyTorch-based recommendation models:
- Explicit feedback: matrix factorization
- Implicit feedback: BPR, WARP
- **Sequence models:** LSTM or Pooling over listening history — "given these N tracks, predict next"

```python
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel

model = ExplicitFactorizationModel(n_iter=10, embedding_dim=32)
model.fit(interactions)
predictions = model.predict(user_id)
```

Best for: sequence-based recommendations ("what to play next given recent session").

### cornac
**GitHub:** https://github.com/PreferredAI/cornac (~900 stars)
**Install:** `pip install cornac`

Multi-modal recommender framework with 40+ algorithms. Notably:
- **CVAECF** — uses side information (audio features) as content alongside CF
- **BiVAECF** — Bilateral Variational Autoencoder
- **VAECF** — standard VAE for CF

Good for: research-grade experimentation with content-aware models.

### microsoft/recommenders
**GitHub:** https://github.com/microsoft/recommenders (~20,000 stars)

Massive toolkit with Jupyter notebooks for every algorithm:
- SAR (Simple Algorithm for Recommendation) — very fast, good baseline
- NCF (Neural Collaborative Filtering)
- Wide & Deep
- LightGCN (Graph Convolutional Network)
- SVD, ALS, and more

Best for: exploring many algorithms with documented notebooks before committing to one.

---

## Bandit / Explore-Exploit

### contextualbandits
**GitHub:** https://github.com/david-cortes/contextualbandits (~1,100 stars)
**Install:** `pip install contextualbandits`

Implements contextual bandit algorithms for recommendation:
- **LinUCB** — linear upper confidence bound; fast, interpretable
- **Thompson Sampling** — Bayesian approach; excellent for cold start
- **epsilon-Greedy** — simple baseline
- Bootstrapped Thompson Sampling, Softmax Explorer

The "online" fit methods update with each observation — directly applicable to a
"show track → user rates → update model" loop.

```python
from contextualbandits.online import BootstrappedTS
import numpy as np

# context: feature vector for current state (time of day, recent tracks, etc.)
# arms: one per candidate track cluster/genre
model = BootstrappedTS(base_algorithm=LogisticRegression(), nchoices=n_arms)
model.fit(X_context, arms_chosen, rewards)
recommended_arm = model.predict(current_context)
```

### vowpalwabbit
**GitHub:** https://github.com/VowpalWabbit/vowpal_wabbit (~8,500 stars)
**Install:** `pip install vowpalwabbit`

Microsoft's industrial-grade contextual bandit framework. Supports:
- `--cb_explore_adf` — contextual bandits with action-dependent features
- Epsilon-greedy, softmax, cover, bag exploration strategies
- Online learning with each example
- Very fast; handles high cardinality feature spaces

More complex to set up than contextualbandits but more production-ready.

### spotify/confidence
**GitHub:** https://github.com/spotify/confidence (~500 stars)

Spotify's internal A/B testing and bandit experimentation framework (open-sourced).
Thompson Sampling and frequentist statistics. Shows how Spotify thinks about
explore/exploit in production.

---

## Nearest Neighbor Search (Embedding Retrieval)

### spotify/annoy
**GitHub:** https://github.com/spotify/annoy (~13,000 stars)
**Install:** `pip install annoy`

Approximate Nearest Neighbors — Spotify's production similarity engine.
Build an index from track embedding vectors, then find nearest neighbors in real time.

```python
from annoy import AnnoyIndex

f = 128  # embedding dimension
index = AnnoyIndex(f, "angular")  # or "euclidean", "dot"

for i, embedding in enumerate(track_embeddings):
    index.add_item(i, embedding)

index.build(10)  # 10 trees — more = better accuracy, slower build
index.save("music.ann")

# Query
nearest = index.get_nns_by_vector(query_embedding, n=20, include_distances=True)
```

### spotify/voyager
**GitHub:** https://github.com/spotify/voyager (~1,300 stars)
**Install:** `pip install voyager`

Successor to annoy using HNSW (Hierarchical Navigable Small World graphs).
Better recall at same speed. Python, Java, TypeScript bindings.

---

## Infrastructure

### gensim
**Install:** `pip install gensim`

Train word2vec/doc2vec on playlist sequences. Treat each playlist as a "sentence"
and each track ID as a "word" — the resulting embeddings capture co-occurrence
context similar to how word2vec captures semantic relationships.

```python
from gensim.models import Word2Vec

# playlists: list of lists of track IDs (as strings)
model = Word2Vec(sentences=playlists, vector_size=128, window=5, min_count=1, workers=4)
model.wv.most_similar("track_id_123", topn=20)
```

### chroma (vector database)
**GitHub:** https://github.com/chroma-core/chroma (~17,000 stars)
**Install:** `pip install chromadb`

Vector database for storing and querying track embeddings. Supports:
- Embedding storage with metadata filters
- Nearest neighbor search
- Integration with OpenAI / local embedding models

Good for LLM-augmented recommendation where you embed track descriptions as text.
