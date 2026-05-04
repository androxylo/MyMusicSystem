# Audio Feature Extraction APIs & Tools

This covers all sources of audio features — the numerical descriptors (tempo, key, mood,
energy, etc.) that power content-based recommendation. Since Spotify's audio features API
is now restricted, this is a critical gap to fill.

---

## Essentia (Open-Source — Self-Hosted)

**Website:** https://essentia.upf.edu
**GitHub:** https://github.com/MTG/essentia (~2,800 stars)
**License:** AGPL v3 / commercial license available
**Developed by:** Music Technology Group, Universitat Pompeu Fabra (Barcelona)

This is the most important tool in this category. Essentia is the library that powered
AcousticBrainz — running it locally gives you the same capabilities with no API dependency.

### What It Extracts

**Rhythm:**
- BPM (tempo)
- Beat grid positions
- Beat loudness profile
- Onset detection
- Rhythm histogram

**Tonal:**
- Key and scale (major/minor)
- Chord progressions
- HPCP — Harmonic Pitch Class Profile (12-dimensional chroma vector)
- Tuning frequency deviation from 440 Hz

**Spectral / Timbre:**
- MFCCs — Mel-Frequency Cepstral Coefficients (13–40 coefficients)
- Spectral centroid, rolloff, flux, contrast
- Zero-crossing rate
- Spectral complexity

**Loudness:**
- Integrated loudness (LUFS)
- Dynamic range
- ReplayGain

**High-Level ML Classifiers (pre-trained TensorFlow models included):**
- Genre classification (multiple taxonomies: Discogs-style, Allmusic, etc.)
- Mood detection: happy / sad / aggressive / relaxed / party / acoustic
- Danceability score
- Voice activity detection (vocal vs. instrumental)
- Instrument recognition
- Approachability / engagement scores

### Installation

```bash
pip install essentia           # Basic audio analysis
pip install essentia-tensorflow  # + ML classifiers (requires TensorFlow)
```

### Basic Usage

```python
import essentia.standard as es
import essentia

# Load audio
loader = es.MonoLoader(filename="track.mp3", sampleRate=44100)
audio = loader()

# Extract rhythm features
rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

# Extract key
key_extractor = es.KeyExtractor()
key, scale, strength = key_extractor(audio)

# Extract MFCCs
w = es.Windowing(type='hann')
spectrum = es.Spectrum()
mfcc = es.MFCC()
# (applied frame by frame — see docs for full pipeline)

# High-level mood classifier (requires essentia-tensorflow)
model = es.TensorflowPredictMusiCNN(
    graphFilename="mood_happy.pb",
    output="model/Softmax"
)
predictions = model(audio)
```

**Full pipeline reference:** https://essentia.upf.edu/tutorial_essentia_python.html
**Pre-trained models:** https://essentia.upf.edu/models.html

### Integration with Tidal

Since Tidal allows streaming to your local device, you can capture audio to a buffer
and run Essentia analysis on it. A simpler approach: download or rip FLAC files and
run Essentia offline to build a features database.

---

## AcousticBrainz

**Status: SHUT DOWN — Service ended November 2022**

AcousticBrainz was a crowd-sourced audio feature database powered by Essentia.
It is no longer active, but the dataset is downloadable.

**Dataset download:** https://acousticbrainz.org/download

- ~2.5 million tracks' worth of features as JSON
- Keyed by MusicBrainz MBID
- Includes both low-level features (MFCCs, spectral, rhythm) and high-level ML predictions
  (mood, genre, danceability, voice/instrumental)
- Static — no new tracks will be added

**How to use the dump:**
1. Download the JSON dumps
2. Build a local lookup table: MBID → feature vector
3. For a known track, resolve its MBID via MusicBrainz/AcoustID, then look up features

Good coverage for well-known music; gaps for obscure or recent releases.

---

## Cyanite.ai

**Website:** https://cyanite.ai
**API docs:** https://api.cyanite.ai (GraphQL)
**Cost:** Commercial — paid plans (pricing not publicly listed; contact sales)
**Target market:** B2B music licensing, playlist curation companies

### Capabilities

- **Mood tags with confidence scores:** Energetic, Sad, Happy, Dark, Romantic, Aggressive,
  Peaceful, Uplifting, Angry, Fear, etc. — mapped to Russell's Valence/Arousal circumplex
- **Genre classification:** Multi-label, with subgenre detection
- **BPM and key detection**
- **Voice/instrumental classification**
- **Instrument detection**
- **Similarity search:** Upload a reference track or URL → get similar tracks from a catalog
- **Auto-tagging:** Batch tag an entire music catalog

### GraphQL Example (Conceptual)

```graphql
query {
  audioAnalysis(id: "track_id") {
    bpm
    key { value scale }
    mood { happy sad energetic relaxed aggressive }
    genre { label confidence }
    instruments { label confidence }
    valence
    arousal
  }
}
```

**Assessment:** Best commercial option if you don't want to self-host Essentia. The similarity
search endpoint is particularly interesting for "find tracks like this one". However, pricing
is likely high for personal use.

---

## ACRCloud

**Docs:** https://docs.acrcloud.com
**Cost:** Freemium — ~1,000 recognitions/month free; paid per-recognition tiers
**Focus:** Audio fingerprinting / identification (not feature extraction)

### What It Does

- Identify a track from a short audio clip (like Shazam)
- Returns: title, artist, album, ISRC, label, streaming links
- Custom catalog fingerprinting: fingerprint your own catalog for recognition
- Broadcast monitoring: detect music in live radio/TV streams
- Humming recognition (identify from humming)

### Python Example

```python
import acrcloud
# Identify from file
result = acrcloud.recognize("clip.mp3", access_key="...", secret="...", host="...")
# Returns JSON with metadata if matched
```

**Assessment:** Useful for identifying tracks captured from audio sources (e.g., "what
song is playing?"), not for feature extraction or recommendations.

---

## AudD

**Docs:** https://docs.audd.io
**Cost:** Freemium — limited free tier; paid plans for volume

Similar to ACRCloud — audio identification returning streaming metadata.
Also has a lyrics detection endpoint.

---

## AcoustID + Chromaprint

**AcoustID API:** https://acoustid.org/webservice
**Chromaprint:** https://acoustid.org/chromaprint
**Cost:** Free (open-source)

**Use case:** Identify a local audio file → get MusicBrainz MBID → look up full metadata.

```bash
# Generate fingerprint
fpcalc track.mp3
# Returns: FINGERPRINT=... DURATION=...
```

```python
# pip install pyacoustid
import acoustid
results = acoustid.match("ACOUSTID_API_KEY", "track.mp3")
for score, recording_id, title, artist in results:
    print(f"Match: {title} by {artist} (MBID: {recording_id})")
```

**Assessment:** Clean pipeline for track identification → MBID resolution without
commercial dependencies.

---

## AudioDB

**Docs:** https://theaudiodb.com/api_guide.php
**Cost:** Free tier (limited) / ~$5/month Patreon for full access

Provides artist bios, album/track metadata, some mood tags, genre info, and YouTube
music video links. Useful as a lightweight free metadata enrichment source.

---

## Summary: Audio Feature Strategy

For this project, the recommended approach is:

**Tier 1 (free, self-hosted):** Essentia
- Run against audio files / streams for any track not in AcousticBrainz
- Full feature vector: BPM, key, mood, MFCCs, genre

**Tier 2 (free, static):** AcousticBrainz dataset dump
- Pre-computed features for ~2.5M tracks, keyed by MBID
- No computation cost for well-known tracks

**Tier 3 (free, lightweight):** Last.fm community tags
- Mood/genre tags for any track without needing audio access
- Not numerical, but rich categorical signal

**Tier 4 (commercial, if budget allows):** Cyanite.ai
- Highest quality ML mood/genre predictions + similarity search
- Only worth it if self-hosting Essentia proves impractical
