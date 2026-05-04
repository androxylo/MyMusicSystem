"""
Essentia audio feature extractor.

Wraps essentia.standard to extract a fixed-length feature vector from a
local audio file. If essentia is not installed, is_available() returns
False and all extract() calls raise ConnectorError.

Feature vector layout (8 dimensions, all values in [0.0, 1.0]):
  [0]  normalized BPM           bpm / 200.0, clamped to [0, 1]
  [1]  key sine                 sin(key_index * 2π / 12)  mapped to [0, 1]
  [2]  key cosine               cos(key_index * 2π / 12)  mapped to [0, 1]
  [3]  mode                     0.0 = minor, 1.0 = major
  [4]  energy                   RMS energy, [0, 1]
  [5]  loudness (normalized)    (integrated_loudness + 60) / 60, clamped [0, 1]
  [6]  spectral centroid        centroid_hz / 8000.0, clamped [0, 1]
  [7]  zero crossing rate       zcr / 0.5, clamped [0, 1]
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

# Key name → chromatic index (0 = C, 1 = C#/Db, …, 11 = B)
_KEY_INDEX: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Detected once at import time
try:
    import essentia  # noqa: F401
    import essentia.standard as _es
    _ESSENTIA_AVAILABLE = True
except ImportError:
    _es = None  # type: ignore[assignment]
    _ESSENTIA_AVAILABLE = False


class ConnectorError(Exception):
    pass


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _build_feature_vector(
    bpm: float,
    key: str,
    mode: str,
    energy: float,
    loudness: float,
    spectral_centroid: float,
    zcr: float,
) -> list[float]:
    """Assemble the 8-dim normalized feature vector."""
    norm_bpm = _clamp(bpm / 200.0)

    key_idx = _KEY_INDEX.get(key, 0)
    angle = key_idx * 2 * math.pi / 12
    key_sin = _clamp((math.sin(angle) + 1.0) / 2.0)   # map [-1,1] → [0,1]
    key_cos = _clamp((math.cos(angle) + 1.0) / 2.0)

    mode_val = 1.0 if mode.lower() == "major" else 0.0
    norm_energy = _clamp(energy)
    norm_loudness = _clamp((loudness + 60.0) / 60.0)
    norm_centroid = _clamp(spectral_centroid / 8000.0)
    norm_zcr = _clamp(zcr / 0.5)

    return [
        norm_bpm, key_sin, key_cos, mode_val,
        norm_energy, norm_loudness, norm_centroid, norm_zcr,
    ]


class EssentiaExtractor:
    """
    Thin wrapper around essentia.standard for audio feature extraction.

    Usage::

        extractor = EssentiaExtractor(config)
        if extractor.is_available():
            features = extractor.extract("/path/to/track.flac")
            # features["features"] is a list[float] of length 8
    """

    FEATURE_DIM = 8

    def __init__(self, config: dict | None = None) -> None:
        self._models_dir = Path(
            (config or {}).get("models_dir", "./models/essentia/")
        )

    def is_available(self) -> bool:
        """True if essentia is importable."""
        return _ESSENTIA_AVAILABLE

    def extract(self, audio_path: str) -> dict:
        """
        Extract audio features from a local file.

        Returns a dict::

            {
                "bpm": float,
                "key": str,           # e.g. "A", "F#"
                "mode": str,          # "major" | "minor"
                "energy": float,      # [0, 1]
                "loudness": float,    # integrated loudness in LUFS (negative)
                "spectral_centroid": float,   # Hz
                "zero_crossing_rate": float,  # [0, 1]
                "features": list[float],      # 8-dim normalized vector
            }

        Raises ConnectorError if essentia is not installed or extraction fails.
        """
        if not _ESSENTIA_AVAILABLE:
            raise ConnectorError("essentia is not installed")

        try:
            return self._run_pipeline(audio_path)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Feature extraction failed for {audio_path!r}: {exc}") from exc

    def _run_pipeline(self, audio_path: str) -> dict:
        """Run the essentia feature extraction pipeline."""
        loader = _es.MonoLoader(filename=audio_path, sampleRate=44100)
        audio = loader()

        # Rhythm
        rhythm_extractor = _es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio)

        # Key / mode
        key_extractor = _es.KeyExtractor()
        key, mode, _ = key_extractor(audio)

        # Energy
        energy_algo = _es.Energy()
        energy = float(energy_algo(audio))
        # Normalize: typical range is 0 to ~50000; clamp after /50000
        energy = _clamp(energy / 50000.0)

        # Loudness (integrated, LUFS-like)
        loudness_algo = _es.Loudness()
        loudness = float(loudness_algo(audio))   # returns a negative value in dB

        # Spectral centroid (frame-level, average)
        frame_size = 2048
        hop_size = 1024
        w = _es.Windowing(type="hann")
        spec = _es.Spectrum()
        centroid_algo = _es.Centroid(range=22050)
        zcr_algo = _es.ZeroCrossingRate()

        centroids = []
        zcrs = []
        for frame in _es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
            windowed = w(frame)
            spectrum = spec(windowed)
            centroids.append(float(centroid_algo(spectrum)))
            zcrs.append(float(zcr_algo(frame)))

        spectral_centroid = sum(centroids) / len(centroids) if centroids else 0.0
        zcr = sum(zcrs) / len(zcrs) if zcrs else 0.0

        features = _build_feature_vector(
            bpm=float(bpm),
            key=key,
            mode=mode,
            energy=energy,
            loudness=loudness,
            spectral_centroid=spectral_centroid,
            zcr=zcr,
        )

        return {
            "bpm": float(bpm),
            "key": key,
            "mode": mode,
            "energy": energy,
            "loudness": loudness,
            "spectral_centroid": spectral_centroid,
            "zero_crossing_rate": zcr,
            "features": features,
        }
