"""Extract voice features from a directory of audio recordings.

The UCI Parkinson's datasets contain precomputed acoustic measurements, but
not the original waveforms. This script is intended for the recordings that
correspond to those observations; it computes the requested waveform-based
features and writes one CSV row per audio file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}


def _safe_statistic(values: np.ndarray, statistic: str) -> float:
    """Return a finite statistic, or NaN when no voiced frames exist."""
    voiced = values[np.isfinite(values) & (values > 0)]
    if voiced.size == 0:
        return float("nan")
    return float(getattr(np, statistic)(voiced))


def _mfcc_features(audio: np.ndarray, sample_rate: int, n_mfcc: int) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    features: dict[str, float] = {}
    for index, coefficient in enumerate(mfcc, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficient))
    return features


def extract_file(path: Path, n_mfcc: int = 13) -> dict[str, object]:
    """Extract pitch, perturbation, harmonicity, and MFCC features."""
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    if audio.size == 0:
        raise ValueError("audio file contains no samples")

    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
    point_process = call([sound, pitch], "To PointProcess (cc)")
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)

    jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    shimmer = call(
        [sound, point_process],
        "Get shimmer (local)",
        0.0,
        0.0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    hnr = call(harmonicity, "Get mean", 0.0, 0.0)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)

    features: dict[str, object] = {
        "file": path.name,
        "duration_seconds": float(len(audio) / sample_rate),
        "sample_rate_hz": sample_rate,
        # Praat returns these perturbation measures as ratios (0.01 = 1%).
        "jitter_local": float(jitter),
        "shimmer_local": float(shimmer),
        "hnr_db": float(hnr),
        "f0_mean_hz": _safe_statistic(f0, "mean"),
        "f0_median_hz": _safe_statistic(f0, "median"),
        "f0_std_hz": _safe_statistic(f0, "std"),
        "f0_min_hz": _safe_statistic(f0, "min"),
        "f0_max_hz": _safe_statistic(f0, "max"),
        "voiced_frame_fraction": float(np.count_nonzero(f0 > 0) / f0.size)
        if f0.size
        else float("nan"),
    }
    features.update(_mfcc_features(audio, sample_rate, n_mfcc))
    return features


def extract_directory(audio_dir: Path, n_mfcc: int = 13) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no supported audio files found under {audio_dir}")

    for path in paths:
        try:
            rows.append(extract_file(path, n_mfcc=n_mfcc))
        except Exception as error:  # Keep the batch usable when one file is bad.
            rows.append({"file": path.name, "error": str(error)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", type=Path, help="directory containing voice recordings")
    parser.add_argument("output_csv", type=Path, help="destination CSV")
    parser.add_argument("--n-mfcc", type=int, default=13, help="number of MFCC coefficients (default: 13)")
    args = parser.parse_args()

    if args.n_mfcc < 1:
        parser.error("--n-mfcc must be at least 1")
    features = extract_directory(args.audio_dir, n_mfcc=args.n_mfcc)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    print(f"Extracted {len(features)} recording(s) to {args.output_csv}")


if __name__ == "__main__":
    main()