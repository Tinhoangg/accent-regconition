"""Acoustic feature extraction for syllable-level Vietnamese accent recognition."""

from __future__ import annotations

import librosa
import numpy as np
import parselmouth

from config import AUDIO, FEATURES
from processing import SyllableSegment, crop_segment
from utils import safe_stats


FEATURE_NAMES = [
    "duration",
    "F1_mean",
    "F1_std",
    "F2_mean",
    "F2_std",
    "F3_mean",
    "F3_std",
    "F4_mean",
    "F4_std",
    "F0_mean",
    "F0_std",
    "F0_min",
    "F0_max",
    "F0_range",
    "F0_slope",
    "F0_start",
    "F0_mid",
    "F0_end",
    "voiced_ratio",
    *[f"MFCC_{idx}_mean" for idx in range(1, 14)],
    *[f"MFCC_{idx}_std" for idx in range(1, 14)],
    *[f"delta_MFCC_{idx}_mean" for idx in range(1, 14)],
    "energy_mean",
    "energy_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
]


def frame_params(audio: np.ndarray) -> tuple[int, int]:
    """Return stable frame parameters for short syllable segments."""
    n_fft = min(1024, max(256, 2 ** int(np.floor(np.log2(max(len(audio), 2))))))
    hop_length = max(80, n_fft // 4)
    return n_fft, hop_length


def extract_duration_feature(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract segment duration in seconds."""
    return np.asarray([len(audio) / max(sr, 1)], dtype=np.float32)


def extract_formants(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract F1-F4 mean and standard deviation."""
    if len(audio) < int(AUDIO.min_syllable_duration * sr):
        return np.zeros(8, dtype=np.float32)
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        formant = sound.to_formant_burg(
            time_step=0.005,
            max_number_of_formants=5,
            maximum_formant=FEATURES.max_formant,
            window_length=0.025,
            pre_emphasis_from=50,
        )
        times = np.arange(0.0125, max(0.0126, sound.duration - 0.0125), 0.005)
        values = []
        for formant_idx in (1, 2, 3, 4):
            track = [
                formant.get_value_at_time(formant_idx, float(time))
                for time in times
            ]
            mean, std = safe_stats(np.asarray(track, dtype=float))
            values.extend([mean, std])
        return np.asarray(values, dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def extract_pitch_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract summary and contour F0 features."""
    if len(audio) < int(AUDIO.min_syllable_duration * sr):
        return np.zeros(10, dtype=np.float32)
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch_ac(
            time_step=0.005,
            pitch_floor=FEATURES.pitch_floor,
            pitch_ceiling=FEATURES.pitch_ceiling,
            voicing_threshold=0.35,
        )
        raw_f0 = pitch.selected_array["frequency"]
        valid_mask = np.isfinite(raw_f0) & (raw_f0 > 0)
        voiced_ratio = float(np.mean(valid_mask)) if raw_f0.size else 0.0
        f0 = raw_f0[valid_mask]
        if f0.size < 2:
            return np.asarray([0.0] * 9 + [voiced_ratio], dtype=np.float32)
        duration = len(audio) / sr
        slope = float((f0[-1] - f0[0]) / max(duration, 1e-6))
        contour_points = np.percentile(f0, [10, 50, 90])
        return np.asarray(
            [
                np.mean(f0),
                np.std(f0),
                np.min(f0),
                np.max(f0),
                np.ptp(f0),
                slope,
                contour_points[0],
                contour_points[1],
                contour_points[2],
                voiced_ratio,
            ],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(10, dtype=np.float32)


def extract_mfcc_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC mean/std and simple delta-MFCC mean."""
    if len(audio) < int(AUDIO.min_syllable_duration * sr):
        return np.zeros(FEATURES.n_mfcc * 3, dtype=np.float32)
    try:
        n_fft, hop_length = frame_params(audio)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=FEATURES.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        if mfcc.shape[1] > 1:
            delta_mean = np.mean(np.diff(mfcc, axis=1), axis=1)
        else:
            delta_mean = np.zeros(FEATURES.n_mfcc, dtype=np.float32)
        return np.concatenate([mfcc_mean, mfcc_std, delta_mean]).astype(np.float32)
    except Exception:
        return np.zeros(FEATURES.n_mfcc * 3, dtype=np.float32)


def extract_energy_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract RMS energy mean and standard deviation."""
    if len(audio) < int(AUDIO.min_syllable_duration * sr):
        return np.zeros(2, dtype=np.float32)
    try:
        n_fft, hop_length = frame_params(audio)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=n_fft,
            hop_length=hop_length,
        ).ravel()
        mean, std = safe_stats(rms)
        return np.asarray([mean, std], dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)


def extract_spectral_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract compact spectral shape and zero-crossing features."""
    if len(audio) < int(AUDIO.min_syllable_duration * sr):
        return np.zeros(8, dtype=np.float32)
    try:
        n_fft, hop_length = frame_params(audio)
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
        ).ravel()
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
        ).ravel()
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
        ).ravel()
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=n_fft,
            hop_length=hop_length,
        ).ravel()
        values = []
        for track in (centroid, bandwidth, rolloff, zcr):
            mean, std = safe_stats(track)
            values.extend([mean, std])
        return np.asarray(values, dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a 68-dimensional feature vector from one syllable segment."""
    features = np.concatenate(
        [
            extract_duration_feature(audio, sr),
            extract_formants(audio, sr),
            extract_pitch_features(audio, sr),
            extract_mfcc_features(audio, sr),
            extract_energy_features(audio, sr),
            extract_spectral_features(audio, sr),
        ]
    )
    if features.shape[0] != FEATURES.feature_dim:
        raise ValueError(f"Expected {FEATURES.feature_dim} features, got {features.shape[0]}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_sequence(
    audio: np.ndarray,
    sr: int,
    segments: list[SyllableSegment],
) -> np.ndarray:
    """Extract a sequence of feature vectors from aligned syllable segments."""
    vectors = []
    for segment in segments:
        syllable_audio = crop_segment(audio, sr, segment.start, segment.end)
        if len(syllable_audio) < int(AUDIO.min_syllable_duration * sr):
            continue
        vectors.append(extract_features(syllable_audio, sr))
    if not vectors:
        return np.empty((0, FEATURES.feature_dim), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)
