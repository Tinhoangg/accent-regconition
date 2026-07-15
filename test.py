"""Single-audio inference entrypoint for Vietnamese accent recognition."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel

from config import FEATURES, PATHS, TRAINING
from feature_extraction import extract_features, extract_sequence
from model import build_model_from_checkpoint
from processing import align_syllables, preprocess_audio, preprocess_transcript, save_temp_wav


def load_audio_file(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load and preprocess a local audio file."""
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    return preprocess_audio(audio, sr)


def transcribe_audio(whisper: WhisperModel, wav_path: Path) -> str:
    """Transcribe a wav file to text using a few Whisper fallback settings."""

    def run_transcribe(**kwargs) -> str:
        segments, _ = whisper.transcribe(str(wav_path), word_timestamps=False, condition_on_previous_text=False, **kwargs)
        parts = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if text:
                parts.append(text.strip())
        return preprocess_transcript(" ".join(parts))

    attempts = [
        {"language": "vi", "beam_size": 5, "vad_filter": False},
        {"language": "vi", "beam_size": 5, "vad_filter": True},
        {"language": None, "beam_size": 5, "vad_filter": False},
        {"language": None, "beam_size": 1, "vad_filter": False},
    ]
    for kwargs in attempts:
        transcript = run_transcribe(**kwargs)
        if transcript:
            return transcript
    return ""


def extract_fallback_sequence(audio: np.ndarray, sr: int, max_len: int) -> np.ndarray:
    """Fallback when no transcript is available: split the audio into equal chunks."""
    duration = len(audio) / max(sr, 1)
    n_chunks = int(np.clip(round(duration / 0.7), 3, min(max_len, 20)))
    if n_chunks <= 0:
        return np.empty((0, FEATURES.feature_dim), dtype=np.float32)
    chunk_len = max(1, len(audio) // n_chunks)
    vectors = []
    for idx in range(n_chunks):
        start = idx * chunk_len
        end = len(audio) if idx == n_chunks - 1 else min(len(audio), (idx + 1) * chunk_len)
        segment = audio[start:end]
        if len(segment) < int(0.04 * sr):
            continue
        vectors.append(extract_features(segment, sr))
    if not vectors:
        return np.empty((0, FEATURES.feature_dim), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def pad_sequence(sequence: np.ndarray, max_len: int, feature_dim: int) -> np.ndarray:
    """Pad or truncate one feature sequence."""
    padded = np.zeros((1, max_len, feature_dim), dtype=np.float32)
    length = min(len(sequence), max_len)
    if length:
        padded[0, :length] = sequence[:length]
    return padded


def normalize_sequence(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply train-set normalization to non-padding frames."""
    mask = np.any(x != 0.0, axis=-1, keepdims=True)
    normalized = (x - mean) / std
    return np.where(mask, normalized, 0.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test one audio file with the accent recognition model.")
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument("--transcript", default=None, help="Transcript of the audio. If omitted, Whisper will generate one automatically.")
    parser.add_argument("--model-path", default=str(PATHS.checkpoints_dir / "final_model.pt"), help="Path to a saved PyTorch model.")
    parser.add_argument("--whisper-model", default=TRAINING.whisper_model)
    parser.add_argument("--whisper-device", default=TRAINING.whisper_device)
    parser.add_argument("--whisper-compute-type", default=TRAINING.whisper_compute_type)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_path = Path(args.model_path)
    if not model_path.exists() and model_path.name == "final_model.pt":
        fallback = PATHS.checkpoints_dir / "best_model.pt"
        if fallback.exists():
            model_path = fallback
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    max_len = int(checkpoint["max_len"])
    feature_dim = int(checkpoint["feature_dim"])
    label_names = list(checkpoint.get("label_names", TRAINING.label2id.keys()))
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    model = build_model_from_checkpoint(checkpoint, feature_dim, num_classes=len(label_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    whisper = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)
    audio, sr = load_audio_file(audio_path)
    with tempfile.TemporaryDirectory(prefix="accent_test_") as tmpdir:
        temp_wav = Path(tmpdir) / f"{audio_path.stem}.wav"
        save_temp_wav(audio, sr, temp_wav)
        transcript = preprocess_transcript(args.transcript) if args.transcript else transcribe_audio(whisper, temp_wav)

        if transcript:
            segments = align_syllables(audio, sr, transcript, whisper, temp_wav)
            sequence = extract_sequence(audio, sr, segments)
            if sequence.size == 0:
                raise RuntimeError("Transcript was available but no syllable features were extracted.")
            print("Mode: transcript-aligned")
        else:
            sequence = extract_fallback_sequence(audio, sr, max_len)
            if sequence.size == 0:
                raise RuntimeError("Whisper could not produce a transcript and fallback chunking also failed.")
            print("Mode: audio-only fallback")
            transcript = "[no transcript available]"

        x = pad_sequence(sequence, max_len, feature_dim)
        x = normalize_sequence(x, mean, std)
        with torch.no_grad():
            logits = model(torch.from_numpy(x).float().to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probabilities))
        ranked = np.argsort(probabilities)[::-1]

        print(f"Audio: {audio_path}")
        print(f"Transcript: {transcript}")
        if transcript != "[no transcript available]":
            print(f"Aligned syllables: {len(segments)}")
        else:
            print(f"Fallback chunks: {len(sequence)}")
        print(f"Prediction: {label_names[pred_idx]} ({probabilities[pred_idx]:.4f})")
        print("Top probabilities:")
        for idx in ranked[:3]:
            print(f"  {label_names[int(idx)]}: {probabilities[int(idx)]:.4f}")


if __name__ == "__main__":
    main()

