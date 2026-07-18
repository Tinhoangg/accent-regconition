"""Near-realtime microphone demo for Vietnamese accent recognition."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from config import AUDIO, PATHS, TRAINING


def resolve_model_path(model_path: str | None) -> Path:
    """Resolve the requested checkpoint path with best/final fallbacks."""
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    candidates.extend([
        PATHS.checkpoints_dir / "best_model.pt",
        PATHS.checkpoints_dir / "final_model.pt",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No PyTorch checkpoint found. Tried: " + ", ".join(str(path) for path in candidates))


def choose_device(requested: str):
    """Choose the PyTorch inference device."""
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(model_path: Path, device) -> tuple[object, dict]:
    """Load a checkpoint and return an eval-ready model plus metadata."""
    import torch

    from model import build_model_from_checkpoint

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    max_len = int(checkpoint["max_len"])
    feature_dim = int(checkpoint["feature_dim"])
    label_names = list(checkpoint.get("label_names", TRAINING.label2id.keys()))
    model = build_model_from_checkpoint(checkpoint, feature_dim, num_classes=len(label_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        "max_len": max_len,
        "feature_dim": feature_dim,
        "label_names": label_names,
        "mean": np.asarray(checkpoint["normalization_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["normalization_std"], dtype=np.float32),
        "wav2vec_config": checkpoint.get("wav2vec_config"),
    }
    return model, metadata


def parse_input_device(value: str | None) -> int | str | None:
    """Convert a numeric device argument to an index; keep device names intact."""
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.lstrip("-").isdigit() else value


def list_input_devices() -> None:
    """Print available microphone devices and the current default input."""
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install sounddevice with `pip install sounddevice`.") from exc

    default_input = sd.default.device[0]
    print("Available input devices:")
    found = False
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) <= 0:
            continue
        found = True
        marker = " (default)" if index == default_input else ""
        print(
            f"  {index}: {device['name']} | channels={int(device['max_input_channels'])} "
            f"| default_sr={float(device['default_samplerate']):.0f}{marker}"
        )
    if not found:
        print("  No input devices found.")


def record_microphone(
    window_seconds: float,
    sample_rate: int,
    channels: int = 1,
    input_device: int | str | None = None,
) -> np.ndarray:
    """Record one microphone window."""
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install sounddevice with `pip install sounddevice`.") from exc

    frames = int(round(window_seconds * sample_rate))
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=input_device,
    )
    sd.wait()
    return np.asarray(recording, dtype=np.float32).squeeze()



def load_audio_file(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load one local audio file without changing its sample rate."""
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install librosa with ``pip install librosa``.") from exc

    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def audio_sample_count(audio: np.ndarray) -> int:
    """Return the number of time samples for mono or channel-first/last audio."""
    y = np.asarray(audio)
    if y.ndim == 0:
        return 0
    if y.ndim == 1:
        return int(y.shape[0])
    if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
        return int(y.shape[1])
    return int(y.shape[0])


def audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    """Compute compact diagnostics without changing the waveform."""
    y = np.asarray(audio, dtype=np.float32)
    finite = y[np.isfinite(y)]
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(finite, dtype=np.float64)))) if finite.size else 0.0
    sample_count = audio_sample_count(y)
    return {
        "shape": tuple(int(value) for value in y.shape),
        "duration": sample_count / float(sample_rate) if sample_rate > 0 else 0.0,
        "peak": peak,
        "rms": rms,
    }


def save_debug_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save debug audio while preserving the waveform passed to the pipeline."""
    from processing import save_temp_wav

    path.parent.mkdir(parents=True, exist_ok=True)
    save_temp_wav(np.asarray(audio, dtype=np.float32), sample_rate, path)


def transcribe_audio(whisper, wav_path: Path) -> str:
    """Transcribe one wav file with conservative Vietnamese settings."""
    from processing import preprocess_transcript

    attempts = [
        {"language": "vi", "beam_size": 5, "vad_filter": False},
        {"language": "vi", "beam_size": 5, "vad_filter": True},
        {"language": None, "beam_size": 5, "vad_filter": False},
        {"language": None, "beam_size": 1, "vad_filter": False},
    ]
    for kwargs in attempts:
        segments, _ = whisper.transcribe(
            str(wav_path),
            word_timestamps=False,
            condition_on_previous_text=False,
            **kwargs,
        )
        transcript = preprocess_transcript(" ".join(getattr(segment, "text", "").strip() for segment in segments))
        if transcript:
            return transcript
    return ""


def pad_sequence(sequence: np.ndarray, max_len: int, feature_dim: int) -> np.ndarray:
    """Pad or truncate a single feature sequence for model inference."""
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


def predict_window(
    raw_audio: np.ndarray,
    sample_rate: int,
    model,
    metadata: dict,
    whisper,
    device,
    min_syllables: int,
    min_raw_peak: float = 1e-4,
    min_raw_rms: float = 1e-5,
    processed_debug_path: Path | None = None,
    wav2vec_extractor=None,
) -> dict:
    """Preprocess, transcribe, align, extract features and predict one audio window."""
    import torch

    from feature_extraction import extract_sequence, extract_sequence_with_wav2vec
    from processing import align_syllables, preprocess_audio, save_temp_wav, segment_syllables

    raw_metrics = audio_stats(raw_audio, sample_rate)
    result = {"raw_metrics": raw_metrics}
    if not np.all(np.isfinite(raw_audio)):
        result.update({"ok": False, "reason": "audio contains NaN or Inf"})
        return result

    audio, sr = preprocess_audio(raw_audio, sample_rate)
    processed_metrics = audio_stats(audio, sr)
    result["processed_metrics"] = processed_metrics
    if processed_debug_path is not None:
        save_debug_wav(audio, sr, processed_debug_path)
    if audio.size == 0 or processed_metrics["duration"] <= 0:
        result.update({"ok": False, "reason": "preprocess produced empty audio"})
        return result

    if raw_metrics["peak"] < min_raw_peak or raw_metrics["rms"] < min_raw_rms:
        result.update(
            {
                "ok": False,
                "reason": (
                    "audio is silent/too quiet "
                    f"(peak={raw_metrics['peak']:.6f}, rms={raw_metrics['rms']:.6f})"
                ),
            }
        )
        return result

    with tempfile.TemporaryDirectory(prefix="accent_realtime_") as tmpdir:
        wav_path = Path(tmpdir) / "window.wav"
        save_temp_wav(audio, sr, wav_path)
        transcript = transcribe_audio(whisper, wav_path)
        if not transcript:
            result.update({"ok": False, "reason": "Whisper returned an empty transcript"})
            return result

        segments = align_syllables(audio, sr, transcript, whisper, wav_path)
        if wav2vec_extractor is None:
            sequence = extract_sequence(audio, sr, segments)
        else:
            sequence = extract_sequence_with_wav2vec(audio, sr, segments, wav2vec_extractor)
        if sequence.shape[0] < min_syllables:
            return {
                "ok": False,
                "reason": f"alignment/feature sequence too short: {sequence.shape[0]} < {min_syllables}",
                "raw_metrics": raw_metrics,
                "processed_metrics": processed_metrics,
                "transcript": transcript,
                "syllables": len(segment_syllables(transcript)),
                "sequence_len": int(sequence.shape[0]),
            }

    x = pad_sequence(sequence, metadata["max_len"], metadata["feature_dim"])
    x = normalize_sequence(x, metadata["mean"], metadata["std"])
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probabilities))
    return {
        "ok": True,
        "raw_metrics": raw_metrics,
        "processed_metrics": processed_metrics,
        "transcript": transcript,
        "syllables": len(segment_syllables(transcript)),
        "sequence_len": int(sequence.shape[0]),
        "prediction": metadata["label_names"][pred_idx],
        "probabilities": probabilities,
    }


def print_result(result: dict, label_names: list[str]) -> None:
    """Print one prediction result."""
    raw = result.get("raw_metrics")
    processed = result.get("processed_metrics")
    if raw:
        print(
            f"Raw: shape={raw['shape']} | duration={raw['duration']:.2f}s "
            f"| peak={raw['peak']:.6f} | rms={raw['rms']:.6f}"
        )
    if processed:
        print(
            f"Processed: duration={processed['duration']:.2f}s "
            f"| peak={processed['peak']:.6f} | rms={processed['rms']:.6f}"
        )
    if not result["ok"]:
        print(f"Skip: {result['reason']}")
        if "transcript" in result:
            print(f"Transcript: {result['transcript']}")
            print(f"Syllables: {result['syllables']} | Sequence length: {result['sequence_len']}")
        return

    probabilities = result["probabilities"]
    ranked = np.argsort(probabilities)[::-1]
    print(f"Transcript: {result['transcript']}")
    print(f"Syllables: {result['syllables']} | Sequence length: {result['sequence_len']}")
    print(f"Prediction: {result['prediction']} ({probabilities[int(ranked[0])]:.4f})")
    print("Probabilities:")
    for idx in ranked:
        print(f"  {label_names[int(idx)]}: {probabilities[int(idx)]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Near-realtime microphone accent recognition demo.")
    parser.add_argument("--model-path", default=str(PATHS.checkpoints_dir / "best_model.pt"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--whisper-model", default=TRAINING.whisper_model)
    parser.add_argument("--whisper-device", default=TRAINING.whisper_device)
    parser.add_argument("--whisper-compute-type", default=TRAINING.whisper_compute_type)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=AUDIO.sample_rate)
    parser.add_argument("--min-syllables", type=int, default=3)
    parser.add_argument("--wav2vec-device", default="cpu", choices=["auto", "cpu", "cuda"], help="Device for Wav2Vec2 feature extraction when the checkpoint requires it.")
    parser.add_argument("--audio-file", default=None, help="Optional local audio file to test instead of recording microphone.")
    parser.add_argument("--list-input-devices", action="store_true", help="List microphone devices and exit.")
    parser.add_argument("--input-device", default=None, help="Microphone index or name. Uses the system default when omitted.")
    parser.add_argument(
        "--save-debug-dir",
        default=None,
        help="Save raw_mic.wav and processed_mic.wav for microphone diagnosis.",
    )
    parser.add_argument(
        "--debug-from-saved-wav",
        action="store_true",
        help="Reload raw_mic.wav before prediction to match the --audio-file path.",
    )
    parser.add_argument("--min-raw-peak", type=float, default=1e-4)
    parser.add_argument("--min-raw-rms", type=float, default=1e-5)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_input_devices:
        list_input_devices()
        return

    import torch
    from faster_whisper import WhisperModel
    sample_rate = int(args.sample_rate)
    if sample_rate <= 0:
        raise ValueError("--sample-rate must be positive.")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive.")
    if args.min_raw_peak < 0 or args.min_raw_rms < 0:
        raise ValueError("--min-raw-peak and --min-raw-rms cannot be negative.")

    device = choose_device(args.device)
    model_path = resolve_model_path(args.model_path)
    model, metadata = load_model(model_path, device)
    whisper = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)
    wav2vec_extractor = None
    if metadata.get("wav2vec_config"):
        from wav2vec_features import wav2vec_extractor_from_checkpoint

        wav2vec_device = "cuda" if args.wav2vec_device == "auto" and torch.cuda.is_available() else args.wav2vec_device
        wav2vec_extractor = wav2vec_extractor_from_checkpoint({"wav2vec_config": metadata["wav2vec_config"]}, wav2vec_device)

    print(f"Loaded model: {model_path}")
    print(f"PyTorch device: {device}")
    print(f"Whisper: {args.whisper_model} | device={args.whisper_device} | compute_type={args.whisper_compute_type}")
    if args.audio_file:
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        print(f"Audio file: {audio_path}")
        raw_audio, file_sample_rate = load_audio_file(audio_path)
        result = predict_window(
            raw_audio=raw_audio,
            sample_rate=file_sample_rate,
            model=model,
            metadata=metadata,
            whisper=whisper,
            device=device,
            min_syllables=args.min_syllables,
            min_raw_peak=args.min_raw_peak,
            min_raw_rms=args.min_raw_rms,
            wav2vec_extractor=wav2vec_extractor,
        )
        print_result(result, metadata["label_names"])
        return

    input_device = parse_input_device(args.input_device)
    debug_dir = Path(args.save_debug_dir) if args.save_debug_dir else None
    if args.debug_from_saved_wav and debug_dir is None:
        raise ValueError("--debug-from-saved-wav requires --save-debug-dir.")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug audio directory: {debug_dir.resolve()}")
    print(f"Microphone device: {input_device if input_device is not None else 'system default'}")
    print(f"Recording {args.window_seconds:.1f}s windows at {sample_rate} Hz. Press Ctrl+C to stop.")

    try:
        while True:
            print("\nListening...")
            raw_audio = record_microphone(
                args.window_seconds,
                sample_rate,
                input_device=input_device,
            )
            prediction_audio = raw_audio
            prediction_sample_rate = sample_rate
            processed_debug_path = None
            if debug_dir is not None:
                raw_debug_path = debug_dir / "raw_mic.wav"
                processed_debug_path = debug_dir / "processed_mic.wav"
                save_debug_wav(raw_audio, sample_rate, raw_debug_path)
                print(f"Saved raw microphone audio: {raw_debug_path.resolve()}")
                if args.debug_from_saved_wav:
                    prediction_audio, prediction_sample_rate = load_audio_file(raw_debug_path)
                    print("Prediction source: reloaded raw_mic.wav")
            result = predict_window(
                raw_audio=prediction_audio,
                sample_rate=prediction_sample_rate,
                model=model,
                metadata=metadata,
                whisper=whisper,
                device=device,
                min_syllables=args.min_syllables,
                min_raw_peak=args.min_raw_peak,
                min_raw_rms=args.min_raw_rms,
                processed_debug_path=processed_debug_path,
                wav2vec_extractor=wav2vec_extractor,
            )
            if processed_debug_path is not None and processed_debug_path.exists():
                print(f"Saved processed microphone audio: {processed_debug_path.resolve()}")
            print_result(result, metadata["label_names"])
            if not args.loop:
                break
            time.sleep(max(0.0, args.pause_seconds))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()








