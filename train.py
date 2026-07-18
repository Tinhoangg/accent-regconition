"""End-to-end PyTorch training script for Vietnamese accent recognition."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import FEATURES, PATHS, TRAINING
from feature_extraction import FEATURE_NAMES, extract_sequence, extract_sequence_with_wav2vec
from model import build_model
from processing import align_syllables, decode_audio_field, preprocess_audio, segment_syllables, stream_raw_samples
from utils import ensure_dirs, print_evaluation, set_seed, setup_logging
from wav2vec_features import Wav2VecConfig, Wav2VecSyllableExtractor, wav2vec_feature_names


LABEL_NAMES = list(TRAINING.label2id.keys())


def uses_wav2vec_features(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_wav2vec_features", False))


def feature_dim_for_args(args: argparse.Namespace) -> int:
    return FEATURES.feature_dim + (int(args.wav2vec_output_dim) if uses_wav2vec_features(args) else 0)


def feature_names_for_dim(feature_dim: int, args: argparse.Namespace | None = None) -> list[str]:
    if args is not None and uses_wav2vec_features(args):
        return FEATURE_NAMES + wav2vec_feature_names(int(args.wav2vec_output_dim))
    if feature_dim == len(FEATURE_NAMES):
        return list(FEATURE_NAMES)
    extra_dim = max(0, feature_dim - len(FEATURE_NAMES))
    return list(FEATURE_NAMES) + wav2vec_feature_names(extra_dim)


def resolve_wav2vec_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--wav2vec-device cuda was requested, but CUDA is not available.")
    return requested


def build_wav2vec_extractor(args: argparse.Namespace, logger: logging.Logger | None = None) -> Wav2VecSyllableExtractor | None:
    if not uses_wav2vec_features(args):
        return None
    device = resolve_wav2vec_device(args.wav2vec_device)
    if logger is not None:
        logger.info(
            "Using Wav2Vec2 features | model=%s | layer=%s | output_dim=%s | device=%s",
            args.wav2vec_model, args.wav2vec_layer, args.wav2vec_output_dim, device,
        )
    return Wav2VecSyllableExtractor(
        Wav2VecConfig(
            model_name=args.wav2vec_model,
            layer=args.wav2vec_layer,
            output_dim=args.wav2vec_output_dim,
            device=device,
            projection_seed=args.wav2vec_projection_seed,
        )
    )


def extract_sequence_for_args(
    audio: np.ndarray,
    sr: int,
    segments,
    args: argparse.Namespace,
    wav2vec_extractor: Wav2VecSyllableExtractor | None,
) -> np.ndarray:
    if wav2vec_extractor is None:
        return extract_sequence(audio, sr, segments)
    return extract_sequence_with_wav2vec(audio, sr, segments, wav2vec_extractor)


def class_count_dict(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y.astype(np.int64), minlength=len(LABEL_NAMES))
    return {label: int(counts[idx]) for idx, label in enumerate(LABEL_NAMES)}


def sequence_lengths(x: np.ndarray) -> np.ndarray:
    return np.any(x != 0.0, axis=-1).sum(axis=1)


def log_feature_diagnostics(logger: logging.Logger, x: np.ndarray, y: np.ndarray, speakers: np.ndarray | None, prefix: str) -> None:
    lengths = sequence_lengths(x)
    logger.info(
        "%s feature diagnostics | samples=%s | class_counts=%s | seq_len min/p25/median/p75/max/mean=%s/%.1f/%.1f/%.1f/%s/%.1f | truncated_at_max_len=%s",
        prefix, len(y), class_count_dict(y), int(np.min(lengths)), float(np.percentile(lengths, 25)),
        float(np.median(lengths)), float(np.percentile(lengths, 75)), int(np.max(lengths)),
        float(np.mean(lengths)), int(np.sum(lengths == x.shape[1])),
    )
    if speakers is not None and len(speakers) == len(y):
        logger.info(
            "%s speaker diagnostics | speakers=%s | speakers_by_class=%s",
            prefix, len(set(speakers)), {label: len(set(speakers[y == idx])) for label, idx in TRAINING.label2id.items()},
        )


def log_split_diagnostics(logger: logging.Logger, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                          train_speakers: np.ndarray | None = None, val_speakers: np.ndarray | None = None,
                          test_speakers: np.ndarray | None = None) -> None:
    logger.info("Split class counts | train=%s | val=%s | test=%s", class_count_dict(y_train), class_count_dict(y_val), class_count_dict(y_test))
    if train_speakers is not None and val_speakers is not None and test_speakers is not None:
        logger.info("Split speaker counts | train=%s | val=%s | test=%s", len(set(train_speakers)), len(set(val_speakers)), len(set(test_speakers)))


def split_balance_score(y_subset: np.ndarray, total_counts: np.ndarray, target_fraction: float) -> float:
    subset_counts = np.bincount(y_subset.astype(np.int64), minlength=len(LABEL_NAMES))
    expected_counts = total_counts * target_fraction
    class_error = np.sum(np.abs(subset_counts - expected_counts) / np.maximum(total_counts, 1))
    size_error = abs(len(y_subset) - np.sum(total_counts) * target_fraction) / max(np.sum(total_counts), 1)
    return float(class_error + size_error)


def stratified_group_holdout(x: np.ndarray, y: np.ndarray, groups: np.ndarray, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, int(round(1.0 / holdout_fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total_counts = np.bincount(y.astype(np.int64), minlength=len(LABEL_NAMES))
    best_train_idx, best_holdout_idx, best_score = None, None, float("inf")
    for train_idx, holdout_idx in splitter.split(x, y, groups=groups):
        score = split_balance_score(y[holdout_idx], total_counts, holdout_fraction)
        if score < best_score:
            best_train_idx, best_holdout_idx, best_score = train_idx, holdout_idx, score
    if best_train_idx is None or best_holdout_idx is None:
        raise RuntimeError("Could not create a stratified group split.")
    return best_train_idx, best_holdout_idx


def collect_features_serial(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logger = setup_logging(Path(getattr(args, "run_log_dir", PATHS.logs_dir)))
    label2id = TRAINING.label2id
    target_counts = {label: 0 for label in label2id}
    sequences, labels, speakers = [], [], []
    alignment_stats = {"segments": 0, "matched_or_replaced": 0, "interpolated_or_fallback": 0}
    whisper = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)
    wav2vec_extractor = build_wav2vec_extractor(args, logger)
    temp_dir = PATHS.data_dir / "tmp_wav"
    temp_dir.mkdir(parents=True, exist_ok=True)
    iterator = stream_raw_samples(split=args.split, shuffle=args.shuffle, seed=args.seed, buffer_size=args.buffer_size)
    for index, sample in enumerate(tqdm(iterator, desc="Streaming ViMD")):
        if index >= args.max_scan or all(count >= args.max_per_class for count in target_counts.values()):
            break
        if sample.region not in label2id or target_counts[sample.region] >= args.max_per_class:
            continue
        try:
            raw_audio, raw_sr = decode_audio_field(sample.audio_field)
            audio, sr = preprocess_audio(raw_audio, raw_sr)
            segments = align_syllables(audio, sr, sample.transcript, whisper, temp_dir / f"{sample.filename}.wav")
            alignment_stats["segments"] += len(segments)
            matched_segments = sum(segment.score is not None for segment in segments)
            matched_ratio = matched_segments / max(len(segments), 1)
            alignment_stats["matched_or_replaced"] += matched_segments
            alignment_stats["interpolated_or_fallback"] += len(segments) - matched_segments
            sequence = extract_sequence_for_args(audio, sr, segments, args, wav2vec_extractor)
            min_sequence_len = max(args.min_syllables, args.min_sequence_len)
            if sequence.shape[0] < min_sequence_len:
                logger.warning("Skipping %s: sequence has fewer than %s frames", sample.filename, min_sequence_len)
                continue
            if matched_ratio < args.min_matched_ratio:
                logger.warning("Skipping %s: matched_ratio %.3f < %.3f", sample.filename, matched_ratio, args.min_matched_ratio)
                continue
            sequences.append(sequence)
            labels.append(label2id[sample.region])
            speakers.append(sample.speaker_id)
            target_counts[sample.region] += 1
            logger.info("Collected valid sample %s/%s | %s | counts=%s", sum(target_counts.values()), args.max_per_class * len(label2id), sample.filename, target_counts)
        except Exception as exc:
            logger.warning("Skipping %s: %s", sample.filename, exc)
    if not sequences:
        raise RuntimeError("No feature sequences were extracted.")
    logger.info("Final collected counts: %s", target_counts)
    if alignment_stats["segments"]:
        logger.info("Alignment diagnostics | segments=%s | matched_or_replaced=%s | interpolated_or_fallback=%s | matched_ratio=%.4f",
                    alignment_stats["segments"], alignment_stats["matched_or_replaced"], alignment_stats["interpolated_or_fallback"],
                    alignment_stats["matched_or_replaced"] / alignment_stats["segments"])
    return np.asarray(sequences, dtype=object), np.asarray(labels, dtype=np.int64), speakers


_WORKER_WHISPER: WhisperModel | None = None
_WORKER_WAV2VEC: Wav2VecSyllableExtractor | None = None
_WORKER_TEMP_DIR: Path | None = None
_WORKER_MIN_SYLLABLES = 3
_WORKER_MIN_MATCHED_RATIO = 0.0
_WORKER_MIN_SEQUENCE_LEN = 0


def init_feature_worker(
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    temp_dir: str,
    min_syllables: int,
    min_matched_ratio: float,
    min_sequence_len: int,
    use_wav2vec_features: bool,
    wav2vec_model: str,
    wav2vec_layer: int,
    wav2vec_output_dim: int,
    wav2vec_device: str,
    wav2vec_projection_seed: int,
) -> None:
    global _WORKER_WHISPER, _WORKER_WAV2VEC, _WORKER_TEMP_DIR, _WORKER_MIN_SYLLABLES, _WORKER_MIN_MATCHED_RATIO, _WORKER_MIN_SEQUENCE_LEN
    _WORKER_WHISPER = WhisperModel(whisper_model, device=whisper_device, compute_type=whisper_compute_type)
    _WORKER_TEMP_DIR = Path(temp_dir)
    _WORKER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    _WORKER_MIN_SYLLABLES = min_syllables
    _WORKER_MIN_MATCHED_RATIO = min_matched_ratio
    _WORKER_MIN_SEQUENCE_LEN = min_sequence_len
    _WORKER_WAV2VEC = None
    if use_wav2vec_features:
        device = resolve_wav2vec_device(wav2vec_device)
        _WORKER_WAV2VEC = Wav2VecSyllableExtractor(
            Wav2VecConfig(
                model_name=wav2vec_model,
                layer=wav2vec_layer,
                output_dim=wav2vec_output_dim,
                device=device,
                projection_seed=wav2vec_projection_seed,
            )
        )


def raw_sample_payload(sample) -> dict:
    return {
        "audio_field": sample.audio_field,
        "transcript": sample.transcript,
        "region": sample.region,
        "speaker_id": sample.speaker_id,
        "filename": sample.filename,
    }


def extract_feature_worker(payload: dict) -> dict:
    try:
        if _WORKER_WHISPER is None or _WORKER_TEMP_DIR is None:
            raise RuntimeError("Feature worker was not initialized.")
        raw_audio, raw_sr = decode_audio_field(payload["audio_field"])
        audio, sr = preprocess_audio(raw_audio, raw_sr)
        safe_filename = str(payload["filename"]).replace("/", "_").replace("\\", "_")
        temp_wav_path = _WORKER_TEMP_DIR / f"{os.getpid()}_{safe_filename}.wav"
        segments = align_syllables(audio, sr, payload["transcript"], _WORKER_WHISPER, temp_wav_path)
        matched_segments = sum(segment.score is not None for segment in segments)
        matched_ratio = matched_segments / max(len(segments), 1)
        if _WORKER_WAV2VEC is None:
            sequence = extract_sequence(audio, sr, segments)
        else:
            sequence = extract_sequence_with_wav2vec(audio, sr, segments, _WORKER_WAV2VEC)
        min_sequence_len = max(_WORKER_MIN_SYLLABLES, _WORKER_MIN_SEQUENCE_LEN)
        if sequence.shape[0] < min_sequence_len:
            return {
                "ok": False,
                "filename": payload["filename"],
                "region": payload["region"],
                "reason": f"sequence has fewer than {min_sequence_len} frames",
                "skip": True,
            }
        if matched_ratio < _WORKER_MIN_MATCHED_RATIO:
            return {
                "ok": False,
                "filename": payload["filename"],
                "region": payload["region"],
                "reason": f"matched_ratio {matched_ratio:.3f} < {_WORKER_MIN_MATCHED_RATIO:.3f}",
                "skip": True,
            }
        return {
            "ok": True,
            "filename": payload["filename"],
            "region": payload["region"],
            "speaker_id": payload["speaker_id"],
            "sequence": sequence,
            "segments": len(segments),
            "matched_or_replaced": matched_segments,
            "interpolated_or_fallback": len(segments) - matched_segments,
            "matched_ratio": matched_ratio,
        }
    except Exception as exc:
        return {
            "ok": False,
            "filename": payload.get("filename", "unknown"),
            "region": payload.get("region", "unknown"),
            "reason": str(exc),
            "skip": False,
        }


def collect_features_parallel(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logger = setup_logging(Path(getattr(args, "run_log_dir", PATHS.logs_dir)))
    label2id = TRAINING.label2id
    target_counts = {label: 0 for label in label2id}
    pending_counts = {label: 0 for label in label2id}
    sequences, labels, speakers = [], [], []
    alignment_stats = {"segments": 0, "matched_or_replaced": 0, "interpolated_or_fallback": 0}
    if args.whisper_device == "cuda":
        logger.warning("num_workers=%s with whisper-device=cuda can increase VRAM use; CPU workers are usually safer.", args.num_workers)

    temp_dir = PATHS.data_dir / "tmp_wav" / "workers"
    temp_dir.mkdir(parents=True, exist_ok=True)
    iterator = enumerate(tqdm(stream_raw_samples(split=args.split, shuffle=args.shuffle, seed=args.seed, buffer_size=args.buffer_size), desc="Streaming ViMD"))
    pending = {}
    max_pending = max(args.num_workers * 2, args.num_workers)
    scan_done = False
    submitted = 0

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_feature_worker,
        initargs=(args.whisper_model, args.whisper_device, args.whisper_compute_type, str(temp_dir), args.min_syllables, args.min_matched_ratio, args.min_sequence_len, args.use_wav2vec_features, args.wav2vec_model, args.wav2vec_layer, args.wav2vec_output_dim, args.wav2vec_device, args.wav2vec_projection_seed),
    ) as executor:
        while True:
            while len(pending) < max_pending and not scan_done:
                if all(count >= args.max_per_class for count in target_counts.values()):
                    scan_done = True
                    break
                try:
                    index, sample = next(iterator)
                except StopIteration:
                    scan_done = True
                    break
                if index >= args.max_scan:
                    scan_done = True
                    break
                if sample.region not in label2id:
                    continue
                if target_counts[sample.region] + pending_counts[sample.region] >= args.max_per_class:
                    continue
                future = executor.submit(extract_feature_worker, raw_sample_payload(sample))
                pending[future] = (sample.region, sample.filename)
                pending_counts[sample.region] += 1
                submitted += 1

            if not pending:
                if scan_done:
                    break
                continue

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                region, filename = pending.pop(future)
                pending_counts[region] -= 1
                result = future.result()
                if result["ok"]:
                    sequences.append(result["sequence"])
                    labels.append(label2id[result["region"]])
                    speakers.append(result["speaker_id"])
                    target_counts[result["region"]] += 1
                    alignment_stats["segments"] += result["segments"]
                    alignment_stats["matched_or_replaced"] += result["matched_or_replaced"]
                    alignment_stats["interpolated_or_fallback"] += result["interpolated_or_fallback"]
                    logger.info("Collected valid sample %s/%s | %s | matched_ratio=%.3f | counts=%s",
                                sum(target_counts.values()), args.max_per_class * len(label2id), result["filename"], result.get("matched_ratio", 0.0), target_counts)
                else:
                    logger.warning("Skipping %s: %s", result["filename"], result["reason"])

            if scan_done and not pending:
                break

    if not sequences:
        raise RuntimeError("No feature sequences were extracted.")
    logger.info("Final collected counts: %s | submitted_jobs=%s", target_counts, submitted)
    if alignment_stats["segments"]:
        logger.info("Alignment diagnostics | segments=%s | matched_or_replaced=%s | interpolated_or_fallback=%s | matched_ratio=%.4f",
                    alignment_stats["segments"], alignment_stats["matched_or_replaced"], alignment_stats["interpolated_or_fallback"],
                    alignment_stats["matched_or_replaced"] / alignment_stats["segments"])
    return np.asarray(sequences, dtype=object), np.asarray(labels, dtype=np.int64), speakers


def default_chunk_dir(cache_path: Path) -> Path:
    """Return the directory used for resumable feature chunks."""
    return cache_path.with_suffix("").parent / f"{cache_path.with_suffix('').name}_chunks"


def chunk_paths(chunk_dir: Path) -> list[Path]:
    """Return existing chunk files in deterministic order."""
    return sorted(chunk_dir.glob("chunk_*.npz"))


def save_feature_chunk(
    chunk_dir: Path,
    chunk_id: int,
    sequences: list[np.ndarray],
    labels: list[int],
    speakers: list[str],
    filenames: list[str],
    matched_ratios: list[float] | None,
    max_len: int,
    logger: logging.Logger,
) -> Path:
    """Pad and save one resumable extraction chunk, then return its path."""
    if not sequences:
        raise ValueError("Cannot save an empty feature chunk.")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    x_chunk, y_chunk = prepare_arrays(np.asarray(sequences, dtype=object), np.asarray(labels, dtype=np.int64), max_len)
    path = chunk_dir / f"chunk_{chunk_id:05d}.npz"
    np.savez_compressed(
        path,
        x=x_chunk,
        y=y_chunk,
        speakers=np.asarray(speakers),
        filenames=np.asarray(filenames),
        matched_ratios=np.asarray(matched_ratios if matched_ratios is not None else []),
        feature_names=np.asarray(feature_names_for_dim(int(x_chunk.shape[-1]))),
        label_names=np.asarray(LABEL_NAMES),
    )
    logger.info("Saved feature chunk: %s | samples=%s | shape=%s", path, len(y_chunk), x_chunk.shape)
    return path


def load_feature_chunks(chunk_dir: Path, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[str], dict[str, int]] | None:
    """Load all existing chunks and derive resume state."""
    paths = chunk_paths(chunk_dir)
    if not paths:
        return None
    xs, ys, speakers, done_filenames = [], [], [], set()
    for path in paths:
        data = np.load(path, allow_pickle=True)
        if "x" not in data or "y" not in data:
            logger.warning("Ignoring invalid feature chunk: %s", path)
            continue
        x_chunk = data["x"].astype(np.float32)
        y_chunk = data["y"].astype(np.int64)
        if x_chunk.ndim != 3:
            raise ValueError(f"Chunk feature shape mismatch in {path}: {x_chunk.shape}")
        xs.append(x_chunk)
        ys.append(y_chunk)
        if "speakers" in data:
            speakers.extend([str(item) for item in data["speakers"]])
        else:
            speakers.extend([""] * len(y_chunk))
        if "filenames" in data:
            done_filenames.update(str(item) for item in data["filenames"])
    if not xs:
        return None
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    counts = class_count_dict(y)
    logger.info("Loaded feature chunks: %s | samples=%s | counts=%s", len(xs), len(y), counts)
    return x, y, np.asarray(speakers), done_filenames, counts


def collect_features_chunked(args: argparse.Namespace, cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features into resumable on-disk chunks, then merge chunks for training."""
    logger = setup_logging(Path(getattr(args, "run_log_dir", PATHS.logs_dir)))
    label2id = TRAINING.label2id
    id2label = {idx: label for label, idx in label2id.items()}
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else default_chunk_dir(cache_path)
    loaded = load_feature_chunks(chunk_dir, logger)
    done_filenames: set[str] = set()
    if loaded is None:
        target_counts = {label: 0 for label in label2id}
        next_chunk_id = 0
    else:
        _, y_existing, _, done_filenames, _ = loaded
        target_counts = {label: 0 for label in label2id}
        for label_id, count in zip(*np.unique(y_existing, return_counts=True)):
            target_counts[id2label[int(label_id)]] = int(count)
        next_chunk_id = len(chunk_paths(chunk_dir))

    if all(count >= args.max_per_class for count in target_counts.values()):
        logger.info("Chunk cache already satisfies target counts: %s", target_counts)
        merged = load_feature_chunks(chunk_dir, logger)
        if merged is None:
            raise RuntimeError("Chunk cache disappeared while loading.")
        return merged[0], merged[1], merged[2]

    if args.num_workers > 1:
        logger.warning("Chunk cache extraction uses one Whisper worker to keep RAM bounded; ignoring --num-workers=%s during extraction.", args.num_workers)

    whisper = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)
    wav2vec_extractor = build_wav2vec_extractor(args, logger)
    temp_dir = PATHS.data_dir / "tmp_wav" / "chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    iterator = stream_raw_samples(split=args.split, shuffle=args.shuffle, seed=args.seed, buffer_size=args.buffer_size)

    chunk_sequences: list[np.ndarray] = []
    chunk_labels: list[int] = []
    chunk_speakers: list[str] = []
    chunk_filenames: list[str] = []
    chunk_matched_ratios: list[float] = []
    alignment_stats = {"segments": 0, "matched_or_replaced": 0, "interpolated_or_fallback": 0}
    scanned = 0

    def flush_chunk() -> None:
        nonlocal next_chunk_id, chunk_sequences, chunk_labels, chunk_speakers, chunk_filenames, chunk_matched_ratios
        if not chunk_sequences:
            return
        save_feature_chunk(chunk_dir, next_chunk_id, chunk_sequences, chunk_labels, chunk_speakers, chunk_filenames, chunk_matched_ratios, args.max_len, logger)
        next_chunk_id += 1
        chunk_sequences.clear()
        chunk_labels.clear()
        chunk_speakers.clear()
        chunk_filenames.clear()
        chunk_matched_ratios.clear()
        gc.collect()

    for index, sample in enumerate(tqdm(iterator, desc="Streaming ViMD")):
        scanned = index + 1
        if index >= args.max_scan or all(count >= args.max_per_class for count in target_counts.values()):
            break
        if sample.filename in done_filenames:
            continue
        if sample.region not in label2id or target_counts[sample.region] >= args.max_per_class:
            continue
        try:
            raw_audio, raw_sr = decode_audio_field(sample.audio_field)
            audio, sr = preprocess_audio(raw_audio, raw_sr)
            safe_filename = str(sample.filename).replace("/", "_").replace("\\", "_")
            segments = align_syllables(audio, sr, sample.transcript, whisper, temp_dir / f"chunk_{safe_filename}.wav")
            matched_segments = sum(segment.score is not None for segment in segments)
            matched_ratio = matched_segments / max(len(segments), 1)
            alignment_stats["segments"] += len(segments)
            alignment_stats["matched_or_replaced"] += matched_segments
            alignment_stats["interpolated_or_fallback"] += len(segments) - matched_segments
            sequence = extract_sequence_for_args(audio, sr, segments, args, wav2vec_extractor)
            min_sequence_len = max(args.min_syllables, args.min_sequence_len)
            if sequence.shape[0] < min_sequence_len:
                logger.warning("Skipping %s: sequence has fewer than %s frames", sample.filename, min_sequence_len)
                continue
            if matched_ratio < args.min_matched_ratio:
                logger.warning("Skipping %s: matched_ratio %.3f < %.3f", sample.filename, matched_ratio, args.min_matched_ratio)
                continue
            chunk_sequences.append(sequence)
            chunk_labels.append(label2id[sample.region])
            chunk_speakers.append(sample.speaker_id)
            chunk_filenames.append(sample.filename)
            chunk_matched_ratios.append(matched_ratio)
            done_filenames.add(sample.filename)
            target_counts[sample.region] += 1
            logger.info("Collected valid sample %s/%s | %s | matched_ratio=%.3f | counts=%s",
                        sum(target_counts.values()), args.max_per_class * len(label2id), sample.filename, matched_ratio, target_counts)
            if len(chunk_sequences) >= args.chunk_size:
                flush_chunk()
        except Exception as exc:
            logger.warning("Skipping %s: %s", sample.filename, exc)

    flush_chunk()
    logger.info("Final chunked collected counts: %s | scanned=%s | chunks=%s", target_counts, scanned, len(chunk_paths(chunk_dir)))
    if alignment_stats["segments"]:
        logger.info("Alignment diagnostics | segments=%s | matched_or_replaced=%s | interpolated_or_fallback=%s | matched_ratio=%.4f",
                    alignment_stats["segments"], alignment_stats["matched_or_replaced"], alignment_stats["interpolated_or_fallback"],
                    alignment_stats["matched_or_replaced"] / alignment_stats["segments"])
    merged = load_feature_chunks(chunk_dir, logger)
    if merged is None:
        raise RuntimeError("No feature chunks were extracted.")
    return merged[0], merged[1], merged[2]


def collect_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if args.num_workers <= 1:
        return collect_features_serial(args)
    return collect_features_parallel(args)



def debug_syllable_extraction(args: argparse.Namespace, logger: logging.Logger) -> None:
    label2id = TRAINING.label2id
    checked = 0
    target_counts = {label: 0 for label in label2id}
    whisper = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)
    wav2vec_extractor = build_wav2vec_extractor(args, logger)
    temp_dir = PATHS.data_dir / "tmp_wav" / "debug"
    temp_dir.mkdir(parents=True, exist_ok=True)

    iterator = stream_raw_samples(split=args.split, shuffle=args.shuffle, seed=args.seed, buffer_size=args.buffer_size)
    for index, sample in enumerate(tqdm(iterator, desc="Debug syllables")):
        if index >= args.max_scan or checked >= args.debug_syllables_only:
            break
        if sample.region not in label2id or target_counts[sample.region] >= args.max_per_class:
            continue
        try:
            raw_audio, raw_sr = decode_audio_field(sample.audio_field)
            raw_array = np.asarray(raw_audio)
            raw_shape = tuple(raw_array.shape)
            raw_duration = raw_array.shape[0] / max(raw_sr, 1) if raw_array.ndim > 0 else 0.0
            raw_peak = float(np.max(np.abs(raw_array))) if raw_array.size else 0.0
            audio, sr = preprocess_audio(raw_audio, raw_sr)
            processed_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            transcript_syllables = segment_syllables(sample.transcript)
            segments = align_syllables(audio, sr, sample.transcript, whisper, temp_dir / f"debug_{sample.filename}.wav")
            sequence = extract_sequence(audio, sr, segments)
            checked += 1
            target_counts[sample.region] += 1
            processed_duration = len(audio) / sr
            last_end = segments[-1].end if segments else 0.0
            coverage_ratio = last_end / max(processed_duration, 1e-6)
            matched_segments = sum(segment.score is not None for segment in segments)
            matched_ratio = matched_segments / max(len(segments), 1)
            preview_tokens = " ".join(transcript_syllables[:20])
            preview_segments = ", ".join(f"{seg.syllable}:{seg.start:.2f}-{seg.end:.2f}" for seg in segments[:10])
            tail_segments = ", ".join(f"{seg.syllable}:{seg.start:.2f}-{seg.end:.2f}" for seg in segments[-10:])
            logger.info(
                "SYLLABLE_DEBUG %s/%s | file=%s | region=%s | raw_shape=%s | raw_duration=%.2fs | processed_duration=%.2fs | last_end=%.2fs | coverage_ratio=%.3f | raw_peak=%.6f | processed_peak=%.6f | transcript_syllables=%s | aligned_segments=%s | sequence_len=%s | matched_ratio=%.3f | first_tokens=%s | first_segments=%s | last_segments=%s",
                checked, args.debug_syllables_only, sample.filename, sample.region, raw_shape, raw_duration, processed_duration, last_end, coverage_ratio, raw_peak, processed_peak,
                len(transcript_syllables), len(segments), sequence.shape[0], matched_ratio, preview_tokens, preview_segments, tail_segments,
            )
        except Exception as exc:
            checked += 1
            logger.warning("SYLLABLE_DEBUG failed %s/%s | file=%s | region=%s | error=%s",
                           checked, args.debug_syllables_only, sample.filename, sample.region, exc)
    logger.info("SYLLABLE_DEBUG done | checked=%s | counts=%s", checked, target_counts)


def default_cache_path(args: argparse.Namespace) -> Path:
    shuffle_tag = "shuffle" if args.shuffle else "ordered"
    feature_dim = feature_dim_for_args(args)
    wav2vec_tag = f"_w2v_layer{args.wav2vec_layer}_{args.wav2vec_output_dim}d" if uses_wav2vec_features(args) else ""
    return PATHS.features_dir / f"{args.split}_{shuffle_tag}_{args.whisper_model}_{args.max_per_class}_per_class_{feature_dim}d{wav2vec_tag}.npz"


def load_cached_features(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    if "x" not in data or "y" not in data:
        return None
    x = data["x"].astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Cached features must be 3D, got shape {x.shape}.")
    speakers = data["speakers"] if "speakers" in data else None
    return x, data["y"].astype(np.int64), speakers

def save_cached_features(cache_path: Path, x: np.ndarray, y: np.ndarray, speakers: list[str] | np.ndarray | None = None, feature_names: list[str] | None = None) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    names = feature_names if feature_names is not None else feature_names_for_dim(int(x.shape[-1]))
    np.savez_compressed(cache_path, x=x, y=y, speakers=np.asarray(speakers if speakers is not None else []),
                        feature_names=np.asarray(names), label_names=np.asarray(LABEL_NAMES))


def prepare_arrays(sequences: np.ndarray, labels: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = int(sequences[0].shape[-1]) if len(sequences) else FEATURES.feature_dim
    padded = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
    for idx, sequence in enumerate(sequences):
        length = min(len(sequence), max_len)
        if length:
            padded[idx, :length] = sequence[:length]
    return padded, labels.astype(np.int64)


def split_data(x: np.ndarray, y: np.ndarray, seed: int, speakers: np.ndarray | None = None,
               logger: logging.Logger | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if speakers is not None and len(speakers) == len(y) and len(set(speakers)) >= 10:
        holdout_fraction = TRAINING.validation_size + TRAINING.test_size
        try:
            train_idx, temp_idx = stratified_group_holdout(x, y, speakers, holdout_fraction, seed)
        except ValueError as exc:
            if logger is not None:
                logger.warning("Falling back to GroupShuffleSplit for train/temp split: %s", exc)
            first_split = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
            train_idx, temp_idx = next(first_split.split(x, y, groups=speakers))
        temp_groups = speakers[temp_idx]
        relative_test_size = TRAINING.test_size / holdout_fraction
        try:
            val_rel_idx, test_rel_idx = stratified_group_holdout(x[temp_idx], y[temp_idx], temp_groups, relative_test_size, seed)
        except ValueError as exc:
            if logger is not None:
                logger.warning("Falling back to GroupShuffleSplit for val/test split: %s", exc)
            second_split = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
            val_rel_idx, test_rel_idx = next(second_split.split(x[temp_idx], y[temp_idx], groups=temp_groups))
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]
        if logger is not None:
            log_split_diagnostics(logger, y[train_idx], y[val_idx], y[test_idx], speakers[train_idx], speakers[val_idx], speakers[test_idx])
        return x[train_idx], x[val_idx], x[test_idx], y[train_idx], y[val_idx], y[test_idx]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=TRAINING.validation_size + TRAINING.test_size, random_state=seed, stratify=y)
    relative_test_size = TRAINING.test_size / (TRAINING.validation_size + TRAINING.test_size)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=relative_test_size, random_state=seed, stratify=y_temp)
    if logger is not None:
        log_split_diagnostics(logger, y_train, y_val, y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def normalize_by_train(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_mask = np.any(x_train != 0.0, axis=-1)
    frames = x_train[frame_mask]
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    std[std < 1e-6] = 1.0

    def transform(x: np.ndarray) -> np.ndarray:
        mask = np.any(x != 0.0, axis=-1, keepdims=True)
        normalized = (x - mean) / std
        return np.where(mask, normalized, 0.0).astype(np.float32)

    return transform(x_train), transform(x_val), transform(x_test), mean.astype(np.float32), std.astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    lengths = sequence_lengths(x).astype(np.int64)
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(lengths).long(),
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def augment_features(batch_x: torch.Tensor, noise_std: float = 0.0, frame_drop_prob: float = 0.0) -> torch.Tensor:
    """Apply train-only augmentation without changing the original sequence lengths."""
    if noise_std <= 0 and frame_drop_prob <= 0:
        return batch_x
    mask = torch.any(batch_x != 0.0, dim=-1, keepdim=True)
    if noise_std > 0:
        noise = torch.randn_like(batch_x) * noise_std
        batch_x = torch.where(mask, batch_x + noise, batch_x)
    if frame_drop_prob > 0:
        frame_mask = mask.squeeze(-1)
        drop_mask = (torch.rand(frame_mask.shape, device=batch_x.device) < frame_drop_prob) & frame_mask
        batch_x = batch_x.masked_fill(drop_mask.unsqueeze(-1), 0.0)
    return batch_x


def symmetric_kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Return the mean bidirectional KL divergence used by R-Drop."""
    log_prob_a = F.log_softmax(logits_a, dim=-1)
    log_prob_b = F.log_softmax(logits_b, dim=-1)
    kl_ab = F.kl_div(log_prob_a, log_prob_b, reduction="batchmean", log_target=True)
    kl_ba = F.kl_div(log_prob_b, log_prob_a, reduction="batchmean", log_target=True)
    return 0.5 * (kl_ab + kl_ba)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    feature_noise_std: float = 0.0,
    frame_drop_prob: float = 0.0,
    rdrop_alpha: float = 0.0,
    mixup_alpha: float = 0.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    all_labels, all_predictions = [], []
    for batch_x, batch_y, batch_lengths in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_lengths = batch_lengths.to(device)
        target_a = batch_y
        target_b = batch_y
        mixup_lambda = 1.0
        if is_train:
            batch_x = augment_features(batch_x, feature_noise_std, frame_drop_prob)
            if mixup_alpha > 0:
                permutation = torch.randperm(len(batch_y), device=device)
                mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
                batch_x = mixup_lambda * batch_x + (1.0 - mixup_lambda) * batch_x[permutation]
                batch_lengths = torch.maximum(batch_lengths, batch_lengths[permutation])
                target_b = batch_y[permutation]
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            if is_train and rdrop_alpha > 0:
                logits_a = model(batch_x, batch_lengths)
                logits_b = model(batch_x, batch_lengths)
                classification_loss = 0.5 * (criterion(logits_a, batch_y) + criterion(logits_b, batch_y))
                loss = classification_loss + rdrop_alpha * symmetric_kl_divergence(logits_a, logits_b)
                logits = 0.5 * (logits_a + logits_b)
            else:
                logits = model(batch_x, batch_lengths)
                if is_train and mixup_alpha > 0:
                    loss = mixup_lambda * criterion(logits, target_a) + (1.0 - mixup_lambda) * criterion(logits, target_b)
                else:
                    loss = criterion(logits, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(batch_y)
        total_samples += len(batch_y)
        all_labels.append(batch_y.detach().cpu().numpy())
        all_predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    return total_loss / max(total_samples, 1), accuracy_score(labels, predictions), labels, predictions


def save_torch_checkpoint(path: Path, model: nn.Module, mean: np.ndarray, std: np.ndarray, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_dim": int(getattr(args, "feature_dim", FEATURES.feature_dim)),
        "max_len": args.max_len,
        "num_classes": len(TRAINING.label2id),
        "label_names": LABEL_NAMES,
        "feature_names": list(getattr(args, "feature_names", FEATURE_NAMES)),
        "normalization_mean": mean,
        "normalization_std": std,
        "training_config": {
            "run_name": getattr(args, "run_name", None),
            "split_seed": getattr(args, "split_seed", getattr(args, "seed", TRAINING.random_state)),
            "train_seed": getattr(args, "train_seed", getattr(args, "seed", TRAINING.random_state)),
            "optimizer": getattr(args, "optimizer", "adam"),
            "scheduler": getattr(args, "scheduler", "none"),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "rdrop_alpha": getattr(args, "rdrop_alpha", 0.0),
            "mixup_alpha": getattr(args, "mixup_alpha", 0.0),
            "feature_noise_std": getattr(args, "feature_noise_std", 0.0),
            "frame_drop_prob": getattr(args, "frame_drop_prob", 0.0),
            "skip_test": getattr(args, "skip_test", False),
        },
        "wav2vec_config": ({
            "model_name": args.wav2vec_model,
            "layer": args.wav2vec_layer,
            "output_dim": args.wav2vec_output_dim,
            "device": args.wav2vec_device,
            "projection_seed": args.wav2vec_projection_seed,
        } if getattr(args, "use_wav2vec_features", False) else None),
        "model_config": {
            "model_type": getattr(model, "model_type", getattr(args, "model_type", "bilstm")),
            "lstm_hidden_dim": getattr(model, "lstm_hidden_dim", getattr(args, "lstm_hidden_dim", 64)),
            "attention_hidden_dim": getattr(model, "attention_hidden_dim", getattr(args, "attention_hidden_dim", 64)),
            "dense_dim": getattr(model, "dense_dim", getattr(args, "dense_dim", 96)),
            "dropout": getattr(model, "dropout", getattr(args, "dropout", 0.35)),
            "transformer_d_model": getattr(model, "transformer_d_model", getattr(args, "transformer_d_model", 96)),
            "transformer_heads": getattr(model, "transformer_heads", getattr(args, "transformer_heads", 4)),
            "transformer_layers": getattr(model, "transformer_layers", getattr(args, "transformer_layers", 2)),
        },
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=TRAINING.split)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--max-scan", type=int, default=50_000)
    parser.add_argument("--min-syllables", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=TRAINING.random_state, help="Backward-compatible fallback for split and train seeds.")
    parser.add_argument("--split-seed", type=int, default=None, help="Seed used only for the speaker-disjoint data split.")
    parser.add_argument("--train-seed", type=int, default=None, help="Seed used for initialization, dropout and DataLoader shuffling.")
    parser.add_argument("--run-name", default=None, help="Optional name for isolated logs and checkpoints.")
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--max-len", type=int, default=TRAINING.max_len)
    parser.add_argument("--whisper-model", default=TRAINING.whisper_model)
    parser.add_argument("--whisper-device", default=TRAINING.whisper_device)
    parser.add_argument("--whisper-compute-type", default=TRAINING.whisper_compute_type)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--debug-syllables-only", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0, help="Save feature extraction chunks every N valid samples; enables resume.")
    parser.add_argument("--chunk-dir", default=None, help="Optional directory for feature chunks.")
    parser.add_argument("--learning-rate", type=float, default=TRAINING.learning_rate)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--scheduler", default="plateau", choices=["none", "plateau"])
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Early stop after N epochs without val_macro_f1 improvement; 0 disables.")
    parser.add_argument("--model-type", default="bilstm", choices=["bilstm", "transformer", "stats_mlp"])
    parser.add_argument("--transformer-d-model", type=int, default=96)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--attention-hidden-dim", type=int, default=64)
    parser.add_argument("--dense-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--frame-drop-prob", type=float, default=0.0)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the held-out test split after training.")
    parser.add_argument("--use-wav2vec-features", action="store_true", help="Concatenate Wav2Vec2 layer embeddings to each syllable feature vector.")
    parser.add_argument("--wav2vec-model", default="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    parser.add_argument("--wav2vec-layer", type=int, default=9)
    parser.add_argument("--wav2vec-output-dim", type=int, default=64)
    parser.add_argument("--wav2vec-device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--wav2vec-projection-seed", type=int, default=42)
    parser.add_argument("--min-matched-ratio", type=float, default=0.0, help="Skip newly extracted samples below this alignment matched ratio.")
    parser.add_argument("--min-sequence-len", type=int, default=0, help="Skip newly extracted samples with fewer feature frames; combined with --min-syllables.")
    args = parser.parse_args()
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    args.train_seed = args.seed if args.train_seed is None else args.train_seed
    if args.rdrop_alpha < 0 or args.mixup_alpha < 0:
        raise ValueError("--rdrop-alpha and --mixup-alpha must be >= 0")
    if args.wav2vec_output_dim < 1:
        raise ValueError("--wav2vec-output-dim must be >= 1")
    if args.rdrop_alpha > 0 and args.mixup_alpha > 0:
        raise ValueError("R-Drop and Mixup cannot be enabled in the same run.")
    if args.run_name:
        if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
            raise ValueError("--run-name must be a single directory name.")
        run_log_dir = PATHS.logs_dir / args.run_name
        run_checkpoint_dir = PATHS.checkpoints_dir / args.run_name
    else:
        run_log_dir = PATHS.logs_dir
        run_checkpoint_dir = PATHS.checkpoints_dir
    args.run_log_dir = str(run_log_dir)

    ensure_dirs([PATHS.data_dir, PATHS.features_dir, PATHS.checkpoints_dir, PATHS.logs_dir, run_log_dir, run_checkpoint_dir])
    logger = setup_logging(run_log_dir)
    set_seed(args.train_seed)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.debug_syllables_only > 0:
        debug_syllable_extraction(args, logger)
        return

    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using PyTorch device: %s", device)
    logger.info("Seeds | split=%s | train=%s", args.split_seed, args.train_seed)

    cache_path = Path(args.cache_path) if args.cache_path else default_cache_path(args)
    cached = None if args.force_extract else load_cached_features(cache_path)
    if cached is not None:
        x, y, speakers = cached
        logger.info("Loaded cached features: %s | shape=%s", cache_path, x.shape)
    else:
        if args.chunk_size > 0:
            x, y, speakers = collect_features_chunked(args, cache_path)
        else:
            sequences, labels, speaker_list = collect_features(args)
            x, y = prepare_arrays(sequences, labels, args.max_len)
            speakers = np.asarray(speaker_list)
        save_cached_features(cache_path, x, y, speakers, feature_names_for_dim(int(x.shape[-1]), args))
        logger.info("Saved feature cache: %s | shape=%s", cache_path, x.shape)
    args.feature_dim = int(x.shape[-1])
    args.feature_names = feature_names_for_dim(args.feature_dim, args if uses_wav2vec_features(args) else None)
    logger.info("Feature dim: %s", args.feature_dim)
    log_feature_diagnostics(logger, x, y, speakers, prefix="Dataset")
    expected_total = args.max_per_class * len(TRAINING.label2id)
    if len(y) != expected_total:
        logger.warning(
            "Expected %s samples, but cache/extraction contains %s samples. This can happen if not enough valid samples were found before max_scan.",
            expected_total, len(y),
        )

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, args.split_seed, speakers, logger)
    x_train, x_val, x_test, mean, std = normalize_by_train(x_train, x_val, x_test)
    logger.info("Train=%s Val=%s Test=%s", len(y_train), len(y_val), len(y_test))

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True, seed=args.train_seed)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = build_model(
        args.max_len,
        args.feature_dim,
        num_classes=len(TRAINING.label2id),
        model_type=args.model_type,
        lstm_hidden_dim=args.lstm_hidden_dim,
        attention_hidden_dim=args.attention_hidden_dim,
        dense_dim=args.dense_dim,
        dropout=args.dropout,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_learning_rate,
        )

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = run_checkpoint_dir / "best_model.pt"
    final_model_path = run_checkpoint_dir / "final_model.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            feature_noise_std=args.feature_noise_std,
            frame_drop_prob=args.frame_drop_prob,
            rdrop_alpha=args.rdrop_alpha,
            mixup_alpha=args.mixup_alpha,
        )
        val_loss, val_acc, val_true, val_predictions = run_epoch(model, val_loader, criterion, device)
        val_macro_f1 = f1_score(val_true, val_predictions, average="macro", zero_division=0)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["learning_rate"].append(current_lr)
        logger.info(
            "Epoch %s/%s | loss=%.4f accuracy=%.4f | val_loss=%.4f val_accuracy=%.4f val_macro_f1=%.4f | lr=%.2e",
            epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc, val_macro_f1, current_lr,
        )
        if scheduler is not None:
            scheduler.step(val_macro_f1)
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            save_torch_checkpoint(best_model_path, model, mean, std, args)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %s | best_val_macro_f1=%.4f", epoch + 1, best_val_macro_f1)
                break

    with (run_log_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    with (run_log_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss", "val_macro_f1", "learning_rate"])
        for idx in range(len(history["loss"])):
            writer.writerow([idx, history["accuracy"][idx], history["loss"][idx], history["val_accuracy"][idx], history["val_loss"][idx], history["val_macro_f1"][idx], history["learning_rate"][idx]])

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if args.skip_test:
        logger.info("Skipped held-out test evaluation. Use ensemble.py for the final test.")
    else:
        _, _, y_true, predictions = run_epoch(model, test_loader, criterion, device)
        logger.info("Accuracy: %.4f", accuracy_score(y_true, predictions))
        logger.info("Precision macro: %.4f", precision_score(y_true, predictions, average="macro", zero_division=0))
        logger.info("Recall macro: %.4f", recall_score(y_true, predictions, average="macro", zero_division=0))
        logger.info("F1 macro: %.4f", f1_score(y_true, predictions, average="macro", zero_division=0))
        print_evaluation(y_true, predictions, LABEL_NAMES, logger)

    save_torch_checkpoint(final_model_path, model, mean, std, args)
    logger.info("Saved best model: %s", best_model_path)
    logger.info("Saved final model: %s", final_model_path)


if __name__ == "__main__":
    main()










