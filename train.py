"""End-to-end PyTorch training script for Vietnamese accent recognition."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import FEATURES, PATHS, TRAINING
from feature_extraction import FEATURE_NAMES, extract_sequence
from model import build_model
from processing import align_syllables, decode_audio_field, preprocess_audio, stream_raw_samples
from utils import ensure_dirs, print_evaluation, set_seed, setup_logging


LABEL_NAMES = list(TRAINING.label2id.keys())


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


def collect_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logger = setup_logging(PATHS.logs_dir)
    label2id = TRAINING.label2id
    target_counts = {label: 0 for label in label2id}
    sequences, labels, speakers = [], [], []
    alignment_stats = {"segments": 0, "matched_or_replaced": 0, "interpolated_or_fallback": 0}
    whisper = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)
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
            alignment_stats["matched_or_replaced"] += matched_segments
            alignment_stats["interpolated_or_fallback"] += len(segments) - matched_segments
            sequence = extract_sequence(audio, sr, segments)
            if sequence.shape[0] < args.min_syllables:
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


def default_cache_path(args: argparse.Namespace) -> Path:
    shuffle_tag = "shuffle" if args.shuffle else "ordered"
    return PATHS.features_dir / f"{args.split}_{shuffle_tag}_{args.whisper_model}_{args.max_per_class}_per_class_{FEATURES.feature_dim}d.npz"


def load_cached_features(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    if "x" not in data or "y" not in data:
        return None
    x = data["x"].astype(np.float32)
    if x.ndim != 3 or x.shape[-1] != FEATURES.feature_dim:
        raise ValueError(f"Cached feature dim {x.shape[-1] if x.ndim == 3 else 'unknown'} does not match configured feature dim {FEATURES.feature_dim}.")
    speakers = data["speakers"] if "speakers" in data else None
    return x, data["y"].astype(np.int64), speakers

def save_cached_features(cache_path: Path, x: np.ndarray, y: np.ndarray, speakers: list[str] | np.ndarray | None = None) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y, speakers=np.asarray(speakers if speakers is not None else []),
                        feature_names=np.asarray(FEATURE_NAMES), label_names=np.asarray(LABEL_NAMES))


def prepare_arrays(sequences: np.ndarray, labels: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((len(sequences), max_len, FEATURES.feature_dim), dtype=np.float32)
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


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    all_labels, all_predictions = [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(batch_x)
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
        "feature_dim": FEATURES.feature_dim,
        "max_len": args.max_len,
        "num_classes": len(TRAINING.label2id),
        "label_names": LABEL_NAMES,
        "feature_names": FEATURE_NAMES,
        "normalization_mean": mean,
        "normalization_std": std,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=TRAINING.split)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--max-scan", type=int, default=50_000)
    parser.add_argument("--min-syllables", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=TRAINING.random_state)
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--max-len", type=int, default=TRAINING.max_len)
    parser.add_argument("--whisper-model", default=TRAINING.whisper_model)
    parser.add_argument("--whisper-device", default=TRAINING.whisper_device)
    parser.add_argument("--whisper-compute-type", default=TRAINING.whisper_compute_type)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    ensure_dirs([PATHS.data_dir, PATHS.features_dir, PATHS.checkpoints_dir, PATHS.logs_dir])
    logger = setup_logging(PATHS.logs_dir)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    logger.info("Using PyTorch device: %s", device)

    cache_path = Path(args.cache_path) if args.cache_path else default_cache_path(args)
    cached = None if args.force_extract else load_cached_features(cache_path)
    if cached is not None:
        x, y, speakers = cached
        logger.info("Loaded cached features: %s | shape=%s", cache_path, x.shape)
    else:
        sequences, labels, speaker_list = collect_features(args)
        x, y = prepare_arrays(sequences, labels, args.max_len)
        speakers = np.asarray(speaker_list)
        save_cached_features(cache_path, x, y, speakers)
        logger.info("Saved feature cache: %s | shape=%s", cache_path, x.shape)
    log_feature_diagnostics(logger, x, y, speakers, prefix="Dataset")
    expected_total = args.max_per_class * len(TRAINING.label2id)
    if len(y) != expected_total:
        logger.warning(
            "Expected %s samples (%s per class), but cache/extraction contains %s samples. This can happen if not enough valid samples were found before max_scan.",
            expected_total, args.max_per_class, len(y),
        )

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, args.seed, speakers, logger)
    x_train, x_val, x_test, mean, std = normalize_by_train(x_train, x_val, x_test)
    logger.info("Train=%s Val=%s Test=%s", len(y_train), len(y_val), len(y_test))

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = build_model(args.max_len, FEATURES.feature_dim, num_classes=len(TRAINING.label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING.learning_rate, weight_decay=1e-4)

    history: dict[str, list[float]] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    best_model_path = PATHS.checkpoints_dir / "best_model.pt"
    final_model_path = PATHS.checkpoints_dir / "final_model.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoch %s/%s | loss=%.4f accuracy=%.4f | val_loss=%.4f val_accuracy=%.4f",
            epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_torch_checkpoint(best_model_path, model, mean, std, args)

    with (PATHS.logs_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    with (PATHS.logs_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        for idx in range(len(history["loss"])):
            writer.writerow([idx, history["accuracy"][idx], history["loss"][idx], history["val_accuracy"][idx], history["val_loss"][idx]])

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
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
