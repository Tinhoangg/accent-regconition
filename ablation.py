"""Feature-group ablation experiments for the 68D accent feature cache."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from config import FEATURES, PATHS, TRAINING
try:
    from feature_extraction import FEATURE_NAMES
except ModuleNotFoundError:
    FEATURE_NAMES = [
        "duration",
        "F1_mean", "F1_std", "F2_mean", "F2_std", "F3_mean", "F3_std", "F4_mean", "F4_std",
        "F0_mean", "F0_std", "F0_min", "F0_max", "F0_range", "F0_slope", "F0_start", "F0_mid", "F0_end", "voiced_ratio",
        *[f"MFCC_{idx}_mean" for idx in range(1, 14)],
        *[f"MFCC_{idx}_std" for idx in range(1, 14)],
        *[f"delta_MFCC_{idx}_mean" for idx in range(1, 14)],
        "energy_mean", "energy_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "zero_crossing_rate_mean", "zero_crossing_rate_std",
    ]


LABEL_NAMES = list(TRAINING.label2id.keys())


DEFAULT_EXPERIMENTS = [
    "all_features",
    "no_duration",
    "no_formant",
    "no_pitch",
    "no_mfcc",
    "no_energy",
    "no_spectral",
    "formant_only",
    "pitch_only",
    "formant_pitch_only",
]


def setup_ablation_logger(log_dir: Path, experiment: str | None = None) -> logging.Logger:
    """Create a logger for ablation runs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "ablation" if experiment is None else f"ablation.{experiment}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_name = "ablation.log" if experiment is None else "train.log"
    file_handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def feature_groups() -> dict[str, list[int]]:
    """Return fixed feature-group indexes for the 68D feature vector."""
    groups = {
        "duration": [0],
        "formant": list(range(1, 9)),
        "pitch": list(range(9, 19)),
        "mfcc": list(range(19, 58)),
        "energy": list(range(58, 60)),
        "spectral": list(range(60, 68)),
    }
    all_indexes = sorted(idx for indexes in groups.values() for idx in indexes)
    expected = list(range(FEATURES.feature_dim))
    if all_indexes != expected:
        raise ValueError(f"Feature groups do not cover 0..{FEATURES.feature_dim - 1}")
    if len(FEATURE_NAMES) != FEATURES.feature_dim:
        raise ValueError(f"Expected {FEATURES.feature_dim} feature names, got {len(FEATURE_NAMES)}")
    return groups


def print_feature_groups() -> None:
    """Print feature groups with indexes and names."""
    for group_name, indexes in feature_groups().items():
        names = [FEATURE_NAMES[idx] for idx in indexes]
        print(f"{group_name}: {len(indexes)} dims | indexes={indexes}")
        print("  " + ", ".join(names))


def expand_experiments(experiments: list[str]) -> list[str]:
    """Expand the default experiment alias and remove duplicates."""
    expanded: list[str] = []
    for experiment in experiments:
        names = DEFAULT_EXPERIMENTS if experiment == "default" else [experiment]
        for name in names:
            if name not in expanded:
                expanded.append(name)
    return expanded


def active_feature_indexes(experiment: str, groups: dict[str, list[int]]) -> list[int]:
    """Return feature indexes kept active for one experiment."""
    all_indexes = set(range(FEATURES.feature_dim))
    if experiment == "all_features":
        return sorted(all_indexes)
    if experiment.startswith("no_"):
        group_name = experiment.removeprefix("no_")
        if group_name not in groups:
            raise ValueError(f"Unknown ablation group: {group_name}")
        return sorted(all_indexes - set(groups[group_name]))
    if experiment == "formant_only":
        return groups["formant"]
    if experiment == "pitch_only":
        return groups["pitch"]
    if experiment == "formant_pitch_only":
        return sorted(groups["formant"] + groups["pitch"])
    raise ValueError(f"Unknown experiment: {experiment}")


def mask_features(x: np.ndarray, active_indexes: list[int]) -> np.ndarray:
    """Return a copy of x with inactive feature dimensions zeroed out."""
    masked = x.copy()
    active_mask = np.zeros(FEATURES.feature_dim, dtype=bool)
    active_mask[active_indexes] = True
    masked[..., ~active_mask] = 0.0
    return masked


def save_history(log_dir: Path, history: dict[str, list[float]]) -> None:
    """Save one experiment history as JSON and CSV."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    with (log_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        for idx in range(len(history["loss"])):
            writer.writerow([
                idx,
                history["accuracy"][idx],
                history["loss"][idx],
                history["val_accuracy"][idx],
                history["val_loss"][idx],
            ])


def save_summary(summary_dir: Path, rows: list[dict]) -> None:
    """Save aggregate ablation summary."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    if not rows:
        return
    fieldnames = [
        "experiment",
        "active_dim",
        "best_epoch",
        "best_val_accuracy",
        "best_val_loss",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        *[f"{label}_f1" for label in LABEL_NAMES],
        "confusion_matrix",
    ]
    with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["confusion_matrix"] = json.dumps(row["confusion_matrix"], ensure_ascii=False)
            writer.writerow(csv_row)


def train_one_experiment(
    experiment: str,
    x: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray | None,
    args: argparse.Namespace,
    groups: dict[str, list[int]],
) -> dict:
    """Run one ablation experiment and return summary metrics."""
    import torch
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from torch import nn

    from model import build_model
    from utils import print_evaluation, set_seed
    from train import (
        make_loader,
        normalize_by_train,
        run_epoch,
        save_torch_checkpoint,
        split_data,
    )

    experiment_log_dir = PATHS.logs_dir / "ablation" / experiment
    logger = setup_ablation_logger(experiment_log_dir, experiment)
    active_indexes = active_feature_indexes(experiment, groups)
    logger.info("Starting experiment=%s | active_dim=%s | active_indexes=%s", experiment, len(active_indexes), active_indexes)

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    x_masked = mask_features(x, active_indexes)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_masked, y, args.seed, speakers, logger)
    x_train, x_val, x_test, mean, std = normalize_by_train(x_train, x_val, x_test)
    logger.info("Train=%s Val=%s Test=%s", len(y_train), len(y_val), len(y_test))

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = build_model(args.max_len, FEATURES.feature_dim, num_classes=len(TRAINING.label2id)).to(args.torch_device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: dict[str, list[float]] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = PATHS.checkpoints_dir / "ablation" / f"{experiment}_best_model.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, args.torch_device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, args.torch_device)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoch %s/%s | loss=%.4f accuracy=%.4f | val_loss=%.4f val_accuracy=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_torch_checkpoint(checkpoint_path, model, mean, std, args)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %s | best_epoch=%s", epoch + 1, best_epoch)
                break

    save_history(experiment_log_dir, history)
    checkpoint = torch.load(checkpoint_path, map_location=args.torch_device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    _, _, y_true, predictions = run_epoch(model, test_loader, criterion, args.torch_device)

    per_class_f1 = f1_score(y_true, predictions, average=None, labels=list(range(len(LABEL_NAMES))), zero_division=0)
    matrix = confusion_matrix(y_true, predictions, labels=list(range(len(LABEL_NAMES))))
    logger.info("Accuracy: %.4f", accuracy_score(y_true, predictions))
    logger.info("Precision macro: %.4f", precision_score(y_true, predictions, average="macro", zero_division=0))
    logger.info("Recall macro: %.4f", recall_score(y_true, predictions, average="macro", zero_division=0))
    logger.info("F1 macro: %.4f", f1_score(y_true, predictions, average="macro", zero_division=0))
    print_evaluation(y_true, predictions, LABEL_NAMES, logger)

    row = {
        "experiment": experiment,
        "active_dim": len(active_indexes),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    for label, value in zip(LABEL_NAMES, per_class_f1):
        row[f"{label}_f1"] = float(value)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=False, default=None)
    parser.add_argument("--print-feature-groups", action="store_true")
    parser.add_argument("--experiments", nargs="+", default=["default"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=TRAINING.random_state)
    parser.add_argument("--max-len", type=int, default=TRAINING.max_len)
    parser.add_argument("--learning-rate", type=float, default=TRAINING.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = feature_groups()
    if args.print_feature_groups:
        print_feature_groups()
        return
    if args.cache_path is None:
        raise ValueError("--cache-path is required unless --print-feature-groups is used.")

    import torch

    from train import load_cached_features
    from utils import ensure_dirs, set_seed

    ensure_dirs([PATHS.logs_dir / "ablation", PATHS.checkpoints_dir / "ablation"])
    logger = setup_ablation_logger(PATHS.logs_dir / "ablation")
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if args.device == "auto":
        args.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.torch_device = torch.device(args.device)
    logger.info("Using PyTorch device: %s", args.torch_device)

    cache_path = Path(args.cache_path)
    cached = load_cached_features(cache_path)
    if cached is None:
        raise RuntimeError(f"Could not load feature cache: {cache_path}")
    x, y, speakers = cached
    logger.info("Loaded cache: %s | shape=%s | samples=%s", cache_path, x.shape, len(y))

    experiments = expand_experiments(args.experiments)
    logger.info("Experiments: %s", experiments)
    rows = []
    for experiment in experiments:
        rows.append(train_one_experiment(experiment, x, y, speakers, args, groups))
        save_summary(PATHS.logs_dir / "ablation", rows)
    logger.info("Saved ablation summary: %s", PATHS.logs_dir / "ablation")


if __name__ == "__main__":
    main()






