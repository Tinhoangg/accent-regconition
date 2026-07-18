"""Evaluate an equal-weight ensemble on one fixed speaker-disjoint test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config import PATHS, TRAINING
from model import build_model_from_checkpoint
from train import load_cached_features, split_data


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize_with_checkpoint(x: np.ndarray, checkpoint: dict) -> np.ndarray:
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    mask = np.any(x != 0.0, axis=-1, keepdims=True)
    return np.where(mask, (x - mean) / std, 0.0).astype(np.float32)


def predict_probabilities(
    checkpoint: dict,
    x_test: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> np.ndarray:
    feature_dim = int(checkpoint["feature_dim"])
    if x_test.shape[-1] != feature_dim:
        raise ValueError(f"Checkpoint feature_dim={feature_dim}, cache feature_dim={x_test.shape[-1]}")
    if int(checkpoint["max_len"]) != x_test.shape[1]:
        raise ValueError(f"Checkpoint max_len={checkpoint['max_len']}, cache max_len={x_test.shape[1]}")

    lengths = np.any(x_test != 0.0, axis=-1).sum(axis=1).astype(np.int64)
    normalized = normalize_with_checkpoint(x_test, checkpoint)
    model = build_model_from_checkpoint(checkpoint, feature_dim, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(normalized[start:end]).float().to(device)
            batch_lengths = torch.from_numpy(lengths[start:end]).long().to(device)
            probabilities = torch.softmax(model(batch_x, batch_lengths), dim=1)
            outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    per_class_recall = recall_score(
        y_true,
        predictions,
        labels=list(range(len(label_names))),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "per_class_recall": {name: float(per_class_recall[idx]) for idx, name in enumerate(label_names)},
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=list(range(len(label_names))),
        ).tolist(),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name} | accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"macro_recall={metrics['macro_recall']:.4f}"
    )
    print(f"  per_class_recall={metrics['per_class_recall']}")
    print(f"  confusion_matrix={metrics['confusion_matrix']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an equal-weight checkpoint ensemble.")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--model-paths", nargs="+", required=True)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=str(PATHS.logs_dir / "ensemble_results.json"))
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    cached = load_cached_features(cache_path)
    if cached is None:
        raise FileNotFoundError(f"Invalid or missing feature cache: {cache_path}")
    x, y, speakers = cached

    model_paths = [Path(path) for path in args.model_paths]
    checkpoints = []
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoints.append(torch.load(path, map_location="cpu", weights_only=False))

    stored_split_seeds = {
        int(checkpoint["training_config"]["split_seed"])
        for checkpoint in checkpoints
        if checkpoint.get("training_config", {}).get("split_seed") is not None
    }
    if len(stored_split_seeds) > 1:
        raise ValueError(f"Checkpoints use different split seeds: {sorted(stored_split_seeds)}")
    if args.split_seed is not None:
        split_seed = args.split_seed
        if stored_split_seeds and split_seed not in stored_split_seeds:
            raise ValueError(f"--split-seed={split_seed} does not match checkpoint split seed {stored_split_seeds}")
    elif stored_split_seeds:
        split_seed = next(iter(stored_split_seeds))
    else:
        split_seed = TRAINING.random_state
        print(f"Warning: old checkpoints have no split metadata; using split_seed={split_seed}.")

    _, _, x_test, _, _, y_test = split_data(x, y, split_seed, speakers)
    label_names = list(checkpoints[0].get("label_names", TRAINING.label2id.keys()))
    for checkpoint in checkpoints[1:]:
        if list(checkpoint.get("label_names", label_names)) != label_names:
            raise ValueError("All checkpoints must use the same label order.")

    device = resolve_device(args.device)
    print(f"Device: {device} | split_seed={split_seed} | test_samples={len(y_test)}")
    member_probabilities = []
    results: dict[str, object] = {
        "cache_path": str(cache_path),
        "split_seed": split_seed,
        "test_samples": int(len(y_test)),
        "members": {},
    }
    for path, checkpoint in zip(model_paths, checkpoints):
        probabilities = predict_probabilities(checkpoint, x_test, device, args.batch_size, len(label_names))
        member_probabilities.append(probabilities)
        metrics = metric_dict(y_test, probabilities, label_names)
        results["members"][str(path)] = metrics
        print_metrics(str(path), metrics)

    ensemble_probabilities = np.mean(np.stack(member_probabilities, axis=0), axis=0)
    ensemble_metrics = metric_dict(y_test, ensemble_probabilities, label_names)
    results["ensemble"] = ensemble_metrics
    print_metrics("ensemble", ensemble_metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()