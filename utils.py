"""Utility helpers for Vietnamese accent recognition."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_dir: Path, name: str = "accent_recognition") -> logging.Logger:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy and TensorFlow if available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def safe_stats(values: np.ndarray) -> tuple[float, float]:
    """Return finite mean/std or zeros for empty arrays."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def print_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    logger: logging.Logger,
) -> None:
    """Log classification report and confusion matrix."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    logger.info("Classification report:\n%s", report)
    logger.info("Confusion matrix:\n%s", matrix)

