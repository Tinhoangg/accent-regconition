"""Configuration for the Vietnamese accent recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    """Audio preprocessing configuration."""

    sample_rate: int = 16_000
    trim_top_db: int = 25
    min_syllable_duration: float = 0.04


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction configuration."""

    feature_dim: int = 68
    n_mfcc: int = 13
    pitch_floor: float = 60.0
    pitch_ceiling: float = 500.0
    max_formant: float = 5500.0


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration."""

    dataset_name: str = "nguyendv02/ViMD_Dataset"
    split: str = "train"
    label2id: dict[str, int] = None
    max_len: int = 128
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    random_state: int = 42
    validation_size: float = 0.1
    test_size: float = 0.1
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    cache_features: bool = True

    def __post_init__(self) -> None:
        """Populate immutable default dictionaries."""
        if self.label2id is None:
            object.__setattr__(
                self,
                "label2id",
                {"North": 0, "Central": 1, "South": 2},
            )


@dataclass(frozen=True)
class PathConfig:
    """Project path configuration."""

    root: Path = Path(__file__).resolve().parent
    data_dir: Path = root / "data"
    features_dir: Path = root / "features"
    checkpoints_dir: Path = root / "checkpoints"
    logs_dir: Path = root / "logs"


AUDIO = AudioConfig()
FEATURES = FeatureConfig()
TRAINING = TrainingConfig()
PATHS = PathConfig()

