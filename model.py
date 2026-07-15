"""PyTorch BiLSTM model for Vietnamese accent classification."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Mask-aware attention pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(inputs).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(inputs * weights, dim=1)


class AccentBiLSTMPooling(nn.Module):
    """BiLSTM with masked average, max, and attention pooling."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 3,
        lstm_hidden_dim: int = 64,
        attention_hidden_dim: int = 64,
        dense_dim: int = 96,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        pooled_dim = lstm_hidden_dim * 2
        self.attention_pool = AttentionPooling(pooled_dim, hidden_dim=attention_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim * 3, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mask = torch.any(features != 0.0, dim=-1)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=features.shape[1],
        )

        float_mask = mask.unsqueeze(-1).to(outputs.dtype)
        avg_pool = torch.sum(outputs * float_mask, dim=1) / float_mask.sum(dim=1).clamp(min=1e-8)
        max_pool = outputs.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        att_pool = self.attention_pool(outputs, mask)
        pooled = torch.cat([avg_pool, max_pool, att_pool], dim=1)
        return self.classifier(pooled)


def build_model(
    max_len: int,
    feature_dim: int,
    num_classes: int = 3,
    lstm_hidden_dim: int = 64,
    attention_hidden_dim: int = 64,
    dense_dim: int = 96,
    dropout: float = 0.35,
) -> AccentBiLSTMPooling:
    """Build the accent model. max_len is kept for API compatibility."""
    _ = max_len
    return AccentBiLSTMPooling(
        feature_dim=feature_dim,
        num_classes=num_classes,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_hidden_dim=attention_hidden_dim,
        dense_dim=dense_dim,
        dropout=dropout,
    )

def infer_model_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int | float]:
    """Infer model dimensions from a saved AccentBiLSTMPooling state dict."""
    lstm_weight = state_dict["lstm.weight_ih_l0"]
    attention_weight = state_dict["attention_pool.attention.0.weight"]
    classifier_weight = state_dict["classifier.0.weight"]
    lstm_hidden_dim = int(lstm_weight.shape[0] // 4)
    attention_hidden_dim = int(attention_weight.shape[0])
    dense_dim = int(classifier_weight.shape[0])
    return {
        "lstm_hidden_dim": lstm_hidden_dim,
        "attention_hidden_dim": attention_hidden_dim,
        "dense_dim": dense_dim,
        "dropout": 0.35,
    }


def build_model_from_checkpoint(
    checkpoint: dict,
    feature_dim: int,
    num_classes: int = 3,
) -> AccentBiLSTMPooling:
    """Build a model compatible with either old or new checkpoints."""
    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = infer_model_config_from_state_dict(checkpoint["state_dict"])
    return AccentBiLSTMPooling(
        feature_dim=feature_dim,
        num_classes=num_classes,
        lstm_hidden_dim=int(model_config.get("lstm_hidden_dim", 64)),
        attention_hidden_dim=int(model_config.get("attention_hidden_dim", 64)),
        dense_dim=int(model_config.get("dense_dim", 96)),
        dropout=float(model_config.get("dropout", 0.35)),
    )

