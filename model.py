"""PyTorch models for Vietnamese accent classification."""

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
        self.model_type = "bilstm"
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
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

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            lengths = torch.any(features != 0.0, dim=-1).sum(dim=1)
        lengths = lengths.to(features.device).long().clamp(min=1, max=features.shape[1])
        positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths.cpu(),
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


class AccentTransformerPooling(nn.Module):
    """Small Transformer encoder with masked mean, max, and attention pooling."""

    def __init__(
        self,
        max_len: int,
        feature_dim: int,
        num_classes: int = 3,
        transformer_d_model: int = 96,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        attention_hidden_dim: int = 64,
        dense_dim: int = 64,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.model_type = "transformer"
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.transformer_d_model = transformer_d_model
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.attention_hidden_dim = attention_hidden_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.input_projection = nn.Linear(feature_dim, transformer_d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, transformer_d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.attention_pool = AttentionPooling(transformer_d_model, hidden_dim=attention_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model * 3, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            lengths = torch.any(features != 0.0, dim=-1).sum(dim=1)
        lengths = lengths.to(features.device).long().clamp(min=1, max=features.shape[1])
        positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        x = self.input_projection(features)
        x = x + self.position_embedding[:, : features.shape[1], :]
        x = x * mask.unsqueeze(-1).to(x.dtype)
        outputs = self.encoder(x, src_key_padding_mask=~mask)

        float_mask = mask.unsqueeze(-1).to(outputs.dtype)
        avg_pool = torch.sum(outputs * float_mask, dim=1) / float_mask.sum(dim=1).clamp(min=1e-8)
        max_pool = outputs.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        att_pool = self.attention_pool(outputs, mask)
        pooled = torch.cat([avg_pool, max_pool, att_pool], dim=1)
        return self.classifier(pooled)


class AccentStatsMLP(nn.Module):
    """MLP over utterance-level statistics from non-padding sequence frames."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 3,
        dense_dim: int = 128,
        dropout: float = 0.45,
    ) -> None:
        super().__init__()
        self.model_type = "stats_mlp"
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            lengths = torch.any(features != 0.0, dim=-1).sum(dim=1)
        lengths = lengths.to(features.device).long().clamp(min=1, max=features.shape[1])
        positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        float_mask = mask.unsqueeze(-1).to(features.dtype)
        count = float_mask.sum(dim=1).clamp(min=1.0)
        mean = torch.sum(features * float_mask, dim=1) / count
        centered = torch.where(mask.unsqueeze(-1), features - mean.unsqueeze(1), torch.zeros_like(features))
        std = torch.sqrt(torch.sum(centered.square() * float_mask, dim=1) / count).clamp(min=0.0)
        min_pool = features.masked_fill(~mask.unsqueeze(-1), 1e9).min(dim=1).values
        max_pool = features.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        pooled = torch.cat([mean, std, min_pool, max_pool], dim=1)
        return self.classifier(pooled)


def build_model(
    max_len: int,
    feature_dim: int,
    num_classes: int = 3,
    model_type: str = "bilstm",
    lstm_hidden_dim: int = 64,
    attention_hidden_dim: int = 64,
    dense_dim: int = 96,
    dropout: float = 0.35,
    transformer_d_model: int = 96,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
) -> nn.Module:
    """Build an accent model for cached 68D sequence features."""
    if model_type == "bilstm":
        return AccentBiLSTMPooling(
            feature_dim=feature_dim,
            num_classes=num_classes,
            lstm_hidden_dim=lstm_hidden_dim,
            attention_hidden_dim=attention_hidden_dim,
            dense_dim=dense_dim,
            dropout=dropout,
        )
    if model_type == "transformer":
        return AccentTransformerPooling(
            max_len=max_len,
            feature_dim=feature_dim,
            num_classes=num_classes,
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            attention_hidden_dim=attention_hidden_dim,
            dense_dim=dense_dim,
            dropout=dropout,
        )
    if model_type == "stats_mlp":
        return AccentStatsMLP(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dense_dim=dense_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def infer_model_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int | float | str]:
    """Infer model dimensions from older checkpoints without model_config."""
    if "lstm.weight_ih_l0" in state_dict:
        lstm_weight = state_dict["lstm.weight_ih_l0"]
        attention_weight = state_dict["attention_pool.attention.0.weight"]
        classifier_weight = state_dict["classifier.0.weight"]
        return {
            "model_type": "bilstm",
            "lstm_hidden_dim": int(lstm_weight.shape[0] // 4),
            "attention_hidden_dim": int(attention_weight.shape[0]),
            "dense_dim": int(classifier_weight.shape[0]),
            "dropout": 0.35,
        }
    if "input_projection.weight" in state_dict:
        return {
            "model_type": "transformer",
            "transformer_d_model": int(state_dict["input_projection.weight"].shape[0]),
            "attention_hidden_dim": int(state_dict["attention_pool.attention.0.weight"].shape[0]),
            "dense_dim": int(state_dict["classifier.0.weight"].shape[0]),
            "dropout": 0.35,
        }
    if "classifier.0.weight" in state_dict:
        return {
            "model_type": "stats_mlp",
            "dense_dim": int(state_dict["classifier.0.weight"].shape[0]),
            "dropout": 0.35,
        }
    raise ValueError("Could not infer model config from checkpoint state_dict.")


def build_model_from_checkpoint(
    checkpoint: dict,
    feature_dim: int,
    num_classes: int = 3,
) -> nn.Module:
    """Build a model compatible with old and current checkpoints."""
    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = infer_model_config_from_state_dict(checkpoint["state_dict"])
    return build_model(
        max_len=int(checkpoint.get("max_len", 0)),
        feature_dim=feature_dim,
        num_classes=num_classes,
        model_type=str(model_config.get("model_type", "bilstm")),
        lstm_hidden_dim=int(model_config.get("lstm_hidden_dim", 64)),
        attention_hidden_dim=int(model_config.get("attention_hidden_dim", 64)),
        dense_dim=int(model_config.get("dense_dim", 96)),
        dropout=float(model_config.get("dropout", 0.35)),
        transformer_d_model=int(model_config.get("transformer_d_model", 96)),
        transformer_heads=int(model_config.get("transformer_heads", 4)),
        transformer_layers=int(model_config.get("transformer_layers", 2)),
    )
