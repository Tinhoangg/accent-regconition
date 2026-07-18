"""Wav2Vec2 temporal embeddings pooled over syllable timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processing import SyllableSegment


@dataclass(frozen=True)
class Wav2VecConfig:
    model_name: str
    layer: int = 9
    output_dim: int = 64
    device: str = "cpu"
    projection_seed: int = 42


def wav2vec_feature_names(output_dim: int) -> list[str]:
    return [f"wav2vec_{idx:03d}" for idx in range(output_dim)]


class Wav2VecSyllableExtractor:
    """Extract fixed-size Wav2Vec2 vectors for aligned syllable segments."""

    def __init__(self, config: Wav2VecConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Wav2Vec2 features require transformers. Install with `pip install transformers`."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name, output_hidden_states=True)
        self.model.to(config.device)
        self.model.eval()
        self.projection: np.ndarray | None = None

    def _get_projection(self, hidden_dim: int) -> np.ndarray:
        if self.projection is None:
            rng = np.random.default_rng(self.config.projection_seed)
            projection = rng.standard_normal((hidden_dim, self.config.output_dim), dtype=np.float32)
            projection /= np.sqrt(float(self.config.output_dim))
            self.projection = projection.astype(np.float32)
        return self.projection

    def _hidden_states(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        layer = self.config.layer
        if layer < 0:
            layer = len(hidden_states) + layer
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(f"Wav2Vec2 layer {self.config.layer} is out of range for {len(hidden_states)} hidden states.")
        return hidden_states[layer][0].detach().cpu().numpy().astype(np.float32)

    def extract_for_segments(self, audio: np.ndarray, sr: int, segments: list[SyllableSegment]) -> np.ndarray:
        if not segments:
            return np.empty((0, self.config.output_dim), dtype=np.float32)

        hidden = self._hidden_states(audio, sr)
        projection = self._get_projection(hidden.shape[-1])
        projected = hidden @ projection
        duration = len(audio) / max(sr, 1)
        frame_count = projected.shape[0]
        vectors = []
        for segment in segments:
            start = int(np.floor((max(segment.start, 0.0) / max(duration, 1e-6)) * frame_count))
            end = int(np.ceil((max(segment.end, segment.start) / max(duration, 1e-6)) * frame_count))
            start = int(np.clip(start, 0, max(frame_count - 1, 0)))
            end = int(np.clip(end, start + 1, frame_count))
            vectors.append(projected[start:end].mean(axis=0))
        return np.asarray(vectors, dtype=np.float32)



def wav2vec_extractor_from_checkpoint(checkpoint: dict, device: str | None = None) -> Wav2VecSyllableExtractor | None:
    config = checkpoint.get("wav2vec_config")
    if not config:
        return None
    return Wav2VecSyllableExtractor(
        Wav2VecConfig(
            model_name=str(config["model_name"]),
            layer=int(config.get("layer", 9)),
            output_dim=int(config.get("output_dim", 64)),
            device=str(device if device is not None else config.get("device", "cpu")),
            projection_seed=int(config.get("projection_seed", 42)),
        )
    )
