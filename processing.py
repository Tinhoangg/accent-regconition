"""Streaming, preprocessing and syllable timestamp alignment utilities."""

from __future__ import annotations

import io
import re
import string
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Generator, Iterable

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel

from config import AUDIO, TRAINING


@dataclass
class MetadataStreamSample:
    """A streaming sample containing only dataset metadata."""

    transcript: str
    region: str
    speaker_id: str
    filename: str


@dataclass
class RawStreamSample:
    """A streaming sample with metadata read before audio decoding."""

    audio_field: dict
    transcript: str
    region: str
    speaker_id: str
    filename: str


@dataclass
class StreamSample:
    """A single decoded streaming sample from ViMD."""

    audio: np.ndarray
    sr: int
    transcript: str
    region: str
    speaker_id: str
    filename: str


@dataclass
class SyllableSegment:
    """One syllable with timestamp boundaries."""

    syllable: str
    start: float
    end: float
    score: float | None = None


def preprocess_audio(audio: np.ndarray, sr: int, target_sr: int = AUDIO.sample_rate) -> tuple[np.ndarray, int]:
    """Convert audio to mono 16 kHz, trim silence and normalize amplitude."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            y = np.mean(y, axis=0)
        else:
            y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    pre_trim_audio = y.copy()
    trimmed, _ = librosa.effects.trim(y, top_db=AUDIO.trim_top_db)
    min_samples = max(1, int(AUDIO.min_syllable_duration * target_sr))
    y = trimmed if len(trimmed) >= min_samples else pre_trim_audio

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), target_sr


def preprocess_transcript(text: str) -> str:
    """Lowercase, remove punctuation and normalize whitespace."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    punctuation = string.punctuation + "“”‘’…–—.,;:!?()[]{}"
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_syllables(transcript: str) -> list[str]:
    """Segment Vietnamese transcript by whitespace; each token is a syllable."""
    transcript = preprocess_transcript(transcript)
    return transcript.split() if transcript else []


def decode_audio_field(audio_field: dict) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face Audio field with decode=False."""
    audio_bytes = audio_field.get("bytes")
    if audio_bytes is None:
        raise ValueError(f"Audio bytes are missing for path={audio_field.get('path')}")
    audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    return audio, int(sr)


def stream_metadata_samples(
    dataset_name: str = TRAINING.dataset_name,
    split: str = TRAINING.split,
    shuffle: bool = False,
    seed: int = TRAINING.random_state,
    buffer_size: int = 1000,
) -> Generator[MetadataStreamSample, None, None]:
    """Stream ViMD metadata without materializing the audio column."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if "audio" in getattr(dataset, "column_names", []):
        dataset = dataset.cast_column("audio", Audio(decode=False))
        dataset = dataset.remove_columns(["audio"])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)

    for item in dataset:
        try:
            yield MetadataStreamSample(
                transcript=preprocess_transcript(item["text"]),
                region=str(item["region"]),
                speaker_id=str(item["speakerID"]),
                filename=str(item["filename"]),
            )
        except Exception:
            continue


def stream_raw_samples(
    dataset_name: str = TRAINING.dataset_name,
    split: str = TRAINING.split,
    shuffle: bool = False,
    seed: int = TRAINING.random_state,
    buffer_size: int = 1000,
) -> Generator[RawStreamSample, None, None]:
    """Stream ViMD metadata before decoding audio bytes."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)

    for item in dataset:
        try:
            yield RawStreamSample(
                audio_field=item["audio"],
                transcript=preprocess_transcript(item["text"]),
                region=str(item["region"]),
                speaker_id=str(item["speakerID"]),
                filename=str(item["filename"]),
            )
        except Exception:
            continue


def stream_samples(
    dataset_name: str = TRAINING.dataset_name,
    split: str = TRAINING.split,
    shuffle: bool = False,
    seed: int = TRAINING.random_state,
    buffer_size: int = 1000,
) -> Generator[StreamSample, None, None]:
    """Stream ViMD samples without loading the full dataset into RAM."""
    for sample in stream_raw_samples(
        dataset_name=dataset_name,
        split=split,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
    ):
        try:
            raw_audio, raw_sr = decode_audio_field(sample.audio_field)
            audio, sr = preprocess_audio(raw_audio, raw_sr)
            yield StreamSample(
                audio=audio,
                sr=sr,
                transcript=sample.transcript,
                region=sample.region,
                speaker_id=sample.speaker_id,
                filename=sample.filename,
            )
        except Exception:
            continue


def save_temp_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    """Write a temporary waveform for faster-whisper timestamp inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def _word_key(word: str) -> str:
    """Normalize a word for fuzzy transcript/ASR matching."""
    word = preprocess_transcript(word).replace("đ", "d")
    word = unicodedata.normalize("NFD", word)
    word = "".join(char for char in word if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]", "", word)


def _similar_word_key(left: str, right: str) -> bool:
    """Return whether two normalized word keys are close enough to share timing."""
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio() >= 0.78

def transcribe_word_timestamps(model: WhisperModel, wav_path: Path) -> list[SyllableSegment]:
    """Use faster-whisper native word timestamps as timing anchors."""
    segments, _ = model.transcribe(
        str(wav_path),
        language="vi",
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    words: list[SyllableSegment] = []
    for segment in segments:
        for item in segment.words or []:
            word = preprocess_transcript(item.word)
            if word and item.start is not None and item.end is not None and item.end > item.start:
                words.append(
                    SyllableSegment(
                        syllable=word,
                        start=float(item.start),
                        end=float(item.end),
                        score=float(item.probability),
                    )
                )
    return words


def map_transcript_to_timestamps(
    syllables: list[str],
    asr_words: list[SyllableSegment],
    audio_duration: float,
) -> list[SyllableSegment]:
    """Keep transcript syllables while borrowing only reliable ASR timing anchors."""
    if not syllables:
        return []
    if not asr_words:
        step = audio_duration / len(syllables)
        return [
            SyllableSegment(syllable=syllable, start=i * step, end=(i + 1) * step)
            for i, syllable in enumerate(syllables)
        ]

    reference_keys = [_word_key(word) for word in syllables]
    asr_keys = [_word_key(item.syllable) for item in asr_words]
    matcher = SequenceMatcher(a=reference_keys, b=asr_keys, autojunk=False)
    mapped: list[SyllableSegment | None] = [None] * len(syllables)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for ref_idx, asr_idx in zip(range(i1, i2), range(j1, j2)):
                asr = asr_words[asr_idx]
                mapped[ref_idx] = SyllableSegment(
                    syllable=syllables[ref_idx],
                    start=asr.start,
                    end=asr.end,
                    score=asr.score,
                )
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for ref_idx, asr_idx in zip(range(i1, i2), range(j1, j2)):
                if not _similar_word_key(reference_keys[ref_idx], asr_keys[asr_idx]):
                    continue
                asr = asr_words[asr_idx]
                mapped[ref_idx] = SyllableSegment(
                    syllable=syllables[ref_idx],
                    start=asr.start,
                    end=asr.end,
                    score=asr.score,
                )

    idx = 0
    while idx < len(mapped):
        if mapped[idx] is not None:
            idx += 1
            continue
        start_idx = idx
        while idx < len(mapped) and mapped[idx] is None:
            idx += 1
        end_idx = idx
        left_time = mapped[start_idx - 1].end if start_idx > 0 and mapped[start_idx - 1] else 0.0
        right_time = mapped[end_idx].start if end_idx < len(mapped) and mapped[end_idx] else audio_duration
        if right_time <= left_time:
            left_time = (start_idx / len(syllables)) * audio_duration
            right_time = (end_idx / len(syllables)) * audio_duration
        step = max(0.0, right_time - left_time) / max(1, end_idx - start_idx)
        for offset, ref_idx in enumerate(range(start_idx, end_idx)):
            mapped[ref_idx] = SyllableSegment(
                syllable=syllables[ref_idx],
                start=left_time + offset * step,
                end=left_time + (offset + 1) * step,
            )

    return [item for item in mapped if item is not None and item.end > item.start]


def align_syllables(
    audio: np.ndarray,
    sr: int,
    transcript: str,
    model: WhisperModel,
    temp_wav_path: Path,
) -> list[SyllableSegment]:
    """Align transcript syllables to audio using faster-whisper timestamps."""
    save_temp_wav(audio, sr, temp_wav_path)
    syllables = segment_syllables(transcript)
    asr_words = transcribe_word_timestamps(model, temp_wav_path)
    return map_transcript_to_timestamps(syllables, asr_words, len(audio) / sr)


def crop_segment(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Crop an audio segment by second boundaries."""
    start_idx = max(0, int(start * sr))
    end_idx = min(len(audio), int(end * sr))
    return audio[start_idx:end_idx]
