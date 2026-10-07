# loadtest/audio.py
"""Build the audio a simulated call leg streams: clips repeated, with pauses, for a fixed call length."""
from __future__ import annotations

from typing import Callable

import numpy as np

SAMPLE_RATE = 16000


def tile_call_audio(wav: np.ndarray, duration_s: float, gap_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Repeat ``wav`` (one utterance) with ``gap_s`` of silence after each copy until the call is
    ``duration_s`` long (the last copy is cut at the end).

    A real call is a series of utterances with pauses, not one clip; the pauses also let the
    streaming logic commit utterances (Qwen3 VAD cut / Whisper silence flush) the way it would live.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    unit = np.concatenate([wav.astype(np.float32), np.zeros(int(max(0.0, gap_s) * sr), dtype=np.float32)])
    if len(unit) == 0:
        raise ValueError("empty audio")
    target = int(duration_s * sr)
    reps = -(-target // len(unit))
    return np.tile(unit, reps)[:target]


def make_call_loader(
    base_loader: Callable[[str], np.ndarray], duration_s: float, gap_s: float
) -> Callable[[str], np.ndarray]:
    """Wrap an audio loader so every file becomes a ``duration_s`` call."""
    return lambda path: tile_call_audio(base_loader(path), duration_s, gap_s)
