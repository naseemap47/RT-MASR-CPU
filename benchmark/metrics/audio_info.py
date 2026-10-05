# benchmark/metrics/audio_info.py
"""Audio file metadata helpers (independent of any ASR engine)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=None)
def audio_duration_s(path: str) -> Optional[float]:
    """
    Duration of an audio file in seconds, read from the file header.

    Returns None if the file cannot be parsed, so callers can fall back to
    engine-reported timing. Using the file itself (rather than the engine's
    own number) keeps RTF comparable across engines.
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / info.samplerate
    except Exception:
        pass
    return None
