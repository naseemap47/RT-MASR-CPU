"""
Sliding-window live streaming for Whisper.

Whisper is an offline model: it transcribes a fixed window (up to 30 s) in one
shot and cannot emit text token-by-token as audio arrives. This module turns it
into a live recogniser with the *sliding window + LocalAgreement* technique
(the approach behind ``whisper_streaming``):

1.  Audio accumulates in a window (``session.audio_buffer``).
2.  Every ``hop_s`` seconds of *new* audio the whole window is re-transcribed.
3.  **LocalAgreement-2**: the longest prefix on which the current and the
    previous hypothesis agree is *committed* (final, never changes again).
    The rest of the current hypothesis is shown as *tentative* text.
4.  **Sliding**: once the window grows past ``max_window_s`` it is cut forward
    to the end of the last Whisper segment that is fully committed, so the cost
    of one pass stays bounded no matter how long the call is. If no such
    segment exists the window is force-committed at ``hard_max_window_s``.
5.  **Silence flush**: when speech is followed by ``silence_flush_s`` of silence,
    everything in the window is committed and the window is emptied.

Typical timeline (hop 1 s)::

    t=1s  window [0,1]   hyp "hello"           -> committed ""        tentative "hello"
    t=2s  window [0,2]   hyp "hello world"     -> committed "hello"   tentative "world"
    t=3s  window [0,3]   hyp "hello world how" -> committed "hello world"  tentative "how"
    ...
    window > max_window_s -> cut at last fully-committed segment end -> window slides

The class is deliberately free of any web-framework code: ``main.py`` feeds PCM
into a :class:`~src.engines.live_call_session.LiveCallSession` and calls
:meth:`WhisperSlidingWindowStreamer.step` (blocking, run it in a thread).
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ── Text units (words, or single characters for CJK) ─────────────────────────

# Scripts written without spaces are compared character by character.
_WIDE = (
    "\u3040-\u30ff"    # Hiragana / Katakana
    "\u3400-\u4dbf"    # CJK extension A
    "\u4e00-\u9fff"    # CJK unified ideographs
    "\uac00-\ud7af"    # Hangul syllables
)
_TOKEN_RE = re.compile(rf"[{_WIDE}]|[^\s{_WIDE}]+")
_NO_SPACE_RE = re.compile(rf"[{_WIDE}\u3000-\u303f\uff00-\uffef]")


@dataclass
class Unit:
    """One comparable piece of a hypothesis."""
    text: str   # as displayed (may carry attached punctuation)
    key: str    # normalised form used for agreement (lower-case, no punctuation)


def _norm_key(token: str) -> str:
    token = unicodedata.normalize("NFKC", token).lower()
    return "".join(c for c in token if not unicodedata.category(c).startswith(("P", "S")))


def split_units(text: str, into: Optional[list[Unit]] = None) -> list[Unit]:
    """
    Split text into units: whitespace-separated words for spaced scripts and
    single characters for CJK. Stray punctuation is attached to the previous
    unit so it never counts as a word of its own.
    """
    units = into if into is not None else []
    for tok in _TOKEN_RE.findall(text):
        key = _norm_key(tok)
        if key:
            units.append(Unit(tok, key))
        elif units:
            units[-1].text += tok
        # leading punctuation with nothing before it is dropped
    return units


def join_units(units: list[Unit]) -> str:
    """Re-assemble units into display text (no spaces around CJK)."""
    out: list[str] = []
    for u in units:
        if out and not (_NO_SPACE_RE.match(out[-1][-1]) or _NO_SPACE_RE.match(u.text[0])):
            out.append(" ")
        out.append(u.text)
    return "".join(out)


def common_prefix_len(a: list[Unit], b: list[Unit]) -> int:
    """Number of leading units whose keys are identical in ``a`` and ``b``."""
    n = 0
    for x, y in zip(a, b):
        if x.key != y.key:
            break
        n += 1
    return n


# ── Configuration / result types ─────────────────────────────────────────────

@dataclass
class StreamingConfig:
    """Tuning knobs; every field can be set from the model YAML ``streaming:`` block."""
    hop_s: float = 1.0               # re-transcribe every this many seconds of new audio
    min_window_s: float = 1.0        # never transcribe less audio than this
    max_window_s: float = 12.0       # slide the window forward past this length
    hard_max_window_s: float = 20.0  # force-commit and cut at this length (Whisper max is 30)
    silence_flush_s: float = 0.8     # trailing silence that finalises the utterance
    preroll_s: float = 0.5           # audio kept when idle-trimming leading silence
    beam_size: int = 1               # 1 = greedy (fast); >1 = beam search
    fallback: bool = False           # temperature-fallback retries (slow; off for live)
    lock_language: bool = True       # after the first detection reuse it for the whole call

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "StreamingConfig":
        cfg = cfg or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})


@dataclass
class StreamUpdate:
    """Result of one streaming pass."""
    committed_text: str          # final text so far (whole call)
    tentative_text: str          # unconfirmed tail of the current window
    full_text: str               # committed + tentative
    changed: bool                # full_text differs from the previous update
    flushed: bool                # utterance was finalised (silence / end of call)
    infer_s: float               # wall-clock of the Whisper pass
    window_s: float              # length of the audio window that was transcribed
    window_start_s: float        # call-time (s) at which that window began
    language: Optional[str]
    timing: dict = field(default_factory=dict)   # engine timing dict (mel/encoder/decode...)


# ── The streamer ─────────────────────────────────────────────────────────────

class WhisperSlidingWindowStreamer:
    """
    Per-call sliding-window streaming state machine.

    Args:
        engine:   A ``WhisperOnnxEngine`` (anything with ``transcribe(audio, language=,
                  beam_size=, fallback=)`` returning Whisper-style segments).
        session:  The call's ``LiveCallSession``. Its ``audio_buffer`` *is* the window;
                  the streamer trims it as the window slides.
        config:   :class:`StreamingConfig`.
        language: Forced language code, or None to auto-detect on the first pass.
    """

    def __init__(self, engine: Any, session: Any, config: Optional[StreamingConfig] = None,
                 language: Optional[str] = None) -> None:
        self.engine = engine
        self.session = session
        self.cfg = config or StreamingConfig()
        self.sr: int = session.sample_rate
        self.reset(language)

    # ── state ────────────────────────────────────────────────────────────
    def reset(self, language: Optional[str] = None) -> None:
        """Start a new call: forget all text and counters."""
        self.language: Optional[str] = language or None
        self.detected_language: Optional[str] = None
        self._committed: list[Unit] = []        # final units, whole call
        self._prev: list[Unit] = []             # previous hypothesis (current window coordinates)
        self._n_committed: int = 0              # leading units of the hypothesis already committed
        self._tentative: list[Unit] = []
        self._pending: bool = False            # speech seen since the last flush
        self._window_start_samples: int = 0     # total samples trimmed off the front so far
        self._last_step_total: int = 0
        self._last_full_text: str = ""

    @property
    def committed_text(self) -> str:
        return join_units(self._committed)

    @property
    def tentative_text(self) -> str:
        return join_units(self._tentative)

    @property
    def full_text(self) -> str:
        return join_units(self._committed + self._tentative)

    # ── scheduling ───────────────────────────────────────────────────────
    def _total_samples(self) -> int:
        return self.session.total_bytes // 2          # int16 PCM

    def ready(self) -> bool:
        """True when ``hop_s`` of new audio arrived and the window is long enough."""
        new = self._total_samples() - self._last_step_total
        return (new >= self.cfg.hop_s * self.sr
                and len(self.session.audio_buffer) >= self.cfg.min_window_s * self.sr)

    # ── window manipulation ──────────────────────────────────────────────
    def _cut(self, n_samples: int) -> None:
        """Drop ``n_samples`` from the front of the window (the window slides)."""
        n_samples = max(0, min(n_samples, len(self.session.audio_buffer)))
        if n_samples:
            self.session.audio_buffer = self.session.audio_buffer[n_samples:]
            self._window_start_samples += n_samples

    def _trim_idle(self) -> None:
        """No speech pending: keep only a short pre-roll so silence cannot pile up."""
        keep = int(self.cfg.preroll_s * self.sr)
        self._cut(len(self.session.audio_buffer) - keep)

    # ── hypothesis handling ──────────────────────────────────────────────
    @staticmethod
    def _hypothesis(result: dict) -> tuple[list[Unit], list[int], list[float]]:
        """
        Flatten Whisper segments into units.

        Returns (units, seg_unit_end, seg_end_time): for segment ``j``,
        ``seg_unit_end[j]`` is the cumulative unit count after it and
        ``seg_end_time[j]`` its end timestamp (s from window start).
        """
        units: list[Unit] = []
        unit_ends: list[int] = []
        end_times: list[float] = []
        for seg in result.get("segments", []):
            split_units(seg.get("text", ""), into=units)
            unit_ends.append(len(units))
            end_times.append(float(seg.get("end", 0.0)))
        return units, unit_ends, end_times

    def _infer(self, flush: bool) -> StreamUpdate:
        cfg = self.cfg
        sr = self.sr
        window = self.session.audio_buffer.copy()
        window_s = len(window) / sr
        window_start_s = self._window_start_samples / sr

        t0 = time.perf_counter()
        try:
            try:
                from src.core.observe import bound
            except ImportError:
                from core.observe import bound
            ctx = bound(window_s=round(window_s, 3), window_start_s=round(window_start_s, 3),
                        flush=flush, pass_kind="sliding_window")
        except ImportError:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            result = self.engine.transcribe(
                window, language=self.language, beam_size=cfg.beam_size, fallback=cfg.fallback,
            )
        infer_s = time.perf_counter() - t0

        lang = result.get("language") or None
        if lang:
            self.detected_language = lang
            if cfg.lock_language and self.language is None and result.get("text", "").strip():
                self.language = lang        # avoid flip-flopping + the per-pass detection cost

        units, unit_ends, end_times = self._hypothesis(result)

        # ── LocalAgreement-2 ────────────────────────────────────────────
        if flush:
            agree = len(units)
        else:
            agree = common_prefix_len(self._prev, units)
            agree = min(max(agree, self._n_committed), len(units))
        if agree > self._n_committed:
            self._committed.extend(units[self._n_committed:agree])
            self._n_committed = agree

        if flush:
            # Utterance finished: everything is final, start from an empty window.
            self._cut(len(self.session.audio_buffer))
            self._prev, self._n_committed, self._tentative = [], 0, []
            self._pending = False
        else:
            self._prev = units
            self._tentative = units[self._n_committed:]
            self._slide(units, unit_ends, end_times, window_s)

        full = self.full_text
        changed = full != self._last_full_text
        self._last_full_text = full
        return StreamUpdate(
            committed_text=self.committed_text,
            tentative_text=self.tentative_text,
            full_text=full,
            changed=changed,
            flushed=flush,
            infer_s=infer_s,
            window_s=window_s,
            window_start_s=window_start_s,
            language=self.language or self.detected_language,
            timing=result.get("timing", {}),
        )

    def _slide(self, units: list[Unit], unit_ends: list[int], end_times: list[float],
               window_s: float) -> None:
        """Cut the window forward once it is too long (keeps pass cost bounded)."""
        cfg, sr = self.cfg, self.sr
        if window_s < cfg.max_window_s or not unit_ends:
            return

        hard = window_s >= cfg.hard_max_window_s
        if hard:
            # Nothing was confirmed in time: accept the current hypothesis as final.
            self._committed.extend(units[self._n_committed:])
            self._n_committed = len(units)
            self._tentative = []
            j = len(unit_ends) - 1
        else:
            # Last segment whose words are all committed.
            j = -1
            for idx, end in enumerate(unit_ends):
                if 0 < end <= self._n_committed:
                    j = idx
            if j < 0:
                return          # nothing safe to cut yet; wait for more agreement

        # Cut where segment j ends. If that is (nearly) the end of the audio, the whole
        # window goes; this only happens on a hard cut, where every unit is committed.
        cut_t = end_times[j]
        if hard and cut_t >= window_s - 0.2:
            cut_samples = len(self.session.audio_buffer)
        else:
            cut_samples = min(int(cut_t * sr), len(self.session.audio_buffer))
        if cut_samples <= 0:
            return
        self._cut(cut_samples)

        # Re-express the hypothesis state in the new (shorter) window's coordinates:
        # the first unit_ends[j] units belonged to the audio that was just removed.
        removed = unit_ends[j]
        self._prev = self._prev[removed:]
        self._n_committed = max(0, self._n_committed - removed)
        self._tentative = self._prev[self._n_committed:]

    # ── public passes ────────────────────────────────────────────────────
    def step(self) -> Optional[StreamUpdate]:
        """
        One scheduled pass (blocking, ~Whisper latency; run in a worker thread).

        Returns None when no inference was needed (silence with nothing pending).
        """
        sr = self.sr
        self._last_step_total = self._total_samples()

        if len(self.session.audio_buffer) < self.cfg.min_window_s * sr:
            return None

        if self.session.has_speech(window_samples=int(self.cfg.hop_s * sr)):
            self._pending = True
        if not self._pending:
            self._trim_idle()
            return None

        tail_silent = not self.session.has_speech(window_samples=int(self.cfg.silence_flush_s * sr))
        return self._infer(flush=tail_silent)

    def finish(self) -> Optional[StreamUpdate]:
        """End of call: transcribe what is left and commit everything."""
        buf = self.session.audio_buffer
        if len(buf) >= 0.1 * self.sr and (self._pending or self.session.has_speech(len(buf))):
            return self._infer(flush=True)

        # Too little / silent audio left: just promote the tentative tail.
        if self._tentative:
            self._committed.extend(self._tentative)
            self._tentative = []
        self._cut(len(self.session.audio_buffer))
        self._prev, self._n_committed, self._pending = [], 0, False
        full = self.full_text
        changed = full != self._last_full_text
        self._last_full_text = full
        return StreamUpdate(
            committed_text=self.committed_text, tentative_text="", full_text=full,
            changed=changed, flushed=True, infer_s=0.0, window_s=0.0,
            window_start_s=self._window_start_samples / self.sr,
            language=self.language or self.detected_language,
        )
