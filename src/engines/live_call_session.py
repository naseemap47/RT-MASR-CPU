"""
LiveCallSession — stateful buffer and telemetry for a single WebSocket call leg.

Timestamp capture points
------------------------
T0  session.start_time          : wall-clock at WebSocket accept + start_call message
T1  session.first_chunk_time    : wall-clock when the first audio PCM chunk arrives
T2  session.first_infer_start   : wall-clock just before the first inference call
T3  session.first_token_time    : wall-clock when the first decoded text delta is produced
Tx  supplied by main.py         : wall-clock just before each inference call  (t_infer_start)
Ty  supplied by main.py         : wall-clock just after each inference call   (t_infer_end)

Latency definitions
-------------------
pipeline_latency_ms
    = (T3 - T0) * 1000
    End-to-end latency: from call-start signal to first recognised text character.
    Captures WS negotiation + audio buffering + full inference pipeline.

infer_latency_ms   (per-pass)
    = (Ty - Tx) * 1000
    Pure inference latency for one transcribe_stream pass over the current audio buffer.
    This is the number shown in the "Inference Latency" card.

ttft_ms   (first-pass only)
    = (T3 - T2) * 1000
    Time-to-First-Token: how long after inference started until the first text delta was emitted.

encoder_ms, prefill_ms, decode_ms
    Internal stage timings extracted from the engine's timing dict (if available).

rtf  (Real-Time Factor)
    = inference_time_s / audio_duration_s
    Values < 1 mean faster-than-real-time; values > 1 mean slower.

throughput_tps  (tokens per second)
    = tokens_generated / decode_time_s
    Decoder throughput in tokens/second.

audio_throughput_bps  (bytes per second, network ingress)
    = total_bytes / elapsed_wall_time_s
    Average bit-rate of audio PCM arriving over the WebSocket.
"""

import time
import resource
import numpy as np
from typing import Dict, Any, Optional


class LiveCallSession:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = np.array([], dtype=np.float32)

        # ── Wall-clock anchors ─────────────────────────────────────────────
        self.start_time: float = time.time()           # T0: call-start signal
        self.first_chunk_time: Optional[float] = None  # T1: first PCM chunk
        self.first_infer_start: Optional[float] = None # T2: first infer call
        self.first_token_time: Optional[float] = None  # T3: first text delta

        # ── Per-pass accumulators ──────────────────────────────────────────
        self.total_bytes: int = 0
        self.chunks_received: int = 0
        self.inference_passes: int = 0

        # ── Memory snapshot ────────────────────────────────────────────────
        self._peak_rss_start_kb: int = self._rss_kb()

        # ── VAD / utterance state ──────────────────────────────────────────
        self.committed_text: str = ""   # finalised utterance transcripts

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rss_kb() -> int:
        """Current resident-set-size in KiB (Linux ru_maxrss is in KiB)."""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            return 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def mark_call_start(self) -> None:
        """Reset the T0 anchor to when start_call message is received."""
        self.start_time = time.time()
        self.first_chunk_time = None
        self.first_infer_start = None
        self.first_token_time = None
        self.total_bytes = 0
        self.chunks_received = 0
        self.inference_passes = 0
        self.audio_buffer = np.array([], dtype=np.float32)
        self._peak_rss_start_kb = self._rss_kb()
        self.committed_text = ""

    def process_pcm_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Convert int16 PCM bytes to float32 and accumulate into audio buffer.

        T1 (first_chunk_time) is set here on the first call.
        """
        self.chunks_received += 1
        self.total_bytes += len(raw_bytes)

        if self.first_chunk_time is None:
            self.first_chunk_time = time.time()  # T1

        int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
        float32_samples = int16_samples.astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, float32_samples])
        buffered_seconds = len(self.audio_buffer) / self.sample_rate

        return {
            "chunks_received": self.chunks_received,
            "total_bytes": self.total_bytes,
            "buffered_seconds": round(buffered_seconds, 3),
            "new_samples_count": len(float32_samples),
        }

    def mark_infer_start(self) -> float:
        """Record T2 (first inference start). Returns the timestamp."""
        t = time.time()
        if self.first_infer_start is None:
            self.first_infer_start = t  # T2
        self.inference_passes += 1
        return t

    def mark_first_token(self) -> None:
        """Record T3 (first text delta produced), idempotent."""
        if self.first_token_time is None:
            self.first_token_time = time.time()  # T3

    # ── Silence / energy gate ──────────────────────────────────────────────
    #
    # Why 0.01? Silence/noise floor is typically < 0.003 RMS (~-50 dBFS).
    # Low-energy noise (room tone, hiss) sits at 0.003-0.008 RMS.
    # Diagnostic test showed RMS 0.004 noise still triggers hallucinations.
    # Speech starts at ~0.01 RMS (-40 dBFS) and is typically 0.03-0.3 RMS.
    # Raising from 0.003 → 0.01 closes the 0.003-0.01 noise band.
    RMS_SPEECH_THRESHOLD: float = 0.01  # ~-40 dBFS; safe floor for real speech

    def has_speech(self, window_samples: int = 16000) -> bool:
        """Return True if the audio buffer contains speech energy.

        Checks RMS energy of the last ``window_samples`` samples (default 1 s
        at 16 kHz). Uses a threshold of 0.01 RMS (~-40 dBFS) — comfortably
        above noise floor (~-60 to -50 dBFS) and below speech (~-30 dBFS).

        The window covers 1 s by default so brief silences between words do
        not suppress inference while a sentence is still being spoken.
        """
        if len(self.audio_buffer) == 0:
            return False
        recent = self.audio_buffer[-window_samples:]
        rms = float(np.sqrt(np.mean(recent ** 2)))
        return rms > self.RMS_SPEECH_THRESHOLD

    # ── Hallucination filter ───────────────────────────────────────────────
    #
    # Whisper-architecture models emit coherent but fabricated sentences when
    # given silence, noise, or audio too short for reliable recognition.
    # These are detected by checking if the output is suspiciously short or
    # matches a known hallucination signal.
    #
    # Reference: github.com/openai/whisper/discussions/1536

    # Minimum tokens a real transcription should produce per second of audio.
    # Silence + hallucination typically produces 5-20 tokens regardless of
    # audio length. Real speech at 150 wpm ≈ 2 tokens/s minimum.
    MIN_TOKENS_PER_SECOND: float = 0.5  # if below this, discard as hallucination

    def is_hallucination(
        self,
        text: str,
        tokens_generated: int,
        audio_duration_s: float,
    ) -> bool:
        """Return True if the output looks like a model hallucination.

        Two signals:
        1. Token density: real speech produces at least 0.5 tokens/s.
           Hallucinations are short fixed phrases regardless of audio length.
        2. Zero text: empty/whitespace after strip() is always discarded.

        Parameters
        ----------
        text            : raw decoded text from the model
        tokens_generated: number of subword tokens produced
        audio_duration_s: duration of the audio that was transcribed
        """
        if not text.strip():
            return True  # nothing produced
        if audio_duration_s <= 0:
            return True
        token_density = tokens_generated / audio_duration_s
        return token_density < self.MIN_TOKENS_PER_SECOND

    # ── VAD sentence chunking ──────────────────────────────────────────────

    def find_vad_boundary(
        self,
        min_silence_samples: int = 3200,      # 0.2 s at 16 kHz
        max_utterance_samples: int = 240000,  # 15 s at 16 kHz
    ) -> int | None:
        """Detect a silence boundary in audio_buffer suitable for committing.

        Scans the buffer for a run of low-energy frames long enough to be a
        natural pause, in the range [min_silence_samples, max_utterance_samples].

        Returns the sample index of the detected boundary, or None if no
        clean boundary exists yet (utterance still in progress).

        Forcing: if the buffer exceeds max_utterance_samples, the boundary
        is forced at that limit to prevent unbounded growth.
        """
        buf = self.audio_buffer
        if len(buf) < min_silence_samples * 2:
            return None  # not enough audio to make a judgement

        # Force-commit if the buffer has grown too long
        if len(buf) >= max_utterance_samples:
            return max_utterance_samples

        # Scan in 100 ms (1600 sample) hops for a silent frame
        hop = 1600
        frame_rms_sq_thresh = self.RMS_SPEECH_THRESHOLD ** 2
        search_start = min_silence_samples

        for i in range(search_start, len(buf) - hop, hop):
            frame = buf[i: i + hop]
            if float(np.mean(frame ** 2)) < frame_rms_sq_thresh:
                # Found a quiet frame — use its midpoint as the boundary
                return i + hop // 2

        return None  # boundary not yet found

    def pop_utterance(self, boundary: int) -> np.ndarray:
        """Remove and return audio_buffer[0:boundary], leaving the remainder.

        Called after find_vad_boundary() returns a boundary index. The
        returned array is the completed utterance to transcribe finally.
        The remaining audio_buffer becomes the start of the next utterance.
        """
        utterance = self.audio_buffer[:boundary].copy()
        self.audio_buffer = self.audio_buffer[boundary:]
        return utterance

    def append_committed(self, text: str) -> None:
        """Append a finalised utterance to committed_text.

        Strips the text and joins with a single space to build the growing
        committed transcript. Empty / whitespace-only strings are ignored.
        """
        text = text.strip()
        if text:
            if self.committed_text:
                self.committed_text += " " + text
            else:
                self.committed_text = text

    def get_metrics(
        self,
        infer_duration_s: float,
        stage_timing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the full telemetry payload to send to the UI.

        Parameters
        ----------
        infer_duration_s:
            Wall-clock seconds for the most-recent inference pass (Ty - Tx).
        stage_timing:
            Optional dict from OnnxAsrPipeline._transcribe_chunk() «timing» key,
            containing mel_s, encoder_s, prepare_s, prefill_s, decode_s,
            tokens_generated.
        """
        audio_dur_s = len(self.audio_buffer) / self.sample_rate
        rtf = infer_duration_s / audio_dur_s if audio_dur_s > 0 else 0.0
        wall_elapsed = time.time() - self.start_time
        audio_throughput_bps = (self.total_bytes / wall_elapsed) if wall_elapsed > 0 else 0

        # TTFT: T3 - T2
        ttft_ms: Optional[float] = None
        if self.first_token_time is not None and self.first_infer_start is not None:
            ttft_ms = round((self.first_token_time - self.first_infer_start) * 1000, 1)

        # Pipeline latency: T3 - T0
        pipeline_latency_ms: Optional[float] = None
        if self.first_token_time is not None:
            pipeline_latency_ms = round((self.first_token_time - self.start_time) * 1000, 1)

        # Stage breakdown from engine
        encoder_ms: Optional[float] = None
        prefill_ms: Optional[float] = None
        decode_ms: Optional[float] = None
        mel_ms: Optional[float] = None
        throughput_tps: Optional[float] = None
        tokens_generated: Optional[int] = None

        if stage_timing:
            mel_ms = round(stage_timing.get("mel_s", 0) * 1000, 1)
            encoder_ms = round(stage_timing.get("encoder_s", 0) * 1000, 1)
            prefill_ms = round(stage_timing.get("prefill_s", 0) * 1000, 1)
            decode_ms = round(stage_timing.get("decode_s", 0) * 1000, 1)
            tokens_generated = stage_timing.get("tokens_generated", 0)
            decode_s = stage_timing.get("decode_s", 0)
            if decode_s > 0 and tokens_generated:
                throughput_tps = round(tokens_generated / decode_s, 1)

        # Memory delta from session start
        rss_delta_mb = round((self._rss_kb() - self._peak_rss_start_kb) / 1024, 1)

        return {
            # ── Core timing ─────────────────────────────────────────────
            "audio_duration_s":       round(audio_dur_s, 3),
            "infer_latency_ms":       round(infer_duration_s * 1000, 1),
            "rtf":                    round(rtf, 4),
            "ttft_ms":                ttft_ms,
            "pipeline_latency_ms":    pipeline_latency_ms,
            # ── Stage breakdown ─────────────────────────────────────────
            "mel_ms":                 mel_ms,
            "encoder_ms":             encoder_ms,
            "prefill_ms":             prefill_ms,
            "decode_ms":              decode_ms,
            # ── Throughput ──────────────────────────────────────────────
            "tokens_generated":       tokens_generated,
            "throughput_tps":         throughput_tps,
            "audio_throughput_bps":   round(audio_throughput_bps, 0),
            # ── Volume / concurrency ────────────────────────────────────
            "chunks_received":        self.chunks_received,
            "inference_passes":       self.inference_passes,
            "total_bytes":            self.total_bytes,
            # ── Memory ──────────────────────────────────────────────────
            "rss_delta_mb":           rss_delta_mb,
        }
