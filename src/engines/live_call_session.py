"""
LiveCallSession — stateful buffer and telemetry for a single WebSocket call leg.

Input audio contract
--------------------
Baseline: 16 kHz, mono, Linear PCM, little-endian Int16 — the format every
engine's feature extractor expects. A client may declare a different PCM format
in the ``start_call`` message (``set_input_format``); multi-channel audio is
averaged to mono and other sample rates are resampled inside
``process_pcm_bytes`` so nothing past this class sees a non-baseline sample.
Non-Linear-PCM encodings are rejected rather than guessed at.

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


SUPPORTED_ENCODINGS = ("pcm_s16le",)


class LiveCallSession:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = np.array([], dtype=np.float32)

        # ── Declared input format (see set_input_format) ───────────────────
        # Defaults are the baseline contract, so a client that declares nothing
        # is taken at its word: 16 kHz mono little-endian Int16.
        self.input_sample_rate: int = sample_rate
        self.input_channels: int = 1
        self.input_encoding: str = "pcm_s16le"
        self._byte_carry: bytes = b""              # partial frame from the last packet
        self._resample_carry = np.array([], dtype=np.float32)
        self._resample_pos: float = 0.0

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
        self._obs_span = None           # AI observability parent trace (call)

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
        self._reset_input_format()
        self._end_observe_trace()
        try:
            from src.core.observe import start_trace
        except ImportError:
            from core.observe import start_trace  # type: ignore
        try:
            self._obs_span = start_trace("call", tags=["call"])
        except Exception:
            self._obs_span = None

    def _end_observe_trace(self, outputs: dict | None = None) -> None:
        sp = getattr(self, "_obs_span", None)
        if sp is None:
            return
        self._obs_span = None
        try:
            sp.finish(outputs=outputs)
        except Exception:
            pass

    def mark_call_end(self, outputs: dict | None = None) -> None:
        """Close the observability trace opened by mark_call_start."""
        self._end_observe_trace(outputs)

    # ── Input format negotiation / normalisation ───────────────────────────

    def _reset_input_format(self) -> None:
        self.input_sample_rate = self.sample_rate
        self.input_channels = 1
        self.input_encoding = "pcm_s16le"
        self._byte_carry = b""
        self._resample_carry = np.array([], dtype=np.float32)
        self._resample_pos = 0.0

    def set_input_format(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Declare the PCM format the client is about to send.

        The baseline is 16 kHz mono little-endian Int16; anything else is
        normalised to it inside process_pcm_bytes() before the audio can reach
        an engine, so the rest of the pipeline only ever sees the baseline.

        Raises
        ------
        ValueError
            If the encoding is not Linear PCM Int16, or the rate / channel
            count is not a usable number. The caller should report this to the
            client rather than transcribing audio it cannot interpret.
        """
        self._reset_input_format()

        if encoding is not None:
            enc = str(encoding).lower()
            if enc not in SUPPORTED_ENCODINGS:
                raise ValueError(
                    f"unsupported encoding '{encoding}'; expected one of {', '.join(SUPPORTED_ENCODINGS)} "
                    "(companded formats such as G.711 mu-law/A-law must be expanded to Linear PCM first)"
                )
            self.input_encoding = enc

        if sample_rate is not None:
            rate = int(sample_rate)
            if not 4000 <= rate <= 192000:
                raise ValueError(f"unsupported sample_rate {sample_rate}; expected 4000–192000 Hz")
            self.input_sample_rate = rate

        if channels is not None:
            ch = int(channels)
            if not 1 <= ch <= 8:
                raise ValueError(f"unsupported channel count {channels}; expected 1–8")
            self.input_channels = ch

        return self.input_format()

    def input_format(self) -> Dict[str, Any]:
        """The accepted input format plus which normalisation steps it triggers."""
        return {
            "sample_rate": self.input_sample_rate,
            "channels": self.input_channels,
            "encoding": self.input_encoding,
            "downmix": self.input_channels > 1,
            "resample": self.input_sample_rate != self.sample_rate,
            "normalized_to": {"sample_rate": self.sample_rate, "channels": 1, "encoding": "pcm_s16le"},
        }

    def _resample_to_baseline(self, mono: np.ndarray) -> np.ndarray:
        """Linear-interpolate `mono` from input_sample_rate up/down to sample_rate.

        Phase is carried across packets (`_resample_pos`, `_resample_carry`) so
        chunk seams do not produce clicks. Linear interpolation is a fallback for
        non-baseline clients: it is cheap and dependency-free but a poorer
        anti-alias filter than the browser's resampler, so clients that can send
        16 kHz should. The realistic case is 8 kHz telephony, which has no
        content above 4 kHz for imaging to fold back.
        """
        src = np.concatenate([self._resample_carry, mono]) if len(self._resample_carry) else mono
        if len(src) < 2:
            self._resample_carry = src
            return np.array([], dtype=np.float32)

        step = self.input_sample_rate / self.sample_rate     # input samples per output sample
        n_out = int(np.floor((len(src) - 1 - self._resample_pos) / step)) + 1
        if n_out <= 0:
            self._resample_carry = src
            return np.array([], dtype=np.float32)

        idx = self._resample_pos + np.arange(n_out) * step
        out = np.interp(idx, np.arange(len(src)), src).astype(np.float32)

        consumed = int(np.floor(idx[-1]))
        self._resample_pos = idx[-1] + step - consumed
        self._resample_carry = src[consumed:]
        return out

    def process_pcm_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Normalise an incoming PCM packet and accumulate it into audio_buffer.

        Int16 → float32/32768, multi-channel interleaved → averaged mono,
        non-baseline rate → resampled to self.sample_rate. A packet that ends
        mid-frame is not an error: the trailing bytes are held back and prefixed
        to the next packet, so an ill-aligned client cannot abort the call leg.

        T1 (first_chunk_time) is set here on the first call.
        """
        self.chunks_received += 1
        self.total_bytes += len(raw_bytes)

        if self.first_chunk_time is None:
            self.first_chunk_time = time.time()  # T1

        if self._byte_carry:
            raw_bytes = self._byte_carry + raw_bytes
        frame_bytes = 2 * self.input_channels
        usable = len(raw_bytes) - len(raw_bytes) % frame_bytes
        self._byte_carry = raw_bytes[usable:]

        int16_samples = np.frombuffer(raw_bytes, dtype=np.int16, count=usable // 2)
        float32_samples = int16_samples.astype(np.float32) / 32768.0

        if self.input_channels > 1:
            float32_samples = float32_samples.reshape(-1, self.input_channels).mean(axis=1)
        if self.input_sample_rate != self.sample_rate:
            float32_samples = self._resample_to_baseline(float32_samples)

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
    # Threshold rationale (measured on real test audio):
    #   Room noise / hiss     : RMS 0.001 – 0.008
    #   Borderline noise      : RMS 0.004 (diagnostic: still hallucinates)
    #   Quiet speech          : RMS 0.02  – 0.04
    #   Normal speech         : RMS 0.04  – 0.12  (measured: 0.045–0.06)
    #   Loud speech           : RMS 0.1   – 0.3
    #   Setting 0.02 gives a 5× margin above max noise and 2× below quiet speech.
    RMS_SPEECH_THRESHOLD: float = 0.02  # ~-34 dBFS

    def has_speech(self, window_samples: int = 8000) -> bool:
        """Return True if the most recent audio window contains speech energy.

        Checks RMS energy of the last ``window_samples`` samples (default 0.5 s
        at 16 kHz). Threshold of 0.02 RMS (~-34 dBFS) provides a 5× margin
        above the measured noise floor (0.004 RMS) and 2× below quiet speech
        (0.02–0.04 RMS).
        """
        if len(self.audio_buffer) == 0:
            return False
        recent = self.audio_buffer[-window_samples:]
        rms = float(np.sqrt(np.mean(recent ** 2)))
        return rms > self.RMS_SPEECH_THRESHOLD

    # ── VAD sentence chunking ──────────────────────────────────────────────

    def find_vad_boundary(
        self,
        min_silence_samples: int = 32000,     # 2.0 s at 16 kHz — minimum committed chunk
        max_utterance_samples: int = 240000,  # 15 s at 16 kHz
    ) -> int | None:
        """Detect a silence boundary in audio_buffer suitable for committing.

        Scans the buffer for a run of low-energy frames long enough to be a
        natural pause, in the range [min_silence_samples, max_utterance_samples].

        ``min_silence_samples`` doubles as the minimum committed chunk length:
        the search only begins at this offset, so any boundary found guarantees
        the committed audio is at least this long. Default 2 s ensures the model
        always receives enough context for reliable (non-hallucinating) output.

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
