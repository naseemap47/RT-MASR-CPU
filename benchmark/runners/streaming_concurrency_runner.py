# benchmark/runners/streaming_concurrency_runner.py
"""
Streaming concurrency benchmark: one leg = one independently streamed audio source.

Every leg plays an audio file into the engine the way a live call does -- in
fixed-size PCM chunks that arrive at *real-time pace* (0.5 s of audio every
0.5 s of wall clock) -- and runs the same stream logic the live server runs
(``main.py`` ``/ws/call-stream``):

  * Whisper backends  -> sliding window + LocalAgreement
                         (``WhisperSlidingWindowStreamer``, backlog is skipped)
  * Qwen3 backends    -> energy-gated, VAD-committed utterances
                         (``LiveCallSession.find_vad_boundary``, chunks queue up)

All legs share ONE loaded engine, as in the server. N legs therefore means N
sources streaming at the same time, and the question answered is *"do all N keep
up with live speech?"* rather than "how fast can N queued requests be drained?"

What is measured
----------------
pass latency     wall time of one inference pass
pass RTF         pass latency / seconds of audio that pass covered
staleness        when a pass finished minus when the newest audio it saw had arrived
                 (how far the transcript lags the speaker; includes queueing)
first text       stream start -> first recognised text
end lag          end of audio -> final transcript committed
kept up          p95 staleness and end lag both <= ``lag_threshold_s`` and no errors
"""
from __future__ import annotations

import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.metrics.statistics import summarise
from benchmark.metrics.system_metrics import SystemSampler

SAMPLE_RATE = 16000
STREAM_MODES = ("vad_utterance", "sliding_window")

_EMPTY_STATS = {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}

# Mirrors main.py::_handle_audio_vad -- inference is only attempted once this much
# audio is buffered, on every 2nd chunk, and only when the newest audio has speech.
_VAD_MIN_BUFFER_SAMPLES = 8000


def _stats(values: list[float]) -> dict:
    return summarise(values) if values else dict(_EMPTY_STATS)


def _default_audio_loader(path: str) -> np.ndarray:
    from utils.audio_utils import load_audio     # 16 kHz mono float32 (same as the engines)
    return load_audio(path)


def _sleep_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


@dataclass
class StreamLegResult:
    """One leg = one streamed call."""
    leg: int
    audio_file: str
    audio_s: float
    passes: int = 0
    errors: int = 0
    first_text_s: Optional[float] = None   # stream start -> first text (None: never produced text)
    end_lag_s: float = 0.0                 # end of audio -> final transcript committed
    p95_staleness_s: float = 0.0
    max_staleness_s: float = 0.0
    kept_up: bool = False
    aborted: bool = False                  # overload guard hit: stopped early (see abort_lag_s)
    final_text: str = ""
    raw_pass_latencies: list[float] = field(default_factory=list)
    raw_pass_rtfs: list[float] = field(default_factory=list)
    raw_staleness: list[float] = field(default_factory=list)


@dataclass
class StreamingConcurrencyResult:
    """Results for one concurrency level: ``n_legs`` calls streaming simultaneously."""
    config_id: str
    n_legs: int
    stream_mode: str            # "vad_utterance" | "sliding_window"
    chunk_s: float
    pace: float                 # 1.0 = real time (anything else is for tests/smoke runs only)
    stagger_s: float
    lag_threshold_s: float
    pass_latency_stats: dict    # per inference pass, all legs pooled
    pass_rtf_stats: dict
    staleness_stats: dict       # per inference pass, all legs pooled
    first_text_stats: dict      # per leg
    end_lag_stats: dict         # per leg
    legs_kept_up: int
    legs_with_text: int
    error_count: int
    system_metrics: dict
    wall_elapsed_s: float = 0.0
    total_audio_s: float = 0.0
    total_passes: int = 0
    legs_aborted: int = 0
    legs: list[StreamLegResult] = field(default_factory=list)


class StreamingConcurrencyRunner:
    """
    Benchmarks one shared engine under N simultaneously streamed call legs.

    Args:
        config_id:        Configuration identifier.
        engine_factory:   Zero-argument callable returning an engine (may return an
                          already-loaded one; called once per run()).
        audio_files:      Audio sources; leg i streams ``audio_files[i % len]``.
        legs_list:        Concurrency levels to test (e.g. [1, 2, 4]).
        stream_mode:      ``"sliding_window"`` (Whisper) or ``"vad_utterance"`` (Qwen3).
        streaming_cfg:    Whisper ``streaming:`` block from the model YAML (dict); ignored
                          for ``vad_utterance``.
        chunk_s:          Seconds of audio per chunk (the browser sends 0.5 s).
        stagger_s:        Delay between consecutive leg start times. 0 = every call starts
                          at the same instant (worst case); real calls are not aligned.
        lag_threshold_s:  A leg "kept up" when p95 staleness and end lag stay below this.
        warmup:           Run one un-timed pass before measuring.
        pace:             Playback speed. 1.0 = real time; >1 only for tests and smoke runs
                          (the latency/lag numbers are then not meaningful).
        audio_loader:     ``path -> float32 16 kHz mono waveform`` (injectable for tests).
        abort_lag_s:      Overload guard: a leg whose transcript falls this far behind stops
                          streaming (marked ``aborted``, never "kept up"). Keeps a badly
                          overloaded level from running for minutes. None = never abort.
    """

    def __init__(
        self,
        config_id: str,
        engine_factory: Callable[[], Any],
        audio_files: list[str],
        legs_list: list[int],
        stream_mode: str = "vad_utterance",
        streaming_cfg: Optional[dict] = None,
        chunk_s: float = 0.5,
        stagger_s: float = 0.0,
        lag_threshold_s: float = 2.0,
        warmup: bool = True,
        pace: float = 1.0,
        audio_loader: Optional[Callable[[str], np.ndarray]] = None,
        abort_lag_s: Optional[float] = None,
    ) -> None:
        if stream_mode not in STREAM_MODES:
            raise ValueError(f"stream_mode must be one of {STREAM_MODES}, got {stream_mode!r}")
        if chunk_s <= 0 or pace <= 0:
            raise ValueError("chunk_s and pace must be positive")
        self.config_id = config_id
        self.engine_factory = engine_factory
        self.audio_files = audio_files
        self.legs_list = legs_list
        self.stream_mode = stream_mode
        self.streaming_cfg = streaming_cfg
        self.chunk_s = chunk_s
        self.stagger_s = stagger_s
        self.lag_threshold_s = lag_threshold_s
        self.warmup = warmup
        self.pace = pace
        self.audio_loader = audio_loader or _default_audio_loader
        self.abort_lag_s = abort_lag_s
        self._waves: dict[str, np.ndarray] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _whisper_cfg(self):
        from engines.whisper_streaming import StreamingConfig
        return StreamingConfig.from_dict(self.streaming_cfg)

    def _qwen_infer(self, engine: Any, audio: np.ndarray) -> list[str]:
        """Same collection as main.py::_run_inference (deltas of one transcribe_stream pass)."""
        return [d for d, _ in engine.transcribe_stream(audio, language=None) if d]

    def _warmup(self, engine: Any, wav: np.ndarray) -> None:
        clip = wav[: int(10 * SAMPLE_RATE)]
        if self.stream_mode == "sliding_window":
            cfg = self._whisper_cfg()
            engine.transcribe(clip, language=None, beam_size=cfg.beam_size, fallback=cfg.fallback)
        else:
            self._qwen_infer(engine, clip)

    # ── one leg ───────────────────────────────────────────────────────────

    def _stream_leg(self, engine: Any, leg: int, path: str, wav: np.ndarray, t0: float) -> StreamLegResult:
        """Stream ``wav`` into the engine, chunk by chunk, starting at wall-clock ``t0``."""
        from engines.live_call_session import LiveCallSession

        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        chunk_samples = max(1, int(self.chunk_s * SAMPLE_RATE))
        chunks = [pcm[i: i + chunk_samples * 2] for i in range(0, len(pcm), chunk_samples * 2)]
        n = len(chunks)
        total_samples = len(pcm) // 2

        res = StreamLegResult(leg=leg, audio_file=path, audio_s=total_samples / SAMPLE_RATE)
        if n == 0:
            res.errors = 1
            return res

        def arrival(k: int) -> float:
            """Wall-clock time at which chunk k has fully arrived."""
            samples = min((k + 1) * chunk_samples, total_samples)
            return t0 + samples / SAMPLE_RATE / self.pace

        session = LiveCallSession(sample_rate=SAMPLE_RATE)
        session.mark_call_start()

        def record(t_start: float, t_end: float, window_s: float, newest: int) -> None:
            lat = t_end - t_start
            res.passes += 1
            res.raw_pass_latencies.append(lat)
            if window_s > 0:
                res.raw_pass_rtfs.append(lat / window_s)
            stale = max(0.0, t_end - arrival(newest))
            res.raw_staleness.append(stale)
            if self.abort_lag_s is not None and stale > self.abort_lag_s:
                res.aborted = True

        def note_text(t: float) -> None:
            if res.first_text_s is None:
                res.first_text_s = max(0.0, t - t0)

        if self.stream_mode == "sliding_window":
            from engines.whisper_streaming import WhisperSlidingWindowStreamer
            streamer = WhisperSlidingWindowStreamer(engine, session, self._whisper_cfg())

            k = 0
            while k < n:
                _sleep_until(arrival(k))
                session.process_pcm_bytes(chunks[k])
                newest = k
                k += 1
                if not streamer.ready():
                    continue
                # Skip any backlog (chunks that already arrived) like main.py::_drain_audio.
                now = time.perf_counter()
                while k < n and arrival(k) <= now:
                    session.process_pcm_bytes(chunks[k])
                    newest = k
                    k += 1
                t_s = time.perf_counter()
                try:
                    update = streamer.step()
                except Exception as exc:
                    res.errors += 1
                    print(f"    [stream-conc] leg {leg} pass failed: {exc}")
                    continue
                t_e = time.perf_counter()
                if update is None:          # silence, nothing to transcribe
                    continue
                record(t_s, t_e, update.window_s, newest)
                if update.full_text:
                    note_text(t_e)
                if res.aborted:
                    break

            t_e = time.perf_counter()
            if not res.aborted:
                t_s = t_e
                try:
                    update = streamer.finish()
                except Exception as exc:
                    res.errors += 1
                    update = None
                    print(f"    [stream-conc] leg {leg} finish failed: {exc}")
                t_e = time.perf_counter()
                if update is not None and update.window_s > 0:
                    record(t_s, t_e, update.window_s, n - 1)
                if update is not None and update.full_text:
                    note_text(t_e)
            res.final_text = streamer.committed_text
            t_done = t_e

        else:   # vad_utterance
            t_done = time.perf_counter()
            for k in range(n):
                _sleep_until(arrival(k))
                stats = session.process_pcm_bytes(chunks[k])
                if not (
                    len(session.audio_buffer) >= _VAD_MIN_BUFFER_SAMPLES
                    and stats["chunks_received"] % 2 == 0
                    and session.has_speech()
                ):
                    continue
                boundary = session.find_vad_boundary()
                commit = boundary is not None
                audio = session.pop_utterance(boundary) if commit else session.audio_buffer
                t_s = time.perf_counter()
                try:
                    deltas = self._qwen_infer(engine, audio)
                except Exception as exc:
                    res.errors += 1
                    print(f"    [stream-conc] leg {leg} pass failed: {exc}")
                    continue
                t_e = time.perf_counter()
                record(t_s, t_e, len(audio) / SAMPLE_RATE, k)
                if deltas:
                    note_text(t_e)
                if commit:
                    session.append_committed("".join(deltas))
                if res.aborted:
                    break

            # end_call: transcribe what is left in the open-utterance buffer
            t_done = time.perf_counter()
            if len(session.audio_buffer) > 0 and not res.aborted:
                audio = session.audio_buffer
                t_s = time.perf_counter()
                try:
                    deltas = self._qwen_infer(engine, audio)
                except Exception as exc:
                    res.errors += 1
                    deltas = []
                    print(f"    [stream-conc] leg {leg} final pass failed: {exc}")
                t_done = time.perf_counter()
                record(t_s, t_done, len(audio) / SAMPLE_RATE, n - 1)
                if deltas:
                    note_text(t_done)
                session.append_committed("".join(deltas))
            res.final_text = session.committed_text

        res.end_lag_s = max(0.0, t_done - arrival(n - 1))
        if res.raw_staleness:
            st = _stats(res.raw_staleness)
            res.p95_staleness_s = st["p95"]
            res.max_staleness_s = st["max"]
        res.kept_up = (
            res.errors == 0
            and not res.aborted
            and res.passes > 0
            and res.p95_staleness_s <= self.lag_threshold_s
            and res.end_lag_s <= self.lag_threshold_s
        )
        return res

    # ── one concurrency level ─────────────────────────────────────────────

    def _run_one_level(
        self,
        engine: Any,
        n_legs: int,
        waves: dict[str, np.ndarray],
        stagger_s: Optional[float] = None,
        leg_offset: int = 0,
        start_at: Optional[float] = None,
    ) -> StreamingConcurrencyResult:
        """
        Run ``n_legs`` simultaneous legs.

        ``leg_offset`` is the global index of the first leg: leg ``i`` streams
        ``audio_files[(leg_offset + i) % len]`` and starts ``(leg_offset + i) * stagger_s``
        after the common start, so several processes can each run a slice of one
        larger level. ``start_at`` (``time.time()`` epoch) is that common start; it
        lets separate processes begin together. Default: now.
        """
        stagger = self.stagger_s if stagger_s is None else stagger_s
        idx = [leg_offset + i for i in range(n_legs)]
        paths = [self.audio_files[g % len(self.audio_files)] for g in idx]
        print(f"  [stream-conc] {n_legs} leg(s) streaming concurrently "
              f"({self.stream_mode}, {self.chunk_s}s chunks, pace x{self.pace:g})")

        legs: list[StreamLegResult] = []
        wall_start = time.perf_counter()
        if start_at is None:
            base = wall_start + 0.05
        else:
            base = wall_start + max(0.0, start_at - time.time())
        with SystemSampler(interval_s=0.1) as sampler:
            with ThreadPoolExecutor(max_workers=max(1, n_legs)) as executor:
                def _run(i: int) -> StreamLegResult:
                    path = paths[i]
                    try:
                        return self._stream_leg(engine, idx[i], path, waves[path], base + idx[i] * stagger)
                    except Exception as exc:
                        print(f"    [stream-conc] leg {idx[i]} crashed: {exc}")
                        return StreamLegResult(leg=idx[i], audio_file=path,
                                               audio_s=len(waves[path]) / SAMPLE_RATE, errors=1)
                legs = list(executor.map(_run, range(n_legs)))
        wall_elapsed = time.perf_counter() - wall_start

        lat = [x for r in legs for x in r.raw_pass_latencies]
        rtf = [x for r in legs for x in r.raw_pass_rtfs]
        stale = [x for r in legs for x in r.raw_staleness]
        first = [r.first_text_s for r in legs if r.first_text_s is not None]
        return StreamingConcurrencyResult(
            config_id=self.config_id,
            n_legs=n_legs,
            stream_mode=self.stream_mode,
            chunk_s=self.chunk_s,
            pace=self.pace,
            stagger_s=stagger,
            lag_threshold_s=self.lag_threshold_s,
            pass_latency_stats=_stats(lat),
            pass_rtf_stats=_stats(rtf),
            staleness_stats=_stats(stale),
            first_text_stats=_stats(first),
            end_lag_stats=_stats([r.end_lag_s for r in legs]),
            legs_kept_up=sum(1 for r in legs if r.kept_up),
            legs_with_text=sum(1 for r in legs if r.final_text.strip()),
            error_count=sum(r.errors for r in legs),
            system_metrics=sampler.result(),
            wall_elapsed_s=wall_elapsed,
            total_audio_s=sum(r.audio_s for r in legs),
            total_passes=sum(r.passes for r in legs),
            legs_aborted=sum(1 for r in legs if r.aborted),
            legs=legs,
        )

    # ── public API ────────────────────────────────────────────────────────

    def prepare(self, engine: Any, max_legs: int) -> None:
        """Decode the audio the first ``max_legs`` legs need and run the un-timed warm-up."""
        if not self.audio_files:
            raise ValueError("StreamingConcurrencyRunner needs at least one audio file")
        needed = {self.audio_files[i % len(self.audio_files)] for i in range(max(1, max_legs))}
        for p in needed - set(self._waves):      # decode up front so file I/O never counts against a leg
            self._waves[p] = np.asarray(self.audio_loader(p), dtype=np.float32)
        if self.warmup and self._waves:
            print("  [stream-conc] warm-up pass (not timed)")
            try:
                self._warmup(engine, next(iter(self._waves.values())))
            except Exception as exc:
                print(f"    [stream-conc] warm-up error: {exc}")
            self.warmup = False

    def run_level(
        self,
        engine: Any,
        n_legs: int,
        stagger_s: Optional[float] = None,
        leg_offset: int = 0,
        start_at: Optional[float] = None,
    ) -> StreamingConcurrencyResult:
        """Run one level on an already-loaded engine (used by the load-test workers)."""
        self.prepare(engine, leg_offset + n_legs)
        return self._run_one_level(engine, n_legs, self._waves, stagger_s, leg_offset, start_at)

    def run(self) -> list[StreamingConcurrencyResult]:
        """Run every concurrency level in ``legs_list`` (one result per level)."""
        engine = self.engine_factory()
        try:
            self.prepare(engine, max(self.legs_list) if self.legs_list else 1)
            return [self._run_one_level(engine, n, self._waves) for n in self.legs_list]
        finally:
            del engine
            gc.collect()
