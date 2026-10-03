# RT-MASR-CPU — Architectural Changes

> This document records every structural change made to the system, why it was
> made, and how it altered the data-flow. Changes are in chronological order.

---

## 1. Telemetry Instrumentation

### Problem

The original system had no timing data — the UI received raw transcript text and
nothing else. There was no way to measure latency, throughput, or system resource
usage.

### Changes

#### `src/engines/live_call_session.py` — Wall-clock anchor model

A **four-anchor timestamp model** was introduced to precisely define latency:

```
T0  mark_call_start()      "start_call" WS message received
T1  process_pcm_bytes()    First PCM binary frame arrives
T2  mark_infer_start()     Just before engine inference call
T3  mark_first_token()     First text delta emitted from decoder
```

From these anchors, three latency metrics are derived:

| Metric | Formula | Meaning |
|---|---|---|
| `pipeline_latency_ms` | T3 − T0 | End-to-end: call signal → first word |
| `ttft_ms` | T3 − T2 | Engine-only: inference start → first token |
| `infer_latency_ms` | Ty − T2 | Full inference pass wall-clock |

The class also tracks process-level counters (`chunks_received`, `inference_passes`,
`total_bytes`) and a memory delta (RSS at session start vs. now).

`get_metrics()` assembles all of these into a single JSON-serialisable dict sent
to the UI on every `transcript_delta` message.

#### `main.py` — Stage timing capture

`psutil` was added to expose process-level CPU, RSS, threads, and VMS via a
`GET /api/health` endpoint polled every 2 s by the dashboard.

#### Frontend (`index.html`, `style.css`, `app.js`)

An 8-metric card grid was added to the UI. Each card maps 1:1 to a field in the
`metrics` dict. A pipeline waterfall bar visualises the proportional time spent in
each stage (mel → encoder → prefill → decode). The stream-state pill cycles
through five states: IDLE → CONNECTING → BUFFERING → TRANSCRIBING → COMPLETED.

---

## 2. `transcribe_stream` Timing Sentinel

### Problem

`OnnxAsrPipeline.transcribe_stream` was a plain generator that yielded `str`
deltas. There was no way to get per-stage timing (encoder time, prefill time,
decode time) from a streaming call — the caller had to use a separate, wasteful
second pass or estimate externally.

### Design decision: final sentinel yield

The generator's return type was changed from `Generator[str, None, None]` to
`Generator[tuple, None, None]`. Every intermediate yield is now `(delta, None)`.
After the decode loop completes, a single **sentinel** is yielded: `("", timing_dict)`.

```
yield "He",       None   ← real-time delta during decode
yield " hoped",   None
...
yield ".",        None
yield "",         timing  ← sentinel: empty delta, full timing dict
```

This design choice avoids:
- A second inference pass (expensive)
- Storing state on the pipeline object (not thread-safe)
- Changing the generator into a class (over-engineered)

The `timing` dict has the same keys as `transcribe()` — `mel_s`, `encoder_s`,
`prepare_s`, `prefill_s`, `decode_s`, `tokens_generated`, `total_s`,
`audio_duration_s`, `rtf` — making both APIs symmetric.

### Impact on `main.py`

All callers changed from:
```python
for delta in engine.transcribe_stream(...):
    deltas.append(delta)
```
to:
```python
for delta, timing in engine.transcribe_stream(...):
    if delta:
        deltas.append(delta)
    if timing is not None:
        stage_timing = timing
```
The `stage_timing` dict is then passed to `session.get_metrics(stage_timing=...)`,
which unpacks `encoder_ms`, `prefill_ms`, `decode_ms`, and `throughput_tps`.

---

## 3. Fix 1 — Async Executor (Event-Loop Safety)

### Problem

`transcribe_stream` is a **synchronous CPU-bound generator**. When iterated
inside an `async def` WebSocket handler, it runs on the event loop thread.
During inference (~2.5 s on a 10 s clip), the entire asyncio event loop is
**frozen**: no new WebSocket messages can arrive, no responses can be sent, and
no other coroutines can run.

```
BEFORE
──────
Event loop thread:
  receive chunk ──► inference (2.5 s, BLOCKING) ──► send result
                    ╔══════════════════════════╗
                    ║  new chunks pile up in   ║
                    ║  OS TCP buffer (invisible)║
                    ╚══════════════════════════╝
```

For short audio this is hidden by TCP buffer size. For calls > 30 s or any
concurrent sessions it causes backpressure and dropped messages.

### Solution: `_run_inference()` helper + `run_in_executor`

A module-level async helper was extracted:

```python
async def _run_inference(engine, audio_buffer, language):
    loop = asyncio.get_event_loop()

    def _collect():               # sync, runs in thread pool
        deltas, stage_timing = [], None
        for delta, timing in engine.transcribe_stream(audio_buffer, language=language):
            if delta:
                deltas.append(delta)
            if timing is not None:
                stage_timing = timing
        return deltas, stage_timing

    return await loop.run_in_executor(None, _collect)
```

`_collect` is dispatched to Python's default `ThreadPoolExecutor`. The event loop
**awaits** the future, staying free to process incoming chunks in parallel.

```
AFTER
─────
Event loop thread:             Thread pool worker:
  receive chunk ──►            ──► inference (2.5 s, non-blocking)
  receive chunk ──►
  receive chunk ──►            ◄── returns (deltas, stage_timing)
  send result  ◄───────────────
```

---

## 4. Fix 3 — RMS Energy Gate

### Problem

The inference trigger fired every 2 PCM chunks (every ~1 s of wall-clock) regardless
of audio content. On a silent segment — background hiss, hold music, or a pause —
the full mel → encoder → prefill → decode pipeline ran and returned empty output.
This wasted ~2.5 s of CPU for each silent pass.

### Solution: `LiveCallSession.has_speech()`

```python
RMS_SPEECH_THRESHOLD = 0.003  # ~−50 dBFS

def has_speech(self, window_samples=8000):
    recent = self.audio_buffer[-window_samples:]
    rms = float(np.sqrt(np.mean(recent ** 2)))
    return rms > self.RMS_SPEECH_THRESHOLD
```

The threshold is placed between typical noise floor (~−60 dBFS, RMS < 0.001)
and speech floor (~−30 dBFS, RMS > 0.03), giving a 10 dB margin on each side.

The inference trigger condition in `main.py` became:

```python
if (
    len(session.audio_buffer) >= 8000
    and stats["chunks_received"] % 2 == 0
    and session.has_speech()          # new gate
):
```

`has_speech()` is a ~5 µs NumPy operation — negligible compared to the inference
cost it prevents (2–3 s).

---

## 5. Fix 2b — VAD Sentence-Boundary Chunking

### Problem: O(n²) re-transcription growth

The original system accumulated all PCM into a single growing `audio_buffer` and
re-transcribed it entirely on every inference pass:

```
Pass 1 (t=2s):   transcribe [0–2s]     →  2s of work
Pass 2 (t=4s):   transcribe [0–4s]     →  4s of work  (2s re-done)
Pass 3 (t=6s):   transcribe [0–6s]     →  6s of work  (4s re-done)
...
Pass N (t=2Ns):  transcribe [0–2Ns]    →  2Ns of work
Total work: O(N²) in call duration
```

On a 5-minute call with inference every second, the last pass re-encodes 300 s of
audio. The encoder uses full self-attention (O(n²) in sequence length), making
this doubly quadratic.

### Architecture: commit + interim dual path

The new design maintains two regions:

```
audio_buffer (unprocessed, bounded ≤ 15s)   ──VAD──►   committed_text (finalised)
```

Three new methods on `LiveCallSession`:

| Method | Behaviour |
|---|---|
| `find_vad_boundary()` | Scans buffer in 100 ms hops for a silent frame. Returns sample index or None. Forces commit at 15 s. |
| `pop_utterance(boundary)` | Removes `audio_buffer[0:boundary]`, returns it as a new array. |
| `append_committed(text)` | Joins text into `committed_text` with a space. |

### Data flow — two execution paths per trigger

```
On each inference trigger (every 2 chunks, has_speech() == True):

find_vad_boundary()
       │
       ├── boundary found ──► pop_utterance()
       │                      transcribe utterance_audio  (once, then gone)
       │                      append_committed(result)
       │                      audio_buffer = remainder
       │
       └── no boundary ──────► transcribe audio_buffer (interim, bounded ≤ 15s)
                                result shown as live partial text

Display = committed_text + " " + interim
```

### On `end_call`

Any audio remaining in `audio_buffer` is flushed with one final inference pass,
committed, and returned as `final_text`. The buffer is then cleared.

### Inference time is now bounded

| Scenario | Before | After |
|---|---|---|
| 5-minute call, last pass | 5 min of audio re-encoded | ≤ 15 s utterance window |
| 30-minute call | 30 min re-encoded per pass | ≤ 15 s always |
| Compute complexity | O(N²) in call length | O(1) — bounded by window |

The force-commit at 15 s (`max_utterance_samples = 240_000`) ensures the buffer
never grows beyond one encoder-context window regardless of speech continuity.

---

## Summary of architectural layers changed

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (index.html / style.css / app.js)             │
│  • 8-metric telemetry grid                              │
│  • 5-state stream-state machine                         │
│  • Waterfall pipeline bar                               │
└─────────────────────────┬───────────────────────────────┘
                          │ WebSocket (unchanged message shapes)
┌─────────────────────────▼───────────────────────────────┐
│  main.py  (FastAPI WebSocket handler)                   │
│  • _run_inference() — async executor wrapper  (Fix 1)   │
│  • has_speech() gate on trigger condition     (Fix 3)   │
│  • VAD commit / interim dual path             (Fix 2b)  │
│  • GET /api/health — psutil process telemetry           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  LiveCallSession  (src/engines/live_call_session.py)    │
│  • T0–T3 timestamp anchors + get_metrics()              │
│  • has_speech() RMS energy gate               (Fix 3)   │
│  • find_vad_boundary() / pop_utterance()      (Fix 2b)  │
│  • append_committed() / committed_text        (Fix 2b)  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  OnnxAsrPipeline  (src/engines/qwen3_engine.py)         │
│  • transcribe_stream yields (delta, timing|None)        │
│  • Per-stage timing: mel_s, encoder_s, prefill_s,       │
│    decode_s, tokens_generated, rtf                      │
└─────────────────────────────────────────────────────────┘
```

---

## Files changed

| File | Change |
|---|---|
| `src/engines/qwen3_engine.py` | `transcribe_stream` now yields `(str, dict\|None)` tuples with per-stage timing |
| `src/engines/live_call_session.py` | T0–T3 anchors, `get_metrics()`, `has_speech()`, `find_vad_boundary()`, `pop_utterance()`, `append_committed()`, `committed_text` |
| `main.py` | `asyncio` import, `_run_inference()` helper, energy gate, VAD dual-path inference, `GET /api/health` |
| `static/index.html` | 8-metric grid, waterfall bar, state pill, tooltips |
| `static/style.css` | Design tokens, RTF colour coding, flash animations, state-aware pill |
| `static/app.js` | 5-state machine, `applyMetrics()`, waterfall update, health polling, `resetUI()` |
| `pyproject.toml` | Added `psutil>=6.0.0`, `[tool.pytest.ini_options]` with `pythonpath = ["."]` |
| `tests/test_live_call_session.py` | 11 unit tests: PCM conversion, energy gate, VAD boundary, pop, commit |
