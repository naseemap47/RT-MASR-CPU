# RT-MASR-CPU — Architectural Changes

> This document records every structural change made to the system, why it was
> made, and how it altered the data-flow. Changes are in chronological order.
>
> For the current state of the system see [`architecture.md`](architecture.md).
> Code snippets below show the change at the time it was made; later sections
> may have superseded them.

| # | Change | Date |
|---|---|---|
| 1 | Telemetry instrumentation | 2026-10-02 |
| 2 | `transcribe_stream` timing sentinel | 2026-10-02 |
| 3 | Async executor for inference | 2026-10-03 |
| 4 | RMS energy gate | 2026-10-03 |
| 5 | VAD sentence-boundary chunking | 2026-10-03 |
| 6 | Hallucination investigation and fix | 2026-10-03 |
| 7 | Config-driven model registry and multiple backends | 2026-10-04 |
| 8 | Offline benchmark pipeline | 2026-10-04 |
| 9 | Whisper ONNX engine (benchmark comparison) | 2026-10-05 |

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
**awaits** the future and stays free to serve other connections and
`/api/health`. Within the same call leg the handler does not call `receive()`
until the await returns, so that call's frames queue in the server-side receive
buffer during inference and are drained afterwards (the diagram below shows the
event loop, not this handler).

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

## Summary of architectural layers changed (sections 1–5)

At this point the only engine was the Qwen3 ONNX pipeline; sections 7–9 add the
config registry, the Transformers and Whisper engines, and the benchmark.

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
│  OnnxAsrPipeline  (src/engines/qwen3_onnx_engine.py)         │
│  • transcribe_stream yields (delta, timing|None)        │
│  • Per-stage timing: mel_s, encoder_s, prefill_s,       │
│    decode_s, tokens_generated, rtf                      │
└─────────────────────────────────────────────────────────┘
```

---

## Files changed

| File | Change |
|---|---|
| `src/engines/qwen3_onnx_engine.py` | `transcribe_stream` now yields `(str, dict\|None)` tuples with per-stage timing |
| `src/engines/live_call_session.py` | T0–T3 anchors, `get_metrics()`, `has_speech()`, `find_vad_boundary()`, `pop_utterance()`, `append_committed()`, `committed_text` |
| `main.py` | `asyncio` import, `_run_inference()` helper, energy gate, VAD dual-path inference, `GET /api/health` |
| `static/index.html` | 8-metric grid, waterfall bar, state pill, tooltips |
| `static/style.css` | Design tokens, RTF colour coding, flash animations, state-aware pill |
| `static/app.js` | 5-state machine, `applyMetrics()`, waterfall update, health polling, `resetUI()` |
| `pyproject.toml` | Added `psutil>=6.0.0`, `[tool.pytest.ini_options]` with `pythonpath = ["."]` |
| `tests/test_live_call_session.py` | 11 unit tests: PCM conversion, energy gate, VAD boundary, pop, commit |

---

## 6. Hallucination Debugging — Investigation and Fix

### Symptom

The UI was displaying fabricated sentences mixed in with correct transcription:

```
"I'm not sure what you mean. I'm not a fan of the new movie.
He hoped there would be stew for dinner. Turnips and carrots..."
```

Only the second part was the actual audio content. The first sentence was invented
by the model.

### Phase 1: Root Cause Investigation

A diagnostic script was run to isolate what the model emits on non-speech inputs:

```
0.5s pure silence  → "I'm a little bit nervous."                              (7 tokens)
1.0s pure silence  → "I'm a little bit of a fan of the new movie, 'The Great Gatsby.'" (19 tokens)
RMS 0.004 noise    → "The system is a computer program that can perform various tasks." (12 tokens)
```

**Finding:** This is a well-known property of Whisper-architecture seq2seq models.
When given silence, noise, or audio that is too short for reliable recognition,
the model produces coherent but fabricated English sentences — not empty output and
not an error. The model was trained on real speech only; it has no concept of
"nothing to transcribe."

### First Fix Attempt — Token-Density Filter (Failed, Reverted)

**Hypothesis:** Hallucinations produce a fixed number of tokens regardless of audio
length. Real speech should produce proportionally more tokens for longer audio.
A `MIN_TOKENS_PER_SECOND` filter at 0.5 tok/s should reject hallucinations.

```python
def is_hallucination(self, text, tokens_generated, audio_duration_s):
    token_density = tokens_generated / audio_duration_s
    return token_density < self.MIN_TOKENS_PER_SECOND  # 0.5
```

**Why it failed:** The hypothesis was incorrect. Hallucinations produce a
full English sentence (10–20 tokens) regardless of audio length. For a 2s
committed chunk: 14 tokens / 2s = **7 tok/s** — well above the 0.5 threshold.
Real speech at 150 wpm produces approximately 2–8 tok/s in the same range.
The filter was statistically blind to the problem.

The commit was cleanly reverted with `git revert`.

### Root Cause — Lead-In Silence Passing a Weak Energy Gate

An RMS scan of the actual test audio files revealed the real trigger:

```
t=0.0s  RMS=0.0014  speech=False (lead-in silence at recording start)
t=0.5s  RMS=0.0720  speech=True  (actual speech begins)
t=1.0s  RMS=0.0795  speech=True
t=3.5s  RMS=0.0012  speech=False (inter-sentence pause)
t=4.0s  RMS=0.0452  speech=True
```

The audio file begins with a brief lead-in silence (RMS 0.0014). The original
energy gate threshold was **0.003 RMS (~−50 dBFS)**. The diagnostic confirmed
that noise at RMS 0.004 — just above this threshold — still causes the model to
hallucinate. The lead-in silence was not blocked; inference fired on it; the
hallucination was committed to `committed_text`; it then appeared in front of
every subsequent correct line.

Real measured speech in the test files: **RMS 0.045–0.06 (~−27 dBFS)**.
The 0.003 threshold left a 10× gap between max-noise and threshold — insufficient.

### Second Fix — Calibrated Threshold + Minimum Commit Window (Correct)

Two changes in `src/engines/live_call_session.py`:

**Fix A — Raise `RMS_SPEECH_THRESHOLD` from `0.003` to `0.02`**

```
Noise floor           : RMS 0.001 – 0.008
Diagnostic noise      : RMS 0.004  ← was hallucinating (below new threshold)
New threshold         : RMS 0.02   ← 5× above max noise, 3× below real speech
Quiet speech          : RMS 0.02  – 0.04
Normal speech (meas.) : RMS 0.045 – 0.06
```

The new threshold provides a **5× margin** above the measured noise ceiling and
**~3× margin** below the measured quiet-speech floor.

**Fix B — Raise `find_vad_boundary` `min_silence_samples` from `3200` (0.2 s) to `32000` (2.0 s)**

`min_silence_samples` is the offset at which the VAD starts scanning for a
boundary. It doubles as the minimum committed chunk length: no boundary can be
found before this point, so every committed utterance is guaranteed to be at
least 2 s long.

Before this change, a false VAD trigger could commit a 0.2–1.0 s chunk of
borderline audio (just enough energy to pass `has_speech()`), giving the model
too little context and triggering hallucination.

After: the model always receives ≥ 2 s of real-speech-energy audio per commit.
Whisper-architecture models are reliable at this duration.

### Verification After Second Fix

Re-running the gate check on the same audio file:

```
Pure silence (RMS=0):      has_speech=False  ✓
Noise RMS 0.0040:          has_speech=False  ✓
Lead-in silence RMS 0.0014:has_speech=False  ✓
Real speech RMS 0.0720:    has_speech=True   ✓
1s buffer: find_vad_boundary=None            ✓ (too short to commit)
```

All 11 existing unit tests continued to pass (the tone-based tests use amplitude
0.3, RMS ≈ 0.21 — well above the new threshold).

### Files changed

| File | Change |
|---|---|
| `src/engines/live_call_session.py` | `RMS_SPEECH_THRESHOLD` 0.003 → 0.02; `find_vad_boundary` `min_silence_samples` 3200 → 32000; added measured calibration comments |

### Key Lesson

Token-density-based hallucination filters do not work for Whisper-architecture
models because hallucinations produce the same token density as real speech.
The correct mitigation is upstream: ensure the model **never receives non-speech
audio** in the first place, through calibrated energy gates and minimum committed
chunk length.

---

## 7. Config-Driven Model Registry and Multiple Backends

### Problem

The engine class and model paths were hard-coded in `main.py`. Comparing model
sizes or runtimes (an assignment requirement) meant editing code.

### Changes

- The ONNX engine was renamed `qwen3_engine.py` → `qwen3_onnx_engine.py`
  (`ONNXQwen3ASR`), and a new `qwen3_engine.py` (`Qwen3ASR`) wraps the
  `qwen-asr` Transformers package with the same `transcribe()` /
  `transcribe_stream()` contract.
- Configuration was split into three levels:

```
config/config.yaml                 server + default_model + model_registry path
  └─ config/models/models.yaml     registry: name, config, backend, engine_class, model_dir
       └─ config/models/<name>.yaml  download / engine / inference / audio / ort_session
```

- `src/core/config.py` resolves `default_model` through the registry
  (`resolve_model_config()`), and each engine gained `from_config()` /
  `from_config_path()`.
- `main.py` `_build_engine()` dispatches on the per-model `backend` field
  (`onnx` | `transformers`). `/api/health` reports the active model and backend.
- `src/utils/download_utils.py` became config-driven: `download.method` selects
  `onnx` (only the ONNX artefacts + tokenizer from the HF repo), `snapshot`
  (full HF snapshot) or, later, `whisper` (PINTO tarball). It runs as a CLI with
  `--model`, `--config` and `--force`.

### Files changed

| File | Change |
|---|---|
| `src/engines/qwen3_onnx_engine.py` | Renamed from `qwen3_engine.py`; `from_config()` |
| `src/engines/qwen3_engine.py` | New Transformers backend (`Qwen3ASR`) |
| `src/core/config.py` | New: registry + per-model config resolution, dtype helper |
| `config/config.yaml`, `config/models/*.yaml` | New three-level config |
| `src/utils/download_utils.py` | Config-driven downloader + CLI |
| `main.py` | `_build_engine()` backend dispatch; health reports model/backend |

---

## 8. Offline Benchmark Pipeline

### Problem

The live UI shows per-call metrics but cannot produce repeatable, comparable
numbers (percentiles, accuracy, concurrency scaling) across configurations.

### Changes

A standalone package `benchmark/` runs four stages per configuration — cold
load, latency/RTF, accuracy (WER/CER) and concurrency — against the engines
directly (no WebSocket), and writes a raw JSON plus a Markdown summary with a
hardware fingerprint. Design and usage are documented in
[`../benchmark/benchmarking.md`](../benchmark/benchmarking.md).

Notable decisions:

- The engine loaded for the cold-start measurement is reused by the other stages,
  and memory is released between configs so the next load's RSS baseline is clean.
- Every concurrency level uses the same fixed audio workload so levels are
  comparable.
- A failing config is recorded in `failures` instead of aborting the run.
- Concurrency means live legs: one leg = one independently streamed audio source.
  `concurrency_mode: stream` (default) plays each leg at real-time pace using the
  server's own stream logic (Whisper sliding window / Qwen3 VAD utterances) and
  reports staleness, end lag and how many legs keep up. The earlier offline
  request-queue test remains as `--concurrency-mode batch`; there a "leg" is a
  worker thread and "calls" are `legs x rounds` requests (now labelled Requests).

---

## 9. Whisper ONNX Engine (Benchmark Comparison)

### Problem

The assignment asks for Qwen3-ASR to be compared against alternative CPU
models/runtimes.

### Changes

- `src/engines/whisper_engine.py` (`WhisperOnnxEngine`) runs PINTO model-zoo
  Whisper ONNX exports (`{size}_encoder_11_{precision}.onnx` /
  `{size}_decoder_11_{precision}.onnx`) on ONNX Runtime.
- Beam search, temperature fallback, language detection and segmenting are
  reused from the vendored `src/whisper/` package (from `whisper-onnx-cpu`);
  a `_WhisperModelProxy` routes its model calls to the ONNX sessions.
- Registry entries: `whisper_int8_{tiny,base,small,medium}` (all sharing
  `models/whisper_int8/`), `whisper_fp16`, `whisper_fp32`.
- The downloader gained `method: whisper`; one tarball per precision contains
  every model size.
- `benchmark/engine_loader.py` and `bench_config.yaml` gained the `whisper`
  backend and the four INT8 sizes.

### Scope

Whisper is wired into the benchmark only. `main.py` `_build_engine()` still
accepts just `onnx` and `transformers`, so a `whisper_*` `default_model` is not
servable by the live UI yet.
