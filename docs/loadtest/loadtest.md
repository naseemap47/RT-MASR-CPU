# Load Test & Capacity Sizing

The load-test pipeline (`loadtest/`) answers: *how many simultaneous live calls can one edge CPU carry, where does it
saturate, and what does that mean if you need 50-1,000 legs on more identical boxes?* It mirrors the benchmark pipeline
(config YAML -> runner -> raw JSON + Markdown) and reuses its engines, stream logic and config.

This project deploys on **edge CPUs**. The pipeline pins threads of the machine under test; it does not model cloud
instance types.

```
loadtest_config.yaml ──> run_loadtest.py ──> <UTC>_loadtest_raw.json / _summary.md      (MEASURED)
                              │  WorkerPool: P pinned processes, each one engine
                              │  ramp: ladder -> bisect -> confirm   (saturation point)
                              ▼
                        run_sizing.py ──> <UTC>_sizing_guide.md / _sizing.json          (DERIVED / EXTRAPOLATED)
```

Final curated results: [`sizing_guide.md`](sizing_guide.md).

## What is simulated

**One leg = one independently streamed audio source**, i.e. one call, fed at real-time pace in 0.5 s chunks (as the browser does)
into the engine with the **same stream logic as the live server**:

- Whisper: `WhisperSlidingWindowStreamer` (sliding window + LocalAgreement, a backlog is skipped like the server does).
- Qwen3: VAD-cut utterances (`LiveCallSession.find_vad_boundary`; inference every 2nd chunk while speech is present).

Calls are 30 s tiled from LibriSpeech clips and **start spread over 6 s**, so they are not phase-aligned. Two **load profiles**:

| Profile | Silence between clips | Speech share | Use |
|---|---|---|---|
| `dense` | 1 s | ~85% | stress / upper bound |
| `conversational` | 7 s | ~47% | planning case |

## Metrics

| Metric | Meaning |
|---|---|
| Pass latency / pass RTF | time of one inference pass / its audio length |
| **Staleness** | pass finish minus arrival of the newest audio it covered = how far the transcript trails the speaker (queueing included) |
| End lag | time from the end of the call audio to the final transcript |
| **Kept up** | leg has p95 staleness and end lag <= `slo.lag_threshold_s` (2.0 s), no error, not aborted |
| **Max legs kept up** (`l_sat`) | highest leg count at which every leg kept up; first failing level is `l_fail` |
| Overload guard | a leg more than `slo.abort_lag_s` (8 s) behind is stopped, so a saturated level ends quickly |

Resources per level (`TreeSampler`): utilisation of the pinned CPUs, CPU-seconds of the worker tree, summed RSS, minimum free RAM.
Caution: ONNX Runtime threads spin-wait, so CPU time overstates useful work and per-leg CPU cost is **not** constant. The sizing is therefore driven
from the measured saturation point, not from CPU-seconds per leg.

## What is run

Each **model × load profile** is one run: pin **all logical CPUs** of this machine, **one process**, ramp the number of
legs until they no longer keep up. That saturation point is the measurement. Thread pinning uses sibling pairs (both SMT
threads of a physical core). `--cpus N` / `--processes P` can shrink a smoke run; they do not build a matrix of layouts.

## Safety on a shared machine

The pool starts workers one at a time and checks RSS against free RAM (`InsufficientMemory` -> scenario skipped and reported), a hard floor kills
workers if free RAM falls below `safety.hard_floor_mb`, and the ramp predicts RAM for the next level before running it (`stop_reason: memory`).

## Running

```bash
python3 loadtest/run_loadtest.py --list                          # show runs
python3 loadtest/run_loadtest.py                                  # every model x profile in loadtest_config.yaml
python3 loadtest/run_loadtest.py --profiles conversational --models whisper_int8_tiny
python3 loadtest/run_loadtest.py --levels 1,2 --duration 15      # smoke run
python3 loadtest/run_sizing.py --input <dense_raw.json> <conv_raw.json>
python3 loadtest/run_sizing.py --headroom 0.6 --serving-overhead 1.25 --spare-fraction 0.2 --legs 50,100,1000
```

On restricted sandboxes set `NUMBA_CACHE_DIR=/tmp/numba_cache` (the scripts do this themselves). Do not run other heavy work while testing;
results are only as clean as the machine.

## Sizing model

Built from the measured saturation point of each layout (never `legs x per-leg cost`):

```
legs/box = max(1, l_sat x headroom(0.70) / serving_overhead(1.10))
boxes    = ceil(N / legs/box) + max(1, ceil(10% x boxes))        # spare edge boxes
RAM/box  = ceil((processes x weights + legs x MB_per_leg) x 1.2 + 2 GB OS)   # whole GB
```

| Factor | Treatment |
|---|---|
| Shared model memory | measured base RSS per layout; one copy per process |
| Per-leg memory | regression over healthy levels |
| Thread contention / diminishing throughput | measured as this machine's saturation (one process, all threads); extra legs need extra boxes |
| Queueing | headroom below the knee (latency-vs-load table shows the knee) |
| Process / NUMA | one process on this one-socket machine; extra processes / extra sockets not swept |
| Batching | **not credited**: engines decode one stream per pass |
| Confidence | falls with N / `l_sat`: <=10x Medium, <=50x Low-Medium, <=200x Low, beyond Very low |

Each number in the guide is tagged MEASURED / DERIVED / ASSUMED / EXTRAPOLATED / NOT MODELLED. Every assumption is a flag of `run_sizing.py`
(or a field of `SizingAssumptions`).

## Tests

```bash
uv run --offline python -m pytest loadtest/tests benchmark/tests -q
```

## Limitations (short)

One laptop-class edge CPU, noisy shared machine, 1-leg resolution, clean read-speech only, no network path, scale-out beyond one box
assumed rather than measured, 30 s calls. See section 6 of the sizing guide.
