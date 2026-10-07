# RT-MASR-CPU — Final Load Test Result

Consolidated from the load-test run `20261007T202154Z`
([raw JSON](../../loadtest/results/20261007T202154Z_loadtest_raw.json) (Not Uploaded) · [summary](../../loadtest/results/20261007T202154Z_loadtest_summary.md))
and the sizing run built from it, `20261007T203650Z`
([sizing JSON](../../loadtest/results/20261007T203650Z_sizing.json) (Not Uploaded) · [sizing guide](../../loadtest/results/20261007T203650Z_sizing_guide.md)).
How the pipeline works: [`loadtest.md`](loadtest.md). Single-request speed and accuracy: [`../benchmark/final_result.md`](../benchmark/final_result.md).

Every number is tagged by where it comes from: **MEASURED** (read from the load test), **DERIVED** (arithmetic on
measured numbers, recomputed from the raw JSON for this page), **ASSUMED** (a sizing input the test cannot measure) or
**EXTRAPOLATED** (a prediction beyond what one machine could run).

## 1. Setup

| Item | Value |
|---|---|
| Machine (one edge box) | AMD Ryzen 7 6800H, 8 physical / 16 logical cores, 14.9 GB RAM, one socket / one NUMA node |
| OS / onnxruntime | Linux 6.8.0-138 (x86_64) / 1.30.0 |
| Layout | whole machine: all 16 logical CPUs pinned, **1 process x 16 threads**, all legs share one engine (same as the live server) |
| Models | `qwen3_onnx_int8_0.6b`, `qwen3_onnx_int4_0.6b` (stream mode `vad_utterance`), `whisper_int8_tiny` (`sliding_window`) |
| One leg | one independently streamed audio source (one call direction), fed in 0.5 s chunks at real-time pace through the live server's stream logic |
| Call | 30 s, tiled from 3 English LibriSpeech clips (leg *i* plays clip *i* mod 3), starts spread over 6 s |
| Load profiles | `dense`: 1 s silence after each clip, **85% speech** (stress case) · `conversational`: 7 s silence, **47% speech** (planning case) |
| "Kept up" | leg p95 staleness **and** end-of-call lag <= **2.0 s**, no errors, not aborted, non-empty transcript |
| Overload guard | a leg more than 8 s behind is stopped |
| Search | ladder 1, 2, 3, 4, 6, … → bisect between last healthy and first failing level → re-run the answer to confirm |
| **Not tested** | `qwen3_onnx_int4_1.7b`, `qwen3_onnx_fp32_0.6b`, the Transformers configs and Whisper base/small/medium (not in this run's roster) |

**Staleness** = when a pass finished minus when the newest audio it covered arrived, i.e. how far the live transcript
trails the speaker, including queueing. **Pass RTF** = time of one inference pass / length of the audio it read.

## 2. Headline: saturation point per box (MEASURED)

| Model | Profile | Max legs kept up | First failing | Confirmed | Stale P95 at max | Pass RTF P95 at max | Base RSS | Load |
|---|---|---|---|---|---|---|---|---|
| `whisper_int8_tiny` | conversational | **3** | 4 | yes | 1.51 s / 1.59 s | 0.43 / 0.41 | 1.0 GB | 2.3 s |
| `whisper_int8_tiny` | dense | **2** | 3 | yes | 1.54 s / 1.63 s | 0.34 / 0.42 | 1.0 GB | 2.3 s |
| `qwen3_onnx_int4_0.6b` | conversational | **1** | 2 | yes | 0.83 s / 0.90 s | 0.31 / 0.31 | 2.5 GB | 2.7 s |
| `qwen3_onnx_int4_0.6b` | dense | **1** ⚠ | 2 | yes | 0.90 s / 0.87 s | 0.31 / 0.31 | 2.5 GB | 3.4 s |
| `qwen3_onnx_int8_0.6b` | conversational | **1** | 2 | yes | 0.98 s / 0.97 s | 0.36 / 0.35 | 3.7 GB | 3.4 s |
| `qwen3_onnx_int8_0.6b` | dense | **1** | 2 | yes | 0.95 s / 0.96 s | 0.36 / 0.35 | 3.7 GB | 4.0 s |

- Two values in a cell = the ramp run and the confirm run of the same level.
- ⚠ INT4 dense passed 2 legs once during the ramp (stale P95 1.72 s), then failed the confirm run at 2 legs (2.30 s). It is right at the 2 s threshold; the planning value is 1.
- Every scenario stopped because the CPU saturated. None was limited by memory: free RAM never fell below 7.9 GB.
- Every leg in every level produced a transcript (no empty outputs; see section 4.5).

## 3. Latency vs load (MEASURED)

Staleness is flat up to the saturation point and then jumps. This is why boxes are planned below it (section 5).

| Model | Profile | Legs | Kept up | Stale P50 | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| Whisper tiny | conversational | 1 | 1/1 | 0.42 s | 0.91 s | 0.19 | 0.00 s | 43% | 1068 MB |
| | | 2 | 2/2 | 0.49 s | 1.04 s | 0.37 | 0.00 s | 47% | 1178 MB |
| | | 3 | 3/3 | 0.61 s | 1.51 s | 0.43 | 0.61 s | 57% | 1303 MB |
| | | 4 | **1/4** | 1.09 s | 2.70 s | 0.58 | 2.62 s | 87% | 1344 MB |
| Whisper tiny | dense | 1 | 1/1 | 0.40 s | 0.91 s | 0.23 | 1.51 s | 52% | 1078 MB |
| | | 2 | 2/2 | 0.49 s | 1.54 s | 0.34 | 1.93 s | 68% | 1191 MB |
| | | 3 | **2/3** | 1.03 s | 3.24 s | 0.59 | 5.82 s | 89% | 1319 MB |
| Qwen3 INT4 0.6B | conversational | 1 | 1/1 | 0.59 s | 0.83 s | 0.31 | 1.01 s | 38% | 2628 MB |
| | | 2 | **0/2** | 0.72 s | 2.76 s | 0.82 | 1.78 s | 52% | 3077 MB |
| Qwen3 INT4 0.6B | dense | 1 | 1/1 | 0.62 s | 0.90 s | 0.31 | 1.17 s | 55% | 2589 MB |
| | | 2 (ramp) | 2/2 | 0.85 s | 1.72 s | 0.60 | 1.93 s | 69% | 2828 MB |
| | | 2 (confirm) | **0/2** | 1.06 s | 2.30 s | 0.92 | 2.07 s | 76% | 2996 MB |
| | | 3 | **0/3**, 2 aborted | 1.87 s | 8.98 s | 1.23 | 1.56 s | 67% | 2991 MB |
| Qwen3 INT8 0.6B | conversational | 1 | 1/1 | 0.68 s | 0.98 s | 0.36 | 1.12 s | 43% | 3895 MB |
| | | 2 | **0/2** | 0.84 s | 3.11 s | 0.80 | 1.36 s | 55% | 4184 MB |
| Qwen3 INT8 0.6B | dense | 1 | 1/1 | 0.71 s | 0.95 s | 0.36 | 1.47 s | 62% | 3869 MB |
| | | 2 | **1/2** | 1.04 s | 2.22 s | 0.73 | 2.41 s | 78% | 4073 MB |

Stale P50/P95 are pooled over all passes of all legs; "kept up" is judged per leg. Time to first text for one leg is
1.4–1.5 s for all three models.

## 4. Findings

### 4.1 One box carries 1–3 live legs

Whisper tiny INT8 sustains **3 conversational / 2 dense** legs. Both Qwen3 0.6B variants sustain **1 leg** in both
profiles. In the benchmark's batch mode the same machine transcribed INT8 0.6B at 6.7x real time with 4 workers. The
two results do not conflict. Batch mode decodes each file once; a live leg re-decodes its open audio about once per
second and must also meet a latency limit.

### 4.2 Why a leg is expensive (DERIVED)

Seconds of inference the engine spends per second of call audio, for **one leg**, compared with the batch RTF of the
same English clips in the benchmark:

| Model | Dense | Conversational | Batch RTF, same EN clips | Streaming / batch |
|---|---|---|---|---|
| Qwen3 INT8 0.6B | 0.65 | 0.44 | 0.19 | 2.3–3.4x |
| Qwen3 INT4 0.6B | 0.57 | 0.38 | 0.16\* | 2.4–3.7x |
| Whisper tiny | 0.53 | 0.41 | 0.28 | 1.5–1.9x |

\* INT4 batch RTF excludes `librispeech_0_1089_0.wav`, which returned empty output in the benchmark (INT8 on the same two clips: 0.17).

- Inference seconds = the sum of all pass latencies of the leg. At 1 leg, the engine is busy **49–60% of the wall
  clock in the dense profile and 35–41% in the conversational profile**. A second Qwen leg (or a third/fourth Whisper
  leg) makes passes overlap on the same 16 threads, every pass slows down, and staleness crosses 2 s.
- Qwen's overhead over batch is larger. Its draft passes re-read the whole growing utterance (up to 15 s) about once per
  second. Whisper re-reads a bounded window and skips a backlog instead of queueing it. Its ratio is also flattered: the
  benchmark decoded Whisper with beam search 5, while live streaming decodes greedily (`beam_size: 1`), so a streaming
  pass is cheaper than a batch pass of the same audio.
- Two conversational Qwen legs failed even though the engine was busy only ~70% of the time. The second leg plays
  the short 3.3 s clip and made only 5 passes, so its P95 staleness is effectively its single slowest pass (3.5 s for
  INT8, 4.3 s for INT4), i.e. one pass that collided with the other leg's long utterance.
- This is a property of the streaming design, not of the hardware. Cross-leg batching, a slower draft rate or a
  different commit policy would change it; none of them is implemented, so none is credited.

### 4.3 INT4 vs INT8 (Qwen3 0.6B)

INT4 has the **same capacity** as INT8 (1 leg) but is cheaper per leg: ~12–14% less inference time per audio second,
~14% lower pass RTF P95, 5–15% lower stale P95, and ~1.2 GB less base memory (2.5 vs 3.7 GB). It is also the only Qwen config
that briefly held 2 dense legs. On a box with slightly more compute, INT4 is the Qwen config most likely to reach 2 legs
first; that is an expectation, not a measurement.

### 4.4 CPU is the constraint, not RAM

Peak RSS stayed under 4.2 GB for Qwen INT8, 3.1 GB for INT4 and 1.5 GB for Whisper tiny. Free RAM never dropped below
7.9 GB. Per-leg memory regressions (DERIVED: Qwen 133–303 MB/leg with R² 0.28–0.57; Whisper 109–117 MB/leg, R² 0.72–0.86)
are rough, because only 1–3 levels were healthy.

### 4.5 Transcripts and the INT4 empty-output issue

All legs produced a transcript at every level, including INT4 on `librispeech_0_1089_0.wav`, the clip that returned 0
tokens in the benchmark. Streaming cuts that clip at pauses into shorter utterances, and the problem did not appear.
The load test checks that each transcript is non-empty and on time; it does **not** score accuracy. Use the benchmark
for WER/CER. Qwen's VAD cuts also land inside sentences ("…and bruised. Potatoes and fat…"), which affects punctuation
but not the words.

### 4.6 Repeatability

The previous whole-machine run, `20261007T170448Z` (INT8 and Whisper tiny only), found exactly the same saturation
points: INT8 1 / 1, Whisper tiny 3 conversational / 2 dense. An earlier run on 2026-10-06 was one leg lower in three
of four cases. Treat ±1 leg as the error bar; at 1–3 legs per box that is large.

## 5. Sizing for 50–1,000 legs (EXTRAPOLATED)

Built from the measured saturation point (never `legs x per-leg cost`):

```
legs/box = max(1, max_legs_kept_up x 0.70 headroom / 1.10 serving overhead)      (ASSUMED factors)
boxes    = ceil(N / legs/box) + max(1, ceil(10% x boxes)) spare boxes             (ASSUMED spares)
RAM/box  = ceil((weights + legs x MB/leg) x 1.20 + 2 GB OS)                       (whole GB)
```

**Conversational (planning case).** Edge boxes identical to the test machine, including spares. In parentheses, the range
if the saturation point were one leg higher or lower.

| Concurrent legs | Whisper tiny (1.91 legs/box, 4 GB) | Qwen3 INT4 0.6B (1 leg/box, 6 GB) | Qwen3 INT8 0.6B (1 leg/box, 7 GB) | Confidence |
|---|---|---|---|---|
| 50 | 30 (22–44) | 55 (44–55) | 55 (44–55) | Whisper Low-Medium · Qwen Low |
| 60 | 36 (27–53) | 66 (53–66) | 66 (53–66) | Whisper Low-Medium · Qwen Low |
| 100 | 59 (44–87) | 110 (87–110) | 110 (87–110) | Whisper Low-Medium · Qwen Low |
| 200 | 116 (87–174) | 220 (174–220) | 220 (174–220) | Low |
| 500 | 289 (217–433) | 550 (433–550) | 550 (433–550) | Whisper Low · Qwen Very low |
| 1,000 | 577 (433–865) | 1,100 (865–1,100) | 1,100 (865–1,100) | Very low |

**Dense (upper bound).**

| Concurrent legs | Whisper tiny (1.27 legs/box, 4 GB) | Qwen3 INT4 0.6B (1 leg/box, 6 GB) | Qwen3 INT8 0.6B (1 leg/box, 7 GB) |
|---|---|---|---|
| 50 | 44 (30–55) | 55 (44–55) | 55 (44–55) |
| 100 | 87 (59–110) | 110 (87–110) | 110 (87–110) |
| 1,000 | 865 (577–1,100) | 1,100 (865–1,100) | 1,100 (865–1,100) |

Targets each box must be operated at (MEASURED at the operating point):

| Model | Profile | Operating point | Target pass RTF | Target P95 staleness |
|---|---|---|---|---|
| Whisper tiny | conversational | 2 legs | <= 0.40 | <= 1.1 s |
| Whisper tiny | dense | 2 legs | <= 0.35 | <= 1.6 s |
| Qwen3 INT4 0.6B | both | 1 leg | <= 0.35 | <= 0.9 s |
| Qwen3 INT8 0.6B | both | 1 leg | <= 0.40 | <= 1.0 s |

**Sensitivity.** Qwen sits on the 1-leg-per-box floor, so headroom (60–80%) and serving overhead (x1.00–x1.25) do not
change its box count; only spares do (100 legs: 100 boxes without spares, 110 with). Whisper moves with every
assumption: at 100 conversational legs it needs 51 boxes at 80% headroom, 69 at 60%, 53 at x1.00 overhead and 66 at
x1.25. The ±1 leg measurement uncertainty moves it from 44 to 87 boxes.

## 6. Caveats

- **One machine.** A laptop-class 8-core CPU with boost and thermal limits, shared with an IDE. Another edge CPU will
  saturate at a different point. Re-run `loadtest/run_loadtest.py` on the target hardware before buying.
- **Scale-out is assumed, not measured.** Boxes are treated as independent with calls sticky to one box. Load-balancer
  skew and correlated peaks are covered only by headroom and spares.
- **Clean English read speech only.** Real calls add noise, other languages (more decoded tokens), overlap and
  different talk ratios. The two profiles bracket speech density, not acoustic difficulty.
- **No network path.** WebSocket, TLS and JSON costs are not in the test; the ASSUMED x1.10 overhead stands in for them.
- **The 2 s lag threshold is a design choice** and sits close to the streaming floor (stale P95 is 0.8–1.0 s even for
  one leg). With a 3 s threshold, the two dense Qwen 2-leg runs would have passed (worst leg: 2.25–2.39 s staleness
  P95, 2.07–2.41 s end lag). Every other failing level had a worst leg at 3.3 s or more, so it would still fail.
  Re-run with `slo.lag_threshold_s` set to the real product SLO.
- **30 s calls.** Memory growth over hour-long calls was not measured; the stream code bounds buffers (Qwen
  force-commits at 15 s, Whisper windows cap at 20 s).
- **Whole-leg resolution, one confirm run.** Not statistically repeated.

## 7. Recommendation

| Goal | Pick |
|---|---|
| Most live legs per edge box | `whisper_int8_tiny`: 3 conversational / 2 dense legs, ~4 GB RAM per box. Accuracy is the price: EN WER 0.130 and ZH CER 0.470 in the benchmark |
| Qwen accuracy at the lowest cost per leg | `qwen3_onnx_int4_0.6b`: 1 leg per box like INT8, but 1.2 GB less RAM, ~12% less compute and the lowest Qwen staleness (P95 0.83–0.90 s). Force the language until the benchmark's empty-output issue is fixed |
| Qwen with no INT4 caveats | `qwen3_onnx_int8_0.6b`: 1 leg per box, ~7 GB RAM per box |
| Planning number (100 conversational legs) | about **59 Whisper-tiny boxes** (range 44–87) or **110 Qwen3 0.6B boxes** (range 87–110), including 10% spares |

On this class of edge CPU, one box is roughly one Qwen call. Real capacity gains need to come from the streaming
design (draft-pass rate, cross-leg batching) or stronger hardware, then be re-measured with the same pipeline.
