# Edge CPU Capacity Sizing Guide (50 / 60 / 100 / 200 / 500 / 1,000 concurrent legs)

> Curated from the load-test run on the development laptop (Ryzen 7 6800H, 8C/16T). Produced by
> `loadtest/run_loadtest.py` + `loadtest/run_sizing.py`; see [`loadtest.md`](loadtest.md) for how it works.
> **Every row in the sizing tables is EXTRAPOLATED**: the test machine saturates at 1-3 legs, so the
> sizing for 50+ legs is a model built on those measurements, not something that was run.

This project deploys on **edge CPUs**. The default load test ramps **this whole machine** (all threads, one process)
once per model and profile. Section 3 still shows an earlier 4/8/16-thread comparison that is kept as measured
history; new runs do not repeat that matrix.

## Key findings

1. **Saturation on this machine is 1-3 legs.** Whisper tiny INT8 sustains 2 conversational legs on 8 CPU threads
   (3 when 16 threads are split into 2 pinned processes); Qwen3-ASR-0.6B INT8 sustains 1 leg on 4 CPU threads and
   gains nothing from more cores. With the stricter *dense* profile both are at 1 leg.
2. **More threads in one process did not help.** Qwen3: 4, 8 and 16 threads all gave 1 leg (16 threads in one
   process was even worse in the dense run). Whisper: 4 -> 8 threads doubled capacity (1 -> 2 legs conversational),
   8 -> 16 added nothing. Extra legs need extra boxes, not a bigger chip on this architecture.
3. **Process layout matters, and more is not better.** Whisper at 16 threads: 1 x 16 = 2 legs, 2 x 8 = 3 legs,
   4 x 4 = 1 leg (conversational). Two processes duplicate the 1 GB weights but isolate contention; four spend too
   few threads per process. For Qwen3 (3.8 GB weights) two processes doubled RAM to 7.7 GB without adding a leg.
4. **Each leg costs a lot of CPU.** Streaming re-decodes a window (Whisper) or a growing utterance (Qwen3) every hop;
   one pass of one leg already takes 0.35-0.45 of real time (pass RTF), so a box is full after a handful of legs.
   This is a property of the streaming architecture, not the hardware: cross-leg batching, a smaller hop rate or a
   smarter commit policy would change it, and none of them is credited here.
5. **The result is sensitive to the 2 s lag threshold.** Streaming staleness is already 1.0-1.3 s for a single leg.
   Whisper at 8 threads failed its 3-leg level at a p95 staleness of 2.16 s: at a 3 s SLO capacity would be higher.
   Re-run with your SLO.
6. **Run-to-run noise is real** (a shared, thermally limited laptop): a repeat of the dense run (`20261006T133427Z`,
   partial) put Qwen3 at 4 threads at 1 leg where the full run measured 0. Treat each saturation point as +/-1 leg;
   section 5 shows what that does to box counts.

## How to use the tables

- Read the **conversational** table for planning and the **dense** table as the upper bound.
- *Edge boxes* is the number of identical machines (central estimate, with the +/-1-leg range in parentheses), including spares.
- *CPU threads / box* and *RAM / box* are the layout of one machine. RAM is the GB that box needs (weights + per-leg RSS,
  headroom and OS, rounded up) — about 4 GB for Whisper tiny, about 7 GB for Qwen3. **CPU is the constraint, not RAM.**
- Before buying hardware, re-run the load test on the target edge CPU; only the YAML changes.

---

**Generated:** 2026-10-07T16:01:53Z  
**Measured data:** `20261006T131842Z_loadtest_raw.json`, `20261006T141231Z_loadtest_raw.json`

## 1. How to read this guide

**One concurrent call leg = one independently streamed audio source** (one call direction), transcribed live.
Every number is tagged by where it comes from:

| Tag | Meaning |
|---|---|
| **MEASURED** | read directly from the load test on the test machine (section 3) |
| **DERIVED** | arithmetic or regression on measured numbers (memory per leg, scaling-law fit) |
| **ASSUMED** | an input the load test cannot measure; listed in section 4 and editable in `SizingAssumptions` |
| **EXTRAPOLATED** | a prediction beyond the measured range. **Every row for 50+ legs below is extrapolated**: the test machine saturates at a handful of legs |

Nothing here is `legs x per-leg cost`. Sizes are built from the measured *saturation point* of this edge CPU, then adjusted
for headroom, serving overhead, spare boxes, and memory (section 4). Unmeasured larger chips are not predicted.

Two **load profiles** were measured because speech density changes the cost of a leg a lot: *conversational* (about half the
call is silence, which skips inference) and *dense* (almost continuous speech, a stress case). Read the conversational
table as the planning case and the dense table as the upper bound.

## 2. Sizing tables (edge CPU)

> **Edge boxes** is how many identical machines of the measured pin-width are needed (central estimate, with the range if saturation is one leg higher or lower in parentheses), including spare boxes. **CPU threads / box** is how many logical CPUs that layout pins. **RAM / box** is the GB this box needs (measured weights + per-leg RSS, with RAM headroom and OS reserve, rounded up to a whole GB). **Target RTF** = P95 real-time factor of one inference pass at the operating point; **Target P95 latency** = P95 *staleness* (how far the live transcript trails the speaker, queueing included) at that point. Both targets are the values *measured* at the operating load on the reference layout; the capacity figures only hold if each box is operated at or below that load.

### Qwen3-ASR-0.6B INT8 (ONNX / CPU) (`qwen3_onnx_int8_0.6b`), conversational (about half of each call is speech): the planning case

**Load profile:** speech about 47% of each call. **Box layout:** 4 CPU threads, 1 process(es) x 4 threads (only shape that sustained a leg). **Measured:** saturation at **1 legs/box** (first failing 2, confirmed). **Planning capacity:** 1.00 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 7 GB (1 legs x 330 MB/leg + 3839 MB model/buffers, x1.20 headroom + 2 GB OS = 6.9 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 55 (44-55) | 4 | 7 GB | <= 0.45 | <= 1.3 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 50+5 spare boxes; confidence Low |
| 60 | 66 (53-66) | 4 | 7 GB | <= 0.45 | <= 1.3 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 60+6 spare boxes; confidence Low |
| 100 | 110 (87-110) | 4 | 7 GB | <= 0.45 | <= 1.3 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 100+10 spare boxes; confidence Low |
| 200 | 220 (174-220) | 4 | 7 GB | <= 0.45 | <= 1.3 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 200+20 spare boxes; confidence Low |
| 500 | 550 (433-550) | 4 | 7 GB | <= 0.45 | <= 1.3 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 500+50 spare boxes; confidence Very low |
| 1,000 | 1,100 (865-1,100) | 4 | 7 GB | <= 0.45 | <= 1.3 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 1000+100 spare boxes; confidence Very low |

> Headroom could not be applied: a box barely sustains one leg, so each box hosts a single leg and capacity is not additive in a meaningful way below that.

### Whisper tiny INT8 (ONNX / CPU) (`whisper_int8_tiny`), conversational (about half of each call is speech): the planning case

**Load profile:** speech about 47% of each call. **Box layout:** 8 CPU threads, 1 process(es) x 8 threads (highest measured saturation legs per CPU thread (2/8 = 0.250) among shapes sustaining >=2 legs). **Measured:** saturation at **2 legs/box** (first failing 3, confirmed). **Planning capacity:** 1.27 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 4 GB (2 legs x 106 MB/leg + 1050 MB model/buffers, x1.20 headroom + 2 GB OS = 3.5 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 44 (30-55) | 8 | 4 GB | <= 0.40 | <= 1.6 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 40+4 spare boxes; confidence Low |
| 60 | 53 (36-66) | 8 | 4 GB | <= 0.40 | <= 1.6 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 48+5 spare boxes; confidence Low |
| 100 | 87 (59-110) | 8 | 4 GB | <= 0.40 | <= 1.6 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 79+8 spare boxes; confidence Low |
| 200 | 174 (116-220) | 8 | 4 GB | <= 0.40 | <= 1.6 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 158+16 spare boxes; confidence Low |
| 500 | 433 (289-550) | 8 | 4 GB | <= 0.40 | <= 1.6 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 393+40 spare boxes; confidence Very low |
| 1,000 | 865 (577-1,100) | 8 | 4 GB | <= 0.40 | <= 1.6 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 786+79 spare boxes; confidence Very low |

### Qwen3-ASR-0.6B INT8 (ONNX / CPU) (`qwen3_onnx_int8_0.6b`), dense speech (almost continuous talking): the worst-case stress profile

**Load profile:** speech about 85% of each call. **Box layout:** 8 CPU threads, 1 process(es) x 8 threads (only shape that sustained a leg). **Measured:** saturation at **1 legs/box** (first failing 2, confirmed). **Planning capacity:** 1.00 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 7 GB (1 legs x 154 MB/leg + 3769 MB model/buffers, x1.20 headroom + 2 GB OS = 6.6 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 55 (44-55) | 8 | 7 GB | <= 0.40 | <= 1.1 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 50+5 spare boxes; confidence Low |
| 60 | 66 (53-66) | 8 | 7 GB | <= 0.40 | <= 1.1 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 60+6 spare boxes; confidence Low |
| 100 | 110 (87-110) | 8 | 7 GB | <= 0.40 | <= 1.1 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 100+10 spare boxes; confidence Low |
| 200 | 220 (174-220) | 8 | 7 GB | <= 0.40 | <= 1.1 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 200+20 spare boxes; confidence Low |
| 500 | 550 (433-550) | 8 | 7 GB | <= 0.40 | <= 1.1 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 500+50 spare boxes; confidence Very low |
| 1,000 | 1,100 (865-1,100) | 8 | 7 GB | <= 0.40 | <= 1.1 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 1000+100 spare boxes; confidence Very low |

> Headroom could not be applied: a box barely sustains one leg, so each box hosts a single leg and capacity is not additive in a meaningful way below that.

### Whisper tiny INT8 (ONNX / CPU) (`whisper_int8_tiny`), dense speech (almost continuous talking): the worst-case stress profile

**Load profile:** speech about 85% of each call. **Box layout:** 4 CPU threads, 1 process(es) x 4 threads (only shape that sustained a leg). **Measured:** saturation at **1 legs/box** (first failing 2, confirmed). **Planning capacity:** 1.00 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 4 GB (1 legs x 148 MB/leg + 1052 MB model/buffers, x1.20 headroom + 2 GB OS = 3.4 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 55 (44-55) | 4 | 4 GB | <= 0.35 | <= 1.9 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 50+5 spare boxes; confidence Low |
| 60 | 66 (53-66) | 4 | 4 GB | <= 0.35 | <= 1.9 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 60+6 spare boxes; confidence Low |
| 100 | 110 (87-110) | 4 | 4 GB | <= 0.35 | <= 1.9 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 100+10 spare boxes; confidence Low |
| 200 | 220 (174-220) | 4 | 4 GB | <= 0.35 | <= 1.9 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 200+20 spare boxes; confidence Low |
| 500 | 550 (433-550) | 4 | 4 GB | <= 0.35 | <= 1.9 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 500+50 spare boxes; confidence Very low |
| 1,000 | 1,100 (865-1,100) | 4 | 4 GB | <= 0.35 | <= 1.9 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 1000+100 spare boxes; confidence Very low |

> Headroom could not be applied: a box barely sustains one leg, so each box hosts a single leg and capacity is not additive in a meaningful way below that.

## 3. Measured results (test machine)

**Machine:** AMD Ryzen 7 6800H with Radeon Graphics, 8 cores / 16 threads, 14.9 GB RAM, one socket / one NUMA node, onnxruntime 1.30.0.

### 3.1 Saturation point per scenario (MEASURED)

> A scenario pins N logical CPUs of this edge machine (hardware-thread sibling pairs, so whole physical cores) across P processes with T ORT threads each. **Max legs kept up** is the highest simultaneous leg count at which every leg stayed within the lag threshold, found by ramping then bisecting, then re-run to confirm. A result of 0 means even one leg could not keep up on that layout. Memory slope is a DERIVED regression over the healthy levels.

| Model | Profile | CPU threads | Procs x threads | Base RSS (all procs) | Load | Max legs kept up | First failing | Confirmed | Memory per leg (DERIVED) | Stopped because |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_onnx_int8_0.6b | conversational | 4 | 1x4 | 3839 MB | 5.3s | 1 | 2 | yes | 330 MB (R2 0.49) | saturated |
| qwen3_onnx_int8_0.6b | conversational | 8 | 1x8 | 3826 MB | 3.7s | 1 | 2 | yes | 346 MB (R2 0.47) | saturated |
| qwen3_onnx_int8_0.6b | conversational | 16 | 1x16 | 3704 MB | 3.7s | 1 | 2 | yes | 343 MB (R2 0.47) | saturated |
| qwen3_onnx_int8_0.6b | conversational | 16 | 2x8 | 7661 MB | 4.7s | 1 | 2 | yes | 264 MB (R2 0.61) | saturated |
| whisper_int8_tiny | conversational | 4 | 1x4 | 1031 MB | 3.1s | 1 | 2 | yes | 70 MB (R2 0.21) | saturated |
| whisper_int8_tiny | conversational | 8 | 1x8 | 1050 MB | 2.6s | 2 | 3 | yes | 106 MB (R2 0.70) | saturated |
| whisper_int8_tiny | conversational | 16 | 1x16 | 1036 MB | 2.6s | 2 | 3 | yes | 111 MB (R2 0.73) | saturated |
| whisper_int8_tiny | conversational | 16 | 2x8 | 2083 MB | 2.6s | 3 | 4 | yes | 84 MB (R2 0.74) | saturated |
| whisper_int8_tiny | conversational | 16 | 4x4 | 4143 MB | 2.8s | 1 | 2 | yes | 38 MB (R2 0.95) | saturated |
| qwen3_onnx_int8_0.6b | dense | 4 | 1x4 | 3670 MB | 5.0s | 0 | 1 | no | n/a | saturated |
| qwen3_onnx_int8_0.6b | dense | 8 | 1x8 | 3769 MB | 5.4s | 1 | 2 | yes | 154 MB (R2 0.48) | saturated |
| qwen3_onnx_int8_0.6b | dense | 16 | 1x16 | 3820 MB | 3.9s | 0 | 1 | no | n/a | saturated |
| qwen3_onnx_int8_0.6b | dense | 16 | 2x8 | 7569 MB | 4.1s | 1 | 2 | yes | 158 MB (R2 0.49) | saturated |
| whisper_int8_tiny | dense | 4 | 1x4 | 1052 MB | 3.0s | 1 | 2 | yes | 148 MB (R2 0.32) | saturated |
| whisper_int8_tiny | dense | 8 | 1x8 | 1048 MB | 2.5s | 1 | 2 | yes | 87 MB (R2 0.40) | saturated |
| whisper_int8_tiny | dense | 16 | 1x16 | 1058 MB | 2.4s | 1 | 2 | yes | 87 MB (R2 0.38) | saturated |
| whisper_int8_tiny | dense | 16 | 2x8 | 2101 MB | 2.4s | 1 | 2 | yes | 19 MB (R2 0.05) | saturated |
| whisper_int8_tiny | dense | 16 | 4x4 | 4196 MB | 2.5s | 0 | 1 | no | n/a | saturated |

### 3.2 More pinned threads on this CPU: `qwen3_onnx_int8_0.6b`, conversational

Single process, using that many logical CPUs of this machine. Adding threads did not add legs in proportion.

| CPU threads pinned | Max legs kept up (MEASURED) |
| --- | --- |
| 4 | 1 |
| 8 | 1 |
| 16 | 1 |

**Universal Scalability Law fit on these measured points only (DERIVED, 3 points, RMSE 0.00 legs):** lambda = 1.000 legs/thread, contention sigma = 1.000, coherency kappa = 0.0000. This describes diminishing returns on *this* chip; it is not used to predict other CPUs.

### 3.3 Process strategy at 16 CPU threads: `qwen3_onnx_int8_0.6b`, conversational

One process shares the weights between all legs (least memory, but all legs contend inside one interpreter and one ORT thread pool). Several pinned processes duplicate the weights but isolate the contention.

| Layout | Base RSS (MEASURED) | Max legs kept up | Legs per GB of weights | Note |
| --- | --- | --- | --- | --- |
| 1 x 16 threads | 3704 MB | 1 | 0.28 |  |
| 2 x 8 threads | 7661 MB | 1 | 0.13 |  |

### 3.4 Latency vs load on the reference layout: `qwen3_onnx_int8_0.6b`, conversational (4 CPU threads, 1x4)

This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | yes | 1.26 s | 0.42 | 1.59 s | 53% | 2.6 | 3974 MB |
| 1 | 1.00 | yes | 1.24 s | 0.42 | 1.59 s | 53% | 2.6 | 4363 MB |
| 2 | 2.00 | no | 5.59 s | 1.36 | 5.85 s | 74% | 3.5 | 4363 MB |

Operating point used for the targets: **1 legs** (stale P95 1.26 s, pass RTF P95 0.42); at the saturation point (1 legs) stale P95 was 1.26 s.

### 3.2 More pinned threads on this CPU: `whisper_int8_tiny`, conversational

Single process, using that many logical CPUs of this machine. Adding threads did not add legs in proportion.

| CPU threads pinned | Max legs kept up (MEASURED) |
| --- | --- |
| 4 | 1 |
| 8 | 2 |
| 16 | 2 |

**Universal Scalability Law fit on these measured points only (DERIVED, 3 points, RMSE 0.13 legs):** lambda = 0.294 legs/thread, contention sigma = 0.000, coherency kappa = 0.0055. This describes diminishing returns on *this* chip; it is not used to predict other CPUs.

### 3.3 Process strategy at 16 CPU threads: `whisper_int8_tiny`, conversational

One process shares the weights between all legs (least memory, but all legs contend inside one interpreter and one ORT thread pool). Several pinned processes duplicate the weights but isolate the contention.

| Layout | Base RSS (MEASURED) | Max legs kept up | Legs per GB of weights | Note |
| --- | --- | --- | --- | --- |
| 1 x 16 threads | 1036 MB | 2 | 1.98 |  |
| 2 x 8 threads | 2083 MB | 3 | 1.48 |  |
| 4 x 4 threads | 4143 MB | 1 | 0.25 |  |

### 3.4 Latency vs load on the reference layout: `whisper_int8_tiny`, conversational (8 CPU threads, 1x8)

This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | yes | 1.26 s | 0.20 | 0.27 s | 51% | 4.4 | 1070 MB |
| 2 | 1.00 | yes | 1.58 s | 0.40 | 0.55 s | 56% | 4.8 | 1180 MB |
| 2 | 1.00 | yes | 1.57 s | 0.42 | 0.54 s | 56% | 4.8 | 1314 MB |
| 3 | 1.50 | no | 2.16 s | 0.46 | 0.74 s | 71% | 6.4 | 1306 MB |

Operating point used for the targets: **2 legs** (stale P95 1.58 s, pass RTF P95 0.40); at the saturation point (2 legs) stale P95 was 1.58 s.

### 3.2 More pinned threads on this CPU: `qwen3_onnx_int8_0.6b`, dense

Single process, using that many logical CPUs of this machine. Adding threads did not add legs in proportion.

| CPU threads pinned | Max legs kept up (MEASURED) |
| --- | --- |
| 4 | 0 |
| 8 | 1 |
| 16 | 0 |

_(not enough measured points with at least one sustained leg for a scaling-law fit)_

### 3.3 Process strategy at 16 CPU threads: `qwen3_onnx_int8_0.6b`, dense

One process shares the weights between all legs (least memory, but all legs contend inside one interpreter and one ORT thread pool). Several pinned processes duplicate the weights but isolate the contention.

| Layout | Base RSS (MEASURED) | Max legs kept up | Legs per GB of weights | Note |
| --- | --- | --- | --- | --- |
| 1 x 16 threads | 3820 MB | 0 | 0.00 |  |
| 2 x 8 threads | 7569 MB | 1 | 0.14 |  |

### 3.4 Latency vs load on the reference layout: `qwen3_onnx_int8_0.6b`, dense (8 CPU threads, 1x8)

This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | yes | 1.04 s | 0.37 | 1.44 s | 68% | 5.7 | 3831 MB |
| 1 | 1.00 | yes | 1.06 s | 0.36 | 1.45 s | 67% | 5.7 | 4016 MB |
| 2 | 2.00 | no | 5.37 s | 1.06 | 6.89 s | 93% | 8.0 | 4016 MB |

Operating point used for the targets: **1 legs** (stale P95 1.04 s, pass RTF P95 0.37); at the saturation point (1 legs) stale P95 was 1.04 s.

### 3.2 More pinned threads on this CPU: `whisper_int8_tiny`, dense

Single process, using that many logical CPUs of this machine. Adding threads did not add legs in proportion.

| CPU threads pinned | Max legs kept up (MEASURED) |
| --- | --- |
| 4 | 1 |
| 8 | 1 |
| 16 | 1 |

**Universal Scalability Law fit on these measured points only (DERIVED, 3 points, RMSE 0.00 legs):** lambda = 1.000 legs/thread, contention sigma = 1.000, coherency kappa = 0.0000. This describes diminishing returns on *this* chip; it is not used to predict other CPUs.

### 3.3 Process strategy at 16 CPU threads: `whisper_int8_tiny`, dense

One process shares the weights between all legs (least memory, but all legs contend inside one interpreter and one ORT thread pool). Several pinned processes duplicate the weights but isolate the contention.

| Layout | Base RSS (MEASURED) | Max legs kept up | Legs per GB of weights | Note |
| --- | --- | --- | --- | --- |
| 1 x 16 threads | 1058 MB | 1 | 0.97 |  |
| 2 x 8 threads | 2101 MB | 1 | 0.49 |  |
| 4 x 4 threads | 4196 MB | 0 | 0.00 |  |

### 3.4 Latency vs load on the reference layout: `whisper_int8_tiny`, dense (4 CPU threads, 1x4)

This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | yes | 1.86 s | 0.35 | 1.54 s | 76% | 3.8 | 1075 MB |
| 1 | 1.00 | yes | 1.94 s | 0.53 | 0.57 s | 75% | 3.8 | 1325 MB |
| 2 | 2.00 | no | 2.68 s | 0.69 | 0.00 s | 72% | 3.6 | 1314 MB |

Operating point used for the targets: **1 legs** (stale P95 1.86 s, pass RTF P95 0.35); at the saturation point (1 legs) stale P95 was 1.86 s.

## 4. Model and assumptions

For each layout, model and profile: `legs/box = max(1, measured_saturation x headroom / serving_overhead)`; `boxes = ceil(legs / legs_per_box) + max(min_spare, ceil(spare_fraction x boxes))`; `RAM/box = ceil(((processes x weights) + legs_on_box x MB_per_leg) x ram_headroom + OS)` in GB.

| Factor | Treatment | Value | Tag |
| --- | --- | --- | --- |
| Saturation point | Highest simultaneous legs that all keep up (p95 staleness and end lag <= threshold), bisected to 1 leg, re-confirmed | section 3.1 | MEASURED |
| Speech density (profile) | Silence skips inference, so a conversational leg costs less than a dense one. Both measured; real traffic must be checked against the profile used | section 2 / 3.1 | MEASURED (profile) + ASSUMED (traffic mix) |
| Headroom / queueing | Waiting time grows without bound as utilisation approaches 1, so each box is run below the measured knee | 70% of saturation | ASSUMED |
| Serving-layer overhead | WebSocket framing, JSON acks, resampling and process supervision are not in the in-process test | x1.10 CPU | ASSUMED |
| Spare capacity (N+k) | A failed or draining box must not push the rest over the knee | max(1, 10% of boxes) | ASSUMED |
| Shared model memory | Weights are one copy per process and shared by all legs in it; a pinned-process layout multiplies them | Base RSS per layout, section 3.1/3.3 | MEASURED |
| Per-leg memory | Linear regression of peak RSS vs legs over healthy levels | section 3.1 | DERIVED |
| RAM headroom / OS | Allocator fragmentation, page cache and bursts; OS and agents. Rounded up to a whole GB for this box | x1.20, 2 GB/box | ASSUMED |
| Thread contention / diminishing throughput | Core sweep on this chip; more threads are not assumed proportionally better, so extra legs need extra boxes | section 3.2 | MEASURED + DERIVED |
| Process strategy | 1 process vs several pinned processes compared at the top CPU-thread count; the best measured layout is the reference | section 3.3 | MEASURED |
| NUMA | Test machine is one socket / one NUMA node: cross-socket effects were not measured | n/a | ASSUMED |
| Batching across legs | The engines decode one stream per pass; no cross-leg batching exists, so none is credited. Batching would raise capacity but is unmeasured | none credited | NOT MODELLED |
| Scale-out | Calls are sticky to a box, boxes share nothing, so capacity is additive across identical edge boxes; confidence falls with the extrapolation ratio | ceil(N / legs per box) | EXTRAPOLATED |
| Target RTF / P95 latency | Values measured at the operating load on the reference layout | section 3.4 | MEASURED |

## 5. Sensitivity (edge boxes needed, including spares)

How much the answer moves when an ASSUMED input or the 1-leg resolution of the measurement changes.

**`qwen3_onnx_int8_0.6b`, conversational**

| Variant | Boxes @ 100 legs | Boxes @ 1,000 legs |
| --- | --- | --- |
| central | 110 | 1,100 |
| headroom 60% | 110 | 1,100 |
| headroom 80% | 110 | 1,100 |
| serving overhead x1.00 | 110 | 1,100 |
| serving overhead x1.25 | 110 | 1,100 |
| no spare boxes | 100 | 1,000 |
| measured saturation 1 leg higher | 87 | 865 |
| measured saturation 1 leg lower | 110 | 1,100 |

**`whisper_int8_tiny`, conversational**

| Variant | Boxes @ 100 legs | Boxes @ 1,000 legs |
| --- | --- | --- |
| central | 87 | 865 |
| headroom 60% | 102 | 1,009 |
| headroom 80% | 76 | 757 |
| serving overhead x1.00 | 80 | 787 |
| serving overhead x1.25 | 99 | 983 |
| no spare boxes | 79 | 786 |
| measured saturation 1 leg higher | 59 | 577 |
| measured saturation 1 leg lower | 110 | 1,100 |

**`qwen3_onnx_int8_0.6b`, dense**

| Variant | Boxes @ 100 legs | Boxes @ 1,000 legs |
| --- | --- | --- |
| central | 110 | 1,100 |
| headroom 60% | 110 | 1,100 |
| headroom 80% | 110 | 1,100 |
| serving overhead x1.00 | 110 | 1,100 |
| serving overhead x1.25 | 110 | 1,100 |
| no spare boxes | 100 | 1,000 |
| measured saturation 1 leg higher | 87 | 865 |
| measured saturation 1 leg lower | 110 | 1,100 |

**`whisper_int8_tiny`, dense**

| Variant | Boxes @ 100 legs | Boxes @ 1,000 legs |
| --- | --- | --- |
| central | 110 | 1,100 |
| headroom 60% | 110 | 1,100 |
| headroom 80% | 110 | 1,100 |
| serving overhead x1.00 | 110 | 1,100 |
| serving overhead x1.25 | 110 | 1,100 |
| no spare boxes | 100 | 1,000 |
| measured saturation 1 leg higher | 87 | 865 |
| measured saturation 1 leg lower | 110 | 1,100 |

## 6. Confidence and limitations

**Confidence rubric (judgement, stated so it can be challenged).** Measured = within the legs one test box sustained.
Beyond that confidence falls with the extrapolation ratio `N / measured saturation legs`: <= 10x Medium, <= 50x Low-Medium,
<= 200x Low, above Very low; one level lower when the saturation point is only 1-2 legs, because its 1-leg resolution is then
a +/-33-100% uncertainty on per-box capacity.

- **One machine, one CPU.** A laptop-class 8-core / 16-thread edge CPU with boost and thermal behaviour. Another edge box
  (different frequency, SIMD, cooling) will shift the saturation point. Re-run the load test on the target hardware; the pipeline is the same.
- **Shared test machine.** The IDE and a browser ran on the same machine and RAM was tight (little free memory, swap in use), which can only make
  results worse, not better, but adds noise. Memory-hungry layouts that did not fit in free RAM were skipped; they are listed in 3.1 with the
  measured per-process RSS.
- **Load profile.** Three clean English read-speech clips looped into 30 s calls. Real calls also have noise, accents, other languages (more decoded
  tokens), overlap and talk ratios that vary by use case; the two profiles bracket speech density but not acoustic difficulty.
- **Scaling beyond one box is assumed, not measured.** Only one machine was available, so the guide treats extra edge boxes as independent
  (calls sticky to a box). Load-balancing skew, correlated peaks and failure domains are covered only by headroom and spares.
- **No network path.** WebSocket, TLS and JSON cost are not in the test; ASSUMED overhead stands in for them.
- **The lag threshold is a design choice.** Saturation is defined as every leg's p95 staleness and end lag <= the threshold in section 3 (2 s by default). Streaming ASR
  has a staleness floor of about 1.0-1.3 s even for one leg (hop + pass time), so the threshold sits close to it; at a 3 s threshold several layouts
  would sustain one more leg (see the latency-vs-load tables in 3.4 for how near the knee the failing level was). Re-run with `slo.lag_threshold_s` set to your product SLO.
- **Saturation resolution is whole legs and was re-confirmed once**, not statistically repeated. The +/-1 leg sensitivity in section 5 is the honest error bar;
  at 1-2 legs per box it is large.
- **Short calls.** 30 s calls; memory growth over hour-long calls was not measured (the streaming code bounds buffers: Qwen force-commits at 15 s, Whisper windows cap at 20 s).
- **No batching credit.** Cross-leg batching of the encoder/decoder would reduce cost per leg but is not implemented here, so it is neither measured nor assumed.
