# Edge CPU Capacity Sizing Guide (50 / 60 / 100 / 200 / 500 / 1,000 concurrent legs)

> Curated from the load-test run on the development laptop (Ryzen 7 6800H, 8C/16T). Produced by
> `loadtest/run_loadtest.py` + `loadtest/run_sizing.py`; see [`loadtest.md`](loadtest.md) for how it works.
> **Every row in the sizing tables is EXTRAPOLATED**: the test machine saturates at 1-3 legs, so the
> sizing for 50+ legs is a model built on those measurements, not something that was run.

This project deploys on **edge CPUs**. The load test ramps **this whole machine** (all 16 threads, one process) once per
model and load profile; one box in the tables below is one such machine.

## Key findings

1. **Saturation on this machine is 1-3 legs.** Whisper tiny INT8 sustains **3 conversational legs** (2 dense);
   Qwen3-ASR-0.6B INT8 sustains **1 leg** in both profiles. All four points were re-confirmed.
2. **For 100 conversational legs:** about **59 Whisper-tiny boxes** (range 44-87, ~4 GB RAM each) or **110 Qwen3 boxes**
   (~7 GB each), including 10% spares. **CPU is the constraint, not RAM.**
3. **Each leg costs a lot of CPU.** Streaming re-decodes a window (Whisper) or a growing utterance (Qwen3) every hop;
   one pass for a single leg already takes 0.17-0.37 of real time (pass RTF), so a box is full after a handful of legs.
   This is a property of the streaming architecture, not the hardware: cross-leg batching, a smaller hop rate or a
   smarter commit policy would change it, and none of them is credited here.
4. **The knee is sharp.** Whisper conversational p95 staleness is 0.93 s at 2 legs, 1.65 s at 3 and 3.02 s at 4;
   Qwen3 goes from 0.97 s at 1 leg to 3.14 s at 2. This is why boxes are planned at 70% of saturation.
5. **The result is sensitive to the 2 s lag threshold.** Streaming staleness is already about 1 s for a single leg,
   and Whisper at its saturation point sits at 1.65-1.69 s. Re-run with your product SLO.
6. **Run-to-run noise is about one leg** (a shared, thermally limited laptop). An earlier whole-machine run
   (`20261006T131842Z` / `20261006T141231Z`) measured Whisper at 2 conversational / 1 dense and Qwen3 dense at 0, one
   leg lower than this run in three of four cases. For Whisper that one leg moves 100 legs between 59 and 87 boxes
   (section 5). For a conservative plan, repeat the run and use the lower result.

## How to use the tables

- Read the **conversational** table for planning and the **dense** table as the upper bound.
- *Edge boxes* is the number of identical machines (central estimate, with the +/-1-leg range in parentheses), including spares.
- *CPU threads / box* and *RAM / box* describe one machine. RAM is the GB that box needs (weights + per-leg RSS,
  headroom and OS, rounded up).
- Before buying hardware, re-run the load test on the target edge CPU; only the YAML changes.

---

**Generated:** 2026-10-07T17:21:23Z  
**Measured data:** `20261007T170448Z_loadtest_raw.json`

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

> **Edge boxes** is how many boxes identical to the measured one are needed (central estimate, with the range if saturation is one leg higher or lower in parentheses), including spare boxes. **CPU threads / box** is how many logical CPUs the measured run pinned. **RAM / box** is the GB this box needs (measured weights + per-leg RSS, with RAM headroom and OS reserve, rounded up to a whole GB). **Target RTF** = P95 real-time factor of one inference pass at the operating point; **Target P95 latency** = P95 *staleness* (how far the live transcript trails the speaker, queueing included) at that point. Both targets are the values *measured* at the operating load on the reference layout; the capacity figures only hold if each box is operated at or below that load.

### Qwen3-ASR-0.6B INT8 (ONNX / CPU) (`qwen3_onnx_int8_0.6b`), conversational (about half of each call is speech): the planning case

**Load profile:** speech about 47% of each call. **Box layout:** 16 CPU threads, 1 process(es) x 16 threads (the measured layout on this machine). **Measured:** saturation at **1 legs/box** (first failing 2, confirmed). **Planning capacity:** 1.00 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 7 GB (1 legs x 308 MB/leg + 3683 MB model/buffers, x1.20 headroom + 2 GB OS = 6.7 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 55 (44-55) | 16 | 7 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 50+5 spare boxes; confidence Low |
| 60 | 66 (53-66) | 16 | 7 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 60+6 spare boxes; confidence Low |
| 100 | 110 (87-110) | 16 | 7 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 100+10 spare boxes; confidence Low |
| 200 | 220 (174-220) | 16 | 7 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 200+20 spare boxes; confidence Low |
| 500 | 550 (433-550) | 16 | 7 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 500+50 spare boxes; confidence Very low |
| 1,000 | 1,100 (865-1,100) | 16 | 7 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 1000+100 spare boxes; confidence Very low |

> Headroom could not be applied: a box barely sustains one leg, so each box hosts a single leg and capacity is not additive in a meaningful way below that.

### Whisper tiny INT8 (ONNX / CPU) (`whisper_int8_tiny`), conversational (about half of each call is speech): the planning case

**Load profile:** speech about 47% of each call. **Box layout:** 16 CPU threads, 1 process(es) x 16 threads (the measured layout on this machine). **Measured:** saturation at **3 legs/box** (first failing 4, confirmed). **Planning capacity:** 1.91 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 4 GB (2 legs x 133 MB/leg + 1035 MB model/buffers, x1.20 headroom + 2 GB OS = 3.5 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 30 (22-44) | 16 | 4 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.91 legs/box = 70% of measured max 3 / 1.10 overhead; 27+3 spare boxes; confidence Low-Medium |
| 60 | 36 (27-53) | 16 | 4 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.91 legs/box = 70% of measured max 3 / 1.10 overhead; 32+4 spare boxes; confidence Low-Medium |
| 100 | 59 (44-87) | 16 | 4 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.91 legs/box = 70% of measured max 3 / 1.10 overhead; 53+6 spare boxes; confidence Low-Medium |
| 200 | 116 (87-174) | 16 | 4 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.91 legs/box = 70% of measured max 3 / 1.10 overhead; 105+11 spare boxes; confidence Low |
| 500 | 289 (217-433) | 16 | 4 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.91 legs/box = 70% of measured max 3 / 1.10 overhead; 262+27 spare boxes; confidence Low |
| 1,000 | 577 (433-865) | 16 | 4 GB | <= 0.35 | <= 1.0 s | EXTRAPOLATED: 1.91 legs/box = 70% of measured max 3 / 1.10 overhead; 524+53 spare boxes; confidence Very low |

### Qwen3-ASR-0.6B INT8 (ONNX / CPU) (`qwen3_onnx_int8_0.6b`), dense speech (almost continuous talking): the worst-case stress profile

**Load profile:** speech about 85% of each call. **Box layout:** 16 CPU threads, 1 process(es) x 16 threads (the measured layout on this machine). **Measured:** saturation at **1 legs/box** (first failing 2, confirmed). **Planning capacity:** 1.00 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 7 GB (1 legs x 164 MB/leg + 3716 MB model/buffers, x1.20 headroom + 2 GB OS = 6.5 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 55 (44-55) | 16 | 7 GB | <= 0.40 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 50+5 spare boxes; confidence Low |
| 60 | 66 (53-66) | 16 | 7 GB | <= 0.40 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 60+6 spare boxes; confidence Low |
| 100 | 110 (87-110) | 16 | 7 GB | <= 0.40 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 100+10 spare boxes; confidence Low |
| 200 | 220 (174-220) | 16 | 7 GB | <= 0.40 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 200+20 spare boxes; confidence Low |
| 500 | 550 (433-550) | 16 | 7 GB | <= 0.40 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 500+50 spare boxes; confidence Very low |
| 1,000 | 1,100 (865-1,100) | 16 | 7 GB | <= 0.40 | <= 1.0 s | EXTRAPOLATED: 1.00 legs/box = 70% of measured max 1 / 1.10 overhead; 1000+100 spare boxes; confidence Very low |

> Headroom could not be applied: a box barely sustains one leg, so each box hosts a single leg and capacity is not additive in a meaningful way below that.

### Whisper tiny INT8 (ONNX / CPU) (`whisper_int8_tiny`), dense speech (almost continuous talking): the worst-case stress profile

**Load profile:** speech about 85% of each call. **Box layout:** 16 CPU threads, 1 process(es) x 16 threads (the measured layout on this machine). **Measured:** saturation at **2 legs/box** (first failing 3, confirmed). **Planning capacity:** 1.27 legs/box (ASSUMED headroom 70%, serving overhead x1.10). **RAM / box:** 4 GB (2 legs x 121 MB/leg + 1052 MB model/buffers, x1.20 headroom + 2 GB OS = 3.5 GB, rounded up).

| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 44 (30-55) | 16 | 4 GB | <= 0.30 | <= 1.4 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 40+4 spare boxes; confidence Low |
| 60 | 53 (36-66) | 16 | 4 GB | <= 0.30 | <= 1.4 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 48+5 spare boxes; confidence Low |
| 100 | 87 (59-110) | 16 | 4 GB | <= 0.30 | <= 1.4 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 79+8 spare boxes; confidence Low |
| 200 | 174 (116-220) | 16 | 4 GB | <= 0.30 | <= 1.4 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 158+16 spare boxes; confidence Low |
| 500 | 433 (289-550) | 16 | 4 GB | <= 0.30 | <= 1.4 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 393+40 spare boxes; confidence Very low |
| 1,000 | 865 (577-1,100) | 16 | 4 GB | <= 0.30 | <= 1.4 s | EXTRAPOLATED: 1.27 legs/box = 70% of measured max 2 / 1.10 overhead; 786+79 spare boxes; confidence Very low |

## 3. Measured results (test machine)

**Machine:** AMD Ryzen 7 6800H with Radeon Graphics, 8 cores / 16 threads, 14.9 GB RAM, one socket / one NUMA node, onnxruntime 1.30.0.

### 3.1 Saturation point per scenario (MEASURED)

> Each run pins the listed logical CPUs of this edge machine (hardware-thread sibling pairs, so whole physical cores; by default all of them, in one process). **Max legs kept up** is the highest simultaneous leg count at which every leg stayed within the lag threshold, found by ramping then bisecting, then re-run to confirm. A result of 0 means even one leg could not keep up on that layout. Memory slope is a DERIVED regression over the healthy levels.

| Model | Profile | CPU threads | Procs x threads | Base RSS (all procs) | Load | Max legs kept up | First failing | Confirmed | Memory per leg (DERIVED) | Stopped because |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_onnx_int8_0.6b | conversational | 16 | 1x16 | 3683 MB | 3.3s | 1 | 2 | yes | 308 MB (R2 0.54) | saturated |
| whisper_int8_tiny | conversational | 16 | 1x16 | 1035 MB | 2.3s | 3 | 4 | yes | 133 MB (R2 0.91) | saturated |
| qwen3_onnx_int8_0.6b | dense | 16 | 1x16 | 3716 MB | 5.0s | 1 | 2 | yes | 164 MB (R2 0.46) | saturated |
| whisper_int8_tiny | dense | 16 | 1x16 | 1052 MB | 2.3s | 2 | 3 | yes | 121 MB (R2 0.74) | saturated |

### 3.2 `qwen3_onnx_int8_0.6b`, conversational (16 CPU threads, 1x16)

**Latency vs load.** This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | yes | 0.97 s | 0.35 | 1.13 s | 42% | 6.7 | 3827 MB |
| 1 | 1.00 | yes | 0.97 s | 0.35 | 1.09 s | 42% | 6.7 | 4155 MB |
| 2 | 2.00 | no | 3.14 s | 0.96 | 1.43 s | 58% | 9.1 | 4155 MB |

Operating point used for the targets: **1 legs** (stale P95 0.97 s, pass RTF P95 0.35); at the saturation point (1 legs) stale P95 was 0.97 s.

### 3.3 `whisper_int8_tiny`, conversational (16 CPU threads, 1x16)

**Latency vs load.** This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.33 | yes | 0.91 s | 0.17 | 0.00 s | 43% | 6.7 | 1069 MB |
| 2 | 0.67 | yes | 0.93 s | 0.32 | 0.00 s | 46% | 7.2 | 1178 MB |
| 3 | 1.00 | yes | 1.65 s | 0.43 | 0.30 s | 60% | 9.5 | 1391 MB |
| 3 | 1.00 | yes | 1.69 s | 0.40 | 0.29 s | 61% | 9.6 | 1431 MB |
| 4 | 1.33 | no | 3.02 s | 0.76 | 3.74 s | 86% | 13.6 | 1431 MB |

Operating point used for the targets: **2 legs** (stale P95 0.93 s, pass RTF P95 0.32); at the saturation point (3 legs) stale P95 was 1.65 s.

### 3.4 `qwen3_onnx_int8_0.6b`, dense (16 CPU threads, 1x16)

**Latency vs load.** This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | yes | 0.99 s | 0.37 | 1.33 s | 61% | 9.7 | 3778 MB |
| 1 | 1.00 | yes | 0.93 s | 0.33 | 1.35 s | 61% | 9.6 | 3982 MB |
| 2 | 2.00 | no | 1.86 s | 0.65 | 2.64 s | 76% | 12.1 | 3977 MB |

Operating point used for the targets: **1 legs** (stale P95 0.99 s, pass RTF P95 0.37); at the saturation point (1 legs) stale P95 was 0.99 s.

### 3.5 `whisper_int8_tiny`, dense (16 CPU threads, 1x16)

**Latency vs load.** This is why a box is not run at its saturation point: staleness is flat, then climbs steeply.

| Legs | Load / saturation | Healthy | Stale P95 | Pass RTF P95 | End lag max | CPU (pinned set) | Cores used | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | yes | 0.90 s | 0.22 | 1.56 s | 52% | 8.2 | 1077 MB |
| 2 | 1.00 | yes | 1.40 s | 0.30 | 1.28 s | 65% | 10.3 | 1212 MB |
| 2 | 1.00 | yes | 1.69 s | 0.46 | 1.98 s | 71% | 11.3 | 1345 MB |
| 3 | 1.50 | no | 2.35 s | 0.53 | 1.87 s | 87% | 13.8 | 1342 MB |

Operating point used for the targets: **2 legs** (stale P95 1.40 s, pass RTF P95 0.30); at the saturation point (2 legs) stale P95 was 1.40 s.

## 4. Model and assumptions

For each model and profile: `legs/box = max(1, measured_saturation x headroom / serving_overhead)`; `boxes = ceil(legs / legs_per_box) + max(min_spare, ceil(spare_fraction x boxes))`; `RAM/box = ceil(((processes x weights) + legs_on_box x MB_per_leg) x ram_headroom + OS)` in GB.

| Factor | Treatment | Value | Tag |
| --- | --- | --- | --- |
| Saturation point | Highest simultaneous legs that all keep up (p95 staleness and end lag <= threshold), bisected to 1 leg, re-confirmed | section 3.1 | MEASURED |
| Speech density (profile) | Silence skips inference, so a conversational leg costs less than a dense one. Both measured; real traffic must be checked against the profile used | section 2 / 3.1 | MEASURED (profile) + ASSUMED (traffic mix) |
| Headroom / queueing | Waiting time grows without bound as utilisation approaches 1, so each box is run below the measured knee | 70% of saturation | ASSUMED |
| Serving-layer overhead | WebSocket framing, JSON acks, resampling and process supervision are not in the in-process test | x1.10 CPU | ASSUMED |
| Spare capacity (N+k) | A failed or draining box must not push the rest over the knee | max(1, 10% of boxes) | ASSUMED |
| Shared model memory | Weights are one copy per process and shared by all legs in it; a pinned-process layout multiplies them | Base RSS, section 3.1 | MEASURED |
| Per-leg memory | Linear regression of peak RSS vs legs over healthy levels | section 3.1 | DERIVED |
| RAM headroom / OS | Allocator fragmentation, page cache and bursts; OS and agents. Rounded up to a whole GB for this box | x1.20, 2 GB/box | ASSUMED |
| Thread contention / diminishing throughput | Measured as the saturation of this machine (one process, all threads). Extra legs need extra boxes; a bigger chip is not assumed to add legs in proportion | section 3.1 | MEASURED |
| Process strategy | Default is one process using the whole machine. Extra processes would duplicate weights; that trade-off is not swept | 1 process | ASSUMED |
| NUMA | Test machine is one socket / one NUMA node: cross-socket effects were not measured | n/a | ASSUMED |
| Batching across legs | The engines decode one stream per pass; no cross-leg batching exists, so none is credited. Batching would raise capacity but is unmeasured | none credited | NOT MODELLED |
| Scale-out | Calls are sticky to a box, boxes share nothing, so capacity is additive across identical edge boxes; confidence falls with the extrapolation ratio | ceil(N / legs per box) | EXTRAPOLATED |
| Target RTF / P95 latency | Values measured at the operating load on the measured box | section 3 (latency vs load) | MEASURED |

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
| central | 59 | 577 |
| headroom 60% | 69 | 674 |
| headroom 80% | 51 | 505 |
| serving overhead x1.00 | 53 | 525 |
| serving overhead x1.25 | 66 | 656 |
| no spare boxes | 53 | 524 |
| measured saturation 1 leg higher | 44 | 433 |
| measured saturation 1 leg lower | 87 | 865 |

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
| central | 87 | 865 |
| headroom 60% | 102 | 1,009 |
| headroom 80% | 76 | 757 |
| serving overhead x1.00 | 80 | 787 |
| serving overhead x1.25 | 99 | 983 |
| no spare boxes | 79 | 786 |
| measured saturation 1 leg higher | 59 | 577 |
| measured saturation 1 leg lower | 110 | 1,100 |

## 6. Confidence and limitations

**Confidence rubric (judgement, stated so it can be challenged).** Measured = within the legs one test box sustained.
Beyond that confidence falls with the extrapolation ratio `N / measured saturation legs`: <= 10x Medium, <= 50x Low-Medium,
<= 200x Low, above Very low; one level lower when the saturation point is only 1-2 legs, because its 1-leg resolution is then
a +/-33-100% uncertainty on per-box capacity.

- **One machine, one CPU.** A laptop-class 8-core / 16-thread edge CPU with boost and thermal behaviour. Another edge box
  (different frequency, SIMD, cooling) will shift the saturation point. Re-run the load test on the target hardware; the pipeline is the same.
- **Shared test machine.** Anything else running on the test machine (IDE, browser, tight RAM) can only make results worse, not better,
  but it adds noise: a repeat run can land one leg higher or lower.
- **Load profile.** Three clean English read-speech clips looped into 30 s calls. Real calls also have noise, accents, other languages (more decoded
  tokens), overlap and talk ratios that vary by use case; the two profiles bracket speech density but not acoustic difficulty.
- **Scaling beyond one box is assumed, not measured.** Only one machine was available, so the guide treats extra edge boxes as independent
  (calls sticky to a box). Load-balancing skew, correlated peaks and failure domains are covered only by headroom and spares.
- **No network path.** WebSocket, TLS and JSON cost are not in the test; ASSUMED overhead stands in for them.
- **The lag threshold is a design choice.** Saturation is defined as every leg's p95 staleness and end lag <= the threshold in section 3 (2 s by default). Streaming ASR
  has a staleness floor of about 1.0-1.3 s even for one leg (hop + pass time), so the threshold sits close to it; at a 3 s threshold some models
  would sustain one more leg (see the latency-vs-load tables in section 3 for how near the knee the failing level was). Re-run with `slo.lag_threshold_s` set to your product SLO.
- **Saturation resolution is whole legs and was re-confirmed once**, not statistically repeated. The +/-1 leg sensitivity in section 5 is the honest error bar;
  at 1-2 legs per box it is large.
- **Short calls.** 30 s calls; memory growth over hour-long calls was not measured (the streaming code bounds buffers: Qwen force-commits at 15 s, Whisper windows cap at 20 s).
- **No batching credit.** Cross-leg batching of the encoder/decoder would reduce cost per leg but is not implemented here, so it is neither measured nor assumed.
