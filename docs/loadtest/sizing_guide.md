# Edge CPU Sizing Guide: 50 / 100 / 200 / 500 / 1,000 Concurrent Call Legs

> **Sources.** Measured: load-test run `20261007T202154Z`
> ([raw](../../loadtest/results/20261007T202154Z_loadtest_raw.json) · [summary](../../loadtest/results/20261007T202154Z_loadtest_summary.md) ·
> [curated results](final_result.md)). Sizing model: run `20261007T203650Z`
> ([JSON](../../loadtest/results/20261007T203650Z_sizing.json) · [generated guide](../../loadtest/results/20261007T203650Z_sizing_guide.md)).
> How the load test and sizing model work: [`loadtest.md`](loadtest.md). Production design: [`../deployment.md`](../deployment.md).
>
> **Every box count in this guide is EXTRAPOLATED.** The test machine saturates at 1-3 legs, so nothing at 50+ legs was run.
> Section 2 explains exactly which numbers were measured and which were derived, assumed or extrapolated.

---

## 1. Summary

**One concurrent call leg** = one independently streamed audio source (one direction of a call), transcribed live with the
2 s latency SLO (section 4). **One box** = one machine identical to the test machine: 8 cores / 16 threads (Ryzen 7 6800H class), one ASR process.

**Conversational traffic (about half of each call is speech): the planning case.** Boxes include spares; the range in
parentheses is the answer if the measured per-box capacity were one leg higher or lower.

| Concurrent legs | Qwen3-ASR-0.6B INT4 or INT8: boxes | CPU threads (physical cores) | Whisper tiny INT8: boxes | CPU threads (physical cores) | Confidence |
|---|---|---|---|---|---|
| 50 | **55** (44-55) | 880 (440) | **30** (22-44) | 480 (240) | Qwen Low · Whisper Low-Medium |
| 100 | **110** (87-110) | 1,760 (880) | **59** (44-87) | 944 (472) | Qwen Low · Whisper Low-Medium |
| 200 | **220** (174-220) | 3,520 (1,760) | **116** (87-174) | 1,856 (928) | Low |
| 500 | **550** (433-550) | 8,800 (4,400) | **289** (217-433) | 4,624 (2,312) | Qwen Very low · Whisper Low |
| 1,000 | **1,100** (865-1,100) | 17,600 (8,800) | **577** (433-865) | 9,232 (4,616) | Very low |

Key points:

1. **CPU is the constraint; RAM is not.** One 16-thread box carries **1 Qwen3 leg** or **3 Whisper tiny legs** (2 in dense speech)
   at the 2 s SLO (MEASURED). Every run left at least 7.9 GB of the box's 14.9 GB free.
2. **Qwen3 needs about one box per live leg** on this CPU class, plus 10% spares. INT4 and INT8 need the same number of boxes;
   INT4 needs 6 GB per box instead of 7 GB and about 12% less compute per pass (MEASURED), so it is the better Qwen choice.
3. **Whisper tiny needs about half as many boxes** in conversational traffic, but its accuracy on Mandarin is much worse
   ([technical report, section 7.2](../report/technical_report.md#72-accuracy)). Choose the model on accuracy first.
4. **Dense speech (section 5.3) does not change the Qwen count** (already one leg per box) but raises Whisper's from 59 to 87 boxes at 100 legs.
5. **The error bar is large.** Saturation is measured in whole legs on a 1-3 leg scale, so ±1 leg is ±33-100% of a box's capacity.
   Re-measure on the target edge CPU before buying hardware (section 9).

---

## 2. Measured vs extrapolated

### 2.1 Tags

Every number in this guide carries one of these tags.

| Tag | Meaning | Examples in this guide |
|---|---|---|
| **MEASURED** | Read directly from the load test on the test machine | Saturation point (max legs kept up), P95 staleness, pass RTF, CPU %, peak RSS, base RSS |
| **DERIVED** | Arithmetic or regression on measured numbers, no new assumptions | Memory per leg (regression slope), CPU threads busy per leg, physical-core totals |
| **ASSUMED** | An input the load test cannot measure; editable in `run_sizing.py` | 70% headroom, x1.10 serving overhead, 10% spare boxes, x1.20 RAM headroom, 2 GB OS reserve |
| **EXTRAPOLATED** | A prediction outside the measured range | Every box count, every fleet total, every row for 50+ legs |

### 2.2 Where measurement ends

| Scope | What was run | Status |
|---|---|---|
| Single box, up to its saturation point | All three models, both profiles; the saturation level was re-run once to confirm | **MEASURED** |
| Single box, first failing level | Whisper 4 legs (conversational) and 3 (dense); Qwen 2 legs (and 3 for INT4 dense) | **MEASURED** (where the knee is) |
| More legs than one box can carry | Not run: only one machine was available | **EXTRAPOLATED** by assuming identical, independent boxes |
| 50 / 100 / 200 / 500 / 1,000 legs | Not run | **EXTRAPOLATED** 17x to 1,000x beyond the measured per-box capacity |
| Network path (WebSocket, TLS, JSON, routing) | Not in the test (legs are fed in-process) | **ASSUMED** x1.10 CPU |
| Other hardware (bigger chips, servers, ARM) | Not run | **Not predicted.** Re-measure |

The extrapolation is only **scale-out**: "N legs need N / (legs per box) identical boxes". The test does not tell us whether
a load balancer, correlated call peaks or a shared network path add cost at fleet scale; headroom and spares are meant to cover that.

### 2.3 Confidence rubric

Confidence falls with the extrapolation ratio `N / measured legs per box`: up to 10x Medium, up to 50x Low-Medium, up to 200x Low,
beyond that Very low. When the measured saturation is only 1-2 legs, Medium and Low-Medium drop one level, because whole-leg
resolution is then a large relative error. The rubric is a judgement, stated so it can be challenged.

| Concurrent legs | Qwen3 (any profile), ratio | Whisper conversational, ratio | Whisper dense, ratio |
|---|---|---|---|
| 50 | 50x, Low | 17x, Low-Medium | 25x, Low |
| 100 | 100x, Low | 33x, Low-Medium | 50x, Low |
| 200 | 200x, Low | 67x, Low | 100x, Low |
| 500 | 500x, Very low | 167x, Low | 250x, Very low |
| 1,000 | 1,000x, Very low | 333x, Very low | 500x, Very low |

---

## 3. Measured inputs (one box)

### 3.1 Test machine

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 6800H, 8 cores / 16 threads, AVX2 + FMA (no AVX-512, VNNI or AMX) |
| Memory | 14.9 GB, one socket, one NUMA node |
| Runtime | ONNX Runtime 1.30.0, CPU execution provider |
| Layout | Whole machine: 16 logical CPUs pinned, **1 process x 16 threads**, all legs share one engine (same as the live server) |
| Calls | 30 s each, starts spread over 6 s, audio sent in 0.5 s chunks at real-time pace, three English LibriSpeech clips looped |
| Profiles | **Conversational**: 7 s silence after each clip, about 47% speech. **Dense**: 1 s silence, about 85% speech |
| Kept up (the SLO) | A leg keeps up if its P95 staleness **and** its end-of-call lag are ≤ 2 s; a level is healthy if every leg keeps up |
| Search | Ramp 1, 2, 3, 4, ... legs until the first unhealthy level, bisect, then re-run the answer to confirm |

*Staleness* is how far the live transcript trails the speaker: the time from the arrival of the newest audio a pass covers to the
moment that pass finishes, queueing included.

### 3.2 Saturation and cost at the operating point (MEASURED unless tagged)

The **operating point** is the measured level the sizing tool takes its latency targets from: 1 leg for Qwen and 2 legs for Whisper
(in dense speech, 2 legs is also Whisper's measured maximum).

| Model | Profile | Max legs kept up | First failing | Operating point | Stale P95 | Pass RTF P95 | CPU busy (of 16 threads) | Threads busy per leg (DERIVED) | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B INT4 | conversational | **1** | 2 | 1 leg | 0.83 s | 0.31 | 38% | 5.9 | 2.6 GB |
| Qwen3-0.6B INT8 | conversational | **1** | 2 | 1 leg | 0.98 s | 0.36 | 43% | 6.7 | 3.9 GB |
| Whisper tiny INT8 | conversational | **3** | 4 | 2 legs | 1.04 s | 0.37 | 47% | 3.7 | 1.2 GB |
| Qwen3-0.6B INT4 | dense | **1** | 2 | 1 leg | 0.90 s | 0.31 | 55% | 8.7 | 2.6 GB |
| Qwen3-0.6B INT8 | dense | **1** | 2 | 1 leg | 0.95 s | 0.36 | 62% | 9.8 | 3.9 GB |
| Whisper tiny INT8 | dense | **2** | 3 | 2 legs | 1.54 s | 0.34 | 68% | 5.4 | 1.2 GB |

All six saturation points were re-confirmed. Two details matter for sizing:

- **The box is not CPU-saturated at the SLO limit.** Even at saturation the 16 threads are only 38-68% busy (MEASURED). Each leg
  is limited by how fast one pass of one stream finishes, not by total CPU, so more threads in one process do not add legs in
  proportion. An earlier run pinned to 4, 8 and 16 threads kept Qwen at 1 leg in all three
  ([`../deployment.md`](../deployment.md), section 9). That is why the guide does not predict a bigger chip.
- **The knee is sharp.** Qwen INT8 conversational goes from 0.98 s stale P95 at 1 leg to 3.11 s at 2; Whisper conversational from
  1.04 s at 2 legs to 1.51 s at 3 and 2.70 s at 4. Planning below saturation is what keeps a box away from that cliff.

The full latency-vs-load ladders are in [`final_result.md`](final_result.md), section 3.

### 3.3 Memory (MEASURED and DERIVED)

| Model | Profile | Weights + buffers (base RSS, MEASURED) | Per leg (DERIVED regression slope, R²) | RAM need per box (DERIVED) | RAM to provision per box |
|---|---|---|---|---|---|
| Qwen3-0.6B INT4 | conversational | 2,549 MB | 303 MB (0.38) | 5.3 GB | **6 GB** |
| Qwen3-0.6B INT4 | dense | 2,561 MB | 133 MB (0.28) | 5.2 GB | **6 GB** |
| Qwen3-0.6B INT8 | conversational | 3,750 MB | 289 MB (0.57) | 6.7 GB | **7 GB** |
| Qwen3-0.6B INT8 | dense | 3,808 MB | 166 MB (0.46) | 6.7 GB | **7 GB** |
| Whisper tiny INT8 | conversational | 1,036 MB | 117 MB (0.86) | 3.5 GB | **4 GB** |
| Whisper tiny INT8 | dense | 1,051 MB | 109 MB (0.72) | 3.5 GB | **4 GB** |

RAM need = (weights + legs on the box x MB per leg) x 1.20 + 2 GB OS (ASSUMED factors), rounded up to a whole GB.
The per-leg slopes have low R² for Qwen, because only 1-2 leg levels exist to fit, so the RAM figure leans on the 1.20 headroom.

---

## 4. Assumptions

### 4.1 Sizing model

```
legs/box  = max(1, measured max legs × 0.70 headroom / 1.10 serving overhead)
boxes     = ceil(N / legs/box) + max(1, ceil(10% × ceil(N / legs/box)))       spare boxes
RAM/box   = ceil((weights + legs on box × MB/leg) × 1.20 + 2 GB)
threads   = boxes × 16        physical cores = boxes × 8
```

| Factor | Value | Tag | Why | Effect on the answer |
|---|---|---|---|---|
| Saturation point | 1 / 3 / 2 legs (Qwen / Whisper conv / Whisper dense) | MEASURED | Highest level at which every leg keeps up, confirmed | The dominant input. ±1 leg moves Whisper at 100 legs between 44 and 87 boxes |
| Headroom below the knee | 70% of saturation | ASSUMED | Waiting time grows sharply near saturation (section 3.2) | Whisper only; Qwen is already at the 1-leg floor |
| Serving overhead | x1.10 CPU | ASSUMED | WebSocket framing, JSON, TLS, resampling and supervision are not in the in-process test | Whisper only |
| Spare boxes (N+k) | max(1, 10% of boxes) | ASSUMED | A failed or draining box must not push the others over the knee | +10% on every row |
| One process per box | 1 x 16 threads | ASSUMED (measured layout) | Weights load once; an earlier run with two Qwen processes doubled RAM (3.7 → 7.7 GB) without adding a leg | RAM/box |
| RAM headroom and OS | x1.20, + 2 GB | ASSUMED | Allocator fragmentation, page cache, bursts, OS and agents | RAM/box only |
| Scale-out | Identical, independent boxes; a call stays on one box | EXTRAPOLATED | Boxes share nothing, so capacity adds up | Every row ≥ 50 legs |
| Cross-leg batching | None credited | NOT MODELLED | The engines run one stream per pass | Would lower box counts if implemented |

### 4.2 Traffic and service assumptions

| Assumption | Value used | Tag | If reality differs |
|---|---|---|---|
| Latency SLO | P95 staleness and end-of-call lag ≤ 2 s per leg | ASSUMED (product choice) | A looser SLO raises capacity; see section 6 |
| Speech density | Conversational (47% speech) for planning; dense (85%) as the upper bound | MEASURED (profile) + ASSUMED (mix) | Denser real traffic costs more per leg; check call recordings against these two profiles |
| Language and audio | Clean English read speech | ASSUMED | English produces the most decoded tokens per audio second for Qwen here (4.1 vs 2.7 ZH, 1.9 ID), so ZH/ID calls should cost slightly less. Noisy audio is not measured |
| Call length | 30 s | ASSUMED | Buffers are bounded (Qwen force-commits at 15 s, Whisper windows cap at 20 s), but hour-long memory growth is not measured |
| Concurrency means peak | N = simultaneous legs at the busy-hour peak | ASSUMED | Size for peak concurrent legs, not total calls per day. A two-party call transcribed on both sides is **2 legs** |
| Load balancing | Least-loaded routing with a hard per-box cap (section 5.4) | ASSUMED | Uneven routing needs more headroom |
| Hardware | Every box identical to the test machine | ASSUMED | Re-measure any other CPU (section 9) |

---

## 5. Sizing tables

All rows are **EXTRAPOLATED** from one box (section 2). "Boxes" includes spares; the range in parentheses is the answer if the
measured saturation were one leg higher (fewer boxes) or lower (more boxes).

### 5.1 One box

| | Qwen3-0.6B INT4 | Qwen3-0.6B INT8 | Whisper tiny INT8 |
|---|---|---|---|
| CPU | 16 threads / 8 cores (test machine class) | same | same |
| RAM to provision | 6 GB | 7 GB | 4 GB |
| Max legs kept up, conversational / dense (MEASURED) | 1 / 1 | 1 / 1 | 3 / 2 |
| Planning legs per box, conversational / dense (DERIVED) | 1.00 / 1.00 | 1.00 / 1.00 | 1.91 / 1.27 |
| Operating target: pass RTF P95 / staleness P95 (MEASURED at the operating point) | ≤ 0.35 / ≤ 0.9 s | ≤ 0.40 / ≤ 1.0 s | conv ≤ 0.40 / ≤ 1.1 s; dense ≤ 0.35 / ≤ 1.6 s |

For Qwen, 1 x 0.70 / 1.10 = 0.64 legs, which rounds up to the 1-leg floor. **Qwen boxes therefore run at their measured saturation
with no capacity headroom.** The latency margin comes from staleness: 0.83-0.98 s at 1 leg against the 2 s SLO.

### 5.2 Conversational profile: the planning case

| Concurrent legs | Qwen3 INT4 or INT8: boxes (base + spares) | Qwen3: CPU threads (physical cores) | Qwen3 fleet RAM, INT4 / INT8 | Whisper tiny: boxes (base + spares) | Whisper: CPU threads (physical cores) | Whisper fleet RAM | Basis |
|---|---|---|---|---|---|---|---|
| 50 | **55** (44-55), 50 + 5 | 880 (440) | 330 / 385 GB | **30** (22-44), 27 + 3 | 480 (240) | 120 GB | EXTRAPOLATED |
| 100 | **110** (87-110), 100 + 10 | 1,760 (880) | 660 / 770 GB | **59** (44-87), 53 + 6 | 944 (472) | 236 GB | EXTRAPOLATED |
| 200 | **220** (174-220), 200 + 20 | 3,520 (1,760) | 1,320 / 1,540 GB | **116** (87-174), 105 + 11 | 1,856 (928) | 464 GB | EXTRAPOLATED |
| 500 | **550** (433-550), 500 + 50 | 8,800 (4,400) | 3,300 / 3,850 GB | **289** (217-433), 262 + 27 | 4,624 (2,312) | 1,156 GB | EXTRAPOLATED |
| 1,000 | **1,100** (865-1,100), 1,000 + 100 | 17,600 (8,800) | 6,600 / 7,700 GB | **577** (433-865), 524 + 53 | 9,232 (4,616) | 2,308 GB | EXTRAPOLATED |

### 5.3 Dense profile: the upper bound

| Concurrent legs | Qwen3 INT4 or INT8: boxes (base + spares) | Qwen3: CPU threads (physical cores) | Qwen3 fleet RAM, INT4 / INT8 | Whisper tiny: boxes (base + spares) | Whisper: CPU threads (physical cores) | Whisper fleet RAM | Basis |
|---|---|---|---|---|---|---|---|
| 50 | **55** (44-55), 50 + 5 | 880 (440) | 330 / 385 GB | **44** (30-55), 40 + 4 | 704 (352) | 176 GB | EXTRAPOLATED |
| 100 | **110** (87-110), 100 + 10 | 1,760 (880) | 660 / 770 GB | **87** (59-110), 79 + 8 | 1,392 (696) | 348 GB | EXTRAPOLATED |
| 200 | **220** (174-220), 200 + 20 | 3,520 (1,760) | 1,320 / 1,540 GB | **174** (116-220), 158 + 16 | 2,784 (1,392) | 696 GB | EXTRAPOLATED |
| 500 | **550** (433-550), 500 + 50 | 8,800 (4,400) | 3,300 / 3,850 GB | **433** (289-550), 393 + 40 | 6,928 (3,464) | 1,732 GB | EXTRAPOLATED |
| 1,000 | **1,100** (865-1,100), 1,000 + 100 | 17,600 (8,800) | 6,600 / 7,700 GB | **865** (577-1,100), 786 + 79 | 13,840 (6,920) | 3,460 GB | EXTRAPOLATED |

### 5.4 From planning capacity to a router cap

Legs are whole numbers, so the router needs an integer cap per box. Use the **measured** maximum as the hard cap and size the
fleet so the **average** load per box stays at the planning capacity:

| Model | Hard cap per box (MEASURED max) | Planned average (DERIVED) | Note |
|---|---|---|---|
| Qwen3-0.6B INT4 / INT8 | 1 leg | 1.00 | Every confirmed 2-leg run failed the SLO (stale P95 2.2-3.1 s); one dense INT4 run passed at 1.72 s, then failed its confirm at 2.30 s |
| Whisper tiny, conversational | 3 legs | 1.91 | If the speech density of real traffic is unknown, cap at 2 (the dense maximum) |
| Whisper tiny, dense | 2 legs | 1.27 | |

Admission control must reject or queue the leg that would exceed the cap. An extra leg is not rejected by the engine; it makes every
call on that box late ([`../deployment.md`](../deployment.md)).

### 5.5 Worked example: 100 conversational legs

- **Whisper tiny.** 3 x 0.70 / 1.10 = 1.91 legs per box. 100 / 1.91 = 52.4, so 53 boxes; spares max(1, ceil(5.3)) = 6; **59 boxes**,
  944 threads, 4 GB each. If the true saturation is 4 legs: 2.55 per box, 40 + 4 = **44**. If it is 2: 1.27 per box, 79 + 8 = **87**.
- **Qwen3.** 1 x 0.70 / 1.10 = 0.64, floored at 1 leg per box; 100 + 10 spares = **110 boxes**, 1,760 threads, 6 GB (INT4) or 7 GB (INT8) each.
  If the true saturation is 2 legs: 1.27 per box, 79 + 8 = **87**. See section 6 for the downside.

---

## 6. Sensitivity

**Whisper tiny, boxes including spares** (from the sizing model; every row EXTRAPOLATED):

| Variant | Conversational @ 100 | Conversational @ 1,000 | Dense @ 100 | Dense @ 1,000 |
|---|---|---|---|---|
| Central (70% headroom, x1.10, 10% spares) | 59 | 577 | 87 | 865 |
| Headroom 60% | 69 | 674 | 102 | 1,009 |
| Headroom 80% | 51 | 505 | 76 | 757 |
| Serving overhead x1.00 | 53 | 525 | 80 | 787 |
| Serving overhead x1.25 | 66 | 656 | 99 | 983 |
| No spare boxes | 53 | 524 | 79 | 786 |
| Measured saturation 1 leg higher | 44 | 433 | 59 | 577 |
| Measured saturation 1 leg lower | 87 | 865 | 110 | 1,100 |

**Qwen3 (INT4 or INT8, either profile).** Headroom (60-80%) and serving overhead (x1.00-x1.25) do not change the answer, because
the box is already at the 1-leg floor; only spares do (100 / 1,000 boxes without them). One leg higher would give 87 / 865 boxes.

The real downside for Qwen is not "more boxes". **If the true capacity is one leg lower, Qwen cannot meet the 2 s SLO on this CPU
at all.** The sizing tool floors that case at 1 leg per box, so the table shows 110, which hides the problem. This is not
hypothetical: an earlier whole-machine run on October 6 measured Qwen INT8 dense at **0** legs ([`../arch/changes.md`](../arch/changes.md), section 11),
and both final runs put a single leg at 0.83-0.98 s stale P95, so one leg fits with margin but run-to-run noise is about one leg.

**SLO.** At a 3 s threshold instead of 2 s, only the dense Qwen 2-leg runs would pass (worst leg 2.25-2.39 s, end lag 2.07-2.41 s),
which would cut dense Qwen to 87 boxes at 100 legs. Conversational Qwen and Whisper would not change, so the planning case stays the same
([`final_result.md`](final_result.md), section 6).

---

## 7. Not covered by this guide

| Gap | Why it matters | Status |
|---|---|---|
| Fleet-level effects (load-balancer skew, correlated peaks, failure domains) | Can push individual boxes over the cap | Covered only by headroom and spares (ASSUMED) |
| Network path (WebSocket, TLS, JSON) | Adds CPU per leg | ASSUMED x1.10, not measured |
| Other CPUs (server parts with AVX-512 / VNNI / AMX, more memory bandwidth) | Could carry more legs per box | Not predicted; re-measure |
| Several smaller processes per box (e.g. 4 x 4 threads) | Could raise legs per box, since the box is only 38-68% busy at saturation | Not measured at whole-box scale; duplicates weights |
| Cross-leg batching, lower hop rate, finals-only mode | Would cut cost per leg | Not implemented, so not credited |
| Noisy, accented, overlapping or non-English audio | Changes decode cost and VAD behaviour | Not measured |
| Hour-long calls | Memory growth and drift | Not measured (30 s calls) |
| Qwen3-1.7B, other Whisper sizes | Larger models | Not load-tested. From the single-leg benchmark, 1.7B INT4 is expected to carry at most 1 leg |

---

## 8. Recommendation

- **Plan with the conversational table (section 5.2) and budget for the dense one (section 5.3)** until real call recordings show
  where the traffic sits.
- **Qwen3-0.6B INT4** if Mandarin / Indonesian accuracy matters: about **1.1 boxes per peak concurrent leg** on this CPU class
  (110 boxes, 1,760 threads, 6 GB each at 100 legs).
- **Whisper tiny INT8** only where its accuracy is acceptable: about **0.6 boxes per leg** conversational, **0.9** dense.
- Treat every number at 50+ legs as an **order of magnitude**, not a purchase order: it is extrapolated 17x-1,000x from a 1-3 leg
  measurement on one laptop-class CPU.
- **Before buying hardware, re-run the load test on the candidate edge box** (section 9). A box that sustains more legs per pass
  is the only thing that brings these counts down without changing the software.

---

## 9. Reproduce or re-measure

```bash
# Measure saturation on the machine you want to deploy (same YAML, nothing else changes)
python3 loadtest/run_loadtest.py                                   # every model x profile in loadtest_config.yaml
python3 loadtest/run_loadtest.py --profiles conversational --models qwen3_onnx_int4_0.6b

# Recompute the tables from a raw file with different assumptions
python3 loadtest/run_sizing.py --input loadtest/results/20261007T202154Z_loadtest_raw.json --legs 50,100,200,500,1000
python3 loadtest/run_sizing.py --headroom 0.6 --serving-overhead 1.25 --spare-fraction 0.2 --legs 50,100,200,500,1000
```

The SLO threshold is `slo.lag_threshold_s` in `loadtest/configs/loadtest_config.yaml`. The previous version of this guide
(run `20261007T170448Z`, without INT4) is in git history.
