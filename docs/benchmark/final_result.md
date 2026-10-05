# RT-MASR-CPU — Final Benchmark Result

Consolidated from the run `20261005T173127Z`
([raw JSON](20261005T173127Z_raw.json) · [full summary](20261005T173127Z_summary.md)).
All numbers below come from that run unless stated otherwise; derived figures (per-language RTF,
speed-ups, the corrected INT4 number) were recomputed from the raw JSON.

## 1. Setup

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 6800H, 8 physical / 16 logical cores |
| RAM | 14.9 GB |
| OS / Python | Linux 6.8.0-138 (x86_64) / 3.12.13 |
| onnxruntime / torch / transformers | 1.30.0 / 2.14.1 / 4.57.6 |
| Latency | 7 files (3 EN, 2 ZH, 2 ID), 3 measured runs + 1 warm-up each |
| Concurrency | 1 / 2 / 4 simultaneous legs, 2 rounds each, same file (`librispeech_0_1089_0.wav`) |
| Models run | 10 configs, 0 load failures, 0 concurrency errors |
| **Not run** | `qwen3_onnx_fp32_1.7b` (disabled in `bench_config.yaml`; ~10 GB of weights on a 14.9 GB machine) |

RTF = processing time / audio time (lower is better, < 1.0 is faster than real time).
"Overall RTF" is duration-weighted over all 21 measured calls (252.3 s of audio per config).

## 2. Headline comparison

| Config | Precision | Disk | Overall RTF | Peak RSS | EN WER | ZH CER\* | ID WER\* |
|---|---|---|---|---|---|---|---|
| `qwen3_onnx_int4_0.6b` | INT4 | 1.9 GB | 0.130 ⚠ (0.141 corrected) | 3.9 GB | 0.537 ⚠ (0.038 corrected) | 0.000 | 0.125 |
| `qwen3_onnx_int8_0.6b` | INT8 | 2.5 GB | 0.151 | 5.1 GB | 0.037 | 0.013 | 0.188 |
| `qwen3_onnx_int4_1.7b` | INT4 | 3.9 GB | 0.245 | 6.3 GB | **0.000** | 0.000 | 0.063 |
| `qwen3_onnx_fp32_0.6b` | FP32 | 3.8 GB | 0.278 | 7.8 GB | 0.037 | 0.000 | 0.188 |
| `qwen3_transformers_bf16_0.6b` | BF16 | 1.8 GB | 0.571 | 3.7 GB | 0.037 | 0.000 | 0.188 |
| `qwen3_transformers_bf16_1.7b` | BF16 | 4.4 GB | 1.070 | 6.3 GB | 0.000 | 0.000 | 0.000 |
| `whisper_int8_tiny` | INT8 | 5.5 GB (shared) | 0.245 | 2.7 GB | 0.130 | 0.470 | 0.000 |
| `whisper_int8_base` | INT8 | 〃 | 0.548 | 3.2 GB | 0.093 | 0.282 | 0.063 |
| `whisper_int8_small` | INT8 | 〃 | 1.716 | 5.0 GB | 0.019 | 0.060 | 0.063 |
| `whisper_int8_medium` | INT8 | 〃 | 6.387 | 10.4 GB | 0.019 | 0.040 | 0.063 |

- Accuracy is the corpus rate (total edits / total reference length). EN and ID are WER, ZH is CER.
- **\*** ZH and ID references are *draft, not human-verified* transcripts. Those scores measure agreement
  with the draft, not true accuracy. The EN set is only 3 files (~54 words), so a single word moves WER by ~2 points.
- Peak RSS is the maximum seen during the latency runs. It includes memory the process kept from earlier audio, so treat it as an upper bound.
- Disk is the size of the model folder under `models/`. The Whisper INT8 folder holds all four sizes.
- ⚠ See section 4: the INT4 0.6B row is distorted by one failed clip.

## 3. Findings

**Speed**
- Every Qwen3 ONNX config runs faster than real time (RTF 0.13–0.28). Of the Transformers and Whisper configs, only
  `qwen3_transformers_bf16_0.6b`, `whisper_int8_tiny` and `whisper_int8_base` do; `qwen3_transformers_bf16_1.7b`,
  `whisper_int8_small` and `whisper_int8_medium` are slower than real time.
- ONNX vs Transformers on the same model size: 0.6B INT8 is **3.8x** faster than BF16 (0.151 vs 0.571). 1.7B INT4 is **4.4x** faster than BF16 (0.245 vs 1.070).
- Per-language RTF (duration-weighted) for the ONNX models:

  | Config | EN | ZH | ID |
  |---|---|---|---|
  | INT8 0.6B | 0.192 | 0.156 | 0.106 |
  | FP32 0.6B | 0.346 | 0.291 | 0.190 |
  | INT4 0.6B | 0.105 ⚠ | 0.155 | 0.107 |
  | INT4 1.7B | 0.287 | 0.253 | 0.192 |

**Precision (0.6B)**
- FP32 gives **no accuracy gain** over INT8 on this data (EN WER and ID WER are identical). The only difference is one ZH character (INT8 CER 0.013 vs 0.000).
  FP32 is 1.8x slower (RTF 0.278 vs 0.151), uses ~2.5 GB more RAM (7.8 vs 5.1 GB peak) and takes longer to load (8.2 s vs 5.2 s).
- INT4 is the smallest (1.9 GB disk, 3.9 GB RSS, 2.8 s load) and, apart from the failure in section 4, as fast as INT8 (corrected RTF 0.141 vs 0.151).

**Model size (INT4)**
- 1.7B INT4 costs about 1.6–1.9x the RTF of 0.6B INT8/INT4 (0.245 vs 0.151 / ~0.13–0.14) and ~2.4 GB more RAM, in exchange for the best ONNX accuracy:
  0.000 EN WER, 0.000 ZH CER and 0.063 ID WER, the same as BF16 1.7B except one ID file.
- 1.7B INT4 matches the BF16 1.7B output (the Transformers 1.7B reference) on EN and ZH while being 4.4x faster.

**Qwen3 vs Whisper (this CPU)**
- Whisper tiny has about the same RTF as Qwen3 1.7B INT4 (0.245) but much worse accuracy (EN WER 0.130, ZH CER 0.470).
- Qwen3 0.6B INT8 is 1.6x faster than Whisper tiny and far more accurate on ZH.
- Only Whisper small and medium reach Qwen-level accuracy on EN (WER 0.019), at RTF 1.7 and 6.4. They cannot keep up with live audio.
- Whisper saturates the CPU (~95% mean) while Qwen ONNX uses ~55%, and Whisper medium reaches 10.4 GB RSS.

**Concurrency (shared engine, one process)**

| Config | Throughput 1 / 2 / 4 legs (x real time) | RTF per call at 4 legs | Keeps RTF < 1 at 4 legs? |
|---|---|---|---|
| INT8 0.6B | 6.1 / 6.5 / 6.7 | 0.59 | yes |
| FP32 0.6B | 3.0 / 3.1 / 3.3 | 1.17 | no (ok at 2: 0.63) |
| INT4 0.6B | ⚠ not valid (see section 4) | – | – |
| INT4 1.7B | 3.2 / 3.7 / 3.5 | 1.10 | no (ok at 2: 0.54) |
| BF16 0.6B | 1.6 / 1.7 / 1.7 | 2.40 | no (ok at 1 only) |
| Whisper tiny | 3.3 / 3.7 / 3.9 | 1.01 | borderline |

- Throughput barely grows with more legs (+10% from 1 to 4 for INT8 0.6B) because a single ONNX Runtime session already uses all cores. More legs mostly add latency: per-call RTF roughly quadruples from 1 to 4 legs.
- Practical capacity on this 8-core machine is about **4 live calls for INT8 0.6B**, about **2 for 1.7B INT4 or FP32 0.6B**, and 1 for BF16 1.7B and Whisper small/medium.

## 4. Issue found: INT4 0.6B returns empty text on one clip

`qwen3_onnx_int4_0.6b` produced **0 tokens** (the model's first predicted token was EOS) on
`librispeech_0_1089_0.wav` in all 3 measured runs, with language auto-detect. I re-ran the engine
separately and got the same empty output. With `language="English"` forced, the same clip is transcribed correctly (37 tokens). The other 6 files decode normally.

Impact on the numbers above:

| Metric | As reported | Without that clip |
|---|---|---|
| EN WER (corpus) | 0.537 | 0.038 (1 error in 26 words) |
| Overall RTF | 0.130 | 0.141 |
| EN RTF | 0.105 | – |
| Concurrency throughput (17.96 / 19.69 / 21.73x) | **invalid**: the concurrency workload is this same clip, so no decoding happened | – |

For the INT4 0.6B concurrency row, expect a figure close to INT8 0.6B (roughly 6–7x), but this run did not measure it.

Language auto-detection is also less stable on INT4: `ind_001.wav` was detected as Malay by 0.6B INT4 and `ind_002.wav` as Malay by 1.7B INT4. Whisper base also reported `ms` for `ind_001.wav`. The text was still correct.

Suggested follow-ups (not done here):
1. Re-run the benchmark with a fixed `language` (or a retry when 0 tokens are generated) for the INT4 0.6B config. Then the concurrency and EN rows become usable.
2. Add a benchmark guard that flags configs where `tokens_generated == 0`, so empty outputs cannot silently inflate throughput.

## 5. Caveats

- **Load times and RSS deltas are only reliable for the first configs.** Configs run in one process, and later rows start with 1.5 GB of leftover memory ("RSS before"). `qwen3_transformers_bf16_1.7b` shows a 0.94 s load and a 208 MB delta, which is clearly not a cold load. The numbers for the Whisper configs are affected too. Use the ONNX rows, which ran first, for load-time comparison: INT4 0.6B 2.8 s, INT8 0.6B 5.2 s, INT4 1.7B 5.9 s, FP32 0.6B 8.2 s.
- Latency is very stable (P95 within ~1–3% of P50 for most files), so single-run differences between configs are real. The exception is rows with only 3 runs on short files.
- The test set is small (7 files, ~4 minutes of audio). Treat accuracy gaps below ~2 points as noise.
- Results are specific to this 8-core laptop CPU with 14.9 GB RAM.

## 6. Recommendation

| Goal | Pick |
|---|---|
| Best speed/accuracy balance for live calls | `qwen3_onnx_int8_0.6b`: RTF 0.15, supports ~4 concurrent calls, 5.1 GB RAM |
| Highest accuracy that still runs in real time | `qwen3_onnx_int4_1.7b`: RTF 0.245, near-perfect EN/ZH, about 2 concurrent calls, 6.3 GB RAM |
| Smallest footprint | `qwen3_onnx_int4_0.6b`, but set the language explicitly (or retry on empty output) until the issue in section 4 is resolved |
| Avoid | `qwen3_onnx_fp32_0.6b` (no accuracy gain over INT8, 1.8x slower, ~2x RAM); Whisper small/medium and `qwen3_transformers_bf16_1.7b` for live use (RTF > 1) |
