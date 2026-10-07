# ASR CPU Load Test Report

**Generated:** 2026-10-07T20:35:31Z

> Everything in this file is **measured** on the hardware below. Sizing for larger deployments is derived from it in the separate sizing guide.

## Test machine

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 7 6800H with Radeon Graphics |
| Physical / logical cores | 8 / 16 |
| RAM | 14.9 GB |
| OS | Linux 6.8.0-138-generic (x86_64) |
| onnxruntime | 1.30.0 |

## Load profile

| Parameter | Value |
| --- | --- |
| call_duration_s | 30 |
| profiles | {'dense': {'gap_s': 1.0}, 'conversational': {'gap_s': 7.0}} |
| start_spread_s | 6 |
| chunk_s | 0.5 |
| audio | librispeech_0_1089_0.wav, librispeech_1_1089_1.wav, librispeech_2_1089_2.wav |
| lag_threshold_s | 2.0 |
| abort_lag_s | 8.0 |
| ramp_levels | [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] |
| ramp_refine | True |
| ramp_refine_steps | 4 |
| ramp_confirm | True |
| reserve_mb | 1500 |

## Saturation summary

> **One leg = one independently streamed audio source** (a simulated call at real-time pace, same stream logic as the live server). **Max legs kept up** is the saturation point: the highest tested leg count at which every leg stayed within the lag threshold. **First failing** is the lowest leg count seen to fail.

| Model | Profile | CPU threads | Procs x threads | Base RSS (all procs) | Load | Max legs kept up | First failing | Confirmed | Stopped because | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_onnx_int8_0.6b | dense | 16 | 1x16 | 3808 MB | 4.0s | 1 | 2 | yes | saturated |  |
| qwen3_onnx_int4_0.6b | dense | 16 | 1x16 | 2561 MB | 3.4s | 1 | 2 | yes | saturated |  |
| whisper_int8_tiny | dense | 16 | 1x16 | 1051 MB | 2.3s | 2 | 3 | yes | saturated |  |
| qwen3_onnx_int8_0.6b | conversational | 16 | 1x16 | 3750 MB | 3.4s | 1 | 2 | yes | saturated |  |
| qwen3_onnx_int4_0.6b | conversational | 16 | 1x16 | 2549 MB | 2.7s | 1 | 2 | yes | saturated |  |
| whisper_int8_tiny | conversational | 16 | 1x16 | 1036 MB | 2.3s | 3 | 4 | yes | saturated |  |

### qwen3_onnx_int8_0.6b [dense] - 16 CPU threads, 1 process(es) x 16 threads (vad_utterance)

| Legs | Healthy | Kept up | Pass P50 | Pass P95 | Pass RTF P95 | Stale P50 | Stale P95 | Stale Max | End Lag Max | CPU (pinned set) | Cores used | CPU-s per audio-s | Passes/s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 1/1 | 0.708s | 0.941s | 0.362 | 0.71s | 0.95s | 1.47s | 1.47s | 62% | 9.8 | 10.59 | 0.8 | 3869 MB |
| 2 | NO: kept_up=1/2 | 1/2 | 0.953s | 1.741s | 0.728 | 1.04s | 2.22s | 2.41s | 2.41s | 78% | 12.4 | 7.42 | 1.1 | 4073 MB |
| 1 | yes | 1/1 | 0.713s | 0.944s | 0.349 | 0.72s | 0.96s | 1.39s | 1.39s | 62% | 9.8 | 10.54 | 0.8 | 4078 MB |

### qwen3_onnx_int4_0.6b [dense] - 16 CPU threads, 1 process(es) x 16 threads (vad_utterance)

| Legs | Healthy | Kept up | Pass P50 | Pass P95 | Pass RTF P95 | Stale P50 | Stale P95 | Stale Max | End Lag Max | CPU (pinned set) | Cores used | CPU-s per audio-s | Passes/s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 1/1 | 0.618s | 0.832s | 0.309 | 0.62s | 0.90s | 1.17s | 1.17s | 55% | 8.7 | 9.32 | 0.8 | 2589 MB |
| 2 | yes | 2/2 | 0.821s | 1.465s | 0.598 | 0.85s | 1.72s | 2.13s | 1.93s | 69% | 10.9 | 6.51 | 1.1 | 2828 MB |
| 3 | NO: overload_abort=2/3, kept_up=0/3 | 0/3 | 1.655s | 3.386s | 1.228 | 1.87s | 8.98s | 10.30s | 1.56s | 67% | 10.6 | 4.07 | 0.8 | 2991 MB |
| 2 | NO: kept_up=0/2 | 0/2 | 0.928s | 1.933s | 0.919 | 1.06s | 2.30s | 2.77s | 2.07s | 76% | 12.1 | 7.15 | 1.1 | 2996 MB |
| 1 | yes | 1/1 | 0.623s | 0.840s | 0.311 | 0.63s | 0.87s | 1.14s | 1.14s | 55% | 8.8 | 9.38 | 0.8 | 2996 MB |

### whisper_int8_tiny [dense] - 16 CPU threads, 1 process(es) x 16 threads (sliding_window)

| Legs | Healthy | Kept up | Pass P50 | Pass P95 | Pass RTF P95 | Stale P50 | Stale P95 | Stale Max | End Lag Max | CPU (pinned set) | Cores used | CPU-s per audio-s | Passes/s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 1/1 | 0.397s | 0.900s | 0.233 | 0.40s | 0.91s | 1.51s | 1.51s | 52% | 8.2 | 8.89 | 0.9 | 1078 MB |
| 2 | yes | 2/2 | 0.473s | 1.266s | 0.340 | 0.49s | 1.54s | 1.93s | 1.93s | 68% | 10.7 | 6.07 | 1.6 | 1191 MB |
| 3 | NO: kept_up=2/3 | 2/3 | 0.939s | 2.489s | 0.588 | 1.03s | 3.24s | 5.82s | 5.82s | 89% | 14.1 | 5.75 | 1.9 | 1319 MB |
| 2 | yes | 2/2 | 0.523s | 1.367s | 0.416 | 0.52s | 1.63s | 2.27s | 1.43s | 74% | 11.7 | 6.66 | 1.5 | 1321 MB |

### qwen3_onnx_int8_0.6b [conversational] - 16 CPU threads, 1 process(es) x 16 threads (vad_utterance)

| Legs | Healthy | Kept up | Pass P50 | Pass P95 | Pass RTF P95 | Stale P50 | Stale P95 | Stale Max | End Lag Max | CPU (pinned set) | Cores used | CPU-s per audio-s | Passes/s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 1/1 | 0.681s | 0.981s | 0.360 | 0.68s | 0.98s | 1.12s | 1.12s | 43% | 6.7 | 7.22 | 0.6 | 3895 MB |
| 2 | NO: kept_up=0/2 | 0/2 | 0.844s | 2.451s | 0.800 | 0.84s | 3.11s | 4.11s | 1.36s | 55% | 8.6 | 5.04 | 0.7 | 4184 MB |
| 1 | yes | 1/1 | 0.683s | 0.970s | 0.346 | 0.68s | 0.97s | 1.11s | 1.11s | 43% | 6.8 | 7.24 | 0.6 | 4184 MB |

### qwen3_onnx_int4_0.6b [conversational] - 16 CPU threads, 1 process(es) x 16 threads (vad_utterance)

| Legs | Healthy | Kept up | Pass P50 | Pass P95 | Pass RTF P95 | Stale P50 | Stale P95 | Stale Max | End Lag Max | CPU (pinned set) | Cores used | CPU-s per audio-s | Passes/s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 1/1 | 0.592s | 0.829s | 0.306 | 0.59s | 0.83s | 1.01s | 1.01s | 38% | 5.9 | 6.28 | 0.6 | 2628 MB |
| 2 | NO: kept_up=0/2 | 0/2 | 0.716s | 1.980s | 0.823 | 0.72s | 2.76s | 5.01s | 1.78s | 52% | 8.2 | 4.84 | 0.7 | 3077 MB |
| 1 | yes | 1/1 | 0.588s | 0.895s | 0.313 | 0.59s | 0.90s | 0.97s | 0.97s | 38% | 5.9 | 6.31 | 0.6 | 3077 MB |

### whisper_int8_tiny [conversational] - 16 CPU threads, 1 process(es) x 16 threads (sliding_window)

| Legs | Healthy | Kept up | Pass P50 | Pass P95 | Pass RTF P95 | Stale P50 | Stale P95 | Stale Max | End Lag Max | CPU (pinned set) | Cores used | CPU-s per audio-s | Passes/s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 1/1 | 0.422s | 0.907s | 0.188 | 0.42s | 0.91s | 0.93s | 0.00s | 43% | 6.6 | 6.87 | 0.7 | 1068 MB |
| 2 | yes | 2/2 | 0.491s | 0.983s | 0.373 | 0.49s | 1.04s | 1.15s | 0.00s | 47% | 7.4 | 4.20 | 0.9 | 1178 MB |
| 3 | yes | 3/3 | 0.602s | 1.373s | 0.428 | 0.61s | 1.51s | 2.77s | 0.61s | 57% | 8.9 | 3.54 | 1.3 | 1303 MB |
| 4 | NO: kept_up=1/4 | 1/4 | 0.967s | 2.541s | 0.576 | 1.09s | 2.70s | 3.71s | 2.62s | 87% | 13.7 | 4.12 | 1.7 | 1344 MB |
| 3 | yes | 3/3 | 0.510s | 1.472s | 0.406 | 0.51s | 1.59s | 1.84s | 0.30s | 61% | 9.6 | 3.78 | 1.4 | 1429 MB |

> **Cores used** = CPU time consumed by the workers / wall time (includes ORT thread spin-wait, so it overstates useful work). **CPU-s per audio-s** = worker CPU seconds per second of audio streamed: the per-leg compute cost at that load.
