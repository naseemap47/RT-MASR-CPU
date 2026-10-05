# Benchmark Pipeline

The benchmark measures each ASR configuration **offline, in-process** (no
WebSocket, no browser) so that model cost is isolated from the streaming layer.
Entry point: `benchmark/run_benchmark.py`.

---

## 1. Flow

```mermaid
flowchart TD
    CLI["run_benchmark.py<br/>CLI overrides"] --> CFG["benchmark/configs/bench_config.yaml"]
    CFG --> LOOP{"for each config id"}
    LOOP --> L1["1 · Load timer<br/>cold-start time + RSS delta"]
    L1 --> L2["2 · Latency runner<br/>warm-up + N runs per file → P50/P95/P99, RTF, CPU/RSS"]
    L2 --> L3["3 · Accuracy runner<br/>WER (en, id) / CER (zh) vs references.yaml"]
    L3 --> L4["4 · Concurrency runner<br/>1/2/4 legs, shared engine, ThreadPoolExecutor"]
    L4 --> FREE["release_memory()<br/>before next config"]
    FREE --> LOOP
    LOOP -->|done| REP["Reporters"]
    REP --> J["benchmark/results/{UTC}_raw.json"]
    REP --> M["benchmark/results/{UTC}_summary.md"]
```

The engine loaded in stage 1 is reused for stages 2–4. If one stage fails, that
config is recorded under `failures` and the run continues with the next config;
`Ctrl+C` writes partial results.

---

## 2. Configuration — `benchmark/configs/bench_config.yaml`

| Key | Meaning | Default |
|---|---|---|
| `configs[]` | `id`, `display_name`, `backend` (`onnx` / `transformers` / `whisper`), `model_config` (per-model YAML), optional `enabled` (default `true`) | 7 Qwen3 + 4 Whisper INT8 (`qwen3_onnx_fp32_1.7b` disabled) |
| `runs` | Measured runs per audio file | 3 |
| `warmup_runs` | Discarded runs per audio file | 1 |
| `concurrency_legs` | Simultaneous legs to test | `[1, 2, 4]` |
| `concurrency_rounds` | Calls per worker (total calls = legs × rounds) | 2 |
| `concurrency_audio` | Fixed workload; call *i* uses file *i* mod len | `librispeech_0_1089_0.wav` |
| `audio` | Latency/accuracy files grouped by `en` / `zh` / `id` | 3 + 2 + 2 files |
| `references_file` | Reference transcripts | `benchmark/data/references.yaml` |
| `output_dir` | Where reports are written | `benchmark/results` |

Available config ids: `qwen3_onnx_int8_0.6b`, `qwen3_onnx_fp32_0.6b`, `qwen3_onnx_int4_0.6b`,
`qwen3_onnx_fp32_1.7b`, `qwen3_onnx_int4_1.7b`, `qwen3_transformers_bf16_0.6b`,
`qwen3_transformers_bf16_1.7b`, `whisper_int8_tiny`, `whisper_int8_base`,
`whisper_int8_small`, `whisper_int8_medium`.

Missing audio files are skipped with a warning. All relative paths are resolved
from the project root, so the script can be launched from anywhere.

### Disabling / enabling a config

Add `enabled: false` to an entry to leave it out of the default run (useful for
models your machine cannot handle). Omitting the key means enabled. Example —
`qwen3_onnx_fp32_1.7b` (~10 GB of weights) is disabled in the shipped config:

```yaml
  - id: "qwen3_onnx_fp32_1.7b"
    enabled: false              # ← disabled: skipped by `run_benchmark.py` with no --models
    display_name: "Qwen3-ASR-1.7B FP32 (ONNX / CPU)"
    backend: "onnx"
    model_config: "config/models/qwen3_onnx_1.7b_fp32.yaml"
```

- **Disable:** set `enabled: false` (or delete/comment out the entry).
- **Enable permanently:** set `enabled: true` or remove the `enabled` line.
- **Enable for one run only:** name it explicitly; `--models` ignores `enabled`:
  `uv run python benchmark/run_benchmark.py --models qwen3_onnx_fp32_1.7b --runs 1`

A default run prints `Skipping disabled config(s): ...` so it is clear what was left out.

---

## 3. CLI

```bash
# Everything in bench_config.yaml
uv run python benchmark/run_benchmark.py

# Quick smoke run: ONNX only, 1 measured run, legs 1 and 2, no accuracy
uv run python benchmark/run_benchmark.py --models qwen3_onnx_int8_0.6b --runs 1 --legs 1,2 --skip-accuracy

# Qwen3 ONNX INT8 vs Transformers 0.6B
uv run python benchmark/run_benchmark.py --models qwen3_onnx_int8_0.6b,qwen3_transformers_bf16_0.6b --runs 3

# Qwen3 ONNX vs Whisper small, latency + accuracy only
uv run python benchmark/run_benchmark.py --models qwen3_onnx_int8_0.6b,whisper_int8_small --skip-concurrency
```

| Flag | Effect |
|---|---|
| `--config PATH` | Alternative bench config |
| `--models a,b` | Subset of config ids |
| `--runs N` | Override `runs` |
| `--legs 1,2,4,8` | Override `concurrency_legs` |
| `--skip-accuracy` | Skip stage 3 |
| `--skip-concurrency` | Skip stage 4 |
| `--output-dir DIR` | Override `output_dir` |

Exit code is `2` if any config failed, `1` for invalid arguments/inputs.

---

## 4. What each stage measures

### 4.1 Load (`runners/load_timer.py`)
Wall-clock time to construct the engine and the RSS before/after. "Cold" means a
fresh process-level load (new ORT sessions / PyTorch modules); model files are
usually in the OS page cache after the first run, so disk read time is mostly
excluded on repeat runs. The 1.7B row can show ~0 MB delta when allocator pages
freed by the previous config are reused.

### 4.2 Latency and RTF (`runners/latency_runner.py`)
For each audio file: `warmup_runs` discarded calls, then `runs` measured calls to
`engine.transcribe(path)`. Reports mean / P50 / P95 / P99 / min / max of latency
and RTF (latency ÷ audio duration). `SystemSampler` (`metrics/system_metrics.py`)
samples machine-wide CPU %, process CPU %, RSS and thread count during the measured
block (the summary table shows the machine-wide CPU %). Every metric is explained
in plain words in [section 5](#5-metrics-explained-in-plain-words).

### 4.3 Accuracy (`runners/accuracy_runner.py`, `metrics/asr_metrics.py`)
- **WER** for English and Indonesian, **CER** for Mandarin. Wagner-Fischer edit
  distance, all edit costs 1.
- Normalisation: NFKC + lowercase + strip punctuation for EN/ID; NFKC + strip
  CJK punctuation for ZH.
- References with `verified: false` in `references.yaml` are drafts taken from
  agreeing Qwen3 outputs. Scores against them measure agreement with Qwen, not
  true accuracy. The Mandarin OSR files contain several sentences each, while
  the current references hold one sentence, which is why CER is > 1 for those
  rows. Fix the references before quoting ZH/ID accuracy.

### 4.4 Concurrency (`runners/concurrency_runner.py`)
One shared, warmed-up engine; `legs` worker threads each perform `rounds`
`transcribe()` calls on the fixed workload. Reports per-call latency
percentiles, RTF, aggregate throughput in audio-hours per wall-hour (> 1 means
faster than real time in aggregate), error count and peak RSS/CPU.

This is an offline capacity test: calls are submitted as fast as workers are
free, not paced at real time, and they do not go through the WebSocket path.

---

## 5. Metrics explained in plain words

You do not need an ML background to read the reports. Each metric below says
**what it is**, **how to read it**, and **where it shows up** in
`<UTC>_summary.md`.

### 5.1 The one-minute version

| Question you have | Look at | Good looks like |
|---|---|---|
| Is it fast enough for live speech? | **RTF** | below 1.0 (lower is better) |
| How long does one request take? | **Latency** (P50 / P95) | small, and P95 close to P50 |
| Is it correct? | **WER** (English, Indonesian) / **CER** (Chinese) | close to 0 % |
| How long until it is ready after start-up? | **Load time** | a few seconds |
| How much memory does it need? | **RSS / Peak RSS** | fits in your RAM with room to spare |
| How many calls can one machine handle? | **Throughput** and the **Concurrency** table | throughput above the number of calls you need |

### 5.2 Speed metrics

**Latency**
The wall-clock time one `transcribe()` call takes from start to finish, in
seconds. Think of it as "how long did I wait for the answer". It includes
reading the audio, the model working, and producing the text.

**RTF — Real-Time Factor**
`RTF = latency ÷ audio length`

It compares how long the computer needed with how long the speech lasted.

| RTF | Meaning (for a 10 s recording) |
|---|---|
| 0.2 | finished in 2 s — 5× faster than live speech |
| 1.0 | finished in 10 s — exactly keeps up |
| 2.0 | finished in 20 s — falls behind live speech |

For live use you want **RTF below 1.0**. The lower, the more spare capacity.

**Mean, P50, P95, P99, min, max**
The same call is measured several times (`runs`), so the report summarises the
list of timings:

| Name | Plain meaning |
|---|---|
| Mean | the average |
| **P50** (median) | half of the calls were faster than this, half slower — the "typical" call |
| **P95** | 95 out of 100 calls were faster than this — the "bad day" number |
| **P99** | 99 out of 100 were faster — the "worst realistic case" |
| Min / Max | the fastest / slowest single call |

*Why not just the average?* A few slow calls hide inside an average. If P50 is
1.0 s but P95 is 4.0 s, one call in twenty feels four times slower, and a live
system has to plan for that.

**Overall RTF** (the table under *Latency*)
Total processing time of all measured calls ÷ total audio time of all calls. Long
files count more than short ones, so this is the fairest single speed number for a
configuration.

**Warm-up runs**
The first call after loading is often slower (caches are cold, memory is being
allocated). A few calls are made first and **thrown away**, so the numbers
describe normal running speed, not the first-call hiccup.

### 5.3 Start-up metrics

**Load time**
How long it takes to create the engine and load the model into memory — the
"cold start". It matters when the service restarts or scales up. Model files are
usually already cached by the operating system after the first run, so repeat
runs mostly measure model set-up, not disk reading.

**RSS (Resident Set Size)**
The amount of real RAM the process occupies, in MB. The load table shows:

| Column | Meaning |
|---|---|
| RSS Before | memory in use just before loading |
| RSS After | memory in use right after loading |
| RSS Delta | the difference — roughly **what the model costs in RAM** |

### 5.4 Resource metrics (CPU and memory while running)

A small background sampler checks the process ten times a second during the
measured runs.

| Metric | Plain meaning |
|---|---|
| **CPU Mean / CPU P95** | how busy the **whole machine** was (0–100 %, across all cores). 100 % means every core was fully used |
| **RSS Mean** | average RAM used by the process |
| **Peak RSS** | the highest RAM use seen — the number to size your server RAM against |
| **Peak Threads** | the most threads the process used at once. Models spread work across cores using threads |

*Reading tip:* CPU near 100 % with RTF below 1.0 means the model is fast but uses
the whole machine; there is no headroom to run other things next to it.

### 5.5 Accuracy metrics

**WER — Word Error Rate** (English, Indonesian)
Count how many word-level fixes are needed to turn the model's text into the
correct text, then divide by the number of words in the correct text.

`WER = (substitutions + deletions + insertions) ÷ words in the reference`

| Fix type | Example (reference: *the cat sat*) |
|---|---|
| Substitution — wrong word | "the **hat** sat" |
| Deletion — missing word | "the sat" |
| Insertion — extra word | "the cat **really** sat" |

One mistake in a 10-word sentence is a WER of 10 %. **Lower is better; 0 % is
perfect.** WER can exceed 100 % if the model writes far more words than were
spoken.

**CER — Character Error Rate** (Chinese)
Exactly the same idea but counted per **character**. Chinese is written without
spaces between words, so "word" is not a clear unit; one character is.

**Normalisation (why capitals and commas don't count)**
Before comparing, both texts are tidied so harmless differences are ignored:
upper/lower case, punctuation, and extra spaces for English/Indonesian
(`Don't!` and `dont` match); punctuation and symbols for Chinese. Only real
wording differences are counted.

**Corpus rate vs Mean rate** (Accuracy → *Aggregate* table)

| | How it is calculated | When it differs |
|---|---|---|
| **Corpus rate** | total mistakes ÷ total reference words (or characters) over all files | long files weigh more — usually the better overall number |
| **Mean rate** | simple average of each file's score | every file counts equally, even a 2-word clip |

**Draft references (`*`)**
Some reference transcripts are drafts (`verified: false`), copied from model
output rather than written by a person. A score against a draft only shows how
much a model **agrees with that draft**, not how correct it is. The report marks
those rows with `*`. Chinese rows can also exceed 1.0 (100 %) when the reference
holds one sentence but the recording has several.

**Detected language**
The language the engine says it heard (when it reports one). A wrong detected
language is a quick explanation for a very high error rate.

### 5.6 Concurrency metrics (many calls at once)

Several worker threads ("legs") share **one** loaded model and transcribe the same
audio as fast as they can, to see how the system copes under load. It is an
offline capacity test, not a live-call simulation.

| Metric | Plain meaning |
|---|---|
| **Legs** | number of simultaneous callers |
| **Calls** | total requests made at that level (legs × rounds) |
| **Wall** | total real time to finish all of them |
| **Lat P50 / P95 / P99** | per-call latency while sharing the machine (see 5.2) |
| **RTF Mean** | average per-call RTF. It rises as legs are added because calls slow each other down |
| **Throughput** | audio seconds transcribed per second of real time, across all legs together |
| **Errors** | calls that raised an exception (should be 0) |
| **Peak RSS** | highest RAM used at that level |

**Throughput, simply:** if the machine transcribed 60 s of audio while 20 s of
real time passed, throughput is **3.0×**. In the raw JSON this is called
`audio-hours per wall-hour` — the same number. A throughput of 3.0× means the
machine can in principle keep roughly three live calls going at once.

**What good scaling looks like:** going from 1 to 2 to 4 legs, throughput should
rise or stay level while latency stays reasonable. If latency grows as fast as
the number of legs and throughput stays flat, the CPU is already saturated and
extra callers just wait in line.

### 5.7 Hardware fingerprint

CPU model, physical and logical cores, RAM, operating system, Python and
key library versions. Benchmarks are only comparable on similar hardware, so every
report records where it was run.

### 5.8 Per-stage timing (inside the engines)

Some engines also report where the time went inside one call. These appear in the
raw JSON (`timing_detail`) and in the live UI cards:

| Stage | Plain meaning |
|---|---|
| **Mel** | turning the sound wave into the picture-like input the model reads |
| **Encoder** | "listening": the model summarises the audio |
| **Prefill** | "getting ready to write": the model reads the summary plus its instructions once |
| **Decode** | "writing": producing the words one at a time — grows with how much was said |
| **Tokens** | how many word-pieces were produced |

### 5.9 Worked example

> Config `X`, one 10 s file, 3 measured runs: latencies 2.1 s, 2.0 s, 2.6 s.

- Mean latency = 2.23 s, P50 = 2.1 s, max = 2.6 s.
- RTF per run = 0.21, 0.20, 0.26 → the model is about 4–5× faster than live
  speech, so it can keep up in a live call.
- If the reference has 20 words and the model made 1 mistake, WER = 1 ÷ 20 = **5 %**.
- With 2 legs, if 4 calls finish in 5 s of wall time: audio processed = 40 s,
  so throughput = 40 ÷ 5 = **8×**.

(These numbers are made up to show the arithmetic; real results are in your
generated `*_summary.md`.)

---

## 6. Outputs

| File | Contents |
|---|---|
| `<UTC>_raw.json` | Every measurement, including raw per-run latencies, hardware fingerprint and run parameters |
| `<UTC>_summary.md` | Hardware/software table, load, latency/RTF, CPU/memory, accuracy and concurrency tables |

Hardware fingerprint (`reporters/hardware_info.py`): CPU model, physical/logical
cores, RAM, OS, Python and key library versions.

---

## 7. Tests

```bash
uv run pytest benchmark/tests -v
```

Unit tests cover statistics, WER/CER, system sampling, each runner, the engine
loader and both reporters. They use fake engines and do not need model weights,
except one Whisper loader test that is skipped when
`models/whisper_int8/tiny_*` is absent.
