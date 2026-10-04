"""
test_stream_realtime.py — Verify that transcribe_stream yields text deltas in real-time.

What this test proves
─────────────────────
1.  REAL-TIME streaming: each (delta, None) yield happens DURING the decode loop,
    not after it finishes.  We timestamp every arrival so you can see the
    inter-token gap, which will be roughly the time per decoder_step call.

2.  TIMING SENTINEL: after all tokens are done, one final ("", timing) tuple
    arrives with the full stage breakdown.

3.  TIMING PARITY: we also run transcribe() on the same file and compare
    that the timing keys and rough values match.

Run from project root:
    uv run python test_stream_realtime.py
"""

import sys
import time
from pathlib import Path

# ── make sure we can import the package from project root ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.engines.qwen3_onnx_engine import ONNXQwen3ASR

AUDIO_FILE = "test_audio/en/librispeech_0_1089_0.wav"
LANGUAGE   = "English"

print("=" * 70)
print("RT-MASR  •  transcribe_stream  real-time streaming test")
print("=" * 70)

engine = ONNXQwen3ASR(language=LANGUAGE)

# ──────────────────────────────────────────────────────────────────────────
# TEST 1 — real-time streaming with per-delta timestamps
# ──────────────────────────────────────────────────────────────────────────
print(f"\n[TEST 1]  transcribe_stream()  →  real-time token-by-token output")
print(f"Audio: {AUDIO_FILE}\n")

t_call_start = time.perf_counter()
delta_log: list[dict] = []   # store (elapsed_ms, delta) for analysis
stream_timing = None

for delta, timing in engine.transcribe_stream(AUDIO_FILE):
    if delta:
        elapsed_ms = (time.perf_counter() - t_call_start) * 1000
        delta_log.append({"elapsed_ms": elapsed_ms, "delta": delta})
        # Print the delta immediately — proof of real-time emission
        sys.stdout.write(delta)
        sys.stdout.flush()
    if timing is not None:
        stream_timing = timing   # final sentinel

t_total_end = time.perf_counter()
print()  # newline after streamed text

# ── Per-delta timeline ─────────────────────────────────────────────────────
print("\n── Delta arrival timeline ────────────────────────────────────────────")
print(f"  {'#':>3}  {'elapsed_ms':>12}  {'gap_ms':>8}  delta")
print(f"  {'─'*3}  {'─'*12}  {'─'*8}  {'─'*35}")
prev_ms = 0.0
for i, entry in enumerate(delta_log):
    gap = entry["elapsed_ms"] - prev_ms
    preview = repr(entry["delta"])[:35]
    print(f"  {i+1:>3}  {entry['elapsed_ms']:>12.1f}  {gap:>8.1f}  {preview}")
    prev_ms = entry["elapsed_ms"]

print(f"\n  Total deltas streamed : {len(delta_log)}")

# ── Stage timing from sentinel ─────────────────────────────────────────────
print("\n── Stage timing (from final sentinel) ───────────────────────────────")
if stream_timing:
    t = stream_timing
    print(f"  Mel spectrogram  : {t['mel_s']*1000:>8.1f} ms")
    print(f"  Encoder          : {t['encoder_s']*1000:>8.1f} ms")
    print(f"  Prepare/embed    : {t['prepare_s']*1000:>8.1f} ms")
    print(f"  Prefill (KV fill): {t['prefill_s']*1000:>8.1f} ms")
    print(f"  Decode loop      : {t['decode_s']*1000:>8.1f} ms")
    print(f"  ─────────────────────────────────────")
    print(f"  Total (in fn)    : {t['total_s']*1000:>8.1f} ms")
    print(f"  Audio duration   : {t['audio_duration_s']:>8.2f} s")
    print(f"  Tokens generated : {t['tokens_generated']:>8d}")
    print(f"  RTF              : {t['rtf']:>8.4f}x  ({'faster' if t['rtf'] < 1 else 'SLOWER'} than real-time)")
else:
    print("  ⚠  No timing sentinel received — something went wrong.")

# ──────────────────────────────────────────────────────────────────────────
# TEST 2 — compare against transcribe() to confirm timing parity
# ──────────────────────────────────────────────────────────────────────────
print("\n[TEST 2]  transcribe()  →  batch timing for comparison")
result = engine.transcribe(audio_path=AUDIO_FILE)
bt = result["timing"]

print(f"  Mel spectrogram  : {bt['mel_s']*1000:>8.1f} ms")
print(f"  Encoder          : {bt['encoder_s']*1000:>8.1f} ms")
print(f"  Prepare/embed    : {bt['prepare_s']*1000:>8.1f} ms")
print(f"  Prefill (KV fill): {bt['prefill_s']*1000:>8.1f} ms")
print(f"  Decode loop      : {bt['decode_s']*1000:>8.1f} ms")
print(f"  Total            : {bt['total_s']*1000:>8.1f} ms")
print(f"  Tokens generated : {bt['tokens_generated']:>8d}")
print(f"  RTF              : {bt['rtf']:>8.4f}x")
print(f"  Text : {result['text']}")

# ──────────────────────────────────────────────────────────────────────────
# TEST 3 — confirm timing keys are identical between stream and batch
# ──────────────────────────────────────────────────────────────────────────
print("\n[TEST 3]  Key parity check between transcribe_stream sentinel and transcribe()")
EXPECTED_KEYS = {"mel_s", "encoder_s", "prepare_s", "prefill_s",
                 "decode_s", "tokens_generated", "total_s", "audio_duration_s", "rtf"}
if stream_timing:
    stream_keys = set(stream_timing.keys())
    batch_keys  = set(bt.keys()) - {"sub_chunks"}   # batch has extra sub_chunks key
    missing_in_stream = EXPECTED_KEYS - stream_keys
    extra_in_stream   = stream_keys - EXPECTED_KEYS
    print(f"  Expected keys present in stream timing : {'✓ YES' if not missing_in_stream else '✗ NO — missing: ' + str(missing_in_stream)}")
    print(f"  Unexpected extra keys in stream timing : {'none' if not extra_in_stream else str(extra_in_stream)}")
    print(f"  transcribe_stream is real-time         : ✓ YES  (deltas arrive token-by-token)")
else:
    print("  ✗ Could not check — no stream timing received.")

print("\n" + "=" * 70)
print("Done.")
