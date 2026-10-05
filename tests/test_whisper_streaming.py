"""
Tests for the Whisper sliding-window streamer (src/engines/whisper_streaming.py)
and its integration into the FastAPI WebSocket handler (main.py).

The unit tests use a scripted fake engine, so they need no model files. The
integration test at the bottom uses the real whisper tiny model and is skipped
if it has not been downloaded.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engines.live_call_session import LiveCallSession
from src.engines.whisper_streaming import (
    StreamingConfig,
    WhisperSlidingWindowStreamer,
    common_prefix_len,
    join_units,
    split_units,
)

SR = 16000
WORD_S = 0.5                       # every scripted word lasts 0.5 s
WORD_N = int(WORD_S * SR)


# ── helpers ──────────────────────────────────────────────────────────────────
def word_audio(k: int) -> np.ndarray:
    """0.5 s of 'speech' whose amplitude encodes the word index k."""
    return np.full(WORD_N, 0.05 + 0.001 * k, dtype=np.float32)


def to_pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class ScriptEngine:
    """
    Fake Whisper. It 'recognises' word k wherever it finds amplitude 0.05+0.001k,
    returns Whisper-style segments (4 words each, window-relative timestamps),
    ignores the cut-off partial word at the window edges, and on every 3rd call
    appends a never-repeated garbage word to mimic an unstable tail.
    """

    def __init__(self, words: list[str], garbage_every: int = 3):
        self.words = words
        self.garbage_every = garbage_every
        self.calls = 0
        self.languages: list = []
        self.window_lengths: list[float] = []

    def transcribe(self, audio, language=None, beam_size=None, fallback=None):
        self.calls += 1
        self.languages.append(language)
        self.window_lengths.append(len(audio) / SR)
        ks = np.rint((audio - 0.05) / 0.001).astype(int)
        voiced = np.nonzero(np.abs(audio) > 0.01)[0]
        found: list[tuple[int, float]] = []              # (word index, start seconds)
        if len(voiced):
            k_voiced = ks[voiced]
            for k in dict.fromkeys(k_voiced.tolist()):
                where = voiced[k_voiced == k]
                if len(where) >= WORD_N - 400:           # complete word only
                    found.append((k, where[0] / SR))
        segments = []
        for i in range(0, len(found), 4):
            chunk = found[i:i + 4]
            segments.append({
                "text": " " + " ".join(self.words[k] for k, _ in chunk),
                "start": chunk[0][1],
                "end": chunk[-1][1] + WORD_S,
            })
        if self.garbage_every and self.calls % self.garbage_every == 0 and segments:
            end = len(audio) / SR
            segments.append({"text": f" garbage{self.calls}", "start": end - 0.3, "end": end})
        return {
            "text": "".join(s["text"] for s in segments),
            "segments": segments,
            "language": language or "en",
            "timing": {"total_s": 0.01, "encoder_s": 0.005, "decode_s": 0.004, "mel_s": 0.001},
        }


def make(words, garbage_every=3, **cfg):
    session = LiveCallSession()
    engine = ScriptEngine(words, garbage_every)
    streamer = WhisperSlidingWindowStreamer(engine, session, StreamingConfig(**cfg))
    return session, engine, streamer


def feed_and_run(session, streamer, chunks: list[np.ndarray]):
    """Feed 0.5 s chunks like the browser does, stepping whenever the streamer is ready."""
    updates = []
    for c in chunks:
        session.process_pcm_bytes(to_pcm(c))
        if streamer.ready():
            u = streamer.step()
            if u is not None:
                updates.append(u)
    return updates


WORDS = [f"w{i}" for i in range(80)]


# ── text units ───────────────────────────────────────────────────────────────
def test_split_units_words_and_punctuation():
    u = split_units(" Hello, world! -  ok")
    assert [x.key for x in u] == ["hello", "world", "ok"]
    assert [x.text for x in u] == ["Hello,", "world!-", "ok"]     # stray '-' attached to 'world!'


def test_split_units_cjk_is_per_character():
    u = split_units("你好，世界。")
    assert [x.key for x in u] == ["你", "好", "世", "界"]
    assert join_units(u) == "你好，世界。"                          # no spaces inserted


def test_join_units_mixed_scripts():
    # no space is inserted next to a CJK character (CJK is written without spaces)
    assert join_units(split_units("hello 世界 world")) == "hello世界world"
    assert join_units(split_units("one two three")) == "one two three"


def test_common_prefix_ignores_case_and_punctuation():
    a, b = split_units("Hello, there friend"), split_units("hello there FRIEND again")
    assert common_prefix_len(a, b) == 3
    assert common_prefix_len(split_units("a b c"), split_units("a x c")) == 1
    assert common_prefix_len([], split_units("a")) == 0


# ── scheduling ───────────────────────────────────────────────────────────────
def test_ready_only_after_hop_of_new_audio():
    session, _, st = make(WORDS, hop_s=1.0)
    session.process_pcm_bytes(to_pcm(word_audio(0)))              # 0.5 s
    assert not st.ready()
    session.process_pcm_bytes(to_pcm(word_audio(1)))              # 1.0 s
    assert st.ready()
    st.step()
    assert not st.ready()                                         # counter reset by the pass


# ── LocalAgreement ───────────────────────────────────────────────────────────
def test_text_is_committed_only_after_two_agreeing_passes():
    session, _, st = make(WORDS, hop_s=1.0, max_window_s=60, hard_max_window_s=60)
    ups = feed_and_run(session, st, [word_audio(k) for k in range(6)])   # 3 s -> 3 passes
    assert ups[0].committed_text == ""                      # first hypothesis: nothing agreed yet
    assert ups[0].tentative_text != ""                      # ...but it is shown as tentative
    assert ups[1].committed_text.startswith("w0")           # second pass confirms the prefix
    assert ups[-1].committed_text != ""


def test_unstable_garbage_tail_is_never_committed():
    session, _, st = make(WORDS, hop_s=1.0, max_window_s=60, hard_max_window_s=60)
    feed_and_run(session, st, [word_audio(k) for k in range(20)])
    assert "garbage" not in st.committed_text
    # final flush on a clean (non-garbage) call commits exactly the words
    st.engine.garbage_every = 0          # a clean final pass
    st.finish()
    assert st.committed_text == " ".join(WORDS[:20])


def test_committed_text_is_append_only_and_has_no_duplicates():
    session, engine, st = make(WORDS, hop_s=1.0, max_window_s=60, hard_max_window_s=60)
    seen = ""
    for c in [word_audio(k) for k in range(24)]:
        session.process_pcm_bytes(to_pcm(c))
        if st.ready():
            u = st.step()
            if u:
                assert u.committed_text.startswith(seen), "committed text must never be rewritten"
                seen = u.committed_text
    words = seen.split()
    assert len(words) == len(set(words))                     # nothing repeated


# ── sliding ──────────────────────────────────────────────────────────────────
def test_window_slides_and_every_word_appears_once_in_order():
    n = 60                                             # 30 s of audio
    session, engine, st = make(WORDS, hop_s=1.0, max_window_s=6.0, hard_max_window_s=10.0)
    feed_and_run(session, st, [word_audio(k) for k in range(n)])
    engine.garbage_every = 0                           # a clean final pass
    st.finish()
    assert st.committed_text.split() == WORDS[:n]
    # The audio actually handed to Whisper stayed bounded by the hard cap (+ one hop of slack).
    assert max(engine.window_lengths) <= 10.0 + 2.0
    # ...and the buffer really was trimmed (not 30 s long).
    assert st._window_start_samples > 0


def test_pass_cost_does_not_grow_with_call_length():
    session, engine, st = make(WORDS, hop_s=1.0, max_window_s=6.0, hard_max_window_s=10.0)
    feed_and_run(session, st, [word_audio(k) for k in range(60)])
    first, last = engine.window_lengths[:5], engine.window_lengths[-5:]
    assert max(last) <= max(10.0 + 2.0, max(first))


def test_hard_cap_force_commits_when_nothing_was_confirmed():
    class Unstable(ScriptEngine):
        def transcribe(self, audio, **kw):               # different garbage prefix each time
            r = super().transcribe(audio, **kw)
            for s in r["segments"]:
                s["text"] = f" x{self.calls} " + s["text"].strip()
            return r

    session = LiveCallSession()
    st = WhisperSlidingWindowStreamer(
        Unstable(WORDS), session,
        StreamingConfig(hop_s=1.0, max_window_s=4.0, hard_max_window_s=6.0))
    feed_and_run(session, st, [word_audio(k) for k in range(16)])   # 8 s
    assert len(session.audio_buffer) / SR < 7.0              # it was cut despite no agreement
    assert st.committed_text != ""                           # ...and the text was force-committed


# ── silence handling ─────────────────────────────────────────────────────────
def test_silence_after_speech_flushes_and_empties_window():
    session, engine, st = make(WORDS, garbage_every=0, hop_s=1.0, max_window_s=60,
                               hard_max_window_s=60, silence_flush_s=0.8)
    speech = [word_audio(k) for k in range(8)]
    silence = [np.zeros(WORD_N, dtype=np.float32) for _ in range(4)]
    ups = feed_and_run(session, st, speech + silence)
    flushes = [u for u in ups if u.flushed]
    assert len(flushes) == 1
    assert flushes[0].committed_text.split() == WORDS[:8]    # whole utterance finalised
    assert flushes[0].tentative_text == ""
    assert not st._pending
    assert len(session.audio_buffer) <= 0.5 * SR              # window emptied (pre-roll at most)


def test_pure_silence_never_calls_the_engine_and_buffer_stays_small():
    session, engine, st = make(WORDS, hop_s=1.0)
    ups = feed_and_run(session, st, [np.zeros(WORD_N, dtype=np.float32) for _ in range(40)])  # 20 s
    assert engine.calls == 0
    assert ups == []
    assert len(session.audio_buffer) / SR <= 1.5             # pre-roll only, not 20 s


def test_finish_with_nothing_pending_is_safe():
    session, engine, st = make(WORDS)
    u = st.finish()
    assert u.full_text == "" and engine.calls == 0


# ── language handling ────────────────────────────────────────────────────────
def test_detected_language_is_locked_after_first_pass():
    session, engine, st = make(WORDS, hop_s=1.0, max_window_s=60, hard_max_window_s=60)
    feed_and_run(session, st, [word_audio(k) for k in range(8)])
    assert engine.languages[0] is None                # auto-detect on pass 1
    assert all(l == "en" for l in engine.languages[1:])
    assert st.detected_language == "en"


def test_forced_language_is_always_used():
    session = LiveCallSession()
    engine = ScriptEngine(WORDS)
    st = WhisperSlidingWindowStreamer(engine, session, StreamingConfig(), language="zh")
    feed_and_run(session, st, [word_audio(k) for k in range(6)])
    assert set(engine.languages) == {"zh"}


def test_reset_clears_all_state():
    session, engine, st = make(WORDS, hop_s=1.0, max_window_s=60, hard_max_window_s=60)
    feed_and_run(session, st, [word_audio(k) for k in range(8)])
    assert st.full_text
    session.mark_call_start()
    st.reset()
    assert st.full_text == "" and st._window_start_samples == 0 and not st.ready()


def test_streaming_config_from_dict_ignores_unknown_keys():
    c = StreamingConfig.from_dict({"hop_s": 2.0, "bogus": 1})
    assert c.hop_s == 2.0 and c.max_window_s == 12.0
    assert StreamingConfig.from_dict(None).hop_s == 1.0


# ── integration: real whisper tiny through the WebSocket ─────────────────────
TINY = "models/whisper_int8/tiny_encoder_11_int8.onnx"


@pytest.mark.skipif(not os.path.exists(TINY), reason="whisper tiny int8 model not downloaded")
def test_websocket_streams_whisper_tiny(monkeypatch):
    import soundfile as sf
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RT_MASR_MODEL", "whisper_int8_tiny")
    import main

    wav, sr = sf.read("test_audio/en/librispeech_2_1089_2.wav", dtype="float32")
    assert sr == SR

    with TestClient(main.app) as client:            # runs lifespan -> loads whisper tiny
        health = client.get("/api/health").json()
        assert health["backend"] == "whisper"
        assert health["stream_mode"] == "sliding_window"

        with client.websocket_connect("/ws/call-stream") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "connected" and hello["stream_mode"] == "sliding_window"
            ws.send_json({"type": "start_call", "language": "en"})
            assert ws.receive_json()["type"] == "call_ready"

            for i in range(0, len(wav), WORD_N):
                ws.send_bytes(to_pcm(wav[i:i + WORD_N]))
            ws.send_json({"type": "end_call"})

            deltas, final = [], None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "transcript_delta":
                    deltas.append(msg)
                elif msg["type"] == "call_ended":
                    final = msg
                    break

    assert deltas, "expected at least one live transcript_delta"
    for d in deltas:
        assert {"full_text", "committed_text", "tentative_text", "metrics"} <= d.keys()
        assert d["metrics"]["stream_mode"] == "sliding_window"
        assert "window_s" in d["metrics"]
    text = final["final_text"].lower()
    assert "yellow" in text and "lamps" in text
    assert final["metrics"]["stream_mode"] == "sliding_window"


@pytest.mark.skipif(not os.path.exists(TINY), reason="whisper tiny int8 model not downloaded")
def test_whisper_timing_reports_nonzero_prefill_and_consistent_stages():
    import soundfile as sf
    from src.engines.whisper_engine import WhisperOnnxEngine

    eng = WhisperOnnxEngine(model_name="tiny", model_dir="models/whisper_int8", precision="int8")
    wav, _ = sf.read("test_audio/en/librispeech_2_1089_2.wav", dtype="float32")

    t = eng.transcribe(wav, language="en", beam_size=1, fallback=False)["timing"]
    assert t["prefill_s"] > 0 and t["decode_s"] > 0 and t["encoder_s"] > 0
    # stages are disjoint slices of the pass
    stages = t["mel_s"] + t["encoder_s"] + t["prefill_s"] + t["decode_s"] + t["other_s"]
    assert stages <= t["total_s"] + 0.05

    # the streaming generator reports the same keys
    final = [tm for _, tm in eng.transcribe_stream(wav, language="en") if tm][-1]
    assert final["prefill_s"] > 0


# ── Qwen Transformers backend: stage timing is measured, not estimated ───────
QWEN06 = "models/qwen3-asr-0.6b"


@pytest.mark.skipif(not os.path.isdir(QWEN06), reason="qwen3-asr-0.6b not downloaded")
def test_qwen_transformers_stage_timing_is_measured():
    import soundfile as sf
    from src.engines.qwen3_engine import Qwen3ASR

    eng = Qwen3ASR(model_path=QWEN06)
    wav, _ = sf.read("test_audio/en/librispeech_2_1089_2.wav", dtype="float32")
    t = [tm for _, tm in eng.transcribe_stream(wav) if tm][-1]
    assert t["encoder_s"] > 0 and t["prefill_s"] > 0 and t["decode_s"] > 0
    # not the old fixed 55/20/25 split
    assert abs(t["encoder_s"] / t["total_s"] - 0.55) > 0.01
    assert t["mel_s"] + t["encoder_s"] + t["prefill_s"] + t["decode_s"] <= t["total_s"] + 0.05
    assert t["tokens_generated"] > 5
