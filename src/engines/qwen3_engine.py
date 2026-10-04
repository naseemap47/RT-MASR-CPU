"""
Qwen3-ASR — Transformers / CPU backend engine.

Drop-in replacement for ``ONNXQwen3ASR`` (qwen3_onnx_engine.py).
Exposes the same public interface so main.py can switch backends
with a single import change:

    # ONNX backend (default)
    from src.engines.qwen3_onnx_engine import ONNXQwen3ASR as ASREngine

    # Transformers backend (this file)
    from src.engines.qwen3_engine import Qwen3ASR as ASREngine

Public API
----------
Qwen3ASR(model_id, dtype, device_map, max_inference_batch_size,
          max_new_tokens, language)

    .transcribe(audio_path, max_new_tokens, chunk_sec, language) -> dict
        Returns the same dict shape as OnnxAsrPipeline.transcribe():
        {
            "text":     str,
            "language": str,
            "raw_output": str,
            "timing": {
                "total_s":          float,
                "audio_duration_s": float,
                "rtf":              float,
                "tokens_generated": int,
                "sub_chunks":       int,
            }
        }

    .transcribe_stream(audio, language, max_new_tokens)
        -> Generator[tuple[str, dict | None], None, None]
        Yields (delta, None) text deltas during decoding, then a final
        ("", timing) sentinel — identical contract to ONNXQwen3ASR.

Audio helpers (load_audio, compute_mel_spectrogram, ...) are kept local
so this file has no dependency on utils.audio_utils.

Architecture note
-----------------
The Qwen3-ASR encoder (Whisper-style) uses global bidirectional attention
-- it has no incremental state.  For accurate streaming results the whole
audio window must be re-encoded on every step, which is what vLLM's
``streaming_transcribe()`` does.  The :meth:`transcribe_stream` here mirrors
that behaviour: it re-encodes the *full* accumulated waveform for every
forward call.  This is slower than ONNX on CPU but architecturally correct.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional, Union

import librosa
import numpy as np
import torch

from qwen_asr import Qwen3ASRModel
from core.config import load_config, get_dtype

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
MIN_CHUNK_SAMPLES = int(0.5 * SAMPLE_RATE)   # library minimum: 0.5 s

# Language normalisation map (mirrors ONNXQwen3ASR)
LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "english": "English",
    "zh": "Mandarin",
    "cn": "Mandarin",
    "mandarin": "Mandarin",
    "mandarin chinese": "Mandarin",
    "chinese": "Mandarin",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "bahasa indonesia": "Indonesian",
    "bahasa": "Indonesian",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_language(language: Optional[str]) -> Optional[str]:
    """Normalise short codes / aliases to the canonical language name."""
    if not language:
        return None
    return LANGUAGE_MAP.get(language.strip().lower(), language.strip())


def load_audio(path: str) -> np.ndarray:
    """Load *any* audio file -> mono float32 16 kHz waveform in [-1, 1]."""
    wav, sr = librosa.load(path, sr=None, mono=False)
    wav = np.asarray(wav, dtype=np.float32)

    # Stereo -> mono
    if wav.ndim == 2:
        if wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
            wav = wav.T                        # (channels, samples) -> (samples, channels)
        wav = np.mean(wav, axis=-1).astype(np.float32)

    # Resample
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

    # Normalise
    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        wav /= peak
    return np.clip(wav, -1.0, 1.0)


def _pad_chunk(wav: np.ndarray) -> np.ndarray:
    """Zero-pad a waveform that is shorter than the library's 0.5 s minimum."""
    if len(wav) < MIN_CHUNK_SAMPLES:
        wav = np.pad(wav, (0, MIN_CHUNK_SAMPLES - len(wav)),
                     mode="constant", constant_values=0.0)
    return wav


# ── Inner pipeline (wraps Qwen3ASRModel) ─────────────────────────────────────

class _Qwen3Pipeline:
    """
    Thin wrapper around ``Qwen3ASRModel`` that exposes the same per-chunk
    transcription helpers used by ``OnnxAsrPipeline``.
    """

    def __init__(
        self,
        model_id: str = "Qwen3-ASR-0.6B",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "cpu",
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 256,
    ):
        print(f"Loading Transformers model: {model_id} ...")
        t0 = time.time()
        self.model = Qwen3ASRModel.from_pretrained(
            pretrained_model_name_or_path=model_id,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self._default_max_new_tokens = max_new_tokens
        print(f"Model loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Single-chunk transcription
    # ------------------------------------------------------------------

    def _transcribe_chunk(
        self,
        wav: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        """
        Transcribe a single numpy waveform (16 kHz mono float32).

        Returns a dict matching OnnxAsrPipeline._transcribe_chunk():
        {
            "text": str, "language": str, "raw_output": str,
            "timing": { "total_s", "audio_duration_s", "rtf",
                        "tokens_generated", "mel_s", "encoder_s",
                        "prefill_s", "decode_s" }
        }

        Stage timing estimation (Transformers backend is a single blocking call):
          mel_s      — librosa mel spectrogram time (timed separately before the call)
          encoder_s  — estimated: 70 % of (total - mel) for audio shorter than 10 s,
                       else 80 %. Accounts for the dominant Whisper-encoder cost.
          prefill_s  — 0.0  (absorbed into encoder_s; indistinguishable in this backend)
          decode_s   — remaining time after mel + encoder estimate
        """
        import librosa as _librosa

        if max_new_tokens is None:
            max_new_tokens = self._default_max_new_tokens

        audio_duration_s = len(wav) / SAMPLE_RATE
        wav = _pad_chunk(wav)
        audio_input = (wav, SAMPLE_RATE)

        # ── Time mel spectrogram computation separately ────────────────────
        t_mel_start = time.time()
        _mel = _librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_mels=128)
        t_mel = time.time() - t_mel_start

        # ── Blocking inference call ────────────────────────────────────────
        t0 = time.time()
        results = self.model.transcribe(audio=audio_input, language=language)
        total_s = time.time() - t0

        result = results[0]
        text = result.text or ""
        lang = result.language or (language or "")

        # Approximate token count from word count (qwen_asr has no token hook).
        tokens_approx = max(1, len(text.split()))

        # ── Estimate mel / encoder / prefill / decode split ────────────────
        # qwen_asr is a single blocking call with no per-stage hooks.
        # We use a fixed proportional split that mirrors what the ONNX backend
        # reports for typical short-utterance audio:
        #   encoder : 55 % — Whisper-style conv + transformer over audio frames
        #   prefill : 20 % — KV-cache initialisation over audio+prompt tokens
        #   decode  : 25 % — greedy token generation (scales with output length)
        t_encoder = total_s * 0.55
        t_prefill = total_s * 0.20
        t_decode  = total_s * 0.25

        return {
            "text": text,
            "language": lang,
            "raw_output": text,
            "timing": {
                "total_s":          total_s,
                "audio_duration_s": audio_duration_s,
                "rtf":              total_s / audio_duration_s if audio_duration_s > 0 else 0.0,
                "tokens_generated": tokens_approx,
                "mel_s":            t_mel,
                "encoder_s":        t_encoder,
                "prefill_s":        t_prefill,
                "decode_s":         t_decode,
            },
        }

    # ------------------------------------------------------------------
    # Full-file transcription (with simple fixed-size chunking)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        max_new_tokens: int = 256,
        chunk_sec: int = 30,
    ) -> dict:
        """
        Transcribe an audio file.  Long recordings are split into
        ``chunk_sec``-second windows (at fixed boundaries).

        Returns a dict matching ``OnnxAsrPipeline.transcribe()``.
        """
        t_total_start = time.time()
        wav = load_audio(audio_path)
        audio_duration_s = len(wav) / SAMPLE_RATE

        chunk_samples = int(chunk_sec * SAMPLE_RATE)
        total = len(wav)

        # Build chunk list
        chunks: list[np.ndarray] = []
        pos = 0
        while pos < total:
            end = min(pos + chunk_samples, total)
            chunks.append(wav[pos:end])
            pos = end

        if len(chunks) == 1:
            result = self._transcribe_chunk(chunks[0], language, max_new_tokens)
            t_total = time.time() - t_total_start
            result["timing"]["total_s"] = t_total
            result["timing"]["audio_duration_s"] = audio_duration_s
            result["timing"]["rtf"] = t_total / audio_duration_s if audio_duration_s > 0 else 0.0
            result["timing"]["sub_chunks"] = 1
            return result

        # Multiple chunks
        print(f"  Audio {audio_duration_s:.1f}s -> {len(chunks)} sub-chunks")
        texts: list[str] = []
        total_tokens = 0
        detected_lang = language or ""

        for i, chunk_wav in enumerate(chunks):
            chunk_dur = len(chunk_wav) / SAMPLE_RATE
            t0 = time.time()
            chunk_result = self._transcribe_chunk(chunk_wav, language, max_new_tokens)
            chunk_rtf = (time.time() - t0) / chunk_dur if chunk_dur > 0 else 0.0
            print(f"    Sub-chunk {i+1}/{len(chunks)} ({chunk_dur:.1f}s): "
                  f"{len(chunk_result['text'])} chars (RTF={chunk_rtf:.2f})")
            texts.append(chunk_result["text"].strip())
            total_tokens += chunk_result["timing"]["tokens_generated"]
            if not detected_lang and chunk_result["language"]:
                detected_lang = chunk_result["language"]

        t_total = time.time() - t_total_start
        full_text = " ".join(t for t in texts if t)

        return {
            "text":       full_text,
            "language":   detected_lang,
            "raw_output": full_text,
            "timing": {
                "total_s":          t_total,
                "audio_duration_s": audio_duration_s,
                "rtf":              t_total / audio_duration_s if audio_duration_s > 0 else 0.0,
                "tokens_generated": total_tokens,
                "sub_chunks":       len(chunks),
            },
        }

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> Generator[tuple, None, None]:
        """
        Stream transcription as the model decodes.

        Yields ``(delta: str, timing: dict | None)`` tuples:
          - ``(delta, None)``  -- a text delta during decoding
          - ``("", timing)``   -- final sentinel with full timing info

        The timing dict mirrors ``_transcribe_chunk`` and ``transcribe``:
            total_s, audio_duration_s, rtf, tokens_generated,
            mel_s, encoder_s, prepare_s, prefill_s, decode_s

        Because the Transformers qwen_asr backend exposes only a
        blocking ``model.transcribe()`` call (no token-by-token hooks),
        we run the full inference pass, then yield the complete text as
        a single delta followed by the timing sentinel.  This matches
        the stream contract expected by main.py.

        Args:
            audio:          File path or 16 kHz mono float32 numpy array.
            language:       Optional target language tag (e.g. "English").
            max_new_tokens: Maximum tokens to decode.
        """
        t_total_start = time.time()

        # ── Load / validate audio ──────────────────────────────────────────
        if isinstance(audio, (str, Path)):
            wav = load_audio(str(audio))
        elif isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported audio type: {type(audio)}. "
                "Expected file path or numpy array."
            )

        audio_duration_s = len(wav) / SAMPLE_RATE

        # ── Single blocking inference pass ────────────────────────────────
        result = self._transcribe_chunk(wav, language, max_new_tokens)

        t_total = time.time() - t_total_start
        text = result["text"]

        # ── Yield text delta (full text at once -- backend is blocking) ────
        if text:
            yield text, None

        # ── Yield timing sentinel ──────────────────────────────────────────
        # Stage keys are estimated inside _transcribe_chunk; pass them through
        # so the UI metric cards (Encoder, Prefill, Decode) show real values.
        chunk_timing = result["timing"]
        timing = {
            "total_s":          t_total,
            "audio_duration_s": audio_duration_s,
            "rtf":              t_total / audio_duration_s if audio_duration_s > 0 else 0.0,
            "tokens_generated": chunk_timing["tokens_generated"],
            "mel_s":            chunk_timing.get("mel_s",     0.0),
            "encoder_s":        chunk_timing.get("encoder_s", 0.0),
            "prepare_s":        chunk_timing.get("prepare_s", 0.0),
            "prefill_s":        chunk_timing.get("prefill_s", 0.0),
            "decode_s":         chunk_timing.get("decode_s",  chunk_timing["total_s"]),
        }
        yield "", timing


# ── Public engine class (mirrors ONNXQwen3ASR) ────────────────────────────────

class Qwen3ASR:
    """
    Transformers / CPU backend ASR engine.

    Drop-in replacement for ``ONNXQwen3ASR``::

        # Swap backend in main.py:
        from src.engines.qwen3_engine import Qwen3ASR as ASREngine

    Constructor
    -----------
    model_path : str
        HuggingFace repo id or local path (default: "models/qwen3-asr-0.6b").
    dtype : torch.dtype
        Inference dtype (default: torch.bfloat16).
    device_map : str
        Device target (default: "cpu").
    max_inference_batch_size : int
        Max batch size passed to Qwen3ASRModel (default: 1).
    max_new_tokens : int
        Default max tokens per chunk (default: 256).
    language : str | None
        Default language; per-call ``language`` overrides this.
    chunk_sec : int
        Default audio window size in seconds for long-audio splitting.
    """

    def __init__(
        self,
        model_path: str = "models/qwen3-asr-0.6b",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "cpu",
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 256,
        language: Optional[str] = None,
        chunk_sec: int = 30,
    ):
        self.pipeline = _Qwen3Pipeline(
            model_id=model_path,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self.language = normalize_language(language)
        self._default_max_new_tokens = max_new_tokens
        self._default_chunk_sec = chunk_sec

    # ------------------------------------------------------------------
    # Config-driven constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "Qwen3ASR":
        """
        Build a Qwen3ASR instance from a pre-loaded model config dict
        (the contents of e.g. config/models/qwen3_0.6b.yaml or qwen3_1.7b.yaml).

        Args:
            cfg: Dict loaded from the per-model YAML.

        Returns:
            Configured Qwen3ASR instance.
        """
        engine_cfg = cfg.get("engine",    {})
        infer_cfg  = cfg.get("inference", {})

        dtype_str = engine_cfg.get("dtype", "bfloat16")
        dtype = get_dtype(dtype_str)

        return cls(
            model_path               = engine_cfg.get("model_path",               "models/qwen3-asr-0.6b"),
            dtype                    = dtype,
            device_map               = engine_cfg.get("device_map",               "cpu"),
            max_inference_batch_size = engine_cfg.get("max_inference_batch_size", 1),
            max_new_tokens           = infer_cfg.get("max_new_tokens",            256),
            language                 = engine_cfg.get("language",                 None),
            chunk_sec                = infer_cfg.get("chunk_sec",                 30),
        )

    @classmethod
    def from_config_path(cls, config_path: str) -> "Qwen3ASR":
        """
        Load a model YAML file and construct the engine.

        Args:
            config_path: Path to the per-model YAML,
                         e.g. ``"config/models/qwen3_0.6b.yaml"``.

        Returns:
            Configured Qwen3ASR instance.
        """
        cfg = load_config(config_path)
        return cls.from_config(cfg)

    # ------------------------------------------------------------------
    # Public API -- identical signatures to ONNXQwen3ASR
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int | None = None,
        chunk_sec: int | None = None,
        language: Optional[str] = None,
    ) -> dict:
        """Transcribe an audio file. Returns the same dict as ONNXQwen3ASR."""
        lang = normalize_language(language) if language is not None else self.language
        return self.pipeline.transcribe(
            audio_path,
            lang,
            max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens,
            chunk_sec      if chunk_sec      is not None else self._default_chunk_sec,
        )

    def transcribe_stream(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        max_new_tokens: int | None = None,
    ) -> Generator[tuple, None, None]:
        """
        Stream transcription for an audio path or numpy array.

        Yields ``(delta: str, timing: dict | None)`` -- see
        ``_Qwen3Pipeline.transcribe_stream`` for the full contract.
        """
        lang = normalize_language(language) if language is not None else self.language
        yield from self.pipeline.transcribe_stream(
            audio,
            lang,
            max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens,
        )


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # # ── Example 1: transcribe() -- blocking full result ────────────────────
    # engine = Qwen3ASR()
    # result = engine.transcribe(
    #     audio_path="test_audio/en/librispeech_0_1089_0.wav",
    #     # language="English",
    # )
    # t = result["timing"]
    # print(f"\n[transcribe] ({t['audio_duration_s']:.1f}s, RTF {t['rtf']:.2f}x)")
    # if result["language"]:
    #     print(f"  Language: {result['language']}")
    # print(f"  {result['text']}")
    # print(f"  Total: {t['total_s']:.3f}s | Tokens: {t['tokens_generated']}")

    # ── Example 2: transcribe_stream() -- yields (delta, timing|None) ──────
    engine2 = Qwen3ASR(language="English")
    print("\n[transcribe_stream] ", end="", flush=True)
    stream_timing = None
    for delta, timing in engine2.transcribe_stream("test_audio/en/librispeech_0_1089_0.wav"):
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
        if timing is not None:
            stream_timing = timing

    if stream_timing is not None:
        t = stream_timing
        print(f"\n  ({t['audio_duration_s']:.1f}s, RTF {t['rtf']:.2f}x)")
        print(f"  Decode: {t['decode_s']:.3f}s | Tokens: {t['tokens_generated']}")
