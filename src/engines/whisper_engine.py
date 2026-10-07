"""
Whisper — Pure ONNX CPU Inference Engine.

No PyTorch dependency. Uses only ONNX Runtime + NumPy + soundfile/librosa.

Architecture:
    Audio → Mel Spectrogram → Encoder (ONNX) → Audio Features
    Beam / Greedy Decoder (ONNX) → Token sequence → Text

Model files must be provided locally via ``model_dir`` + ``precision``.
The engine constructs the filename as::

    {model_dir}/{model}_{component}_11_{precision}.onnx   # e.g. medium_encoder_11_int8.onnx
    {model_dir}/{model}_{component}_11.onnx               # when precision="none"

Usage (standalone)::

    python whisper_engine.py audio.wav --model medium --model_dir whisper_int8 --precision int8
    python whisper_engine.py audio.wav --model small  --model_dir whisper_fp32 --precision fp32 --stream

Drop-in integration::

    from engines.whisper_engine import WhisperOnnxEngine

    engine = WhisperOnnxEngine(
        model_name="medium",
        model_dir="whisper_int8",
        precision="int8",
    )
    result = engine.transcribe("audio.mp4")
    print(result["text"])
"""

import io
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Generator, List, Literal, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnx.serialization import ProtoSerializer

logger = logging.getLogger("rtmasr.engines.whisper")

# ── suppress noisy runtime warnings ────────────────────────────────────────
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants (mirror whisper/audio.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
N_FFT: int = 400
HOP_LENGTH: int = 160
CHUNK_LENGTH: int = 30          # seconds per Whisper context window
N_SAMPLES: int = CHUNK_LENGTH * SAMPLE_RATE   # 480 000
N_FRAMES: int = N_SAMPLES // HOP_LENGTH       # 3 000
N_MELS: int = 80

# Beam / decode defaults
DEFAULT_BEAM_SIZE: int = 5
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_BEST_OF: int = 5

# Supported precision suffixes
AVAILABLE_PRECISIONS = ["int8", "fp16", "fp32", "none"]

# ---------------------------------------------------------------------------
# Model dimension configs
# ---------------------------------------------------------------------------

_DIMS: dict = {
    "tiny":      {"n_mels": 80, "n_vocab": 51865, "n_audio_ctx": 1500, "n_audio_state": 384,  "n_audio_head": 6,  "n_audio_layer": 4,  "n_text_ctx": 448, "n_text_state": 384,  "n_text_head": 6,  "n_text_layer": 4},
    "tiny.en":   {"n_mels": 80, "n_vocab": 51864, "n_audio_ctx": 1500, "n_audio_state": 384,  "n_audio_head": 6,  "n_audio_layer": 4,  "n_text_ctx": 448, "n_text_state": 384,  "n_text_head": 6,  "n_text_layer": 4},
    "base":      {"n_mels": 80, "n_vocab": 51865, "n_audio_ctx": 1500, "n_audio_state": 512,  "n_audio_head": 8,  "n_audio_layer": 6,  "n_text_ctx": 448, "n_text_state": 512,  "n_text_head": 8,  "n_text_layer": 6},
    "base.en":   {"n_mels": 80, "n_vocab": 51864, "n_audio_ctx": 1500, "n_audio_state": 512,  "n_audio_head": 8,  "n_audio_layer": 6,  "n_text_ctx": 448, "n_text_state": 512,  "n_text_head": 8,  "n_text_layer": 6},
    "small":     {"n_mels": 80, "n_vocab": 51865, "n_audio_ctx": 1500, "n_audio_state": 768,  "n_audio_head": 12, "n_audio_layer": 12, "n_text_ctx": 448, "n_text_state": 768,  "n_text_head": 12, "n_text_layer": 12},
    "small.en":  {"n_mels": 80, "n_vocab": 51864, "n_audio_ctx": 1500, "n_audio_state": 768,  "n_audio_head": 12, "n_audio_layer": 12, "n_text_ctx": 448, "n_text_state": 768,  "n_text_head": 12, "n_text_layer": 12},
    "medium":    {"n_mels": 80, "n_vocab": 51865, "n_audio_ctx": 1500, "n_audio_state": 1024, "n_audio_head": 16, "n_audio_layer": 24, "n_text_ctx": 448, "n_text_state": 1024, "n_text_head": 16, "n_text_layer": 24},
    "medium.en": {"n_mels": 80, "n_vocab": 51864, "n_audio_ctx": 1500, "n_audio_state": 1024, "n_audio_head": 16, "n_audio_layer": 24, "n_text_ctx": 448, "n_text_state": 1024, "n_text_head": 16, "n_text_layer": 24},
}

AVAILABLE_MODELS: List[str] = list(_DIMS.keys())

# KV-cache layer counts per model
_KV_LAYERS: dict = {
    "tiny": 8, "tiny.en": 8,
    "base": 12, "base.en": 12,
    "small": 24, "small.en": 24,
    "medium": 48, "medium.en": 48,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ort_graph_opt(level: str) -> ort.GraphOptimizationLevel:
    _map = {
        "none":     ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return _map.get((level or "all").lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)


def _onnx_dtype_to_np(dtype_str: str) -> np.dtype:
    _map = {
        "tensor(float)":   np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)":  np.float64,
        "tensor(int8)":    np.int8,
        "tensor(uint8)":   np.uint8,
        "tensor(int16)":   np.int16,
        "tensor(int32)":   np.int32,
        "tensor(int64)":   np.int64,
        "tensor(bool)":    np.bool_,
    }
    return _map.get(dtype_str, np.float32)


# ---------------------------------------------------------------------------
# ONNX model loader — explicit model_dir + precision
# ---------------------------------------------------------------------------


def _load_onnx(
    component: str,
    model_dir: str,
    precision: str,
) -> bytes:
    """
    Load and serialise an ONNX model from ``model_dir``.

    The filename is constructed as::

        {model_dir}/{component}_11_{precision}.onnx   # e.g. medium_encoder_11_int8.onnx
        {model_dir}/{component}_11.onnx               # when precision="none"

    Parameters
    ----------
    component:  Model component name, e.g. ``"medium_encoder"`` or
                ``"medium_decoder"``.
    model_dir:  Directory that contains the ONNX files.  Must exist.
    precision:  One of ``"int8"``, ``"fp16"``, ``"fp32"``, ``"none"``.
                ``"none"`` means no precision suffix in the filename.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model_dir '{model_dir}' does not exist or is not a directory."
        )

    suffix = "" if precision == "none" else f"_{precision}"
    filename = f"{component}_11{suffix}.onnx"
    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ONNX file not found: {path}\n"
            f"  model_dir : {model_dir}\n"
            f"  component : {component}\n"
            f"  precision : {precision}  →  filename: {filename}\n"
            f"  Available precisions: {AVAILABLE_PRECISIONS}"
        )

    logger.info("Loading: %s", path)
    serializer: ProtoSerializer = onnx._get_serializer(fmt="protobuf")
    graph = onnx.load(path)
    return serializer.serialize_proto(proto=graph)


# ---------------------------------------------------------------------------
# Audio utilities (self-contained — no whisper package needed for loading)
# ---------------------------------------------------------------------------


def _load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file as float32 mono at ``sr`` Hz."""
    try:
        import soundfile as sf
        wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
            except ImportError:
                from scipy.signal import resample
                wav = resample(wav, int(len(wav) * sr / file_sr)).astype(np.float32)
        return wav.astype(np.float32)
    except Exception:
        # ffmpeg fallback
        import subprocess
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-threads", "0", "-i", path,
             "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", f"-ar", str(sr), "-"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def _pad_or_trim(arr: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    if arr.shape[-1] > length:
        arr = arr[..., :length]
    if arr.shape[-1] < length:
        pad = [(0, 0)] * (arr.ndim - 1) + [(0, length - arr.shape[-1])]
        arr = np.pad(arr, pad)
    return arr


# NOTE: the log-Mel spectrogram is computed by ``whisper.audio.log_mel_spectrogram``
# (called from ``whisper.transcribe``) using the official mel_filters.npz.


# ---------------------------------------------------------------------------
# Duck-type model proxy — lets whisper.transcribe() drive our ORT sessions
# ---------------------------------------------------------------------------


class _WhisperModelProxy:
    """
    Shim that satisfies the interface expected by ``whisper.transcribe.transcribe()``
    and ``whisper.decoding.decode()`` without loading the original PyTorch model.

    All heavy computation (encode + decode step) is delegated back to the
    live ONNX ``InferenceSession`` objects held in ``_pipeline``.
    """

    class _DimObj:
        pass

    def __init__(
        self,
        pipeline: "WhisperOnnxPipeline",
        model_name: str,
        dims: dict,
    ):
        self._pipeline = pipeline
        self.model_name = model_name

        # Per-call accumulators (the proxy is created fresh for every transcribe()
        # call, so these are safe under concurrent calls on a shared pipeline).
        self.encoder_s: float = 0.0
        # Decoder time is split like the Qwen engine does:
        #   prefill_s – decoder passes at KV offset 0: the prompt (SOT/language/task
        #               tokens) that fills the KV cache, plus the language-id pass.
        #   decode_s  – every later single-token step of the autoregressive loop.
        self.prefill_s: float = 0.0
        self.decoder_s: float = 0.0     # token-generation steps only (excludes prefill)
        self.decoder_calls: int = 0
        self.prefill_calls: int = 0

        d = self._DimObj()
        for k, v in dims.items():
            setattr(d, k, v)
        self.dims = d
        self.is_multilingual = dims["n_vocab"] == 51865

    # ── whisper.decoding / transcribe API ───────────────────────────────────

    def embed_audio(self, mel: np.ndarray) -> np.ndarray:
        return self.encoder(mel)

    def encoder(self, mel: np.ndarray) -> np.ndarray:
        """Callable encoder shim for whisper.decoding._get_audio_features."""
        t0 = time.perf_counter()
        out = self._pipeline._encode(mel)
        self.encoder_s += time.perf_counter() - t0
        return out

    def decoder(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
    ):
        """Callable decoder shim for whisper.decoding.PyTorchInference.logits."""
        t0 = time.perf_counter()
        out = self._pipeline._decode_step(tokens, audio_features, kv_cache, offset)
        dt = time.perf_counter() - t0
        if offset == 0:
            self.prefill_s += dt
            self.prefill_calls += 1
        else:
            self.decoder_s += dt
            self.decoder_calls += 1
        return out

    def logits(self, tokens: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        kv = self._pipeline._new_kv_cache(tokens.shape[0], tokens.shape[-1])
        logits, _ = self.decoder(tokens, audio_features, kv, 0)
        return logits

    def __call__(self, mel: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        return self.logits(tokens, self.embed_audio(mel))

    def new_kv_cache(self, n_group: int, length: int) -> np.ndarray:
        return self._pipeline._new_kv_cache(n_group, length)

    # Bound via property so whisper.decoding can call model.detect_language(...)
    @property
    def detect_language(self):
        _ensure_whisper_in_path()
        from whisper.decoding import detect_language as _dl
        proxy = self

        def _detect(mel, tokenizer=None):
            return _dl(proxy, mel, tokenizer)
        return _detect

    @property
    def decode(self):
        _ensure_whisper_in_path()
        from whisper.decoding import decode as _decode
        proxy = self

        def _do_decode(segment, options):
            return _decode(proxy, segment, options)
        return _do_decode


def _ensure_whisper_in_path():
    """Add ``src/`` to sys.path so the vendored ``whisper`` package is importable."""
    # The engine lives at <repo>/src/engines/whisper_engine.py; the vendored
    # ``whisper`` package lives at <repo>/src/whisper, i.e. two levels up.
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# ---------------------------------------------------------------------------
# Low-level pipeline
# ---------------------------------------------------------------------------


class WhisperOnnxPipeline:
    """
    Low-level Whisper inference pipeline — ONNX encoder + decoder sessions.

    Parameters
    ----------
    model_name:  One of ``AVAILABLE_MODELS`` (e.g. ``"medium"``).
    model_dir:   Directory that contains the ONNX files.  Required.
    precision:   Precision suffix of the model files:
                 ``"int8"`` → ``medium_encoder_11_int8.onnx``
                 ``"fp16"`` → ``medium_encoder_11_fp16.onnx``
                 ``"fp32"`` → ``medium_encoder_11_fp32.onnx``
                 ``"none"`` → ``medium_encoder_11.onnx``
    num_threads: ORT intra-op thread count (0 = ``logical_cpus - 1``).
    graph_opt:   ORT graph optimisation level
                 (``"none" | "basic" | "extended" | "all"``).
    """

    def __init__(
        self,
        model_name: str = "small",
        model_dir: str = ".",
        precision: str = "int8",
        num_threads: int = 0,
        graph_opt: str = "all",
    ):
        if model_name not in _DIMS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {AVAILABLE_MODELS}"
            )
        if precision not in AVAILABLE_PRECISIONS:
            raise ValueError(
                f"Unknown precision '{precision}'. Available: {AVAILABLE_PRECISIONS}"
            )

        self.model_name = model_name
        self.dims = _DIMS[model_name]
        self.is_multilingual = self.dims["n_vocab"] == 51865

        # ── ORT session options ──────────────────────────────────────────────
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = _ort_graph_opt(graph_opt)
        n_threads = num_threads if num_threads > 0 else max(1, psutil.cpu_count(logical=True) - 1)
        sess_opts.intra_op_num_threads = n_threads

        providers = ["CPUExecutionProvider"]

        # ── Encoder ─────────────────────────────────────────────────────────
        logger.info("Loading %s encoder (%s) …", model_name, precision)
        enc_bytes = _load_onnx(f"{model_name}_encoder", model_dir, precision)
        self._encoder = ort.InferenceSession(
            path_or_bytes=enc_bytes,
            sess_options=sess_opts,
            providers=providers,
        )
        self._enc_dtypes = {
            inp.name: _onnx_dtype_to_np(inp.type)
            for inp in self._encoder.get_inputs()
        }

        # ── Decoder ─────────────────────────────────────────────────────────
        logger.info("Loading %s decoder (%s) …", model_name, precision)
        dec_bytes = _load_onnx(f"{model_name}_decoder", model_dir, precision)
        self._decoder = ort.InferenceSession(
            path_or_bytes=dec_bytes,
            sess_options=sess_opts,
            providers=providers,
        )
        self._dec_dtypes = {
            inp.name: _onnx_dtype_to_np(inp.type)
            for inp in self._decoder.get_inputs()
        }

        # Import the decoding package and build the tokenizer now, so the first
        # real request does not pay ~2 s of import / tokenizer-load time.
        self._warm_decoding_stack()

        logger.info("Pipeline ready  model=%s  precision=%s.", model_name, precision)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warm_decoding_stack(self) -> None:
        _ensure_whisper_in_path()
        try:
            import whisper.transcribe  # noqa: F401  (pulls in transformers' tokenizer)
            from whisper.tokenizer import get_tokenizer
            get_tokenizer(self.is_multilingual)
        except ImportError:
            pass  # reported with a clear message on the first transcribe() call

    def _new_kv_cache(self, n_group: int = 1, length: int = 1) -> np.ndarray:
        layers = _KV_LAYERS[self.model_name]
        state  = self.dims["n_audio_state"]
        return np.zeros([layers, n_group, length, state], dtype=np.float32)

    def _encode(self, mel: np.ndarray) -> np.ndarray:
        """mel [1, n_mels, T] → audio_features [1, n_audio_ctx, n_audio_state]"""
        return self._encoder.run(
            ["output"],
            {"mel": mel.astype(self._enc_dtypes["mel"])},
        )[0]

    def _decode_step(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
    ):
        """One decoder forward pass → (logits float32, new_kv_cache float32)."""
        logits, new_kv, _ = self._decoder.run(
            ["logits", "output_kv_cache", "cross_attention_qks"],
            {
                "tokens":         tokens.astype(self._dec_dtypes["tokens"]),
                "audio_features": audio_features.astype(self._dec_dtypes["audio_features"]),
                "kv_cache":       kv_cache.astype(self._dec_dtypes["kv_cache"]),
                "offset":         np.array([offset], dtype=self._dec_dtypes["offset"]),
            },
        )
        return logits.astype(np.float32), new_kv.astype(np.float32)

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: Optional[int] = DEFAULT_BEAM_SIZE,
        temperature: Union[float, tuple] = DEFAULT_TEMPERATURE,
        best_of: Optional[int] = DEFAULT_BEST_OF,
        condition_on_previous_text: bool = True,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        verbose: bool = False,
        fallback: bool = True,
    ) -> dict:
        """
        Transcribe audio using the Whisper ONNX model.

        ``fallback=False`` disables the temperature-fallback retries (a single
        pass at the given temperature); ``beam_size`` of ``None`` or ``1`` means
        greedy decoding. Both are used by the streaming path to bound latency.

        Delegates segment-level logic (timestamp parsing, beam search,
        temperature fallback) to ``whisper.transcribe.transcribe()`` from
        the whisper-onnx-cpu package, while routing all ONNX inference
        through our live ``InferenceSession`` objects.

        Returns
        -------
        dict:
            text      – full transcript string
            segments  – list of timed segment dicts (start, end, text, …)
            language  – detected or forced language code
            timing    – wall-clock breakdown + RTF
        """
        t_total_start = time.time()

        # ── Load audio ──────────────────────────────────────────────────────
        t0 = time.time()
        if isinstance(audio, (str, Path)):
            wav = _load_audio(str(audio))
        elif isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        audio_duration_s = len(wav) / SAMPLE_RATE
        t_load = time.time() - t0

        # Too short to contain a mel frame (< 0.1 s): nothing to transcribe.
        if audio_duration_s < 0.1:
            return {
                "text": "", "segments": [], "language": language or "",
                "timing": {
                    "load_s": t_load, "mel_s": 0.0, "encoder_s": 0.0, "prefill_s": 0.0,
                    "decode_s": 0.0, "other_s": 0.0, "tokens_generated": 0,
                    "total_s": time.time() - t_total_start,
                    "audio_duration_s": audio_duration_s, "rtf": 0.0, "segments": 0,
                },
            }

        # ── Build model proxy and run whisper.transcribe() ──────────────────
        _ensure_whisper_in_path()
        try:
            from whisper.audio import log_mel_spectrogram
            from whisper.transcribe import transcribe as _whisper_transcribe
        except ImportError as exc:
            raise ImportError(
                "Cannot import whisper.transcribe. "
                "The vendored package src/whisper/ (from whisper-onnx-cpu) "
                "must be present next to src/engines/."
            ) from exc

        proxy = _WhisperModelProxy(
            pipeline=self,
            model_name=self.model_name,
            dims=self.dims,
        )

        # ── Mel spectrogram (timed here, then handed to whisper.transcribe) ──
        t0 = time.time()
        mel = log_mel_spectrogram(wav, self.dims["n_mels"])
        t_mel = time.time() - t0

        if beam_size is not None and beam_size <= 1:
            beam_size = None          # greedy

        # Build temperature tuple (mirrors cli() logic in transcribe.py)
        if isinstance(temperature, (int, float)):
            if temperature == 0.0:
                temp_arg = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if fallback else (0.0,)
            else:
                temp_arg = (float(temperature),)
        else:
            temp_arg = tuple(temperature)

        t0 = time.time()
        result = _whisper_transcribe(
            model=proxy,
            audio=wav,
            mel=mel,
            verbose=True if verbose else None,   # None = silent (no tqdm bar)
            temperature=temp_arg,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
        )
        t_transcribe = time.time() - t0
        t_total  = time.time() - t_total_start

        n_tokens = sum(len(seg.get("tokens", [])) for seg in result.get("segments", []))
        result["timing"] = {
            "load_s":           t_load,
            "mel_s":            t_mel,
            "encoder_s":        proxy.encoder_s,            # sum of encoder ONNX runs
            "prefill_s":        proxy.prefill_s,            # decoder prompt pass(es) at KV offset 0 (+ language id)
            "decode_s":         proxy.decoder_s,            # per-token decoder steps after the prefill
            "other_s":          max(0.0, t_transcribe - proxy.encoder_s - proxy.prefill_s - proxy.decoder_s),  # beam search, tokenizer, python
            "tokens_generated": n_tokens,
            "total_s":          t_total,
            "audio_duration_s": audio_duration_s,
            "rtf":              t_total / audio_duration_s if audio_duration_s > 0 else 0.0,
            "segments":         len(result.get("segments", [])),
        }
        return result


# ---------------------------------------------------------------------------
# High-level engine (mirrors ONNXQwen3ASR interface)
# ---------------------------------------------------------------------------


class WhisperOnnxEngine:
    """
    High-level ONNX Whisper ASR engine, mirroring the ``ONNXQwen3ASR`` interface.

    Can be instantiated directly or via :meth:`from_config` /
    :meth:`from_config_path` for YAML/JSON-driven configuration.

    Parameters
    ----------
    model_name:  Whisper model size — one of ``AVAILABLE_MODELS``.
    model_dir:   Directory that contains the ONNX model files.
    precision:   Precision of the ONNX files:

                 - ``"int8"``  → ``medium_encoder_11_int8.onnx``
                 - ``"fp16"``  → ``medium_encoder_11_fp16.onnx``
                 - ``"fp32"``  → ``medium_encoder_11_fp32.onnx``
                 - ``"none"``  → ``medium_encoder_11.onnx`` (no suffix)

    num_threads: ORT intra-op thread count (0 = auto).
    language:    Default language code (e.g. ``"en"``); ``None`` = auto-detect.
    graph_opt:   ORT graph optimisation level string.
    beam_size:   Beam search width (instance default, overridable per call).
    temperature: Sampling temperature (0.0 = greedy / beam).

    Example
    -------
    >>> engine = WhisperOnnxEngine(
    ...     model_name="medium",
    ...     model_dir="whisper_int8",
    ...     precision="int8",
    ...     language="en",
    ... )
    >>> result = engine.transcribe("audio.mp4")
    >>> print(result["text"])
    """

    def __init__(
        self,
        model_name: str = "small",
        model_dir: str = ".",
        precision: str = "int8",
        num_threads: int = 0,
        language: Optional[str] = None,
        graph_opt: str = "all",
        beam_size: int = DEFAULT_BEAM_SIZE,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.language    = language
        self.beam_size   = beam_size
        self.temperature = temperature

        self.pipeline = WhisperOnnxPipeline(
            model_name  = model_name,
            model_dir   = model_dir,
            precision   = precision,
            num_threads = num_threads,
            graph_opt   = graph_opt,
        )

    # ------------------------------------------------------------------
    # Config-driven constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "WhisperOnnxEngine":
        """
        Build a ``WhisperOnnxEngine`` from a pre-loaded config dict.

        Expected structure (all keys optional)::

            engine:
              model_name:  medium
              model_dir:   whisper_int8
              precision:   int8          # int8 | fp16 | fp32 | none
              num_threads: 0
              language:    null
              graph_opt:   all
            inference:
              beam_size:   5
              temperature: 0.0
        """
        engine_cfg = cfg.get("engine", {})
        infer_cfg  = cfg.get("inference", {})
        engine = cls(
            model_name  = engine_cfg.get("model_name",  "small"),
            model_dir   = engine_cfg.get("model_dir",   "."),
            precision   = engine_cfg.get("precision",   "int8"),
            num_threads = engine_cfg.get("num_threads", 0),
            language    = engine_cfg.get("language",    None),
            graph_opt   = engine_cfg.get("graph_opt",   "all"),
            beam_size   = infer_cfg.get("beam_size",    DEFAULT_BEAM_SIZE),
            temperature = infer_cfg.get("temperature",  DEFAULT_TEMPERATURE),
        )
        try:
            try:
                from src.core.observe import annotate_from_config
            except ImportError:
                from core.observe import annotate_from_config
            annotate_from_config(engine, cfg, precision=engine_cfg.get("precision"),
                                 whisper_model=engine_cfg.get("model_name"))
        except Exception:
            pass
        return engine

    @classmethod
    def from_config_path(cls, config_path: str) -> "WhisperOnnxEngine":
        """Load a YAML or JSON config and build the engine."""
        path = Path(config_path)
        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
                with open(path) as f:
                    cfg = yaml.safe_load(f)
            except ImportError:
                raise ImportError("pip install pyyaml to use YAML configs")
        elif path.suffix == ".json":
            import json
            with open(path) as f:
                cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls.from_config(cfg)

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: Optional[int] = None,
        verbose: bool = False,
        fallback: bool = True,
    ) -> dict:
        """
        Transcribe an audio file or numpy waveform.

        Parameters
        ----------
        audio:     File path or float32 numpy array at 16 kHz.
        language:  Override the engine-level language setting.
        task:      ``"transcribe"`` or ``"translate"``.
        beam_size: Override beam width (``None`` = use engine default).
        verbose:   Print segment timestamps while decoding.
        fallback:  ``False`` = no temperature-fallback retries (single pass; faster,
                   used by the live-streaming path).

        Returns
        -------
        dict:
            text      – full transcript string
            segments  – list of timed segment dicts
            language  – detected / forced language code
            timing    – mel_s / encoder_s / decode_s / other_s breakdown, total_s, RTF
        """
        lang = language if language is not None else self.language
        kwargs = dict(
            audio=audio, language=lang, task=task,
            beam_size=beam_size if beam_size is not None else self.beam_size,
            temperature=self.temperature, verbose=verbose, fallback=fallback,
        )
        try:
            try:
                from src.core.observe import audio_inputs, record_asr, span, use_engine
            except ImportError:
                from core.observe import audio_inputs, record_asr, span, use_engine
            use_engine(self)
            with span("transcribe", "llm",
                      inputs=audio_inputs(audio, language=lang, beam_size=kwargs["beam_size"],
                                          fallback=fallback)) as sp:
                result = self.pipeline.transcribe(**kwargs)
                record_asr(sp, result)
                return result
        except ImportError:
            pass
        return self.pipeline.transcribe(**kwargs)

    def transcribe_stream(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> Generator[tuple, None, None]:
        """
        Streaming transcription — yields ``(segment_text: str, timing: dict | None)``.

        Each completed Whisper segment is yielded as ``(text, None)`` as soon
        as it is decoded.  After all segments, a final ``("", timing)`` sentinel
        carries the full timing dict.

        The timing dict mirrors ``ONNXQwen3ASR.transcribe_stream``:
            mel_s       – log-mel spectrogram
            encoder_s   – audio encoder runs
            prefill_s   – decoder prompt pass(es) at KV offset 0 (SOT/language/task
                          tokens that fill the KV cache, plus language detection)
            decode_s    – per-token decoder steps after the prefill
            other_s     – beam search / tokenizer / Python overhead
            tokens_generated, total_s, audio_duration_s, rtf, segments

        Parameters
        ----------
        audio:    File path or float32 numpy array at 16 kHz.
        language: Override the engine-level language setting.
        verbose:  Print timestamps to stdout during decode.
        """
        t_start = time.time()
        lang = language if language is not None else self.language

        def _stream():
            yield from self._transcribe_stream_body(audio, lang, verbose, t_start)

        try:
            try:
                from src.core.observe import audio_inputs, trace_stream, use_engine
            except ImportError:
                from core.observe import audio_inputs, trace_stream, use_engine
            use_engine(self)
            yield from trace_stream(
                "transcribe_stream", _stream(), inputs=audio_inputs(audio, language=lang),
            )
        except ImportError:
            yield from _stream()

    def _transcribe_stream_body(self, audio, lang, verbose, t_start):
        if isinstance(audio, (str, Path)):
            wav = _load_audio(str(audio))
        elif isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        audio_duration_s = len(wav) / SAMPLE_RATE

        result = self.pipeline.transcribe(
            audio       = wav,
            language    = lang,
            task        = "transcribe",
            beam_size   = self.beam_size,
            temperature = self.temperature,
            verbose     = verbose,
        )

        for seg in result.get("segments", []):
            yield seg["text"], None

        t_total = time.time() - t_start
        timing  = result.get("timing", {})
        timing["total_s"]          = t_total
        timing["audio_duration_s"] = audio_duration_s
        timing["rtf"]              = t_total / audio_duration_s if audio_duration_s > 0 else 0.0
        yield "", timing


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Whisper ONNX Engine — standalone CPU inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument(
        "--model", default="small", choices=AVAILABLE_MODELS,
        help="Whisper model size",
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory that contains the ONNX model files",
    )
    parser.add_argument(
        "--precision", default="int8", choices=AVAILABLE_PRECISIONS,
        help=(
            "Precision of the ONNX files in model_dir.  "
            "Determines the filename suffix: "
            "'int8' → {model}_encoder_11_int8.onnx | "
            "'fp16' → {model}_encoder_11_fp16.onnx | "
            "'fp32' → {model}_encoder_11_fp32.onnx | "
            "'none' → {model}_encoder_11.onnx"
        ),
    )
    parser.add_argument("--language", default=None, help="Language code (e.g. 'en'); None = auto-detect")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--beam_size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--threads", type=int, default=0, help="ORT thread count (0 = auto)")
    parser.add_argument("--verbose", action="store_true", help="Print segment timestamps")
    parser.add_argument("--stream", action="store_true", help="Streaming (segment-by-segment) output")
    args = parser.parse_args()

    engine = WhisperOnnxEngine(
        model_name  = args.model,
        model_dir   = args.model_dir,
        precision   = args.precision,
        num_threads = args.threads,
        language    = args.language,
        beam_size   = args.beam_size,
        temperature = args.temperature,
    )

    for audio_path in args.audio:
        print(f"\n{'='*60}")
        print(f"File: {audio_path}")
        print("=" * 60)

        if args.stream:
            print("Streaming transcription:")
            stream_timing = None
            for text, timing in engine.transcribe_stream(audio_path, verbose=args.verbose):
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                if timing is not None:
                    stream_timing = timing
            print()
            if stream_timing:
                t = stream_timing
                print(f"\n  RTF {t['rtf']:.2f}x | Duration {t['audio_duration_s']:.1f}s | "
                      f"Total {t['total_s']:.1f}s | Segments {t.get('segments', '?')}")
        else:
            result = engine.transcribe(audio_path, task=args.task, verbose=args.verbose)
            t = result["timing"]
            if result.get("language"):
                print(f"Language : {result['language']}")
            print(f"Text     : {result['text']}")
            print(
                f"\nTiming   : mel={t['mel_s']:.2f}s | enc={t['encoder_s']:.2f}s | dec={t['decode_s']:.2f}s | "
                f"total={t['total_s']:.2f}s | RTF={t['rtf']:.2f}x | "
                f"Segments={t['segments']}"
            )


if __name__ == "__main__":
    _cli()
