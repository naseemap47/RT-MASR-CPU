"""
Qwen3-ASR — Pure ONNX Inference Pipeline (0.6B and 1.7B).

No PyTorch dependency. Uses only ONNX Runtime + NumPy + librosa.

Architecture:
    Audio → Mel → Encoder (ONNX) → Audio Features
    Prompt tokens → Embed (numpy) → Replace audio placeholders → Decoder Init (ONNX) → Logits + KV Cache
    Greedy decode loop: Decoder Step (ONNX) → next token until EOS

Two on-disk model layouts are supported; the layout is auto-detected from the
files present in ``onnx_dir``:

  "split" (legacy, Daumee/Qwen3-ASR-0.6B-ONNX-CPU, INT8 decoder)
      encoder_conv.onnx + encoder_transformer.onnx, decoder_{init,step}[.int8].onnx,
      embed_tokens.bin (FP32, 151936 x 1024).

  "fused" (andrewleech/qwen3-asr-{0.6b,1.7b}-onnx, FP32 or INT4)
      encoder[.int4].onnx (single graph, windowed internally),
      decoder_{init,step}[.int4].onnx (+ decoder_weights[.int4].data),
      embed_tokens.bin (FP16, shape/hidden size read from config.json).
      decoder_init may take either ``input_embeds`` (v1) or
      ``input_ids + audio_features + audio_offset`` (v3); both are handled.

Precision is selected with ``quantize``: "fp32" | "int8" (split layout only) |
"int4" (fused layout only).

Usage:
    python src/engines/qwen3_onnx_engine.py audio.wav
    python src/engines/qwen3_onnx_engine.py audio.wav --config config/models/qwen3_onnx_0.6b_int4.yaml
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Literal, Generator, Union

import numpy as np
import onnxruntime as ort

from utils.audio_utils import (
    load_audio, compute_mel_spectrogram, get_mel_filters,
    get_feat_extract_output_lengths, find_silence_split_points
)
from core.config import load_config

logger = logging.getLogger("rtmasr.engines.qwen3_onnx")


def _ort_graph_opt(level_str: str) -> ort.GraphOptimizationLevel:
    """Convert the config string to an ORT GraphOptimizationLevel enum value."""
    _map = {
        "none":     ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return _map.get((level_str or "all").lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

# ── Constants ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
# N_FFT = 400
# HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100  # n_window * 2

# Special token IDs
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645      # EOS
ENDOFTEXT_ID = 151643   # EOS alt
NEWLINE_ID = 198        # '\n'

# Vocab
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024

# Language mapping dictionary for supported target languages
LANGUAGE_MAP = {
    "en": "English",
    "english": "English",
    # "Chinese" is the canonical name (the model itself reports "language Chinese"
    # when it auto-detects, and qwen_asr's supported list uses it). Mandarin is an alias.
    "zh": "Chinese",
    "cn": "Chinese",
    "mandarin": "Chinese",
    "mandarin chinese": "Chinese",
    "chinese": "Chinese",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "bahasa indonesia": "Indonesian",
    "bahasa": "Indonesian",
}


def normalize_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    lang_clean = language.strip().lower()
    return LANGUAGE_MAP.get(lang_clean, language.strip())


# ── Tokenizer (minimal, no HuggingFace dependency) ─────────────────────

class SimpleTokenizer:
    """Minimal tokenizer using tokenizers library (or HF tokenizer.json)."""

    def __init__(self, tokenizer_path: str = None):
        if tokenizer_path and Path(tokenizer_path).exists():
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Fall back to HF tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
            self._is_hf = True
            return
        self._is_hf = False

    def encode(self, text: str) -> list:
        if self._is_hf:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        if self._is_hf:
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.tokenizer.decode(ids, skip_special_tokens=True)


# ── ONNX Pipeline ──────────────────────────────────────────────────────

class OnnxAsrPipeline:
    """End-to-end ASR pipeline using only ONNX Runtime (split or fused model layout)."""

    # quantize value → filename suffix inserted before ".onnx" (e.g. decoder_init.int4.onnx)
    _SUFFIX = {"fp32": "", "none": "", "int8": ".int8", "int4": ".int4"}

    def __init__(
        self,
        onnx_dir: str = "models/qwen3-asr-onnx-0.6b-int8",
        num_threads: int = 0,
        quantize: str = "int8",
        ort_session: dict | None = None,
    ):
        """
        Args:
            onnx_dir:    Directory containing all ONNX model artefacts.
            num_threads: ORT intra-op thread count (0 = auto).
            quantize:    Model precision: "fp32" | "int8" | "int4".
                         "int8" applies to the split layout's decoder only;
                         "int4" applies to every graph of the fused layout.
                         A missing quantised file falls back to the FP32 one.
            ort_session: Optional dict from the ``ort_session`` config section:
                         { graph_optimization_level, log_severity_level, providers }
        """
        onnx_path = Path(onnx_dir)
        if not onnx_path.is_dir():
            raise FileNotFoundError(
                f"ONNX model directory '{onnx_path}' not found. "
                "Download it first: python src/utils/download_utils.py --model <name>"
            )
        ort_cfg = ort_session or {}

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = _ort_graph_opt(
            ort_cfg.get("graph_optimization_level", "all")
        )
        if num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
        sess_opts.log_severity_level = int(ort_cfg.get("log_severity_level", 3))

        providers = ort_cfg.get("providers", ["CPUExecutionProvider"])

        quantize = (quantize or "fp32").lower()
        if quantize not in self._SUFFIX:
            raise ValueError(f"Unknown quantize '{quantize}'. Expected one of {list(self._SUFFIX)}.")
        suffix = self._SUFFIX[quantize]

        def pick(name: str) -> Path:
            """Return {name}{suffix}.onnx if present, else {name}.onnx."""
            if suffix:
                quant_path = onnx_path / f"{name}{suffix}.onnx"
                if quant_path.exists():
                    return quant_path
                logger.warning("%s not found - falling back to %s.onnx (FP32)", quant_path.name, name)
            return onnx_path / f"{name}.onnx"

        def load(path: Path) -> ort.InferenceSession:
            logger.info("  %s", path.name)
            return ort.InferenceSession(str(path), sess_opts, providers=providers)

        self.layout = "split" if (onnx_path / "encoder_conv.onnx").exists() else "fused"
        logger.info("Loading ONNX models (layout: %s, precision: %s)...", self.layout, quantize.upper())

        if self.layout == "split":
            # Legacy layout: only the decoder is quantised (INT8).
            self.encoder_conv = load(onnx_path / "encoder_conv.onnx")
            self.encoder_transformer = load(onnx_path / "encoder_transformer.onnx")
            self.encoder = None
            self.decoder_init = load(pick("decoder_init"))
            self.decoder_step = load(pick("decoder_step"))
        else:
            self.encoder = load(pick("encoder"))
            self.encoder_conv = self.encoder_transformer = None
            self.decoder_init = load(pick("decoder_init"))
            self.decoder_step = load(pick("decoder_step"))

        # v3 decoder_init takes token ids + audio features; v1 takes pre-fused embeds.
        init_inputs = {i.name for i in self.decoder_init.get_inputs()}
        self._init_takes_ids = "input_ids" in init_inputs

        # Load embedding matrix (kept in its on-disk dtype; rows are cast to FP32 on lookup)
        self.embed_tokens = self._load_embeddings(onnx_path)

        # Mel filterbank
        self.mel_filters = get_mel_filters()

        # Tokenizer
        tokenizer_path = onnx_path / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = None
        self.tokenizer = SimpleTokenizer(str(tokenizer_path) if tokenizer_path else None)

        logger.info("Pipeline ready.")

    def _load_embeddings(self, onnx_path: Path) -> np.ndarray:
        """Load embed_tokens.bin as [vocab, hidden] (FP32 for split layout, per config.json otherwise)."""
        embed_path = onnx_path / "embed_tokens.bin"
        logger.info("Loading embeddings (%.0f MB)...", embed_path.stat().st_size / 1e6)

        cfg_path = onnx_path / "config.json"
        if self.layout == "split" or not cfg_path.exists():
            return np.fromfile(str(embed_path), dtype=np.float32).reshape(VOCAB_SIZE, HIDDEN_SIZE)

        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        dec = cfg.get("decoder", {})
        shape = cfg.get("embed_tokens_shape") or [
            dec.get("vocab_size", VOCAB_SIZE), dec.get("hidden_size", HIDDEN_SIZE)
        ]
        dtype = np.float16 if cfg.get("embed_tokens_dtype") == "float16" else np.float32
        return np.fromfile(str(embed_path), dtype=dtype).reshape(shape)

    def _embed_rows(self, ids) -> np.ndarray:
        """Look up embedding rows and return them as a fresh FP32 array."""
        return self.embed_tokens[ids].astype(np.float32)

    def _compute_mel(self, wav: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram [n_mels, frames] matching the active encoder layout."""
        mel = compute_mel_spectrogram(wav, self.mel_filters)
        if self.layout == "fused":
            # The fused encoder was exported against WhisperFeatureExtractor, which drops the
            # last STFT frame.
            mel = mel[:, :-1]
        return mel

    def _encode_audio(self, mel: np.ndarray, mel_len: int) -> np.ndarray:
        """Run the encoder: mel → audio features [N, hidden]."""
        if self.layout == "fused":
            features = self.encoder.run(
                ["audio_features"], {"mel": np.ascontiguousarray(mel[np.newaxis, :, :mel_len])}
            )[0]
            return features[0]  # [1, N, hidden] → [N, hidden]
        return self._encode_audio_split(mel, mel_len)

    def _encode_audio_split(self, mel: np.ndarray, mel_len: int) -> np.ndarray:
        """Legacy encoder (conv + transformer graphs): mel → audio features [N, 1024]."""
        mel_valid = mel[:, :mel_len]
        chunk_num = int(np.ceil(mel_len / CHUNK_SIZE))

        chunk_lengths = []
        for i in range(chunk_num):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, mel_len)
            chunk_lengths.append(end - start)

        # Pad chunks
        max_chunk_len = max(chunk_lengths)
        padded = np.zeros((chunk_num, 1, N_MELS, max_chunk_len), dtype=np.float32)
        start = 0
        for i, cl in enumerate(chunk_lengths):
            padded[i, 0, :, :cl] = mel_valid[:, start:start + cl]
            start += cl

        # Conv output lengths
        lens_after_cnn = get_feat_extract_output_lengths(np.array(chunk_lengths))

        # Conv block
        conv_out = self.encoder_conv.run(None, {"padded_mel_chunks": padded})[0]

        # Pack features (remove padding)
        features = []
        for i, l in enumerate(lens_after_cnn):
            features.append(conv_out[i, :l, :])
        hidden_states = np.concatenate(features, axis=0)

        # Transformer block (all-to-all attention)
        total_tokens = hidden_states.shape[0]
        attn_mask = np.zeros((1, 1, total_tokens, total_tokens), dtype=np.float32)
        encoder_output = self.encoder_transformer.run(None, {
            "hidden_states": hidden_states,
            "attention_mask": attn_mask,
        })[0]

        return encoder_output  # [N, 1024]

    def _build_prompt_ids(self, num_audio_tokens: int, language: Optional[str] = None) -> list:
        """Build prompt token IDs with audio placeholders."""
        # <|im_start|>system\n<|im_end|>\n
        ids = [IM_START_ID] + self.tokenizer.encode("system") + [NEWLINE_ID, IM_END_ID, NEWLINE_ID]
        # <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
        ids += [IM_START_ID] + self.tokenizer.encode("user") + [NEWLINE_ID]
        ids += [AUDIO_START_ID] + [AUDIO_PAD_ID] * num_audio_tokens + [AUDIO_END_ID]
        ids += [IM_END_ID, NEWLINE_ID]
        # <|im_start|>assistant\n
        ids += [IM_START_ID] + self.tokenizer.encode("assistant") + [NEWLINE_ID]
        lang = normalize_language(language)
        if lang:
            lang_tokens = self.tokenizer.encode(f"language {lang}<asr_text>")
            ids += lang_tokens
        return ids

    def _prepare_decoder_inputs(self, token_ids: list, audio_features: np.ndarray) -> dict:
        """Build the decoder_init feed dict for the detected decoder format (v1 or v3)."""
        ids_array = np.asarray(token_ids, dtype=np.int64)
        audio_positions = np.where(ids_array == AUDIO_PAD_ID)[0]
        if len(audio_positions) != audio_features.shape[0]:
            raise ValueError(
                f"Audio token count mismatch: {len(audio_positions)} vs {audio_features.shape[0]}"
            )
        position_ids = np.arange(len(ids_array), dtype=np.int64)[np.newaxis, :]
        audio_features = audio_features.astype(np.float32, copy=False)

        if self._init_takes_ids:
            # v3: the graph embeds the tokens itself and splices in the audio features.
            return {
                "input_ids": ids_array[np.newaxis, :],
                "position_ids": position_ids,
                "audio_features": audio_features[np.newaxis, :, :],
                "audio_offset": np.array([audio_positions[0]], dtype=np.int64),
            }

        # v1: fuse embeddings and audio features here.
        embeds = self._embed_rows(ids_array)  # [seq_len, hidden]
        embeds[audio_positions] = audio_features
        return {"input_embeds": embeds[np.newaxis, :, :], "position_ids": position_ids}

    def _prefill(self, feeds: dict):
        """Run decoder_init → (logits, present_keys, present_values)."""
        logits, present_keys, present_values = self.decoder_init.run(
            ["logits", "present_keys", "present_values"], feeds
        )
        return logits, present_keys, present_values

    def _step(self, token: int, pos: int, past_keys: np.ndarray, past_values: np.ndarray):
        """Run one decoder_step → (logits, present_keys, present_values)."""
        return self.decoder_step.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": self._embed_rows(token)[np.newaxis, np.newaxis, :],
                "position_ids": np.array([[pos]], dtype=np.int64),
                "past_keys": past_keys,
                "past_values": past_values,
            },
        )

    def _transcribe_chunk(
        self,
        wav: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        """Transcribe a single audio chunk (≤45s recommended)."""
        t0 = time.time()
        mel = self._compute_mel(wav)
        mel_len = mel.shape[1]
        t_mel = time.time() - t0

        t0 = time.time()
        audio_features = self._encode_audio(mel, mel_len) if mel_len > 0 else np.zeros((0, 0), np.float32)
        num_audio_tokens = audio_features.shape[0]
        t_encoder = time.time() - t0

        if num_audio_tokens == 0:  # audio too short to yield any encoder frame
            return {
                "text": "", "language": language or "", "raw_output": "",
                "timing": {
                    "mel_s": t_mel, "encoder_s": t_encoder, "prepare_s": 0.0,
                    "prefill_s": 0.0, "decode_s": 0.0, "tokens_generated": 0,
                },
            }

        t0 = time.time()
        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        feeds = self._prepare_decoder_inputs(token_ids, audio_features)
        seq_len = len(token_ids)
        t_prepare = time.time() - t0

        t0 = time.time()
        logits, present_keys, present_values = self._prefill(feeds)
        t_prefill = time.time() - t0

        t0 = time.time()
        next_token = int(np.argmax(logits[0, -1, :]))
        generated = [next_token]
        cur_pos = seq_len

        for _ in range(max_new_tokens - 1):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            logits, present_keys, present_values = self._step(
                next_token, cur_pos, present_keys, present_values)

            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
            cur_pos += 1

        if generated and generated[-1] in (IM_END_ID, ENDOFTEXT_ID):
            generated = generated[:-1]

        raw_text = self.tokenizer.decode(generated)
        t_decode = time.time() - t0

        parsed_lang = ""
        parsed_text = raw_text
        if "language " in raw_text and "<asr_text>" in raw_text:
            parts = raw_text.split("<asr_text>", 1)
            lang_part = parts[0]
            if lang_part.startswith("language "):
                parsed_lang = lang_part[len("language "):]
            parsed_text = parts[1] if len(parts) > 1 else ""
        elif language:
            parsed_lang = language
            parsed_text = raw_text

        return {
            "text": parsed_text,
            "language": parsed_lang,
            "raw_output": raw_text,
            "timing": {
                "mel_s": t_mel,
                "encoder_s": t_encoder,
                "prepare_s": t_prepare,
                "prefill_s": t_prefill,
                "decode_s": t_decode,
                "tokens_generated": len(generated),
            },
        }

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
        chunk_sec: int = 30,
    ) -> dict:
        """Transcribe an audio file. Long audio is automatically split at silence."""
        t_total_start = time.time()

        wav = load_audio(audio_path)
        audio_duration = len(wav) / SAMPLE_RATE

        # Split long audio at silence boundaries
        split_points = find_silence_split_points(wav, target_sec=chunk_sec)

        if not split_points:
            # Short audio — single pass
            result = self._transcribe_chunk(wav, language, max_new_tokens)
            t_total = time.time() - t_total_start
            result["timing"]["total_s"] = t_total
            result["timing"]["audio_duration_s"] = audio_duration
            result["timing"]["rtf"] = t_total / audio_duration
            result["timing"]["sub_chunks"] = 1
            return result

        # Long audio — VAD chunking
        boundaries = [0] + split_points + [len(wav)]
        num_chunks = len(boundaries) - 1
        logger.info("  Audio %.1fs → %s sub-chunks (split at silence)", audio_duration, num_chunks)

        texts = []
        total_tokens = 0
        detected_lang = language or ""

        for i in range(num_chunks):
            chunk_wav = wav[boundaries[i]:boundaries[i + 1]]
            chunk_dur = len(chunk_wav) / SAMPLE_RATE
            t0 = time.time()

            chunk_result = self._transcribe_chunk(chunk_wav, language, max_new_tokens)

            chunk_rtf = (time.time() - t0) / chunk_dur
            chunk_chars = len(chunk_result["text"])
            logger.info("    Sub-chunk %s/%s (%.1fs): %s chars (RTF=%.2f)",
                        i + 1, num_chunks, chunk_dur, chunk_chars, chunk_rtf)

            texts.append(chunk_result["text"].strip())
            total_tokens += chunk_result["timing"]["tokens_generated"]
            if not detected_lang and chunk_result["language"]:
                detected_lang = chunk_result["language"]

        t_total = time.time() - t_total_start
        full_text = " ".join(t for t in texts if t)

        return {
            "text": full_text,
            "language": detected_lang,
            "raw_output": full_text,
            "timing": {
                "total_s": t_total,
                "audio_duration_s": audio_duration,
                "rtf": t_total / audio_duration,
                "tokens_generated": total_tokens,
                "sub_chunks": num_chunks,
            },
        }

    def transcribe_stream(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Generator[tuple, None, None]:
        """
        Stream transcription text deltas in real-time as the model decodes tokens.

        Yields ``(delta: str, timing: dict | None)`` tuples:
          - During decoding every non-empty text chunk is yielded as ``(delta, None)``.
          - After the decode loop completes, a final ``("", timing)`` sentinel is
            yielded so callers can access full per-stage timing without a second pass.

        The ``timing`` dict mirrors ``_transcribe_chunk`` and ``transcribe``:
            mel_s            – mel-spectrogram computation time (s)
            encoder_s        – encoder_conv + encoder_transformer time (s)
            prepare_s        – prompt-build + embed-fuse time (s)
            prefill_s        – decoder_init (KV-cache fill) time (s)
            decode_s         – greedy token loop time (s)
            tokens_generated – number of subword tokens decoded
            total_s          – end-to-end time for this call (s)
            audio_duration_s – length of the input waveform (s)
            rtf              – total_s / audio_duration_s

        Args:
            audio: Path to audio file or float32 waveform numpy array (16 kHz).
            language: Optional target language tag.
            max_new_tokens: Maximum number of tokens to generate.
        """
        t_total_start = time.time()

        # ── Audio loading ──────────────────────────────────────────────────
        if isinstance(audio, (str, Path)):
            wav = load_audio(str(audio))
        elif isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}. Expected file path or numpy array.")

        audio_duration_s = len(wav) / SAMPLE_RATE

        # ── Mel spectrogram ────────────────────────────────────────────────
        t0 = time.time()
        mel = self._compute_mel(wav)
        mel_len = mel.shape[1]
        t_mel = time.time() - t0

        # ── Encoder ────────────────────────────────────────────────────────
        t0 = time.time()
        audio_features = self._encode_audio(mel, mel_len) if mel_len > 0 else np.zeros((0, 0), np.float32)
        num_audio_tokens = audio_features.shape[0]
        t_encoder = time.time() - t0

        if num_audio_tokens == 0:  # audio too short to yield any encoder frame
            t_total = time.time() - t_total_start
            yield "", {
                "mel_s": t_mel, "encoder_s": t_encoder, "prepare_s": 0.0,
                "prefill_s": 0.0, "decode_s": 0.0, "tokens_generated": 0,
                "total_s": t_total, "audio_duration_s": audio_duration_s,
                "rtf": t_total / audio_duration_s if audio_duration_s > 0 else 0.0,
            }
            return

        # ── Prompt prep (+ embedding fuse for v1 decoders) ─────────────────
        t0 = time.time()
        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        feeds = self._prepare_decoder_inputs(token_ids, audio_features)
        seq_len = len(token_ids)
        t_prepare = time.time() - t0

        # ── Decoder prefill (KV-cache init) ────────────────────────────────
        t0 = time.time()
        logits, present_keys, present_values = self._prefill(feeds)
        t_prefill = time.time() - t0

        # ── Greedy decode loop ─────────────────────────────────────────────
        t_decode_start = time.time()
        next_token = int(np.argmax(logits[0, -1, :]))
        cur_pos = seq_len
        generated = []
        printed_text = ""

        for _ in range(max_new_tokens):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            generated.append(next_token)

            raw_text = self.tokenizer.decode(generated)
            if language is not None:
                asr_text = raw_text
            elif "<asr_text>" in raw_text:
                asr_text = raw_text.split("<asr_text>", 1)[1]
            else:
                asr_text = ""

            if len(asr_text) > len(printed_text):
                delta = asr_text[len(printed_text):]
                yield delta, None  # text delta; timing not yet available
                printed_text = asr_text

            logits, present_keys, present_values = self._step(
                next_token, cur_pos, present_keys, present_values)

            next_token = int(np.argmax(logits[0, -1, :]))
            cur_pos += 1

        t_decode = time.time() - t_decode_start
        t_total  = time.time() - t_total_start

        # ── Final sentinel: empty delta + full timing dict ─────────────────
        timing = {
            "mel_s":            t_mel,
            "encoder_s":        t_encoder,
            "prepare_s":        t_prepare,
            "prefill_s":        t_prefill,
            "decode_s":         t_decode,
            "tokens_generated": len(generated),
            "total_s":          t_total,
            "audio_duration_s": audio_duration_s,
            "rtf":              t_total / audio_duration_s if audio_duration_s > 0 else 0.0,
        }
        yield "", timing


class ONNXQwen3ASR:
    """
    ONNX Runtime ASR engine for Qwen3-ASR (0.6B / 1.7B; FP32, INT8 or INT4).

    Can be instantiated directly with keyword arguments or via
    :meth:`from_config` / :meth:`from_config_path` to load all settings
    from a model YAML file.
    """

    def __init__(
        self,
        onnx_dir: str = "models/qwen3-asr-onnx-0.6b-int8",
        num_threads: int = 0,
        quantize: Literal["int8", "int4", "fp32"] = "int8",
        language: Optional[str] = None,
        ort_session: dict | None = None,
        # inference defaults (read from config, overridable per-call)
        max_new_tokens: int = 512,
        chunk_sec: int = 30,
    ):
        self.pipeline = OnnxAsrPipeline(
            onnx_dir=onnx_dir,
            num_threads=num_threads,
            quantize=quantize,
            ort_session=ort_session,
        )
        self.language = normalize_language(language)
        self._default_max_new_tokens = max_new_tokens
        self._default_chunk_sec = chunk_sec

    # ------------------------------------------------------------------
    # Config-driven constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "ONNXQwen3ASR":
        """
        Build an ONNXQwen3ASR instance from a pre-loaded model config dict
        (the contents of e.g. config/models/qwen3_onnx_0.6b_int8.yaml).

        Args:
            cfg: Dict loaded from the per-model YAML.

        Returns:
            Configured ONNXQwen3ASR instance.
        """
        engine_cfg   = cfg.get("engine",    {})
        infer_cfg    = cfg.get("inference", {})
        ort_cfg      = cfg.get("ort_session", {})

        return cls(
            onnx_dir       = engine_cfg.get("onnx_dir",     "models/qwen3-asr-onnx-0.6b-int8"),
            num_threads    = engine_cfg.get("num_threads",  0),
            quantize       = engine_cfg.get("quantize",     "int8"),
            language       = engine_cfg.get("language",     None),
            ort_session    = ort_cfg or None,
            max_new_tokens = infer_cfg.get("max_new_tokens", 512),
            chunk_sec      = infer_cfg.get("chunk_sec",      30),
        )

    @classmethod
    def from_config_path(cls, config_path: str) -> "ONNXQwen3ASR":
        """
        Load a model YAML file and construct the engine.

        Args:
            config_path: Path to the per-model YAML,
                         e.g. ``"config/models/qwen3_onnx_0.6b_int8.yaml"``.

        Returns:
            Configured ONNXQwen3ASR instance.
        """
        cfg = load_config(config_path)
        return cls.from_config(cfg)

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int | None = None,
        chunk_sec: int | None = None,
        language: Optional[str] = None,
    ) -> dict:
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
        """Stream real-time transcription for an audio file path or numpy array.

        Yields ``(delta: str, timing: dict | None)`` — see
        ``OnnxAsrPipeline.transcribe_stream`` for the full contract.
        """
        lang = normalize_language(language) if language is not None else self.language
        yield from self.pipeline.transcribe_stream(
            audio,
            lang,
            max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens,
        )


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Qwen3-ASR Pure ONNX Inference")
    parser.add_argument("audio", nargs="+", help="Audio file(s)")
    parser.add_argument("--config", default="config/models/qwen3_onnx_0.6b_int8.yaml",
                        help="Per-model YAML (default: config/models/qwen3_onnx_0.6b_int8.yaml)")
    parser.add_argument("--language", default=None, help="Force a language, e.g. English")
    parser.add_argument("--onnx-dir", default=None, help="Override engine.onnx_dir from the config")
    parser.add_argument("--quantize", default=None, choices=["fp32", "int8", "int4"],
                        help="Override engine.quantize from the config")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--stream", action="store_true", help="Print text deltas as they are decoded")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.onnx_dir:
        cfg.setdefault("engine", {})["onnx_dir"] = args.onnx_dir
    if args.quantize:
        cfg.setdefault("engine", {})["quantize"] = args.quantize
    engine = ONNXQwen3ASR.from_config(cfg)

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"File not found: {audio_path}", file=sys.stderr)
            continue

        if args.stream:
            print(f"\n[{audio_path}] ", end="", flush=True)
            t = None
            for delta, timing in engine.transcribe_stream(
                audio_path, language=args.language, max_new_tokens=args.max_new_tokens
            ):
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                if timing is not None:
                    t = timing
            print()
        else:
            result = engine.transcribe(
                audio_path, language=args.language, max_new_tokens=args.max_new_tokens
            )
            t = result["timing"]
            print(f"\n[{audio_path}]")
            if result["language"]:
                print(f"  Language: {result['language']}")
            print(f"  {result['text']}")

        if t:
            print(f"  ({t['audio_duration_s']:.1f}s audio, {t['total_s']:.2f}s, RTF {t['rtf']:.2f}x, "
                  f"{t['tokens_generated']} tokens)")


if __name__ == "__main__":
    main()
