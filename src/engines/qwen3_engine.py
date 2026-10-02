"""
Qwen3-ASR-0.6B — Pure ONNX Inference Pipeline.

No PyTorch dependency. Uses only ONNX Runtime + NumPy + librosa.

Architecture:
    Audio → Mel → Encoder (ONNX) → Audio Features
    Prompt tokens → Embed (numpy) → Replace audio placeholders → Decoder Init (ONNX) → Logits + KV Cache
    Greedy decode loop: Decoder Step (ONNX) → next token until EOS

Usage:
    python onnx_inference.py audio.wav
    python onnx_inference.py audio1.wav audio2.wav --language Korean
"""

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
    """End-to-end ASR pipeline using only ONNX Runtime."""

    def __init__(self, onnx_dir: str = "models/qwen3-asr-onnx", num_threads: int = 0,
                 quantize: str = "int8"):
        onnx_path = Path(onnx_dir)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
        sess_opts.log_severity_level = 3  # Suppress warnings

        # Choose decoder model files based on quantization
        if quantize == "int8" and (onnx_path / "decoder_init.int8.onnx").exists():
            decoder_init_path = "decoder_init.int8.onnx"
            decoder_step_path = "decoder_step.int8.onnx"
            print(f"Loading ONNX models (decoder: INT8)...")
        else:
            decoder_init_path = "decoder_init.onnx"
            decoder_step_path = "decoder_step.onnx"
            print(f"Loading ONNX models (decoder: FP32)...")

        self.encoder_conv = ort.InferenceSession(
            str(onnx_path / "encoder_conv.onnx"), sess_opts,
            providers=["CPUExecutionProvider"])
        self.encoder_transformer = ort.InferenceSession(
            str(onnx_path / "encoder_transformer.onnx"), sess_opts,
            providers=["CPUExecutionProvider"])
        self.decoder_init = ort.InferenceSession(
            str(onnx_path / decoder_init_path), sess_opts,
            providers=["CPUExecutionProvider"])
        self.decoder_step = ort.InferenceSession(
            str(onnx_path / decoder_step_path), sess_opts,
            providers=["CPUExecutionProvider"])

        # Load embedding matrix
        embed_path = onnx_path / "embed_tokens.bin"
        print(f"Loading embeddings ({embed_path.stat().st_size / 1e6:.0f} MB)...")
        self.embed_tokens = np.fromfile(
            str(embed_path), dtype=np.float32
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)

        # Mel filterbank
        self.mel_filters = get_mel_filters()

        # Tokenizer
        tokenizer_path = onnx_path / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = None
        self.tokenizer = SimpleTokenizer(str(tokenizer_path) if tokenizer_path else None)

        print("Pipeline ready.")

    def _encode_audio(self, mel: np.ndarray, mel_len: int) -> np.ndarray:
        """Run encoder: mel → audio features [N, 1024]."""
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

    def _embed_and_fuse(self, token_ids: list, audio_features: np.ndarray) -> np.ndarray:
        """Embed tokens and replace audio placeholders with encoder output."""
        ids_array = np.array(token_ids)
        embeds = self.embed_tokens[ids_array]  # [seq_len, 1024]

        # Replace audio_pad positions
        audio_mask = (ids_array == AUDIO_PAD_ID)
        audio_positions = np.where(audio_mask)[0]
        assert len(audio_positions) == audio_features.shape[0], \
            f"Audio token count mismatch: {len(audio_positions)} vs {audio_features.shape[0]}"
        embeds[audio_positions] = audio_features

        return embeds[np.newaxis, :, :]  # [1, seq_len, 1024]

    def _transcribe_chunk(
        self,
        wav: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        """Transcribe a single audio chunk (≤45s recommended)."""
        t0 = time.time()
        mel = compute_mel_spectrogram(wav, self.mel_filters)
        mel_len = mel.shape[1]
        t_mel = time.time() - t0

        t0 = time.time()
        audio_features = self._encode_audio(mel, mel_len)
        num_audio_tokens = audio_features.shape[0]
        t_encoder = time.time() - t0

        t0 = time.time()
        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        input_embeds = self._embed_and_fuse(token_ids, audio_features)
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        t_prepare = time.time() - t0

        t0 = time.time()
        logits, present_keys, present_values = self.decoder_init.run(None, {
            "input_embeds": input_embeds,
            "position_ids": position_ids,
        })
        t_prefill = time.time() - t0

        t0 = time.time()
        next_token = int(np.argmax(logits[0, -1, :]))
        generated = [next_token]
        cur_pos = seq_len

        for _ in range(max_new_tokens - 1):
            if next_token in (IM_END_ID, ENDOFTEXT_ID):
                break

            token_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            pos = np.array([[cur_pos]], dtype=np.int64)

            logits, present_keys, present_values = self.decoder_step.run(None, {
                "input_embeds": token_embed,
                "position_ids": pos,
                "past_keys": present_keys,
                "past_values": present_values,
            })

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
        print(f"  Audio {audio_duration:.1f}s → {num_chunks} sub-chunks (split at silence)")

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
            print(f"    Sub-chunk {i+1}/{num_chunks} ({chunk_dur:.1f}s): "
                  f"{chunk_chars} chars (RTF={chunk_rtf:.2f})")

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
    ) -> Generator[str, None, None]:
        """
        Stream transcription text deltas in real-time as the model decodes tokens.

        Args:
            audio: Path to audio file or float32 audio waveform numpy array (16kHz).
            language: Optional target language tag.
            max_new_tokens: Maximum number of tokens to generate.

        Yields:
            str: Newly generated text delta chunks.
        """
        if isinstance(audio, (str, Path)):
            wav = load_audio(str(audio))
        elif isinstance(audio, np.ndarray):
            wav = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}. Expected file path or numpy array.")

        mel = compute_mel_spectrogram(wav, self.mel_filters)
        mel_len = mel.shape[1]

        audio_features = self._encode_audio(mel, mel_len)
        num_audio_tokens = audio_features.shape[0]

        token_ids = self._build_prompt_ids(num_audio_tokens, language)
        input_embeds = self._embed_and_fuse(token_ids, audio_features)
        seq_len = input_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, present_keys, present_values = self.decoder_init.run(None, {
            "input_embeds": input_embeds,
            "position_ids": position_ids,
        })

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
                yield delta
                printed_text = asr_text

            token_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            pos = np.array([[cur_pos]], dtype=np.int64)

            logits, present_keys, present_values = self.decoder_step.run(None, {
                "input_embeds": token_embed,
                "position_ids": pos,
                "past_keys": present_keys,
                "past_values": present_values,
            })

            next_token = int(np.argmax(logits[0, -1, :]))
            cur_pos += 1


class ONNXQwen3ASR:
    def __init__(
        self, onnx_dir: str = "models/qwen3-asr-onnx", num_threads: int = 0,
        quantize: Literal["int8", "fp32"] = "int8", language: Optional[str] = None,
    ):
        self.pipeline = OnnxAsrPipeline(
            onnx_dir, num_threads, quantize
        )
        self.language = normalize_language(language)

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int = 512,
        chunk_sec: int = 30,
        language: Optional[str] = None,
    ) -> dict:
        lang = normalize_language(language) if language is not None else self.language
        return self.pipeline.transcribe(audio_path, lang, max_new_tokens, chunk_sec)

    def transcribe_stream(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Generator[str, None, None]:
        """Stream real-time transcription text deltas for an audio file path or numpy array."""
        lang = normalize_language(language) if language is not None else self.language
        yield from self.pipeline.transcribe_stream(audio, lang, max_new_tokens)


# ── CLI ─────────────────────────────────────────────────────────────────

# def main():
    # parser = argparse.ArgumentParser(description="Qwen3-ASR Pure ONNX Inference")
    # parser.add_argument("audio", nargs="+", help="Audio file(s)")
    # parser.add_argument("--language", type=str, default=None)
    # parser.add_argument("--onnx-dir", type=str, default="models/qwen3-asr-onnx")
    # parser.add_argument("--max-new-tokens", type=int, default=512)
    # parser.add_argument("--quantize", type=str, default="int8", choices=["none", "int8"],
    #                     help="Decoder quantization: none (FP32) or int8 (default)")
    # parser.add_argument("--chunk-sec", type=int, default=30,
    #                     help="Target chunk length for long audio splitting (default: 30)")
    # parser.add_argument("--threads", type=int, default=0, help="Number of threads (0=all)")
    # args = parser.parse_args()

    # pipeline = OnnxAsrPipeline(onnx_dir=args.onnx_dir, num_threads=args.threads,
    #                            quantize=args.quantize)

    # for audio_path in args.audio:
    #     if not Path(audio_path).exists():
    #         print(f"File not found: {audio_path}", file=sys.stderr)
    #         continue

        # result = pipeline.transcribe(
        #     audio_path, language=args.language,
        #     max_new_tokens=args.max_new_tokens,
        #     chunk_sec=args.chunk_sec,
        # )
            # results.append({
            #     "file": audio_path, "language": result["language"],
            #     "text": result["text"],
            #     "audio_duration_s": result["timing"]["audio_duration_s"],
            #     "processing_time_s": result["timing"]["total_s"],
            #     "rtf": result["timing"]["rtf"],
            # })
        
        # t = result["timing"]
        # print(f"\n[{audio_path}] ({t['audio_duration_s']:.1f}s, RTF {t['rtf']:.2f}x)")
        # if result["language"]:
        #     print(f"  Language: {result['language']}")
        # print(f"  {result['text']}")
        # print(f"  Encoder: {t['encoder_s']:.3f}s | Prefill: {t['prefill_s']:.3f}s | Decode: {t['decode_s']:.3f}s | Tokens: {t['tokens_generated']}")


if __name__ == "__main__":
    # qwen3_asr_engine = ONNXQwen3ASR()
    # result = qwen3_asr_engine.transcribe(
    #     audio_path="test_audio/librispeech_0_1089_0.wav",
    #     # language="English",
    # )
    # print(result)

    # from src.engines.qwen3_engine import ONNXQwen3ASR
    import sys

    engine = ONNXQwen3ASR(
        # language="English",
        # language="Mandarin",
        # language="Indonesian",
    )

    # Stream text deltas real-time from an audio file or array
    for delta in engine.transcribe_stream("test_audio/cn/OSR_cn_000_0073_8k.wav"):
        sys.stdout.write(delta)
        sys.stdout.flush()

