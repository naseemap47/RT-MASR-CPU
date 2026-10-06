# benchmark/engine_loader.py
"""
Engine loader: resolves a bench_config entry to an ASR engine instance.

Supports three backends:
  - "onnx"         → ONNXQwen3ASR      (from src/engines/qwen3_onnx_engine.py)
  - "transformers" → Qwen3ASR          (from src/engines/qwen3_engine.py)
  - "whisper"      → WhisperOnnxEngine (from src/engines/whisper_engine.py)

Each entry in bench_config.yaml["configs"] looks like:
    id: "qwen3_onnx_int8_0.6b"
    backend: "onnx"
    model_config: "config/models/qwen3_onnx_0.6b_int8.yaml"
"""
from __future__ import annotations

import sys
import os
from typing import Any, Callable

# Ensure project root is on the path so src.engines is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Also add src/ so 'from engines.xxx import ...' works
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)


SUPPORTED_BACKENDS = ("onnx", "transformers", "whisper")


def _deep_update(base: dict, overrides: dict | None) -> dict:
    """Merge ``overrides`` into ``base`` in place (nested dicts are merged, other values replaced)."""
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_engine(config_entry: dict) -> Any:
    """
    Instantiate an ASR engine from a bench_config entry.

    Args:
        config_entry: Dict with keys: id, backend, model_config, and optionally
                      ``overrides`` (a dict deep-merged into the model YAML, e.g.
                      ``{"engine": {"num_threads": 4}}`` -- used by the load test to
                      size the ORT thread pool per process).

    Returns:
        A live engine instance with a .transcribe(audio_path) -> dict method.

    Raises:
        ValueError: If backend is not one of SUPPORTED_BACKENDS.
        FileNotFoundError: If model_config path does not exist.
    """
    import yaml

    backend = config_entry.get("backend", "")
    model_config_path = config_entry.get("model_config", "")

    if not os.path.exists(model_config_path):
        raise FileNotFoundError(
            f"Model config not found: {model_config_path}"
        )

    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    _deep_update(model_cfg, config_entry.get("overrides"))

    if backend == "onnx":
        from engines.qwen3_onnx_engine import ONNXQwen3ASR
        return ONNXQwen3ASR.from_config(model_cfg)

    elif backend == "transformers":
        from engines.qwen3_engine import Qwen3ASR
        return Qwen3ASR.from_config(model_cfg)

    elif backend == "whisper":
        from engines.whisper_engine import WhisperOnnxEngine
        return WhisperOnnxEngine.from_config(model_cfg)

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}."
        )


def stream_mode_for(config_entry: dict) -> str:
    """
    Live-server streaming strategy for a bench config (same rule as main.py):
    Whisper backends use a sliding window, every Qwen3 backend VAD-cut utterances.
    """
    return "sliding_window" if config_entry.get("backend", "") == "whisper" else "vad_utterance"


def streaming_settings(config_entry: dict) -> dict | None:
    """The ``streaming:`` block of the entry's model YAML (None if absent/unreadable)."""
    import yaml

    path = config_entry.get("model_config", "")
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return (yaml.safe_load(f) or {}).get("streaming")


def engine_factory(config_entry: dict) -> Callable[[], Any]:
    """
    Return a zero-argument factory function that creates a fresh engine.

    Used by the concurrency runners, which need to instantiate engines on demand.

    Args:
        config_entry: Same dict as load_engine().

    Returns:
        Callable[[], engine] — calling it returns a new engine instance.
    """
    def _factory() -> Any:
        return load_engine(config_entry)
    return _factory
