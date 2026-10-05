"""
src/core/config.py  —  Configuration loading helpers for RT-MASR.

Functions
---------
load_config(config_path)
    Load any YAML file and return it as a plain dict.

load_server_config(config_path)
    Load config/config.yaml and return the full top-level dict.

load_model_registry(registry_path)
    Load the model registry YAML (config/models/models.yaml) and
    return the list of model entries.

resolve_model_config(config_path, model_name)
    Given the top-level config path and a model name, resolve:
      1. Read config.yaml → find model_registry path.
      2. Read the registry → find the matching entry.
      3. Load and return the per-model YAML dict.
    Raises KeyError if the model name is not in the registry.

get_active_model_config(config_path)
    Shortcut: resolve_model_config using the default_model from config.yaml.

get_dtype(dtype_str)
    Convert a dtype string ("bfloat16" | "float32" | "float16") to a
    torch.dtype.  Returns torch.bfloat16 if the string is unrecognised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ── Low-level loader ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Server config ─────────────────────────────────────────────────────────────

def load_server_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Load the top-level server configuration (config/config.yaml).

    Returns the full dict, e.g.:
        {
            "server":         { "host": ..., "port": ..., "log_level": ... },
            "default_model":  "qwen3_onnx_0.6b_int8",
            "model_registry": "config/models/models.yaml",
        }
    """
    return load_config(config_path)


# ── Model registry ────────────────────────────────────────────────────────────

def load_model_registry(registry_path: str) -> list[dict[str, Any]]:
    """
    Load the model registry YAML (config/models/models.yaml).

    Returns the list under the ``models`` key:
        [
            {"name": "qwen3_onnx_0.6b_int8", "config": "...", "backend": "onnx", ...},
            {"name": "qwen3_0.6b", "config": "...", "backend": "transformers", ...},
            ...
        ]
    """
    data = load_config(registry_path)
    return data.get("models", [])


def get_registry_entry(registry_path: str, model_name: str) -> dict[str, Any]:
    """
    Find and return a single registry entry by its ``name`` field.

    Args:
        registry_path: Path to the model registry YAML.
        model_name:    The ``name`` value to look up (e.g. "qwen3_onnx_0.6b_int8").

    Raises:
        KeyError: If no entry with the given name exists.
    """
    models = load_model_registry(registry_path)
    for entry in models:
        if entry.get("name") == model_name:
            return entry
    available = [m.get("name") for m in models]
    raise KeyError(
        f"Model '{model_name}' not found in registry '{registry_path}'. "
        f"Available: {available}"
    )


# ── Per-model config ──────────────────────────────────────────────────────────

def resolve_model_config(
    config_path: str = "config/config.yaml",
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Resolve the per-model YAML for a given model name.

    Steps:
      1. Load config.yaml → ``default_model`` / ``model_registry``.
      2. Load the registry → find the matching entry.
      3. Load and return the per-model YAML (e.g. config/models/qwen3_onnx_0.6b_int8.yaml).

    Args:
        config_path: Path to config/config.yaml (default).
        model_name:  Model name to resolve.  If None, uses ``default_model``
                     from config.yaml.

    Returns:
        The per-model config dict, e.g.:
        {
            "name": "qwen3_onnx_0.6b_int8",
            "backend": "onnx",
            "download": { ... },
            "engine":   { "onnx_dir": ..., "num_threads": ..., ... },
            "inference": { ... },
            ...
        }
    """
    server_cfg = load_server_config(config_path)

    if model_name is None:
        model_name = server_cfg.get("default_model")

    # Registry path is relative to the project root (where config.yaml lives).
    cfg_dir = Path(config_path).parent.parent   # config/config.yaml → project root
    registry_path = cfg_dir / server_cfg["model_registry"]

    entry = get_registry_entry(str(registry_path), model_name)
    model_cfg_path = cfg_dir / entry["config"]
    return load_config(str(model_cfg_path))


def get_active_model_config(
    config_path: str = "config/config.yaml",
) -> dict[str, Any]:
    """
    Shortcut: load the per-model config for the active ``default_model``.

    Returns the same dict as :func:`resolve_model_config`.
    """
    return resolve_model_config(config_path, model_name=None)


# ── Dtype helper ──────────────────────────────────────────────────────────────

def get_dtype(dtype_str: str):
    """
    Convert a YAML dtype string to a torch.dtype.

    Args:
        dtype_str: One of "bfloat16", "float32", "float16".

    Returns:
        torch.dtype  (torch.bfloat16 if the string is unrecognised).
    """
    import torch

    _map = {
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
        "float16":  torch.float16,
    }
    dtype = _map.get(dtype_str.lower() if dtype_str else "")
    if dtype is None:
        import warnings
        warnings.warn(
            f"Unrecognised dtype '{dtype_str}'; falling back to bfloat16.",
            stacklevel=2,
        )
        dtype = torch.bfloat16
    return dtype