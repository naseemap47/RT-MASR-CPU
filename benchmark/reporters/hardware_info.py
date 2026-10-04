# benchmark/reporters/hardware_info.py
"""
Hardware and software environment fingerprinting.
Auto-collects CPU, RAM, OS, Python, and library versions.
"""
from __future__ import annotations

import platform
import sys
from typing import Any

import psutil


def _lib_version(module_name: str) -> str:
    """Return the version string of an importable module, or 'not installed'."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


def collect_hardware_info() -> dict[str, Any]:
    """
    Collect hardware and software environment details.

    Returns:
        Dict with keys: cpu_model, physical_cores, logical_cores, ram_gb,
        os, python_version, lib_versions (dict of library→version strings).
    """
    vm = psutil.virtual_memory()
    ram_gb = vm.total / (1024 ** 3)

    lib_versions = {
        "onnxruntime":   _lib_version("onnxruntime"),
        "torch":         _lib_version("torch"),
        "transformers":  _lib_version("transformers"),
        "qwen_asr":      _lib_version("qwen_asr"),
        "psutil":        _lib_version("psutil"),
        "librosa":       _lib_version("librosa"),
    }

    return {
        "cpu_model":      platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores":  psutil.cpu_count(logical=True) or 0,
        "ram_gb":         round(ram_gb, 2),
        "os":             f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version,
        "lib_versions":   lib_versions,
    }
