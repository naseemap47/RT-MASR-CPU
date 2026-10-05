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


def _cpu_model() -> str:
    """Human-readable CPU name (platform.processor() is just 'x86_64' on Linux)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


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
        "numpy":         _lib_version("numpy"),
        "onnx":          _lib_version("onnx"),
        "onnxruntime":   _lib_version("onnxruntime"),
        "torch":         _lib_version("torch"),
        "transformers":  _lib_version("transformers"),
        "qwen_asr":      _lib_version("qwen_asr"),
        "psutil":        _lib_version("psutil"),
        "librosa":       _lib_version("librosa"),
        "soundfile":     _lib_version("soundfile"),
    }

    return {
        "cpu_model":      _cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores":  psutil.cpu_count(logical=True) or 0,
        "ram_gb":         round(ram_gb, 2),
        "os":             f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version,
        "lib_versions":   lib_versions,
    }
