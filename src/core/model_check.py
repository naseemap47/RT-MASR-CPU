"""
Model preflight: is the requested model known, and are its files on disk?

Used by the live server, the benchmark, the load test, ``check_models`` and the
downloader, so every entry point gives the same terminal message:

* unknown name / id   -> "did you mean ..." plus a table of what is available
                         (with a downloaded yes/no column);
* files not on disk   -> what is missing and the exact download command.

Registry names (``config/models/models.yaml``, e.g. ``qwen3_onnx_0.6b_int8``)
are what the downloader and ``RT_MASR_MODEL`` take. Benchmark / load-test ids
(``bench_config.yaml``, e.g. ``qwen3_onnx_int8_0.6b``) are a different
namespace; they are mapped back to the registry through their ``model_config``.
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = "src/utils/download_utils.py"

_DEFAULT_ONNX_FILES = [
    "decoder_init.int8.onnx", "decoder_step.int8.onnx", "embed_tokens.bin",
    "encoder_conv.onnx", "encoder_conv.onnx.data",
    "encoder_transformer.onnx", "encoder_transformer.onnx.data", "tokenizer.json",
]


class ModelSetupError(RuntimeError):
    """The requested model is unknown or not downloaded. ``str()`` is the terminal message."""


# ── Paths / registry ──────────────────────────────────────────────────────────

def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _rel(path: str | Path) -> str:
    p = _abs(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _load_yaml(path: str | Path) -> dict:
    with open(_abs(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def registry_path(config_path: str | Path = "config/config.yaml") -> Path:
    top = _load_yaml(config_path)
    return _abs(top.get("model_registry", "config/models/models.yaml"))


def load_registry(config_path: str | Path = "config/config.yaml") -> list[dict]:
    return list(_load_yaml(registry_path(config_path)).get("models", []))


def registry_name_for_config(model_config: str | Path,
                             registry: Optional[list[dict]] = None) -> Optional[str]:
    """Registry ``name`` whose ``config`` is ``model_config`` (what ``--model`` of the downloader takes)."""
    target = _rel(model_config)
    for entry in registry if registry is not None else load_registry():
        if _rel(entry.get("config", "")) == target:
            return entry.get("name")
    return None


# ── Files on disk ─────────────────────────────────────────────────────────────

def _size(paths: Iterable[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def missing_files(cfg: dict) -> tuple[list[str], int]:
    """Return (missing file descriptions, bytes on disk) for one per-model config dict."""
    dl = cfg.get("download", {}) or {}
    eng = cfg.get("engine", {}) or {}
    method = dl.get("method", "snapshot")
    backend = cfg.get("backend")

    if method == "hf_files":
        d = _abs(dl["target_dir"])
        files = dl["files"]
        missing = [f for f in files if not (d / f).is_file() or (d / f).stat().st_size == 0]
        return missing, _size(d / f for f in files)

    if method == "onnx":
        d = _abs(dl["target_dir"])
        files = dl.get("required_files") or _DEFAULT_ONNX_FILES
        missing = [f for f in files if not (d / f).is_file()]
        return missing, _size(d / f for f in files)

    if method == "whisper" or backend == "whisper":
        d = _abs(eng.get("model_dir", dl.get("target_dir", "")))
        size = eng.get("model_name", "")
        enc = sorted(d.glob(f"{size}_encoder*.onnx")) if d.is_dir() else []
        dec = sorted(d.glob(f"{size}_decoder*.onnx")) if d.is_dir() else []
        missing = []
        if not enc:
            missing.append(f"{size}_encoder*.onnx")
        if not dec:
            missing.append(f"{size}_decoder*.onnx")
        # Prefer files of the configured precision when several exist.
        prec = eng.get("precision", "")
        pick = lambda fs: [f for f in fs if prec and prec in f.name] or fs[:1]  # noqa: E731
        return missing, _size(pick(enc) + pick(dec))

    # snapshot (Transformers): a directory with config.json + safetensors weights
    d = _abs(dl.get("local_dir", eng.get("model_path", "")))
    weights = sorted(d.glob("*.safetensors")) if d.is_dir() else []
    missing = []
    if not (d / "config.json").is_file():
        missing.append("config.json")
    if not weights:
        missing.append("*.safetensors")
    return missing, _size(weights)


def model_dir(cfg: dict) -> str:
    dl = cfg.get("download", {}) or {}
    eng = cfg.get("engine", {}) or {}
    return (dl.get("target_dir") or dl.get("local_dir") or eng.get("model_dir")
            or eng.get("onnx_dir") or eng.get("model_path") or "?")


def is_downloaded(cfg: dict) -> bool:
    try:
        return not missing_files(cfg)[0]
    except Exception:
        return False


def _cfg_downloaded(model_config: str | Path) -> Optional[bool]:
    try:
        return is_downloaded(_load_yaml(model_config))
    except Exception:
        return None


# ── Messages ──────────────────────────────────────────────────────────────────

def download_command(registry_name: str) -> str:
    return f"uv run python {DOWNLOAD_SCRIPT} --model {registry_name}"


def close_matches(name: str, choices: Iterable[str], n: int = 3) -> list[str]:
    choices = list(choices)
    hits = difflib.get_close_matches(name, choices, n=n, cutoff=0.5)
    low = name.lower()
    hits += [c for c in choices if low and low in c.lower() and c not in hits]
    return hits[:n]


def _yes_no(flag: Optional[bool]) -> str:
    return "?" if flag is None else ("yes" if flag else "NO")


def format_table(header: Sequence[str], rows: Sequence[Sequence[Any]], indent: str = "  ") -> str:
    cols = [list(map(str, c)) for c in zip(header, *rows)] if rows else [[h] for h in header]
    widths = [max(len(v) for v in col) for col in cols]
    lines = []
    for row in [header, *rows]:
        cells = [str(v).ljust(w) for v, w in zip(row, widths)]
        lines.append(indent + "  ".join(cells).rstrip())
    return "\n".join(lines)


def registry_table(registry: Optional[list[dict]] = None) -> str:
    rows = []
    for e in registry if registry is not None else load_registry():
        rows.append((e.get("name", "?"), e.get("backend", "?"), _yes_no(_cfg_downloaded(e.get("config", "")))))
    return format_table(("NAME", "BACKEND", "DOWNLOADED"), rows)


def bench_table(entries: Iterable[dict], registry: Optional[list[dict]] = None) -> str:
    """Table of benchmark / load-test ids with the registry name the downloader needs."""
    registry = registry if registry is not None else load_registry()
    rows = []
    for e in entries:
        mc = e.get("model_config", "")
        rows.append((e.get("id", "?"), e.get("backend", "?"),
                     registry_name_for_config(mc, registry) or "-", _yes_no(_cfg_downloaded(mc))))
    return format_table(("ID", "BACKEND", "REGISTRY NAME", "DOWNLOADED"), rows)


def not_downloaded_message(label: str, cfg: dict, missing: list[str],
                           registry_name: Optional[str]) -> str:
    shown = ", ".join(missing[:4]) + (f" (+{len(missing) - 4} more)" if len(missing) > 4 else "")
    lines = [
        f"Model '{label}' is not downloaded.",
        f"  missing in {model_dir(cfg)}: {shown}",
    ]
    if registry_name:
        lines += ["Download it with:", f"  {download_command(registry_name)}"]
    else:
        lines += [f"Download it with {DOWNLOAD_SCRIPT} (see the `download:` block of its model YAML)."]
    return "\n".join(lines)


def unknown_name_message(kind: str, names: str | Iterable[str], choices: Iterable[str], table: str,
                         where: str = "", hint: str = "") -> str:
    names = [names] if isinstance(names, str) else list(names)
    choices = list(choices)
    suffix = f" ({where})" if where else ""
    lines = []
    for name in names:
        near = close_matches(name, choices)
        tip = f"  Did you mean: {', '.join(near)}?" if near else ""
        lines.append(f"Unknown {kind} '{name}'{suffix}.{tip}")
    lines += ["", f"Available {kind}s:", table]
    if hint:
        lines += ["", hint]
    return "\n".join(lines)


# ── Checks (raise ModelSetupError with the terminal message) ──────────────────

def check_registry_model(name: str, *, config_path: str | Path = "config/config.yaml",
                         where: str = "", require_download: bool = True) -> dict:
    """Validate a registry model name and its files; return the per-model config dict."""
    registry = load_registry(config_path)
    entry = next((e for e in registry if e.get("name") == name), None)
    if entry is None:
        raise ModelSetupError(unknown_name_message(
            "model", [str(name)], [e.get("name", "") for e in registry], registry_table(registry),
            where=where,
            hint=("Pick a NAME above: set `default_model` in config/config.yaml, or for one run\n"
                  "  RT_MASR_MODEL=<name> uv run uvicorn main:app"),
        ))
    cfg = _load_yaml(entry["config"])
    if require_download:
        missing, _ = missing_files(cfg)
        if missing:
            raise ModelSetupError(not_downloaded_message(name, cfg, missing, name))
    return cfg


def bench_entries_not_downloaded(entries: Iterable[dict],
                                 registry: Optional[list[dict]] = None) -> list[tuple[dict, str]]:
    """``(entry, message)`` for every benchmark / load-test entry whose files are missing."""
    registry = registry if registry is not None else load_registry()
    out = []
    for e in entries:
        mc = e.get("model_config", "")
        try:
            cfg = _load_yaml(mc)
        except FileNotFoundError:
            out.append((e, f"Model '{e.get('id')}': model config not found: {mc}"))
            continue
        missing, _ = missing_files(cfg)
        if missing:
            out.append((e, not_downloaded_message(
                e.get("id", "?"), cfg, missing, registry_name_for_config(mc, registry))))
    return out


def bench_id_hint(wanted: Iterable[str], entries: Iterable[dict],
                  registry: Optional[list[dict]] = None) -> str:
    """If the user typed registry names instead of benchmark ids, say which id to use."""
    registry = registry if registry is not None else load_registry()
    by_reg = {}
    for e in entries:
        reg = registry_name_for_config(e.get("model_config", ""), registry)
        if reg:
            by_reg[reg] = e.get("id")
    tips = [f"  '{w}' is a registry name; use id '{by_reg[w]}'"
            for w in wanted if w in by_reg and by_reg[w] != w]
    return "\n".join(tips)
