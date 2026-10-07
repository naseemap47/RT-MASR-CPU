"""
Model preflight: is the requested model known, and are its files on disk?

Used by the live server, the benchmark, the load test, ``check_models`` and the
downloader, so every entry point prints the same boxed terminal panel:

* unknown name / id   -> "did you mean ..." plus a table of what is available
                         (with a downloaded yes/NO column);
* files not on disk   -> what is missing and the exact download command;
* test audio missing  -> which clips are missing and the ``--test-audio`` command.

Registry names (``config/models/models.yaml``, e.g. ``qwen3_onnx_0.6b_int8``)
are what the downloader and ``RT_MASR_MODEL`` take. Benchmark / load-test ids
(``bench_config.yaml``, e.g. ``qwen3_onnx_int8_0.6b``) are a different
namespace; they are mapped back to the registry through their ``model_config``.

Panels use ANSI colour only on a terminal (``NO_COLOR`` disables it,
``FORCE_COLOR`` forces it); ``run.log`` gets the plain text.
"""
from __future__ import annotations

import difflib
import logging
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = "src/utils/download_utils.py"
TEST_AUDIO_DIR = "test_audio"

_DEFAULT_ONNX_FILES = [
    "decoder_init.int8.onnx", "decoder_step.int8.onnx", "embed_tokens.bin",
    "encoder_conv.onnx", "encoder_conv.onnx.data",
    "encoder_transformer.onnx", "encoder_transformer.onnx.data", "tokenizer.json",
]


class ModelSetupError(RuntimeError):
    """The requested model is unknown or not downloaded. ``.panel`` is the terminal message."""

    def __init__(self, panel: "Panel") -> None:
        super().__init__(panel.render(color=False))
        self.panel = panel


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


# ── Terminal panel ────────────────────────────────────────────────────────────

_SGR = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36"}
_LEVEL_STYLE = {"error": ("red", "✖"), "warning": ("yellow", "▲"), "info": ("cyan", "●")}
_LABEL_W = 14
_MAX_W = 100

Seg = tuple[str, str]           # (text, space-separated style names)


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR", "").strip().lower()
    if force and force not in ("0", "false", "no"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                return True
        except Exception:
            pass
    return False


def _paint(text: str, style: str, color: bool) -> str:
    if not color or not style or not text:
        return text
    codes = ";".join(_SGR[s] for s in style.split() if s in _SGR)
    return f"\033[{codes}m{text}\033[0m" if codes else text


def _content_width() -> int:
    cols = shutil.get_terminal_size((_MAX_W + 4, 24)).columns
    return max(60, min(_MAX_W, cols - 4))


@dataclass
class Panel:
    """A boxed, aligned terminal message: fields, tables, ``$`` commands, notes."""

    title: str
    level: str = "error"
    lines: list[list[Seg]] = field(default_factory=list)

    def blank(self) -> "Panel":
        if self.lines and self.lines[-1]:
            self.lines.append([])
        return self

    def text(self, s: str, style: str = "") -> "Panel":
        for chunk in textwrap.wrap(s, _content_width()) or [""]:
            self.lines.append([(chunk, style)])
        return self

    def heading(self, s: str) -> "Panel":
        self.lines.append([(s, "bold")])
        return self

    def field(self, label: str, value: str, style: str = "") -> "Panel":
        width = max(20, _content_width() - _LABEL_W)
        chunks = textwrap.wrap(value, width) or [""]
        self.lines.append([(f"{label:<{_LABEL_W}}", "dim"), (chunks[0], style)])
        for chunk in chunks[1:]:
            self.lines.append([(" " * _LABEL_W, ""), (chunk, style)])
        return self

    def command(self, cmd: str) -> "Panel":
        self.lines.append([("  $ ", "dim"), (cmd, "cyan bold")])
        return self

    def table(self, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> "Panel":
        cells = [[str(v) for v in r] for r in rows]
        widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
                  for i, h in enumerate(header)]
        status_col = next((i for i, h in enumerate(header) if h == "DOWNLOADED"), None)

        def row(values: Sequence[str], base: str) -> list[Seg]:
            segs: list[Seg] = [("  ", "")]
            for i, (v, w) in enumerate(zip(values, widths)):
                style = base
                if i == status_col and base != "bold dim":
                    style = {"yes": "green", "NO": "red bold"}.get(v, "dim")
                segs.append((v.ljust(w) if i < len(widths) - 1 else v, style))
                if i < len(widths) - 1:
                    segs.append(("  ", ""))
            return segs

        self.lines.append(row([str(h) for h in header], "bold dim"))
        self.lines.append([("  " + "  ".join("─" * w for w in widths), "dim")])
        for r in cells:
            self.lines.append(row(r, ""))
        return self

    def note(self, s: str) -> "Panel":
        return self.text(s, "dim")

    def render(self, color: Optional[bool] = None) -> str:
        color = use_color() if color is None else color
        accent, icon = _LEVEL_STYLE.get(self.level, _LEVEL_STYLE["info"])
        body = list(self.lines)
        while body and not body[-1]:
            body.pop()
        inner = max([len(self.title) + 6, 56, *(sum(len(t) for t, _ in ln) for ln in body)])

        title = f" {icon} {self.title} "
        top = (_paint("╭─", accent, color) + _paint(title, f"{accent} bold", color)
               + _paint("─" * (inner + 1 - len(title)) + "╮", accent, color))
        side = _paint("│", accent, color)
        out = [top, f"{side}{' ' * (inner + 2)}{side}"]
        for ln in body:
            plain = sum(len(t) for t, _ in ln)
            text = "".join(_paint(t, s, color) for t, s in ln)
            out.append(f"{side} {text}{' ' * (inner - plain)} {side}")
        out.append(f"{side}{' ' * (inner + 2)}{side}")
        out.append(_paint("╰" + "─" * (inner + 2) + "╯", accent, color))
        return "\n".join(out)

    def __str__(self) -> str:
        return self.render(color=False)


def report(logger: logging.Logger, panel: Panel) -> None:
    """Log a panel on its own lines (error / warning / info by ``panel.level``)."""
    level = {"error": logging.ERROR, "warning": logging.WARNING}.get(panel.level, logging.INFO)
    logger.log(level, "\n%s\n", panel.render())


# ── Panels ────────────────────────────────────────────────────────────────────

def download_command(registry_name: str) -> str:
    return f"uv run python {DOWNLOAD_SCRIPT} --model {registry_name}"


def audio_download_command() -> str:
    return f"uv run python {DOWNLOAD_SCRIPT} --test-audio"


def _shorten(items: Sequence[str], n: int = 6) -> str:
    return ", ".join(items[:n]) + (f"  (+{len(items) - n} more)" if len(items) > n else "")


def close_matches(name: str, choices: Iterable[str], n: int = 3) -> list[str]:
    choices = list(choices)
    hits = difflib.get_close_matches(name, choices, n=n, cutoff=0.5)
    low = name.lower()
    hits += [c for c in choices if low and low in c.lower() and c not in hits]
    return hits[:n]


def _yes_no(flag: Optional[bool]) -> str:
    return "?" if flag is None else ("yes" if flag else "NO")


Table = tuple[tuple[str, ...], list[tuple[str, ...]]]


def registry_table(registry: Optional[list[dict]] = None) -> Table:
    rows = [(e.get("name", "?"), e.get("backend", "?"), _yes_no(_cfg_downloaded(e.get("config", ""))))
            for e in (registry if registry is not None else load_registry())]
    return ("NAME", "BACKEND", "DOWNLOADED"), rows


def bench_table(entries: Iterable[dict], registry: Optional[list[dict]] = None) -> Table:
    """Benchmark / load-test ids with the registry name the downloader needs."""
    registry = registry if registry is not None else load_registry()
    rows = []
    for e in entries:
        mc = e.get("model_config", "")
        rows.append((e.get("id", "?"), e.get("backend", "?"),
                     registry_name_for_config(mc, registry) or "-", _yes_no(_cfg_downloaded(mc))))
    return ("ID", "BACKEND", "REGISTRY NAME", "DOWNLOADED"), rows


def not_downloaded_panel(label: str, cfg: dict, missing: list[str],
                         registry_name: Optional[str], level: str = "error") -> Panel:
    p = Panel("Model not downloaded", level)
    p.field("Model", label, "bold")
    if registry_name and registry_name != label:
        p.field("Registry", registry_name)
    p.field("Location", model_dir(cfg))
    p.field("Missing", _shorten(missing), "yellow")
    p.blank().heading("Download it with")
    if registry_name:
        p.command(download_command(registry_name))
    else:
        p.note(f"{DOWNLOAD_SCRIPT}, using the `download:` block of its model YAML.")
    return p


def is_test_audio(path: str | Path) -> bool:
    """True if *path* lies inside ``test_audio/`` (i.e. ``--test-audio`` would fetch it)."""
    return _rel(path).split(os.sep, 1)[0] == TEST_AUDIO_DIR


def has_test_audio() -> bool:
    """True if ``test_audio/`` holds at least one ``.wav`` file."""
    return next((ROOT / TEST_AUDIO_DIR).glob("**/*.wav"), None) is not None


def audio_missing_panel(missing: Sequence[str] = (), level: str = "error", where: str = "") -> Panel:
    """Test clips are not on disk. ``missing`` empty means ``test_audio/`` has no WAV at all."""
    p = Panel("Test audio not downloaded", level)
    p.field("Location", f"{TEST_AUDIO_DIR}/")
    if where:
        p.field("Listed in", where)
    p.field("Missing", _shorten([_rel(m) for m in missing]) if missing else "no .wav files", "yellow")
    p.blank().heading("Download it with").command(audio_download_command())
    outside = [m for m in missing if not is_test_audio(m)]
    if outside:
        p.blank().note(f"{_shorten([_rel(m) for m in outside], 3)} is outside {TEST_AUDIO_DIR}/ and "
                       f"not part of the download; fix the path{f' in {where}' if where else ''}.")
    return p


def unknown_name_panel(kind: str, names: str | Iterable[str], choices: Iterable[str], table: Table,
                       where: str = "", hints: Iterable[str] = ()) -> Panel:
    """``hints``: plain lines; ``$ `` lines are commands, ``""`` is a blank line."""
    names = [names] if isinstance(names, str) else list(names)
    choices = list(choices)
    p = Panel(f"Unknown {kind}", "error")
    for name in names:
        p.field("Requested", name + (f"   (from {where})" if where else ""), "bold")
        near = close_matches(name, choices)
        if near:
            p.field("Did you mean", ", ".join(near), "green")
    p.blank().heading(f"Available {kind}s")
    p.table(*table)
    hints = list(hints)
    if any(hints):
        p.blank()
        for h in hints:
            if not h:
                p.blank()
            elif h.startswith("$ "):
                p.command(h[2:])
            else:
                p.note(h)
    return p


# ── Checks ────────────────────────────────────────────────────────────────────

def check_registry_model(name: str, *, config_path: str | Path = "config/config.yaml",
                         where: str = "", require_download: bool = True) -> dict:
    """Validate a registry model name and its files; return the per-model config dict."""
    registry = load_registry(config_path)
    entry = next((e for e in registry if e.get("name") == name), None)
    if entry is None:
        raise ModelSetupError(unknown_name_panel(
            "model", [str(name)], [e.get("name", "") for e in registry], registry_table(registry),
            where=where,
            hints=["Set `default_model` in config/config.yaml, or pick one for a single run:",
                   "$ RT_MASR_MODEL=<name> uv run uvicorn main:app"],
        ))
    cfg = _load_yaml(entry["config"])
    if require_download:
        missing, _ = missing_files(cfg)
        if missing:
            raise ModelSetupError(not_downloaded_panel(name, cfg, missing, name))
    return cfg


def bench_entries_not_downloaded(entries: Iterable[dict],
                                 registry: Optional[list[dict]] = None) -> list[tuple[dict, Panel]]:
    """``(entry, panel)`` for every benchmark / load-test entry whose files are missing."""
    registry = registry if registry is not None else load_registry()
    out = []
    for e in entries:
        mc = e.get("model_config", "")
        try:
            cfg = _load_yaml(mc)
        except FileNotFoundError:
            p = Panel("Model config not found").field("Model", e.get("id", "?"), "bold").field("Config", mc)
            out.append((e, p))
            continue
        missing, _ = missing_files(cfg)
        if missing:
            out.append((e, not_downloaded_panel(
                e.get("id", "?"), cfg, missing, registry_name_for_config(mc, registry))))
    return out


def bench_id_hints(wanted: Iterable[str], entries: Iterable[dict],
                   registry: Optional[list[dict]] = None) -> list[str]:
    """If the user typed registry names instead of benchmark ids, say which id to use."""
    registry = registry if registry is not None else load_registry()
    by_reg = {}
    for e in entries:
        reg = registry_name_for_config(e.get("model_config", ""), registry)
        if reg:
            by_reg[reg] = e.get("id")
    return [f"'{w}' is a registry name; the id for it is '{by_reg[w]}'."
            for w in wanted if w in by_reg and by_reg[w] != w]
