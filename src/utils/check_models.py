#!/usr/bin/env python3
"""
src/utils/check_models.py  —  "Which model can MY PC run?" self-test.

For every model in the registry (config/models/models.yaml) this script:

  1. checks the model files are downloaded and the Python deps are installed;
  2. estimates the RAM it needs and compares it with the RAM available now;
  3. actually loads the model in an isolated child process, transcribes a short
     clip and measures load time, latency, real-time factor (RTF) and peak RAM;
  4. prints a verdict per model plus a recommendation.

Each model runs in its own subprocess, so a model that runs out of memory or
hangs is killed and reported -- it cannot take the script (or your desktop) down.

Usage
-----
    uv run python src/utils/check_models.py                       # all downloaded models
    uv run python src/utils/check_models.py --models whisper_int8_tiny,qwen3_onnx_0.6b_int8
    uv run python src/utils/check_models.py --no-run              # static checks only (seconds)
    uv run python src/utils/check_models.py --force               # also try models predicted too big
    uv run python src/utils/check_models.py --json report.json    # machine-readable report

Verdicts
--------
    REAL-TIME    RTF <= 0.5   comfortable for live streaming
    BORDERLINE   RTF <= 1.0   keeps up on one stream, little headroom
    OFFLINE ONLY RTF  > 1.0   slower than real time: fine for files, not for live calls

RTF = processing time / audio duration, measured on one short clip with a
single stream. Live streaming and concurrent calls cost more than this number.

Exit code: 0 if at least one model is usable, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULT_TAG = "@@RESULT@@ "

# ── Tunables ──────────────────────────────────────────────────────────────────
REALTIME_RTF   = 0.5      # <= this: REAL-TIME
BORDERLINE_RTF = 1.0      # <= this: BORDERLINE, above: OFFLINE ONLY
RAM_OVERHEAD   = 1.5      # estimated RAM = model files on disk * this + RAM_FIXED
RAM_FIXED      = 400 * 1024 ** 2
KILL_RAM_FRAC  = 0.90     # kill a model that grows beyond this share of available RAM
MIN_FREE_BYTES = 300 * 1024 ** 2   # kill if the whole system gets this low
DEFAULT_AUDIO  = "test_audio/en/librispeech_0_1089_0.wav"

# Approximate accuracy ranking, only used to pick "best model that still keeps up".
_SIZE_RANK = {"tiny": 1.0, "base": 2.0, "small": 3.0, "medium": 4.0, "0.6b": 4.5, "1.7b": 6.0}
_PRECISION_PENALTY = {"int4": 0.3, "int8": 0.1}

_DEFAULT_ONNX_FILES = [
    "decoder_init.int8.onnx", "decoder_step.int8.onnx", "embed_tokens.bin",
    "encoder_conv.onnx", "encoder_conv.onnx.data",
    "encoder_transformer.onnx", "encoder_transformer.onnx.data", "tokenizer.json",
]


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelReport:
    name: str
    display_name: str
    backend: str
    config: str
    # static checks
    downloaded: bool = False
    missing: list = field(default_factory=list)
    missing_deps: list = field(default_factory=list)
    disk_gb: float = 0.0
    est_ram_gb: float = 0.0
    quality: float = 0.0
    # dynamic run
    ran: bool = False
    load_s: Optional[float] = None
    latency_s: Optional[float] = None
    audio_s: Optional[float] = None
    rtf: Optional[float] = None
    peak_ram_gb: Optional[float] = None
    transcript: str = ""
    # outcome
    status: str = "PENDING"      # REAL-TIME | BORDERLINE | OFFLINE ONLY | NOT DOWNLOADED | MISSING DEPS |
                                 # TOO BIG | OUT OF MEMORY | TIMEOUT | ERROR | NO OUTPUT | OK (not run)
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.status in ("REAL-TIME", "BORDERLINE", "OFFLINE ONLY")


# ── Registry + static inspection ──────────────────────────────────────────────

def load_registry() -> list[dict]:
    top = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    reg = yaml.safe_load((ROOT / top.get("model_registry", "config/models/models.yaml")).read_text())
    return reg["models"]


def _size(paths) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def inspect_files(cfg: dict) -> tuple[list[str], int]:
    """Return (missing file descriptions, bytes on disk) for one model config."""
    dl = cfg.get("download", {})
    eng = cfg.get("engine", {})
    method = dl.get("method", "snapshot")
    backend = cfg.get("backend")

    if method == "hf_files":
        d = ROOT / dl["target_dir"]
        files = dl["files"]
        missing = [f for f in files if not (d / f).is_file() or (d / f).stat().st_size == 0]
        return missing, _size(d / f for f in files)

    if method == "onnx":
        d = ROOT / dl["target_dir"]
        files = dl.get("required_files") or _DEFAULT_ONNX_FILES
        missing = [f for f in files if not (d / f).is_file()]
        return missing, _size(d / f for f in files)

    if method == "whisper" or backend == "whisper":
        d = ROOT / eng.get("model_dir", dl.get("target_dir", ""))
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
        pick = lambda fs: [f for f in fs if prec and prec in f.name] or fs[:1]
        return missing, _size(pick(enc) + pick(dec))

    # snapshot (Transformers): a directory with config.json + safetensors weights
    d = ROOT / dl.get("local_dir", eng.get("model_path", ""))
    weights = sorted(d.glob("*.safetensors")) if d.is_dir() else []
    missing = []
    if not (d / "config.json").is_file():
        missing.append("config.json")
    if not weights:
        missing.append("*.safetensors")
    return missing, _size(weights)


def missing_dependencies(backend: str) -> list[str]:
    need = {
        "onnx": ["onnxruntime", "soundfile", "tokenizers"],
        "whisper": ["onnxruntime", "soundfile"],
        "transformers": ["torch", "transformers", "qwen_asr"],
    }.get(backend, [])
    return [m for m in need if importlib.util.find_spec(m) is None]


def quality_score(name: str, cfg: dict) -> float:
    """Rough accuracy tier (higher = more accurate); only used for recommendations."""
    size = cfg.get("engine", {}).get("model_name") or ""
    tokens = name.lower().split("_")
    if size not in _SIZE_RANK:
        size = next((t for t in tokens if t in _SIZE_RANK), "")
    score = _SIZE_RANK.get(size, 0.0)
    for prec, pen in _PRECISION_PENALTY.items():
        if prec in tokens:
            score -= pen
    return score


def inspect_model(entry: dict) -> ModelReport:
    cfg_path = ROOT / entry["config"]
    cfg = yaml.safe_load(cfg_path.read_text())
    backend = entry.get("backend", cfg.get("backend", ""))
    rep = ModelReport(
        name=entry["name"],
        display_name=cfg.get("display_name", entry["name"]),
        backend=backend,
        config=entry["config"],
        quality=quality_score(entry["name"], cfg),
    )
    missing, disk = inspect_files(cfg)
    rep.missing, rep.disk_gb = missing, disk / 1024 ** 3
    rep.downloaded = not missing
    rep.missing_deps = missing_dependencies(backend)

    est = disk * RAM_OVERHEAD + RAM_FIXED
    if backend == "transformers" and str(cfg.get("engine", {}).get("dtype", "")).endswith("32"):
        est *= 2          # fp32 activations/weights are twice the bf16 files on disk
    rep.est_ram_gb = est / 1024 ** 3
    return rep


# ── Dynamic run (child process) ───────────────────────────────────────────────

def worker_main(args: argparse.Namespace) -> int:
    """Runs inside the child: load engine, transcribe, print one result line."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    out: dict[str, Any] = {"ok": False}
    try:
        from benchmark.engine_loader import load_engine
        from benchmark.metrics.audio_info import audio_duration_s

        t0 = time.perf_counter()
        engine = load_engine({"backend": args.backend, "model_config": args.model_config})
        out["load_s"] = time.perf_counter() - t0

        out["audio_s"] = audio_duration_s(args.audio) or 0.0
        engine.transcribe(args.audio)                      # warm-up (not measured)
        lat, text = [], ""
        for _ in range(args.runs):
            t = time.perf_counter()
            res = engine.transcribe(args.audio)
            lat.append(time.perf_counter() - t)
            text = (res or {}).get("text", "")
        out.update(ok=True, latency_s=statistics.median(lat), text=text)
    except BaseException as exc:                           # noqa: BLE001 - report everything
        out["error"] = f"{type(exc).__name__}: {exc}"
    print(RESULT_TAG + json.dumps(out), flush=True)
    return 0 if out["ok"] else 1


def _tree_rss(proc: psutil.Process) -> int:
    try:
        total = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def _kill_tree(proc: psutil.Process) -> None:
    try:
        for c in proc.children(recursive=True):
            c.kill()
        proc.kill()
    except psutil.Error:
        pass


def run_model(rep: ModelReport, audio: str, runs: int, timeout: float) -> None:
    """Load + transcribe `rep` in a child process and fill in the dynamic fields."""
    budget = int(psutil.virtual_memory().available * KILL_RAM_FRAC)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker",
           "--backend", rep.backend, "--model-config", rep.config,
           "--audio", audio, "--runs", str(runs)]
    reason, peak = "", 0
    # librosa/numba need a writable JIT cache; fall back to a temp dir if the default is read-only.
    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "rt_masr_numba_cache"))
    with tempfile.TemporaryFile("w+") as fout, tempfile.TemporaryFile("w+") as ferr:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                 stdout=fout, stderr=ferr, text=True)
        ps = psutil.Process(child.pid)
        start = time.time()
        while child.poll() is None:
            rss = _tree_rss(ps)
            peak = max(peak, rss)
            if rss > budget or psutil.virtual_memory().available < MIN_FREE_BYTES:
                reason = "oom"
            elif time.time() - start > timeout:
                reason = "timeout"
            if reason:
                _kill_tree(ps)
                child.wait()
                break
            time.sleep(0.2)
        fout.seek(0); ferr.seek(0)
        stdout, stderr = fout.read(), ferr.read()

    rep.ran = True
    rep.peak_ram_gb = peak / 1024 ** 3
    if reason == "oom":
        rep.status = "OUT OF MEMORY"
        rep.note = f"killed at {peak / 1024 ** 3:.1f} GB (only ~{budget / 1024 ** 3:.1f} GB was free)"
        return
    if reason == "timeout":
        rep.status = "TIMEOUT"
        rep.note = f"no result within {timeout:.0f}s (use --timeout to wait longer)"
        return

    line = next((l for l in reversed(stdout.splitlines()) if l.startswith(RESULT_TAG)), None)
    if line is None:
        rep.status = "OUT OF MEMORY" if child.returncode in (-9, 137) else "ERROR"
        tail = (stderr.strip().splitlines() or ["no output"])[-1]
        rep.note = "process killed (likely out of memory)" if rep.status == "OUT OF MEMORY" else tail[:160]
        return
    res = json.loads(line[len(RESULT_TAG):])
    if not res.get("ok"):
        rep.status, rep.note = "ERROR", str(res.get("error", ""))[:160]
        return

    rep.load_s, rep.latency_s, rep.audio_s = res["load_s"], res["latency_s"], res["audio_s"]
    rep.rtf = rep.latency_s / rep.audio_s if rep.audio_s else None
    rep.transcript = res.get("text", "").strip()
    if rep.rtf is None:
        rep.status = "ERROR"
        rep.note = "could not read audio duration"
    elif rep.rtf <= REALTIME_RTF:
        rep.status = "REAL-TIME"
    elif rep.rtf <= BORDERLINE_RTF:
        rep.status = "BORDERLINE"
    else:
        rep.status = "OFFLINE ONLY"
    if rep.usable and not rep.transcript:
        rep.status = "NO OUTPUT"
        rep.note = "loaded and ran, but the transcript was empty: model/export problem, don't use"
    elif rep.usable and budget and peak > 0.8 * (budget / KILL_RAM_FRAC):
        rep.note = "uses most of your free RAM: close other apps / avoid concurrency"


# ── Reporting ─────────────────────────────────────────────────────────────────

def system_summary() -> dict:
    vm = psutil.virtual_memory()
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception:
        ort_v = "not installed"
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1024 ** 3, 1),
        "ram_available_gb": round(vm.available / 1024 ** 3, 1),
        "python": platform.python_version(),
        "onnxruntime": ort_v,
        "torch": "installed" if importlib.util.find_spec("torch") else "not installed",
    }


def print_system(info: dict) -> None:
    print("=" * 78)
    print(" RT-MASR model compatibility check")
    print("=" * 78)
    print(f" CPU    : {info['cpu']}  ({info['physical_cores']} cores / {info['logical_cores']} threads)")
    print(f" RAM    : {info['ram_total_gb']} GB total, {info['ram_available_gb']} GB available now")
    print(f" Python : {info['python']}   onnxruntime: {info['onnxruntime']}   PyTorch: {info['torch']}")
    print("-" * 78)


def _fmt(v: Optional[float], spec: str, unit: str = "") -> str:
    return f"{v:{spec}}{unit}" if v is not None else "-"


def print_table(reports: list[ModelReport]) -> None:
    head = f"{'Model':<24}{'Verdict':<16}{'Load':>7}{'Latency':>9}{'RTF':>6}{'Peak RAM':>10}{'Est. RAM':>10}"
    print("\n" + head)
    print("-" * len(head))
    for r in reports:
        print(f"{r.name:<24}{r.status:<16}"
              f"{_fmt(r.load_s, '.1f', 's'):>7}{_fmt(r.latency_s, '.2f', 's'):>9}"
              f"{_fmt(r.rtf, '.2f'):>6}{_fmt(r.peak_ram_gb, '.1f', ' GB'):>10}"
              f"{_fmt(r.est_ram_gb if r.downloaded else None, '.1f', ' GB'):>10}")
        if r.note:
            print(f"    ↳ {r.note}")


def recommend(reports: list[ModelReport]) -> None:
    ok = [r for r in reports if r.usable]
    rt = [r for r in ok if r.status == "REAL-TIME"]
    print("\n" + "=" * 78)
    print(" Recommendation")
    print("=" * 78)
    if not ok:
        print(" No model could run on this machine. Check the notes above (download the")
        print(" models, free up RAM, or try a smaller model such as whisper_int8_tiny).")
        return
    if rt:
        best = max(rt, key=lambda r: (r.quality, -(r.rtf or 9)))
        fast = min(rt, key=lambda r: r.rtf or 9)
        light = min(rt, key=lambda r: r.peak_ram_gb or 99)
        print(f" Best accuracy that still keeps up live : {best.name}  (RTF {best.rtf:.2f})")
        print(f" Fastest                                : {fast.name}  (RTF {fast.rtf:.2f})")
        print(f" Lightest on RAM                        : {light.name}  ({light.peak_ram_gb:.1f} GB)")
        print(f"\n Set it in config/config.yaml:  default_model: \"{best.name}\"")
        print(f" or for one run:                RT_MASR_MODEL={best.name} uv run python main.py")
    else:
        slow = min(ok, key=lambda r: r.rtf or 99)
        print(f" No model is comfortably real-time here. Closest: {slow.name} (RTF {slow.rtf:.2f}).")
        print(" It is fine for transcribing files, but live calls will lag.")
    off = [r for r in ok if r.status in ("BORDERLINE", "OFFLINE ONLY")]
    if off and rt:
        print("\n Usable for files only (too slow for live): " + ", ".join(r.name for r in off))
    print("\n Note: RTF is for ONE stream on a ~10 s clip. Whisper live streaming re-decodes a")
    print(" sliding window and concurrent calls share the CPU, so keep headroom (RTF <~ 0.3).")


def explain_skips(reports: list[ModelReport]) -> None:
    for r in reports:
        if r.status == "NOT DOWNLOADED":
            r.note = f"missing {', '.join(r.missing[:3])}. Download: uv run python src/utils/download_utils.py --model {r.name}"
        elif r.status == "MISSING DEPS":
            r.note = f"python package(s) not installed: {', '.join(r.missing_deps)}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check which RT-MASR models run on this PC.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--models", help="Comma-separated model names (default: all in the registry)")
    p.add_argument("--audio", default=DEFAULT_AUDIO, help=f"Test clip (default: {DEFAULT_AUDIO})")
    p.add_argument("--runs", type=int, default=2, help="Measured runs per model, median used (default: 2)")
    p.add_argument("--timeout", type=float, default=600, help="Seconds allowed per model (default: 600)")
    p.add_argument("--no-run", action="store_true", help="Only check files, deps and estimated RAM")
    p.add_argument("--force", action="store_true", help="Also try models predicted to need more RAM than is free")
    p.add_argument("--json", metavar="FILE", help="Write the full report to FILE as JSON")
    # internal: child-process entry point
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend", help=argparse.SUPPRESS)
    p.add_argument("--model-config", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args._worker:
        return worker_main(args)

    os.chdir(ROOT)
    registry = load_registry()
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in {e["name"] for e in registry}]
        if unknown:
            print(f"Unknown model(s): {unknown}. Available: {[e['name'] for e in registry]}")
            return 1
        registry = [e for e in registry if e["name"] in wanted]

    audio = args.audio
    if not (ROOT / audio).is_file() and not Path(audio).is_file():
        print(f"Test audio not found: {audio}")
        return 1
    audio = str(Path(audio).resolve()) if Path(audio).is_file() else str(ROOT / audio)

    info = system_summary()
    print_system(info)
    available = psutil.virtual_memory().available / 1024 ** 3

    reports: list[ModelReport] = []
    for i, entry in enumerate(registry, 1):
        rep = inspect_model(entry)
        reports.append(rep)
        tag = f"[{i}/{len(registry)}] {rep.name}"

        if not rep.downloaded:
            rep.status = "NOT DOWNLOADED"
        elif rep.missing_deps:
            rep.status = "MISSING DEPS"
        elif rep.est_ram_gb > available and not args.force:
            rep.status = "TOO BIG"
            rep.note = (f"needs ~{rep.est_ram_gb:.1f} GB RAM but only {available:.1f} GB is free "
                        f"(--force to try anyway)")
        elif args.no_run:
            rep.status = "OK (not run)"
            rep.note = f"files present, ~{rep.est_ram_gb:.1f} GB RAM needed"
        else:
            print(f"{tag}: loading + transcribing ...", flush=True)
            run_model(rep, audio, args.runs, args.timeout)
            print(f"{tag}: {rep.status}" + (f"  (RTF {rep.rtf:.2f})" if rep.rtf else ""), flush=True)
            continue
        print(f"{tag}: {rep.status}", flush=True)

    explain_skips(reports)
    print_table(reports)
    if not args.no_run:
        recommend(reports)
    else:
        print("\n(--no-run: nothing was executed, so no speed verdicts. Re-run without it to measure.)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"system": info, "audio": audio, "models": [asdict(r) for r in reports]}, indent=2))
        print(f"\nJSON report written to {args.json}")

    return 0 if any(r.usable or r.status == "OK (not run)" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
