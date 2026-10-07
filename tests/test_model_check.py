"""Model preflight: unknown names list what is available, missing files give the download command."""
from __future__ import annotations

import io

import pytest
import yaml

from src.core import model_check as mc
from src.core import runlog


def _write_registry(tmp_path, model_dir):
    model_cfg = {
        "name": "fake_onnx",
        "backend": "onnx",
        "download": {"method": "hf_files", "target_dir": str(model_dir), "files": ["a.onnx", "b.bin"]},
        "engine": {"onnx_dir": str(model_dir)},
    }
    model_yaml = tmp_path / "fake_onnx.yaml"
    model_yaml.write_text(yaml.safe_dump(model_cfg))
    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": [
        {"name": "fake_onnx", "config": str(model_yaml), "backend": "onnx"},
    ]}))
    top = tmp_path / "config.yaml"
    top.write_text(yaml.safe_dump({"default_model": "fake_onnx", "model_registry": str(registry)}))
    return top, model_yaml


def test_unknown_model_lists_available_and_suggests(tmp_path):
    top, _ = _write_registry(tmp_path, tmp_path / "weights")
    with pytest.raises(mc.ModelSetupError) as ei:
        mc.check_registry_model("fake_onx", config_path=top, where="RT_MASR_MODEL")
    msg = str(ei.value)
    assert "Unknown model" in msg
    assert "fake_onx   (from RT_MASR_MODEL)" in msg
    assert "Did you mean" in msg and "fake_onnx" in msg
    assert "Available models" in msg and "DOWNLOADED" in msg
    assert "$ RT_MASR_MODEL=<name> uv run uvicorn main:app" in msg


def test_not_downloaded_shows_download_command(tmp_path):
    top, _ = _write_registry(tmp_path, tmp_path / "weights")
    with pytest.raises(mc.ModelSetupError) as ei:
        mc.check_registry_model("fake_onnx", config_path=top)
    msg = str(ei.value)
    assert "Model not downloaded" in msg
    assert "a.onnx" in msg and "b.bin" in msg
    assert "$ " + mc.download_command("fake_onnx") in msg


def test_downloaded_model_returns_config(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "a.onnx").write_bytes(b"x")
    (weights / "b.bin").write_bytes(b"x")
    top, _ = _write_registry(tmp_path, weights)
    cfg = mc.check_registry_model("fake_onnx", config_path=top)
    assert cfg["name"] == "fake_onnx"


def test_bench_entries_map_to_registry_name_for_download(tmp_path):
    _, model_yaml = _write_registry(tmp_path, tmp_path / "weights")
    registry = [{"name": "fake_onnx", "config": str(model_yaml), "backend": "onnx"}]
    entry = {"id": "fake_bench_id", "backend": "onnx", "model_config": str(model_yaml)}
    out = mc.bench_entries_not_downloaded([entry], registry)
    assert len(out) == 1
    assert out[0][0] is entry
    text = str(out[0][1])
    assert "fake_bench_id" in text and "Registry" in text
    assert mc.download_command("fake_onnx") in text


def test_bench_id_hints_for_registry_name(tmp_path):
    _, model_yaml = _write_registry(tmp_path, tmp_path / "weights")
    registry = [{"name": "fake_onnx", "config": str(model_yaml), "backend": "onnx"}]
    entries = [{"id": "fake_bench_id", "model_config": str(model_yaml)}]
    assert any("'fake_bench_id'" in h for h in mc.bench_id_hints(["fake_onnx"], entries, registry))
    assert mc.bench_id_hints(["nothing"], entries, registry) == []


def test_panel_is_a_closed_box_with_aligned_borders():
    p = mc.Panel("Model not downloaded").field("Model", "x").blank().command("echo hi")
    p.table(("NAME", "DOWNLOADED"), [("a", "yes"), ("bbbbbb", "NO")])
    lines = p.render(color=False).splitlines()
    assert lines[0].startswith("╭─") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    assert len({len(ln) for ln in lines}) == 1
    assert all(ln.startswith("│") and ln.endswith("│") for ln in lines[1:-1])


def test_color_only_when_asked(monkeypatch):
    p = mc.Panel("t").command("echo hi")
    assert "\033[" not in p.render(color=False)
    assert "\033[" in p.render(color=True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert mc.use_color() is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert mc.use_color() is True
    monkeypatch.setenv("FORCE_COLOR", "0")
    assert mc.use_color() is False         # pytest captures stdout, so no TTY either


def test_run_log_gets_plain_text():
    term, log = io.StringIO(), io.StringIO()
    tee = runlog._Tee(term, log)
    tee.write("\033[31;1mred\033[0m plain\n")
    assert "\033[" in term.getvalue()
    assert log.getvalue() == "red plain\n"


def test_real_registry_is_consistent():
    """Every registry entry points at a readable model YAML (no network, no weights needed)."""
    registry = mc.load_registry()
    assert registry
    for e in registry:
        assert mc.registry_name_for_config(e["config"], registry) == e["name"]
    header, rows = mc.registry_table(registry)
    assert header[0] == "NAME" and len(rows) == len(registry)
