"""Model preflight: unknown names list what is available, missing files give the download command."""
from __future__ import annotations

import pytest
import yaml

from src.core import model_check as mc


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
    assert "Unknown model 'fake_onx' (RT_MASR_MODEL)" in msg
    assert "Did you mean: fake_onnx" in msg
    assert "Available models:" in msg and "DOWNLOADED" in msg
    assert "fake_onnx" in msg.split("Available models:")[1]


def test_not_downloaded_shows_download_command(tmp_path):
    top, _ = _write_registry(tmp_path, tmp_path / "weights")
    with pytest.raises(mc.ModelSetupError) as ei:
        mc.check_registry_model("fake_onnx", config_path=top)
    msg = str(ei.value)
    assert "is not downloaded" in msg
    assert "a.onnx" in msg and "b.bin" in msg
    assert mc.download_command("fake_onnx") in msg


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
    assert "Model 'fake_bench_id' is not downloaded" in out[0][1]
    assert mc.download_command("fake_onnx") in out[0][1]


def test_bench_id_hint_for_registry_name(tmp_path):
    _, model_yaml = _write_registry(tmp_path, tmp_path / "weights")
    registry = [{"name": "fake_onnx", "config": str(model_yaml), "backend": "onnx"}]
    entries = [{"id": "fake_bench_id", "model_config": str(model_yaml)}]
    assert "use id 'fake_bench_id'" in mc.bench_id_hint(["fake_onnx"], entries, registry)
    assert mc.bench_id_hint(["nothing"], entries, registry) == ""


def test_real_registry_is_consistent():
    """Every registry entry points at a readable model YAML (no network, no weights needed)."""
    registry = mc.load_registry()
    assert registry
    for e in registry:
        assert mc.registry_name_for_config(e["config"], registry) == e["name"]
    assert "NAME" in mc.registry_table(registry)
