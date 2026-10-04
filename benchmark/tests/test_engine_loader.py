# benchmark/tests/test_engine_loader.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.engine_loader import engine_factory


def test_engine_factory_returns_callable():
    # Use a fake config that doesn't actually load a model
    config_entry = {
        "id": "fake",
        "backend": "fake_backend",
        "model_config": "nonexistent.yaml"
    }
    factory = engine_factory(config_entry)
    assert callable(factory)


def test_engine_factory_raises_on_unknown_backend():
    config_entry = {
        "id": "bad",
        "backend": "unknown_xyz",
        "model_config": "nonexistent.yaml"
    }
    factory = engine_factory(config_entry)
    import pytest
    with pytest.raises(Exception):
        factory()
