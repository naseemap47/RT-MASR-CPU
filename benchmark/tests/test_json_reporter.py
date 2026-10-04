# benchmark/tests/test_json_reporter.py
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.reporters.json_reporter import JsonReporter, to_serialisable
from dataclasses import dataclass


@dataclass
class _DC:
    x: int
    y: str


def test_to_serialisable_dataclass():
    obj = _DC(x=1, y="hello")
    result = to_serialisable(obj)
    assert result == {"x": 1, "y": "hello"}


def test_to_serialisable_nested():
    obj = {"a": _DC(x=2, y="world"), "b": [_DC(x=3, y="!"), 42]}
    result = to_serialisable(obj)
    assert result["a"] == {"x": 2, "y": "world"}
    assert result["b"][0] == {"x": 3, "y": "!"}
    assert result["b"][1] == 42


def test_json_reporter_saves_file(tmp_path):
    reporter = JsonReporter(output_dir=str(tmp_path))
    data = {"config": "test", "score": 0.5, "nested": _DC(x=99, y="z")}
    path = reporter.save(data)

    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["config"] == "test"
    assert loaded["score"] == 0.5
    assert loaded["nested"]["x"] == 99


def test_json_reporter_filename_has_timestamp(tmp_path):
    reporter = JsonReporter(output_dir=str(tmp_path))
    path = reporter.save({"x": 1})
    filename = os.path.basename(path)
    assert "_raw.json" in filename
    # Filename should start with ISO date-like prefix
    assert filename[0].isdigit()
