# benchmark/tests/test_summary_reporter.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.reporters.hardware_info import collect_hardware_info
from benchmark.reporters.summary_reporter import SummaryReporter


def test_hardware_info_has_required_keys():
    info = collect_hardware_info()
    for key in ["cpu_model", "physical_cores", "logical_cores", "ram_gb",
                "os", "python_version"]:
        assert key in info, f"Missing key: {key}"


def test_hardware_info_values_non_empty():
    info = collect_hardware_info()
    assert info["physical_cores"] > 0
    assert info["logical_cores"] > 0
    assert info["ram_gb"] > 0


def test_summary_reporter_creates_file(tmp_path):
    reporter = SummaryReporter(output_dir=str(tmp_path))
    from benchmark.reporters.hardware_info import collect_hardware_info
    results = {"hardware": collect_hardware_info(), "load": [], "latency": [],
               "accuracy": [], "concurrency": []}
    path = reporter.render(results)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "# Qwen3-ASR CPU Benchmark Report" in content
    assert "Hardware" in content


def test_summary_reporter_filename_has_timestamp(tmp_path):
    reporter = SummaryReporter(output_dir=str(tmp_path))
    path = reporter.render({"hardware": {}, "load": [], "latency": [],
                            "accuracy": [], "concurrency": []})
    assert "_summary.md" in os.path.basename(path)
