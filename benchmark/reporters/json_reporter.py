# benchmark/reporters/json_reporter.py
"""
Raw results serialiser — writes timestamped JSON to the output directory.
"""
from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def to_serialisable(obj: Any) -> Any:
    """
    Recursively convert dataclasses, lists, and dicts to JSON-serialisable types.

    Handles: dataclass → dict, list → list, dict → dict, primitives pass-through.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_serialisable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(v) for v in obj]
    return obj


class JsonReporter:
    """
    Saves benchmark results as a timestamped JSON file.

    Args:
        output_dir: Directory where JSON files are written (created if needed).
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: Any) -> str:
        """
        Serialise `results` to a timestamped JSON file.

        Args:
            results: Any JSON-serialisable object (dicts, lists, dataclasses).

        Returns:
            Absolute path to the written JSON file.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_raw.json"
        path = self.output_dir / filename

        serialisable = to_serialisable(results)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)

        print(f"  [reporter] Raw JSON saved: {path}")
        return str(path)
