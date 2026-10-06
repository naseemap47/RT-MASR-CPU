#!/usr/bin/env python3
# loadtest/run_sizing.py
"""
Turn measured load-test results into the capacity sizing guide (Markdown + JSON).

Usage:
    python3 loadtest/run_sizing.py                                  # newest *_loadtest_raw.json
    python3 loadtest/run_sizing.py --input a_raw.json b_raw.json    # merge several runs
    python3 loadtest/run_sizing.py --headroom 0.6 --serving-overhead 1.25 --spare-fraction 0.2
    python3 loadtest/run_sizing.py --legs 50,60,100,200,500,1000

Every ASSUMED input of the model is a flag here; measured data is never altered.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loadtest.sizing.model import SizingAssumptions, load_raw_files, to_jsonable
from loadtest.sizing.report import render_sizing_guide


def main() -> None:
    p = argparse.ArgumentParser(description="Capacity sizing from load-test data", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="*", default=None, help="Raw load-test JSON file(s); default: newest in loadtest/results")
    p.add_argument("--results-dir", default=str(_PROJECT_ROOT / "loadtest" / "results"))
    p.add_argument("--output-dir", default=None, help="default: same as --results-dir")
    d = SizingAssumptions()
    p.add_argument("--headroom", type=float, default=d.headroom)
    p.add_argument("--serving-overhead", type=float, default=d.serving_overhead)
    p.add_argument("--spare-fraction", type=float, default=d.spare_fraction)
    p.add_argument("--min-spare-nodes", type=int, default=d.min_spare_nodes)
    p.add_argument("--os-reserve-gb", type=float, default=d.os_reserve_gb)
    p.add_argument("--ram-headroom", type=float, default=d.ram_headroom)
    p.add_argument("--legs", default=",".join(str(x) for x in d.targets), help="Concurrent-leg targets")
    args = p.parse_args()

    files = args.input or sorted(Path(args.results_dir).glob("*_loadtest_raw.json"))[-1:]
    if not files:
        sys.exit(f"No *_loadtest_raw.json in {args.results_dir}. Run loadtest/run_loadtest.py first.")
    scenarios, hardware = load_raw_files(files)
    if not scenarios:
        sys.exit("The input files contain no scenarios.")

    a = SizingAssumptions(
        headroom=args.headroom, serving_overhead=args.serving_overhead, spare_fraction=args.spare_fraction,
        min_spare_nodes=args.min_spare_nodes, os_reserve_gb=args.os_reserve_gb, ram_headroom=args.ram_headroom,
        targets=tuple(int(x) for x in args.legs.split(",") if x.strip()),
    )
    text, data = render_sizing_guide(scenarios, hardware, a, sources=[Path(f).name for f in files])

    out_dir = Path(args.output_dir or args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md, js = out_dir / f"{stamp}_sizing_guide.md", out_dir / f"{stamp}_sizing.json"
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps({"assumptions": to_jsonable(a), "models": to_jsonable(data)}, indent=2), encoding="utf-8")
    print(text)
    print(f"\n✓ Sizing guide: {md}\n✓ Sizing data : {js}")


if __name__ == "__main__":
    main()
