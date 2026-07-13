#!/usr/bin/env python3
"""Export stable record IDs and missing fields for estimator correction."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from correction_queue import build_correction_queue


def _csv(queue: list[dict]) -> str:
    columns = [
        "record_id", "family", "subtype", "source", "date", "description",
        "unit", "rate", "quality_issues", "missing_required_fields",
        "missing_recommended_fields", "blocking", "correction_ids",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for item in queue:
        row = dict(item)
        for field in ("quality_issues", "missing_required_fields",
                      "missing_recommended_fields", "correction_ids"):
            row[field] = "; ".join(str(value) for value in row[field])
        writer.writerow(row)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--corrections", default=config.CORRECTIONS_PATH)
    parser.add_argument("--family")
    parser.add_argument("--blocking-only", action="store_true")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output")
    args = parser.parse_args()
    queue = build_correction_queue(
        args.data, args.corrections, args.family, args.blocking_only
    )
    rendered = (_csv(queue) if args.format == "csv"
                else json.dumps(queue, indent=2) + "\n")
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
