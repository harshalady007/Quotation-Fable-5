#!/usr/bin/env python3
"""Export a V3 estimator template for missing contextual evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from context_enrichment import (build_context_review_queue,
                                write_context_review)
from context_readiness import CONTEXT_FIELDS
from production_pricing import SUPPORTED_INPUT_FAMILIES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--corrections", default=config.CORRECTIONS_PATH)
    parser.add_argument("--family", choices=sorted(SUPPORTED_INPUT_FAMILIES))
    parser.add_argument(
        "--field", action="append", choices=CONTEXT_FIELDS,
        help="Repeat to select fields; defaults to all V3 context fields.",
    )
    parser.add_argument(
        "--include-quarantined", action="store_true",
        help="Include current-revision rows blocked by another quality defect.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include rows whose selected context fields are already populated.",
    )
    parser.add_argument("--format", choices=("csv", "xlsx", "json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows, summary = build_context_review_queue(
        args.data,
        args.corrections,
        args.family,
        args.field,
        include_quarantined=args.include_quarantined,
        missing_only=not args.all,
    )
    output = write_context_review(
        args.output, rows, summary, file_format=args.format
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
