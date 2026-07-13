#!/usr/bin/env python3
"""Validate and optionally apply an estimator-reviewed V3 context template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from context_enrichment import (merge_context_reviews,
                                write_correction_manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_file")
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--corrections", default=config.CORRECTIONS_PATH)
    parser.add_argument(
        "--output", default=config.CORRECTIONS_PATH,
        help="Destination correction manifest; defaults to the active manifest.",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Atomically write the merged manifest. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    manifest, summary = merge_context_reviews(
        args.review_file, args.data, args.corrections
    )
    result = {**summary, "write_requested": args.write, "output": args.output}
    if args.write:
        write_correction_manifest(args.output, manifest)
        result["written"] = True
    else:
        result["written"] = False
        result["next_step"] = "Re-run with --write after reviewing this dry run."
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
