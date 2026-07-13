#!/usr/bin/env python3
"""Generate the V2 quotation-lineage-held-out readiness scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from family_readiness import evaluate_family_readiness, load_readiness_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--corrections", default=config.CORRECTIONS_PATH)
    parser.add_argument("--output", help="Optional JSON snapshot path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--check-snapshot",
        action="store_true",
        help="Fail when data/family_readiness.json does not match this evaluation.",
    )
    parser.add_argument(
        "--enforce-production",
        action="store_true",
        help="Fail when an already-approved family no longer passes V2 gates.",
    )
    args = parser.parse_args()
    report = evaluate_family_readiness(args.data, args.corrections)
    rendered = json.dumps(report, indent=2)
    if not args.quiet:
        print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    stale_snapshot = False
    if args.check_snapshot:
        try:
            stale_snapshot = load_readiness_snapshot() != report
        except ValueError:
            stale_snapshot = True
        if stale_snapshot:
            print(
                "Family readiness snapshot is stale; regenerate it with "
                "--output data/family_readiness.json.",
                file=sys.stderr,
            )
    if args.enforce_production:
        failed = [
            item["family"] for item in report["families"]
            if item["approved_for_automatic_pricing"]
            and not item["release_gate_passed"]
        ]
        return 1 if failed or stale_snapshot else 0
    return 1 if stale_snapshot else 0


if __name__ == "__main__":
    raise SystemExit(main())
