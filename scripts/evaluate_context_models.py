#!/usr/bin/env python3
"""Generate/check leakage-safe V3 context-model shadow readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from context_modeling import (evaluate_context_model_readiness,
                              load_context_model_snapshot)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--corrections", default=config.CORRECTIONS_PATH)
    parser.add_argument("--output", help="Optional deterministic JSON snapshot path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--check-snapshot",
        action="store_true",
        help="Fail when data/context_model_readiness.json is stale.",
    )
    parser.add_argument(
        "--enforce-production",
        action="store_true",
        help="Fail if an approved context adjustment no longer passes its gates.",
    )
    args = parser.parse_args()

    report = evaluate_context_model_readiness(args.data, args.corrections)
    rendered = json.dumps(report, indent=2)
    if not args.quiet:
        print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    stale_snapshot = False
    if args.check_snapshot:
        try:
            stale_snapshot = load_context_model_snapshot() != report
        except (ValueError, RuntimeError):
            stale_snapshot = True
        if stale_snapshot:
            print(
                "Context model readiness snapshot is stale; regenerate it with "
                "--output data/context_model_readiness.json.",
                file=sys.stderr,
            )
    failed_approved = []
    if args.enforce_production:
        failed_approved = [
            item["adjustment_id"] for item in report["adjustments"]
            if item["approved_for_price_adjustment"]
            and (
                not item["production_gate_passed"]
                or not item["base_family_approved_for_automatic_pricing"]
                or not item["implemented_in_pricing_engine"]
            )
        ]
    return 1 if stale_snapshot or failed_approved else 0


if __name__ == "__main__":
    raise SystemExit(main())
