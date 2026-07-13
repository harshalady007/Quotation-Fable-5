#!/usr/bin/env python3
"""Initialize and verify an explicitly selected local V4 SQLite store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quotation_workflow import SQLiteWorkflowStore
from quotation_workflow.domain import new_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", required=True,
        help="Local SQLite file to initialize (never defaults to production).",
    )
    parser.add_argument("--bootstrap-email")
    parser.add_argument("--bootstrap-name")
    parser.add_argument(
        "--request-id",
        help="Idempotency key for admin bootstrap; generated if omitted.",
    )
    args = parser.parse_args()
    if bool(args.bootstrap_email) != bool(args.bootstrap_name):
        parser.error("--bootstrap-email and --bootstrap-name must be supplied together")

    store = SQLiteWorkflowStore(args.database)
    result = {"initialization": store.initialize()}
    if args.bootstrap_email:
        result["bootstrap_admin"] = store.bootstrap_admin(
            email=args.bootstrap_email,
            display_name=args.bootstrap_name,
            request_id=args.request_id or new_id("request"),
        )
    result["integrity"] = store.verify_integrity()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
