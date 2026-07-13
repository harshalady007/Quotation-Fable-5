"""Read-only V4 capability and safety report for the deployed API."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import config

from .domain import WORKFLOW_SCHEMA_VERSION, WORKFLOW_VERSION, WorkflowError
from .store import SQLiteWorkflowStore


IMPLEMENTED_CAPABILITIES = (
    "role-checked domain commands",
    "project and quotation aggregates",
    "decimal line-item totals",
    "immutable submitted revisions",
    "four-eyes approval control",
    "optimistic concurrency",
    "payload-bound idempotency keys",
    "hash-chained audit events",
    "SQLite transactions and integrity verification",
)

PENDING_CAPABILITIES = (
    "local estimator quotation workspace",
    "PDF quotation rendering",
    "Excel import/export",
    "attachments and source-page evidence",
)


def workflow_readiness() -> dict[str, Any]:
    """Describe what V4 can safely do without changing application state."""
    database_path = config.WORKFLOW_DB_PATH
    running_on_vercel = bool(os.environ.get("VERCEL"))
    storage: dict[str, Any] = {
        "backend": "sqlite" if database_path else "unconfigured",
        "configured": bool(database_path),
        "initialized": False,
        "durable_for_current_runtime": False,
        "integrity": None,
    }
    if database_path:
        path = Path(database_path).expanduser()
        storage["initialized"] = path.is_file()
        storage["durable_for_current_runtime"] = (
            path.is_file()
            and not running_on_vercel
            and not str(path.resolve()).startswith("/tmp/")
        )
        if path.is_file():
            try:
                storage["integrity"] = SQLiteWorkflowStore(path).verify_integrity()
            except (WorkflowError, sqlite3.Error, OSError, ValueError) as exc:
                storage["integrity"] = {"valid": False, "error": str(exc)}

    constraints = [
        "Workflow writes are intentionally local-only and are not exposed "
        "from the Vercel API.",
        "PDF and Excel document workflows are not implemented yet.",
    ]
    if not storage["configured"]:
        constraints.insert(
            1,
            "No local workflow database is configured in this runtime.",
        )
    if storage["integrity"] and not storage["integrity"].get("valid"):
        constraints.insert(0, "Configured workflow store failed integrity checks.")

    return {
        "workflow_version": WORKFLOW_VERSION,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "stage": "foundation",
        "deployment_mode": "local_only",
        "production_ready": False,
        "mutations_exposed": False,
        "storage": storage,
        "state_transitions": {
            "draft": ["submitted"],
            "submitted": ["approved", "rejected"],
            "approved": ["new draft revision"],
            "rejected": ["new draft revision"],
        },
        "implemented_capabilities": list(IMPLEMENTED_CAPABILITIES),
        "pending_capabilities": list(PENDING_CAPABILITIES),
        "deployment_constraints": constraints,
    }


def compact_workflow_readiness() -> dict[str, Any]:
    report = workflow_readiness()
    return {
        "workflow_version": report["workflow_version"],
        "stage": report["stage"],
        "deployment_mode": report["deployment_mode"],
        "production_ready": report["production_ready"],
        "mutations_exposed": report["mutations_exposed"],
        "storage": report["storage"],
        "deployment_constraints": report["deployment_constraints"],
    }
