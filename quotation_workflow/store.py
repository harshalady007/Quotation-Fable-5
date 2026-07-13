"""Transactional SQLite repository for the V4 quotation workflow.

SQLite is the supported local-development store for V4.1.  It provides real
transactions, foreign keys and optimistic concurrency, but it is intentionally
not opened by the Vercel API: a serverless filesystem is not durable storage.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .domain import (
    GENESIS_AUDIT_HASH,
    MONEY_QUANTUM,
    WORKFLOW_SCHEMA_VERSION,
    ApprovalDecision,
    QuotationStatus,
    UserRole,
    WorkflowAuthorizationError,
    WorkflowConflictError,
    WorkflowConfigurationError,
    WorkflowIntegrityError,
    WorkflowNotFoundError,
    WorkflowStateError,
    WorkflowValidationError,
    audit_event_hash,
    canonical_json,
    decimal_text,
    money_value,
    new_id,
    normalize_currency,
    normalize_decision,
    normalize_email,
    normalize_line_items,
    normalize_request_id,
    normalize_role,
    optional_text,
    require_text,
    utc_now,
)


_EMPTY_ITEM_DIGEST = sha256(canonical_json([]).encode("utf-8")).hexdigest()
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SQLiteWorkflowStore:
    """Execute guarded quotation commands against one SQLite database file."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        id_factory: Callable[[str], str] = new_id,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        raw_path = str(database_path or "").strip()
        if not raw_path:
            raise WorkflowConfigurationError("workflow database path is required.")
        if "://" in raw_path:
            raise WorkflowConfigurationError(
                "SQLiteWorkflowStore accepts a filesystem path, not a database URL."
            )
        if raw_path == ":memory:":
            raise WorkflowConfigurationError(
                "Use a temporary database file; separate transactions cannot share "
                "an isolated :memory: database."
            )
        self.database_path = Path(raw_path).expanduser().resolve()
        self._id_factory = id_factory
        self._clock = clock

    def initialize(self) -> dict[str, Any]:
        """Create schema version 1 or verify an existing compatible schema."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path.touch(exist_ok=True)
        try:
            schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkflowConfigurationError(
                f"could not read workflow schema: {exc}"
            ) from exc
        with self._connect() as conn:
            existing_tables = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if existing_tables and "workflow_meta" not in existing_tables:
                raise WorkflowConfigurationError(
                    "refusing to initialize workflow tables inside an unrelated "
                    "non-empty SQLite database."
                )
            conn.executescript(schema)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            row = conn.execute(
                "SELECT value FROM workflow_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO workflow_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(WORKFLOW_SCHEMA_VERSION)),
                )
            else:
                try:
                    current_version = int(row["value"])
                except (TypeError, ValueError) as exc:
                    raise WorkflowConfigurationError(
                        "workflow schema version metadata is invalid."
                    ) from exc
                if current_version != WORKFLOW_SCHEMA_VERSION:
                    raise WorkflowConfigurationError(
                        "workflow schema version is "
                        f"{row['value']}; expected {WORKFLOW_SCHEMA_VERSION}."
                    )
        return {
            "database_path": str(self.database_path),
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "initialized": True,
        }

    def bootstrap_admin(
        self,
        *,
        email: str,
        display_name: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Create the one initial admin; disabled permanently after first use."""
        normalized_email = normalize_email(email)
        normalized_name = require_text(
            display_name, "display_name", max_length=200
        )
        request_id = normalize_request_id(request_id)
        payload = {
            "email": normalized_email,
            "display_name": normalized_name,
            "role": UserRole.ADMIN.value,
        }
        request_hash = self._request_hash(payload)
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT command_name, request_hash, response_json "
                "FROM idempotency_keys WHERE request_id = ? "
                "AND command_name = 'bootstrap_admin' LIMIT 1",
                (request_id,),
            ).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise WorkflowConflictError(
                        "request_id was already used with different bootstrap data."
                    )
                return json.loads(existing["response_json"])
            if conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()[
                "count"
            ]:
                raise WorkflowStateError(
                    "bootstrap_admin is disabled after the first user is created."
                )
            now = self._clock()
            user_id = self._id_factory("usr")
            conn.execute(
                "INSERT INTO users(user_id, email, display_name, role, active, "
                "created_at, updated_at, row_version) VALUES (?, ?, ?, ?, 1, ?, ?, 1)",
                (user_id, normalized_email, normalized_name,
                 UserRole.ADMIN.value, now, now),
            )
            self._append_audit(
                conn,
                aggregate_type="user",
                aggregate_id=user_id,
                action="admin.bootstrapped",
                actor_user_id=user_id,
                payload={**payload, "request_id": request_id},
            )
            response = self._user_snapshot(conn, user_id)
            self._save_idempotency(
                conn, user_id, request_id, "bootstrap_admin",
                request_hash, response,
            )
            return response

    def create_user(
        self,
        *,
        actor_user_id: str,
        email: str,
        display_name: str,
        role: UserRole | str,
        request_id: str,
    ) -> dict[str, Any]:
        normalized_email = normalize_email(email)
        normalized_name = require_text(
            display_name, "display_name", max_length=200
        )
        normalized_role = normalize_role(role)
        request_id = normalize_request_id(request_id)
        request = {
            "email": normalized_email,
            "display_name": normalized_name,
            "role": normalized_role.value,
        }
        with self._transaction() as conn:
            self._require_actor(conn, actor_user_id, {UserRole.ADMIN})
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "create_user", request
            )
            if replay is not None:
                return replay
            user_id = self._id_factory("usr")
            now = self._clock()
            try:
                conn.execute(
                    "INSERT INTO users(user_id, email, display_name, role, active, "
                    "created_at, updated_at, row_version) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?, 1)",
                    (user_id, normalized_email, normalized_name,
                     normalized_role.value, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise WorkflowConflictError(
                    f"a user with email {normalized_email!r} already exists."
                ) from exc
            self._append_audit(
                conn,
                aggregate_type="user",
                aggregate_id=user_id,
                action="user.created",
                actor_user_id=actor_user_id,
                payload=request,
            )
            response = self._user_snapshot(conn, user_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "create_user", request, response
            )
            return response

    def create_project(
        self,
        *,
        actor_user_id: str,
        project_code: str,
        name: str,
        client_name: str,
        currency: str,
        request_id: str,
    ) -> dict[str, Any]:
        code = require_text(
            project_code, "project_code", max_length=50
        ).upper()
        project_name = require_text(name, "name", max_length=300)
        client = require_text(client_name, "client_name", max_length=300)
        normalized_currency = normalize_currency(currency)
        request_id = normalize_request_id(request_id)
        request = {
            "project_code": code,
            "name": project_name,
            "client_name": client,
            "currency": normalized_currency,
        }
        with self._transaction() as conn:
            self._require_actor(
                conn, actor_user_id, {UserRole.ESTIMATOR, UserRole.ADMIN}
            )
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "create_project", request
            )
            if replay is not None:
                return replay
            project_id = self._id_factory("prj")
            now = self._clock()
            try:
                conn.execute(
                    "INSERT INTO projects(project_id, project_code, name, "
                    "client_name, currency, status, created_by, created_at, "
                    "updated_at, row_version) "
                    "VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, 1)",
                    (project_id, code, project_name, client,
                     normalized_currency, actor_user_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise WorkflowConflictError(
                    f"a project with code {code!r} already exists."
                ) from exc
            self._append_audit(
                conn,
                aggregate_type="project",
                aggregate_id=project_id,
                action="project.created",
                actor_user_id=actor_user_id,
                payload=request,
            )
            response = self._project_snapshot(conn, project_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "create_project", request,
                response,
            )
            return response

    def archive_project(
        self,
        *,
        actor_user_id: str,
        project_id: str,
        expected_version: int,
        request_id: str,
    ) -> dict[str, Any]:
        project_id = require_text(project_id, "project_id", max_length=100)
        expected_version = self._expected_version(expected_version)
        request_id = normalize_request_id(request_id)
        request = {
            "project_id": project_id,
            "expected_version": expected_version,
        }
        with self._transaction() as conn:
            self._require_actor(conn, actor_user_id, {UserRole.ADMIN})
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "archive_project", request
            )
            if replay is not None:
                return replay
            project = self._project_row(conn, project_id)
            self._assert_version(project, expected_version, "project")
            if project["status"] == "archived":
                raise WorkflowStateError("project is already archived.")
            open_count = conn.execute(
                "SELECT COUNT(*) AS count FROM quotations WHERE project_id = ? "
                "AND status IN ('draft', 'submitted')",
                (project_id,),
            ).fetchone()["count"]
            if open_count:
                raise WorkflowStateError(
                    "project cannot be archived while draft/submitted quotations exist."
                )
            now = self._clock()
            self._versioned_update(
                conn,
                "UPDATE projects SET status = 'archived', updated_at = ?, "
                "row_version = row_version + 1 WHERE project_id = ? "
                "AND row_version = ?",
                (now, project_id, expected_version),
                "project",
            )
            self._append_audit(
                conn,
                aggregate_type="project",
                aggregate_id=project_id,
                action="project.archived",
                actor_user_id=actor_user_id,
                payload={"new_version": expected_version + 1},
            )
            response = self._project_snapshot(conn, project_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "archive_project", request,
                response,
            )
            return response

    def create_quotation(
        self,
        *,
        actor_user_id: str,
        project_id: str,
        quotation_number: str,
        notes: str = "",
        request_id: str,
    ) -> dict[str, Any]:
        project_id = require_text(project_id, "project_id", max_length=100)
        number = require_text(
            quotation_number, "quotation_number", max_length=100
        ).upper()
        normalized_notes = optional_text(notes, "notes", max_length=4000)
        request_id = normalize_request_id(request_id)
        request = {
            "project_id": project_id,
            "quotation_number": number,
            "notes": normalized_notes,
        }
        with self._transaction() as conn:
            self._require_actor(
                conn, actor_user_id, {UserRole.ESTIMATOR, UserRole.ADMIN}
            )
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "create_quotation", request
            )
            if replay is not None:
                return replay
            project = self._project_row(conn, project_id)
            if project["status"] != "active":
                raise WorkflowStateError(
                    "quotations cannot be created for an archived project."
                )
            quotation_id = self._id_factory("qtn")
            revision_id = self._id_factory("rev")
            now = self._clock()
            try:
                conn.execute(
                    "INSERT INTO quotations(quotation_id, project_id, "
                    "quotation_number, status, current_revision, created_by, "
                    "created_at, updated_at, row_version) "
                    "VALUES (?, ?, ?, 'draft', 1, ?, ?, ?, 1)",
                    (quotation_id, project_id, number, actor_user_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise WorkflowConflictError(
                    f"quotation {number!r} already exists in this project."
                ) from exc
            conn.execute(
                "INSERT INTO quotation_revisions(revision_id, quotation_id, "
                "revision_number, status, notes, currency, subtotal, item_digest, "
                "created_by, created_at) "
                "VALUES (?, ?, 1, 'draft', ?, ?, '0.00', ?, ?, ?)",
                (revision_id, quotation_id, normalized_notes,
                 project["currency"], _EMPTY_ITEM_DIGEST, actor_user_id, now),
            )
            self._append_audit(
                conn,
                aggregate_type="quotation",
                aggregate_id=quotation_id,
                action="quotation.created",
                actor_user_id=actor_user_id,
                payload={
                    "project_id": project_id,
                    "quotation_number": number,
                    "revision_number": 1,
                    "currency": project["currency"],
                },
            )
            response = self._quotation_snapshot(conn, quotation_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "create_quotation", request,
                response,
            )
            return response

    def replace_draft_items(
        self,
        *,
        actor_user_id: str,
        quotation_id: str,
        items: Sequence[Mapping[str, Any]],
        expected_version: int,
        request_id: str,
    ) -> dict[str, Any]:
        quotation_id = require_text(
            quotation_id, "quotation_id", max_length=100
        )
        expected_version = self._expected_version(expected_version)
        request_id = normalize_request_id(request_id)
        normalized, subtotal, item_digest = normalize_line_items(
            items, id_factory=self._id_factory
        )
        request = {
            "quotation_id": quotation_id,
            "expected_version": expected_version,
            "item_digest": item_digest,
            "item_count": len(normalized),
        }
        with self._transaction() as conn:
            actor = self._require_actor(
                conn, actor_user_id, {UserRole.ESTIMATOR, UserRole.ADMIN}
            )
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "replace_draft_items", request
            )
            if replay is not None:
                return replay
            quotation = self._quotation_row(conn, quotation_id)
            self._assert_owner_or_admin(quotation, actor)
            self._assert_version(quotation, expected_version, "quotation")
            if quotation["status"] != QuotationStatus.DRAFT.value:
                raise WorkflowStateError(
                    "only the current draft revision may be edited."
                )
            revision = self._current_revision_row(conn, quotation)
            if revision["status"] != QuotationStatus.DRAFT.value:
                raise WorkflowIntegrityError(
                    "quotation and current revision states do not agree."
                )
            conn.execute(
                "DELETE FROM quotation_items WHERE revision_id = ?",
                (revision["revision_id"],),
            )
            conn.executemany(
                "INSERT INTO quotation_items(item_id, revision_id, line_number, "
                "description, unit, quantity, unit_price, line_total, "
                "pricing_snapshot_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item["item_id"], revision["revision_id"],
                        item["line_number"], item["description"], item["unit"],
                        item["quantity"], item["unit_price"], item["line_total"],
                        item["pricing_snapshot_json"],
                    )
                    for item in normalized
                ],
            )
            conn.execute(
                "UPDATE quotation_revisions SET subtotal = ?, item_digest = ? "
                "WHERE revision_id = ?",
                (subtotal, item_digest, revision["revision_id"]),
            )
            now = self._clock()
            self._versioned_update(
                conn,
                "UPDATE quotations SET updated_at = ?, row_version = "
                "row_version + 1 WHERE quotation_id = ? AND row_version = ?",
                (now, quotation_id, expected_version),
                "quotation",
            )
            self._append_audit(
                conn,
                aggregate_type="quotation",
                aggregate_id=quotation_id,
                action="revision.items_replaced",
                actor_user_id=actor_user_id,
                payload={
                    "revision_number": quotation["current_revision"],
                    "item_count": len(normalized),
                    "subtotal": subtotal,
                    "item_digest": item_digest,
                    "new_version": expected_version + 1,
                },
            )
            response = self._quotation_snapshot(conn, quotation_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "replace_draft_items", request,
                response,
            )
            return response

    def submit_quotation(
        self,
        *,
        actor_user_id: str,
        quotation_id: str,
        expected_version: int,
        request_id: str,
    ) -> dict[str, Any]:
        quotation_id = require_text(
            quotation_id, "quotation_id", max_length=100
        )
        expected_version = self._expected_version(expected_version)
        request_id = normalize_request_id(request_id)
        request = {
            "quotation_id": quotation_id,
            "expected_version": expected_version,
        }
        with self._transaction() as conn:
            actor = self._require_actor(
                conn, actor_user_id, {UserRole.ESTIMATOR, UserRole.ADMIN}
            )
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "submit_quotation", request
            )
            if replay is not None:
                return replay
            quotation = self._quotation_row(conn, quotation_id)
            self._assert_owner_or_admin(quotation, actor)
            self._assert_version(quotation, expected_version, "quotation")
            if quotation["status"] != QuotationStatus.DRAFT.value:
                raise WorkflowStateError("only a draft quotation may be submitted.")
            revision = self._current_revision_row(conn, quotation)
            item_count = conn.execute(
                "SELECT COUNT(*) AS count FROM quotation_items "
                "WHERE revision_id = ?", (revision["revision_id"],)
            ).fetchone()["count"]
            if not item_count:
                raise WorkflowStateError(
                    "a quotation cannot be submitted without line items."
                )
            now = self._clock()
            conn.execute(
                "UPDATE quotation_revisions SET status = 'submitted', "
                "submitted_by = ?, submitted_at = ?, locked_at = ? "
                "WHERE revision_id = ? AND status = 'draft'",
                (actor_user_id, now, now, revision["revision_id"]),
            )
            self._versioned_update(
                conn,
                "UPDATE quotations SET status = 'submitted', submitted_by = ?, "
                "approved_by = NULL, updated_at = ?, row_version = row_version + 1 "
                "WHERE quotation_id = ? AND row_version = ? AND status = 'draft'",
                (actor_user_id, now, quotation_id, expected_version),
                "quotation",
            )
            self._append_audit(
                conn,
                aggregate_type="quotation",
                aggregate_id=quotation_id,
                action="quotation.submitted",
                actor_user_id=actor_user_id,
                payload={
                    "revision_number": quotation["current_revision"],
                    "item_count": item_count,
                    "subtotal": revision["subtotal"],
                    "item_digest": revision["item_digest"],
                    "new_version": expected_version + 1,
                },
            )
            response = self._quotation_snapshot(conn, quotation_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "submit_quotation", request,
                response,
            )
            return response

    def decide_quotation(
        self,
        *,
        actor_user_id: str,
        quotation_id: str,
        decision: ApprovalDecision | str,
        comment: str = "",
        expected_version: int,
        request_id: str,
    ) -> dict[str, Any]:
        quotation_id = require_text(
            quotation_id, "quotation_id", max_length=100
        )
        normalized_decision = normalize_decision(decision)
        normalized_comment = optional_text(comment, "comment", max_length=4000)
        if (normalized_decision is ApprovalDecision.REJECT
                and len(normalized_comment) < 3):
            raise WorkflowValidationError(
                "a rejection requires a comment of at least 3 characters."
            )
        expected_version = self._expected_version(expected_version)
        request_id = normalize_request_id(request_id)
        request = {
            "quotation_id": quotation_id,
            "decision": normalized_decision.value,
            "comment": normalized_comment,
            "expected_version": expected_version,
        }
        with self._transaction() as conn:
            self._require_actor(
                conn, actor_user_id, {UserRole.APPROVER, UserRole.ADMIN}
            )
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "decide_quotation", request
            )
            if replay is not None:
                return replay
            quotation = self._quotation_row(conn, quotation_id)
            self._assert_version(quotation, expected_version, "quotation")
            if quotation["status"] != QuotationStatus.SUBMITTED.value:
                raise WorkflowStateError(
                    "only a submitted quotation may be approved or rejected."
                )
            if actor_user_id in {
                quotation["created_by"], quotation["submitted_by"]
            }:
                raise WorkflowAuthorizationError(
                    "four-eyes control: a quotation creator/submitter cannot "
                    "approve or reject their own revision."
                )
            revision = self._current_revision_row(conn, quotation)
            if revision["status"] != QuotationStatus.SUBMITTED.value:
                raise WorkflowIntegrityError(
                    "quotation and current revision states do not agree."
                )
            now = self._clock()
            decision_id = self._id_factory("dec")
            conn.execute(
                "INSERT INTO approval_decisions(decision_id, quotation_id, "
                "revision_id, decision, comment, actor_user_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (decision_id, quotation_id, revision["revision_id"],
                 normalized_decision.value, normalized_comment,
                 actor_user_id, now),
            )
            conn.execute(
                "UPDATE quotation_revisions SET status = ? "
                "WHERE revision_id = ? AND status = 'submitted'",
                (normalized_decision.value, revision["revision_id"]),
            )
            approved_by = (
                actor_user_id
                if normalized_decision is ApprovalDecision.APPROVE else None
            )
            self._versioned_update(
                conn,
                "UPDATE quotations SET status = ?, approved_by = ?, "
                "updated_at = ?, row_version = row_version + 1 "
                "WHERE quotation_id = ? AND row_version = ? "
                "AND status = 'submitted'",
                (normalized_decision.value, approved_by, now,
                 quotation_id, expected_version),
                "quotation",
            )
            self._append_audit(
                conn,
                aggregate_type="quotation",
                aggregate_id=quotation_id,
                action=f"quotation.{normalized_decision.value}",
                actor_user_id=actor_user_id,
                payload={
                    "revision_number": quotation["current_revision"],
                    "decision_id": decision_id,
                    "comment": normalized_comment,
                    "new_version": expected_version + 1,
                },
            )
            response = self._quotation_snapshot(conn, quotation_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "decide_quotation", request,
                response,
            )
            return response

    def create_revision(
        self,
        *,
        actor_user_id: str,
        quotation_id: str,
        expected_version: int,
        request_id: str,
        notes: str = "",
    ) -> dict[str, Any]:
        quotation_id = require_text(
            quotation_id, "quotation_id", max_length=100
        )
        expected_version = self._expected_version(expected_version)
        request_id = normalize_request_id(request_id)
        normalized_notes = optional_text(notes, "notes", max_length=4000)
        request = {
            "quotation_id": quotation_id,
            "expected_version": expected_version,
            "notes": normalized_notes,
        }
        with self._transaction() as conn:
            actor = self._require_actor(
                conn, actor_user_id, {UserRole.ESTIMATOR, UserRole.ADMIN}
            )
            replay = self._idempotent_replay(
                conn, actor_user_id, request_id, "create_revision", request
            )
            if replay is not None:
                return replay
            quotation = self._quotation_row(conn, quotation_id)
            self._assert_owner_or_admin(quotation, actor)
            self._assert_version(quotation, expected_version, "quotation")
            if quotation["status"] not in {
                QuotationStatus.APPROVED.value,
                QuotationStatus.REJECTED.value,
            }:
                raise WorkflowStateError(
                    "a new revision may be created only after approval/rejection."
                )
            project = self._project_row(conn, quotation["project_id"])
            if project["status"] != "active":
                raise WorkflowStateError(
                    "a new revision cannot be created for an archived project."
                )
            previous = self._current_revision_row(conn, quotation)
            new_number = int(quotation["current_revision"]) + 1
            new_revision_id = self._id_factory("rev")
            now = self._clock()
            conn.execute(
                "INSERT INTO quotation_revisions(revision_id, quotation_id, "
                "revision_number, status, notes, currency, subtotal, item_digest, "
                "created_by, created_at) "
                "VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?)",
                (new_revision_id, quotation_id, new_number, normalized_notes,
                 previous["currency"], previous["subtotal"],
                 previous["item_digest"], actor_user_id, now),
            )
            previous_items = conn.execute(
                "SELECT line_number, description, unit, quantity, unit_price, "
                "line_total, pricing_snapshot_json FROM quotation_items "
                "WHERE revision_id = ? ORDER BY line_number",
                (previous["revision_id"],),
            ).fetchall()
            conn.executemany(
                "INSERT INTO quotation_items(item_id, revision_id, line_number, "
                "description, unit, quantity, unit_price, line_total, "
                "pricing_snapshot_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        self._id_factory("item"), new_revision_id,
                        item["line_number"], item["description"], item["unit"],
                        item["quantity"], item["unit_price"], item["line_total"],
                        item["pricing_snapshot_json"],
                    )
                    for item in previous_items
                ],
            )
            self._versioned_update(
                conn,
                "UPDATE quotations SET status = 'draft', current_revision = ?, "
                "submitted_by = NULL, approved_by = NULL, updated_at = ?, "
                "row_version = row_version + 1 WHERE quotation_id = ? "
                "AND row_version = ?",
                (new_number, now, quotation_id, expected_version),
                "quotation",
            )
            self._append_audit(
                conn,
                aggregate_type="quotation",
                aggregate_id=quotation_id,
                action="revision.created",
                actor_user_id=actor_user_id,
                payload={
                    "from_revision": quotation["current_revision"],
                    "revision_number": new_number,
                    "copied_item_count": len(previous_items),
                    "item_digest": previous["item_digest"],
                    "new_version": expected_version + 1,
                },
            )
            response = self._quotation_snapshot(conn, quotation_id)
            self._save_idempotency_for_request(
                conn, actor_user_id, request_id, "create_revision", request,
                response,
            )
            return response

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            return self._user_snapshot(conn, user_id)

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            return self._project_snapshot(conn, project_id)

    def get_quotation(self, quotation_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            return self._quotation_snapshot(conn, quotation_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT project_id FROM projects ORDER BY created_at, project_id"
            ).fetchall()
            return [self._project_snapshot(conn, row["project_id"]) for row in rows]

    def list_quotations(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._project_row(conn, project_id)
            rows = conn.execute(
                "SELECT quotation_id FROM quotations WHERE project_id = ? "
                "ORDER BY updated_at DESC, quotation_id",
                (project_id,),
            ).fetchall()
            return [
                self._quotation_snapshot(conn, row["quotation_id"])
                for row in rows
            ]

    def get_audit_events(
        self, *, aggregate_type: str | None = None,
        aggregate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[str] = []
        if aggregate_type:
            clauses.append("aggregate_type = ?")
            params.append(aggregate_type)
        if aggregate_id:
            clauses.append("aggregate_id = ?")
            params.append(aggregate_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events" + where + " ORDER BY sequence",
                params,
            ).fetchall()
            return [self._audit_snapshot(row) for row in rows]

    def verify_integrity(self) -> dict[str, Any]:
        """Verify SQLite, foreign keys, revision totals/digests and audit chain."""
        with self._connect() as conn:
            self._verify_schema(conn)
            integrity = [
                row[0] for row in conn.execute("PRAGMA integrity_check").fetchall()
            ]
            if integrity != ["ok"]:
                raise WorkflowIntegrityError(
                    "SQLite integrity_check failed: " + "; ".join(integrity)
                )
            foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise WorkflowIntegrityError(
                    f"foreign_key_check found {len(foreign_keys)} violation(s)."
                )
            revision_count = self._verify_revisions(conn)
            audit = self._verify_audit_chain(conn)
            state_count = self._verify_current_states(conn)
            return {
                "valid": True,
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "revision_count": revision_count,
                "quotation_count": state_count,
                "audit_event_count": audit["event_count"],
                "audit_head_hash": audit["head_hash"],
            }

    @staticmethod
    def _verify_schema(conn: sqlite3.Connection) -> None:
        try:
            row = conn.execute(
                "SELECT value FROM workflow_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise WorkflowIntegrityError(
                "workflow schema metadata is unavailable."
            ) from exc
        if row is None or row["value"] != str(WORKFLOW_SCHEMA_VERSION):
            actual = row["value"] if row else "missing"
            raise WorkflowIntegrityError(
                f"workflow schema version is {actual}; expected "
                f"{WORKFLOW_SCHEMA_VERSION}."
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self.database_path.exists():
            raise WorkflowConfigurationError(
                "workflow database is not initialized; call initialize() first."
            )
        conn = sqlite3.connect(
            str(self.database_path), timeout=10, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA synchronous = FULL")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _expected_version(value: Any) -> int:
        if isinstance(value, bool):
            raise WorkflowValidationError("expected_version must be a positive integer.")
        try:
            version = int(value)
        except (TypeError, ValueError) as exc:
            raise WorkflowValidationError(
                "expected_version must be a positive integer."
            ) from exc
        if version < 1 or str(value).strip() != str(version):
            raise WorkflowValidationError(
                "expected_version must be a positive integer."
            )
        return version

    @staticmethod
    def _request_hash(payload: Mapping[str, Any]) -> str:
        return sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()

    def _idempotent_replay(
        self,
        conn: sqlite3.Connection,
        actor_user_id: str,
        request_id: str,
        command_name: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT command_name, request_hash, response_json "
            "FROM idempotency_keys WHERE actor_user_id = ? AND request_id = ?",
            (actor_user_id, request_id),
        ).fetchone()
        if row is None:
            return None
        request_hash = self._request_hash(request)
        if (row["command_name"] != command_name
                or row["request_hash"] != request_hash):
            raise WorkflowConflictError(
                "request_id was already used for a different workflow command "
                "or payload."
            )
        return json.loads(row["response_json"])

    def _save_idempotency_for_request(
        self,
        conn: sqlite3.Connection,
        actor_user_id: str,
        request_id: str,
        command_name: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        self._save_idempotency(
            conn, actor_user_id, request_id, command_name,
            self._request_hash(request), response,
        )

    def _save_idempotency(
        self,
        conn: sqlite3.Connection,
        actor_user_id: str,
        request_id: str,
        command_name: str,
        request_hash: str,
        response: Mapping[str, Any],
    ) -> None:
        conn.execute(
            "INSERT INTO idempotency_keys(actor_user_id, request_id, "
            "command_name, request_hash, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                actor_user_id,
                request_id,
                command_name,
                request_hash,
                canonical_json(dict(response)),
                self._clock(),
            ),
        )

    def _require_actor(
        self,
        conn: sqlite3.Connection,
        actor_user_id: str,
        allowed_roles: set[UserRole],
    ) -> sqlite3.Row:
        actor = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (actor_user_id,)
        ).fetchone()
        if actor is None:
            raise WorkflowAuthorizationError("workflow actor does not exist.")
        if not actor["active"]:
            raise WorkflowAuthorizationError("workflow actor is inactive.")
        if UserRole(actor["role"]) not in allowed_roles:
            names = ", ".join(sorted(role.value for role in allowed_roles))
            raise WorkflowAuthorizationError(
                f"workflow command requires one of these roles: {names}."
            )
        return actor

    @staticmethod
    def _assert_owner_or_admin(
        quotation: sqlite3.Row, actor: sqlite3.Row
    ) -> None:
        if (quotation["created_by"] != actor["user_id"]
                and actor["role"] != UserRole.ADMIN.value):
            raise WorkflowAuthorizationError(
                "only the quotation owner or an administrator may edit/submit it."
            )

    @staticmethod
    def _assert_version(row: sqlite3.Row, expected: int, entity: str) -> None:
        actual = int(row["row_version"])
        if actual != expected:
            raise WorkflowConflictError(
                f"stale {entity} version: expected {expected}, current is {actual}."
            )

    @staticmethod
    def _versioned_update(
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[Any, ...],
        entity: str,
    ) -> None:
        cursor = conn.execute(sql, params)
        if cursor.rowcount != 1:
            raise WorkflowConflictError(
                f"{entity} changed concurrently; reload it before retrying."
            )

    def _append_audit(
        self,
        conn: sqlite3.Connection,
        *,
        aggregate_type: str,
        aggregate_id: str,
        action: str,
        actor_user_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        previous = conn.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else GENESIS_AUDIT_HASH
        event_id = self._id_factory("evt")
        occurred_at = self._clock()
        payload_json = canonical_json(dict(payload))
        event_hash = audit_event_hash(
            previous_hash=previous_hash,
            event_id=event_id,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            action=action,
            actor_user_id=actor_user_id,
            occurred_at=occurred_at,
            payload_json=payload_json,
        )
        conn.execute(
            "INSERT INTO audit_events(event_id, aggregate_type, aggregate_id, "
            "action, actor_user_id, occurred_at, payload_json, previous_hash, "
            "event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id, aggregate_type, aggregate_id, action, actor_user_id,
                occurred_at, payload_json, previous_hash, event_hash,
            ),
        )

    @staticmethod
    def _user_row(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise WorkflowNotFoundError(f"user {user_id!r} was not found.")
        return row

    @staticmethod
    def _project_row(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise WorkflowNotFoundError(f"project {project_id!r} was not found.")
        return row

    @staticmethod
    def _quotation_row(conn: sqlite3.Connection, quotation_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM quotations WHERE quotation_id = ?", (quotation_id,)
        ).fetchone()
        if row is None:
            raise WorkflowNotFoundError(
                f"quotation {quotation_id!r} was not found."
            )
        return row

    @staticmethod
    def _current_revision_row(
        conn: sqlite3.Connection, quotation: sqlite3.Row
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM quotation_revisions WHERE quotation_id = ? "
            "AND revision_number = ?",
            (quotation["quotation_id"], quotation["current_revision"]),
        ).fetchone()
        if row is None:
            raise WorkflowIntegrityError(
                "quotation current_revision does not reference a revision."
            )
        return row

    def _user_snapshot(
        self, conn: sqlite3.Connection, user_id: str
    ) -> dict[str, Any]:
        row = self._user_row(conn, user_id)
        return {
            "user_id": row["user_id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "role": row["role"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "row_version": int(row["row_version"]),
        }

    def _project_snapshot(
        self, conn: sqlite3.Connection, project_id: str
    ) -> dict[str, Any]:
        row = self._project_row(conn, project_id)
        quotation_count = conn.execute(
            "SELECT COUNT(*) AS count FROM quotations WHERE project_id = ?",
            (project_id,),
        ).fetchone()["count"]
        return {
            "project_id": row["project_id"],
            "project_code": row["project_code"],
            "name": row["name"],
            "client_name": row["client_name"],
            "currency": row["currency"],
            "status": row["status"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "row_version": int(row["row_version"]),
            "quotation_count": int(quotation_count),
        }

    def _quotation_snapshot(
        self, conn: sqlite3.Connection, quotation_id: str
    ) -> dict[str, Any]:
        quotation = self._quotation_row(conn, quotation_id)
        project = self._project_row(conn, quotation["project_id"])
        revisions = conn.execute(
            "SELECT * FROM quotation_revisions WHERE quotation_id = ? "
            "ORDER BY revision_number",
            (quotation_id,),
        ).fetchall()
        revision_snapshots = []
        for revision in revisions:
            items = conn.execute(
                "SELECT * FROM quotation_items WHERE revision_id = ? "
                "ORDER BY line_number",
                (revision["revision_id"],),
            ).fetchall()
            decision = conn.execute(
                "SELECT * FROM approval_decisions WHERE revision_id = ?",
                (revision["revision_id"],),
            ).fetchone()
            revision_snapshots.append({
                "revision_id": revision["revision_id"],
                "revision_number": int(revision["revision_number"]),
                "status": revision["status"],
                "notes": revision["notes"],
                "currency": revision["currency"],
                "subtotal": revision["subtotal"],
                "item_digest": revision["item_digest"],
                "created_by": revision["created_by"],
                "submitted_by": revision["submitted_by"],
                "created_at": revision["created_at"],
                "submitted_at": revision["submitted_at"],
                "locked_at": revision["locked_at"],
                "items": [self._item_snapshot(item) for item in items],
                "decision": self._decision_snapshot(decision) if decision else None,
            })
        return {
            "quotation_id": quotation["quotation_id"],
            "project_id": quotation["project_id"],
            "project_code": project["project_code"],
            "quotation_number": quotation["quotation_number"],
            "status": quotation["status"],
            "current_revision": int(quotation["current_revision"]),
            "created_by": quotation["created_by"],
            "submitted_by": quotation["submitted_by"],
            "approved_by": quotation["approved_by"],
            "created_at": quotation["created_at"],
            "updated_at": quotation["updated_at"],
            "row_version": int(quotation["row_version"]),
            "revisions": revision_snapshots,
        }

    @staticmethod
    def _item_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "item_id": row["item_id"],
            "line_number": int(row["line_number"]),
            "description": row["description"],
            "unit": row["unit"],
            "quantity": row["quantity"],
            "unit_price": row["unit_price"],
            "line_total": row["line_total"],
            "pricing_snapshot": (
                json.loads(row["pricing_snapshot_json"])
                if row["pricing_snapshot_json"] else None
            ),
        }

    @staticmethod
    def _decision_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "decision_id": row["decision_id"],
            "decision": row["decision"],
            "comment": row["comment"],
            "actor_user_id": row["actor_user_id"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _audit_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": int(row["sequence"]),
            "event_id": row["event_id"],
            "aggregate_type": row["aggregate_type"],
            "aggregate_id": row["aggregate_id"],
            "action": row["action"],
            "actor_user_id": row["actor_user_id"],
            "occurred_at": row["occurred_at"],
            "payload": json.loads(row["payload_json"]),
            "previous_hash": row["previous_hash"],
            "event_hash": row["event_hash"],
        }

    def _verify_audit_chain(self, conn: sqlite3.Connection) -> dict[str, Any]:
        expected_previous = GENESIS_AUDIT_HASH
        count = 0
        rows = conn.execute(
            "SELECT * FROM audit_events ORDER BY sequence"
        ).fetchall()
        for row in rows:
            if row["previous_hash"] != expected_previous:
                raise WorkflowIntegrityError(
                    f"audit chain breaks before sequence {row['sequence']}."
                )
            calculated = audit_event_hash(
                previous_hash=row["previous_hash"],
                event_id=row["event_id"],
                aggregate_type=row["aggregate_type"],
                aggregate_id=row["aggregate_id"],
                action=row["action"],
                actor_user_id=row["actor_user_id"],
                occurred_at=row["occurred_at"],
                payload_json=row["payload_json"],
            )
            if calculated != row["event_hash"]:
                raise WorkflowIntegrityError(
                    f"audit event hash is invalid at sequence {row['sequence']}."
                )
            expected_previous = row["event_hash"]
            count += 1
        return {"event_count": count, "head_hash": expected_previous}

    def _verify_revisions(self, conn: sqlite3.Connection) -> int:
        revisions = conn.execute(
            "SELECT * FROM quotation_revisions ORDER BY quotation_id, "
            "revision_number"
        ).fetchall()
        for revision in revisions:
            items = conn.execute(
                "SELECT line_number, description, unit, quantity, unit_price, "
                "line_total, pricing_snapshot_json FROM quotation_items "
                "WHERE revision_id = ? ORDER BY line_number",
                (revision["revision_id"],),
            ).fetchall()
            subtotal = money_value("0")
            digest_items = []
            for expected_line, item in enumerate(items, start=1):
                if int(item["line_number"]) != expected_line:
                    raise WorkflowIntegrityError(
                        f"revision {revision['revision_id']} has a line-number gap."
                    )
                try:
                    expected_total = (
                        money_value(item["unit_price"])
                        * self._stored_quantity(item["quantity"])
                    ).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
                except WorkflowValidationError as exc:
                    raise WorkflowIntegrityError(
                        f"revision {revision['revision_id']} contains invalid "
                        "decimal data."
                    ) from exc
                if decimal_text(expected_total, quantum=MONEY_QUANTUM) != item[
                    "line_total"
                ]:
                    raise WorkflowIntegrityError(
                        f"revision {revision['revision_id']} contains an invalid "
                        "line total."
                    )
                subtotal += expected_total
                digest_items.append({
                    key: item[key]
                    for key in (
                        "line_number", "description", "unit", "quantity",
                        "unit_price", "line_total", "pricing_snapshot_json",
                    )
                })
            subtotal_text = decimal_text(subtotal, quantum=MONEY_QUANTUM)
            if subtotal_text != revision["subtotal"]:
                raise WorkflowIntegrityError(
                    f"revision {revision['revision_id']} subtotal is invalid."
                )
            digest = sha256(
                canonical_json(digest_items).encode("utf-8")
            ).hexdigest()
            if digest != revision["item_digest"]:
                raise WorkflowIntegrityError(
                    f"revision {revision['revision_id']} item digest is invalid."
                )
            if revision["status"] != "draft" and not revision["locked_at"]:
                raise WorkflowIntegrityError(
                    f"non-draft revision {revision['revision_id']} is not locked."
                )
        return len(revisions)

    @staticmethod
    def _stored_quantity(value: str):
        try:
            quantity = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise WorkflowIntegrityError("stored quantity is not a decimal.") from exc
        if not quantity.is_finite() or quantity <= 0:
            raise WorkflowIntegrityError("stored quantity must be positive.")
        return quantity

    def _verify_current_states(self, conn: sqlite3.Connection) -> int:
        rows = conn.execute(
            "SELECT q.quotation_id, q.status AS quotation_status, "
            "r.status AS revision_status FROM quotations q "
            "LEFT JOIN quotation_revisions r ON r.quotation_id = q.quotation_id "
            "AND r.revision_number = q.current_revision"
        ).fetchall()
        for row in rows:
            if row["revision_status"] is None:
                raise WorkflowIntegrityError(
                    f"quotation {row['quotation_id']} has no current revision."
                )
            if row["quotation_status"] != row["revision_status"]:
                raise WorkflowIntegrityError(
                    f"quotation {row['quotation_id']} state disagrees with its "
                    "current revision."
                )
        return len(rows)
