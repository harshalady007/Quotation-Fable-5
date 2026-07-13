"""V4.1 quotation workflow transaction, approval and integrity tests."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from quotation_workflow import (
    SQLiteWorkflowStore,
    WorkflowAuthorizationError,
    WorkflowConflictError,
    WorkflowConfigurationError,
    WorkflowIntegrityError,
    WorkflowStateError,
    WorkflowValidationError,
)
from quotation_workflow.readiness import workflow_readiness


@dataclass
class WorkflowFixture:
    store: SQLiteWorkflowStore
    database: str
    admin_id: str
    estimator_id: str
    approver_id: str
    project_id: str


@pytest.fixture
def workflow(tmp_path) -> WorkflowFixture:
    database = tmp_path / "workflow.db"
    store = SQLiteWorkflowStore(database)
    initialized = store.initialize()
    assert initialized["schema_version"] == 1
    admin = store.bootstrap_admin(
        email="admin@example.com",
        display_name="Workflow Admin",
        request_id="bootstrap-admin-001",
    )
    estimator = store.create_user(
        actor_user_id=admin["user_id"],
        email="estimator@example.com",
        display_name="Estimator One",
        role="estimator",
        request_id="create-user-estimator-001",
    )
    approver = store.create_user(
        actor_user_id=admin["user_id"],
        email="approver@example.com",
        display_name="Approver One",
        role="approver",
        request_id="create-user-approver-001",
    )
    project = store.create_project(
        actor_user_id=estimator["user_id"],
        project_code="PRJ-001",
        name="Test Park",
        client_name="Test Client",
        currency="aed",
        request_id="create-project-001",
    )
    return WorkflowFixture(
        store=store,
        database=str(database),
        admin_id=admin["user_id"],
        estimator_id=estimator["user_id"],
        approver_id=approver["user_id"],
        project_id=project["project_id"],
    )


def _new_quote(workflow: WorkflowFixture, suffix: str = "001") -> dict:
    return workflow.store.create_quotation(
        actor_user_id=workflow.estimator_id,
        project_id=workflow.project_id,
        quotation_number=f"QT-{suffix}",
        notes="Initial estimator draft",
        request_id=f"create-quotation-{suffix}",
    )


def _items(price: str = "100.00") -> list[dict]:
    return [
        {
            "description": "Mild steel backless bench L1800 x W500 x H450mm",
            "unit": "Nos",
            "quantity": "1.2500",
            "unit_price": price,
            "pricing_snapshot": {
                "pricing_version": "3.3.0-universal-estimate",
                "pricing_policy": "always_estimate",
                "status": "priced",
                "predicted_unit_price": float(price),
                "confidence": "Low",
            },
        },
        {
            "description": "Project identification plate",
            "unit": "No",
            "quantity": "3",
            "unit_price": "0.10",
        },
    ]


def _prepare_submitted(workflow: WorkflowFixture, suffix: str = "001") -> dict:
    quote = _new_quote(workflow, suffix)
    quote = workflow.store.replace_draft_items(
        actor_user_id=workflow.estimator_id,
        quotation_id=quote["quotation_id"],
        items=_items(),
        expected_version=quote["row_version"],
        request_id=f"replace-items-{suffix}",
    )
    return workflow.store.submit_quotation(
        actor_user_id=workflow.estimator_id,
        quotation_id=quote["quotation_id"],
        expected_version=quote["row_version"],
        request_id=f"submit-quotation-{suffix}",
    )


def test_complete_approval_lifecycle_uses_decimal_totals_and_locked_revision(
    workflow,
):
    quote = _new_quote(workflow)
    assert quote["status"] == "draft"
    assert quote["row_version"] == 1

    quote = workflow.store.replace_draft_items(
        actor_user_id=workflow.estimator_id,
        quotation_id=quote["quotation_id"],
        items=_items(),
        expected_version=1,
        request_id="replace-items-approval-001",
    )
    revision = quote["revisions"][0]
    assert quote["row_version"] == 2
    assert revision["subtotal"] == "125.30"
    assert revision["items"][0]["quantity"] == "1.2500"
    assert revision["items"][0]["line_total"] == "125.00"

    submitted = workflow.store.submit_quotation(
        actor_user_id=workflow.estimator_id,
        quotation_id=quote["quotation_id"],
        expected_version=2,
        request_id="submit-approval-001",
    )
    assert submitted["status"] == "submitted"
    assert submitted["revisions"][0]["locked_at"]

    with pytest.raises(WorkflowStateError, match="only the current draft"):
        workflow.store.replace_draft_items(
            actor_user_id=workflow.estimator_id,
            quotation_id=quote["quotation_id"],
            items=_items("120.00"),
            expected_version=3,
            request_id="edit-locked-revision-001",
        )

    approved = workflow.store.decide_quotation(
        actor_user_id=workflow.approver_id,
        quotation_id=quote["quotation_id"],
        decision="approved",
        comment="Commercial review complete.",
        expected_version=3,
        request_id="approve-quotation-001",
    )
    assert approved["status"] == "approved"
    assert approved["approved_by"] == workflow.approver_id
    assert approved["revisions"][0]["decision"]["decision"] == "approved"
    assert workflow.store.verify_integrity()["valid"]


def test_four_eyes_control_rolls_back_self_approval(workflow):
    quote = workflow.store.create_quotation(
        actor_user_id=workflow.admin_id,
        project_id=workflow.project_id,
        quotation_number="QT-ADMIN",
        request_id="create-admin-quote-001",
    )
    quote = workflow.store.replace_draft_items(
        actor_user_id=workflow.admin_id,
        quotation_id=quote["quotation_id"],
        items=_items(),
        expected_version=1,
        request_id="replace-admin-items-001",
    )
    submitted = workflow.store.submit_quotation(
        actor_user_id=workflow.admin_id,
        quotation_id=quote["quotation_id"],
        expected_version=2,
        request_id="submit-admin-quote-001",
    )
    audit_before = len(workflow.store.get_audit_events())
    with pytest.raises(WorkflowAuthorizationError, match="four-eyes"):
        workflow.store.decide_quotation(
            actor_user_id=workflow.admin_id,
            quotation_id=quote["quotation_id"],
            decision="approved",
            expected_version=3,
            request_id="self-approve-admin-001",
        )
    after = workflow.store.get_quotation(quote["quotation_id"])
    assert after["status"] == "submitted"
    assert after["row_version"] == submitted["row_version"]
    assert len(workflow.store.get_audit_events()) == audit_before


def test_rejection_creates_a_new_draft_without_mutating_old_revision(workflow):
    submitted = _prepare_submitted(workflow, "REV")
    rejected = workflow.store.decide_quotation(
        actor_user_id=workflow.approver_id,
        quotation_id=submitted["quotation_id"],
        decision="rejected",
        comment="Update the material specification.",
        expected_version=3,
        request_id="reject-revision-001",
    )
    old_digest = rejected["revisions"][0]["item_digest"]
    revised = workflow.store.create_revision(
        actor_user_id=workflow.estimator_id,
        quotation_id=submitted["quotation_id"],
        expected_version=4,
        notes="Material corrected after review",
        request_id="create-revision-002",
    )
    assert revised["status"] == "draft"
    assert revised["current_revision"] == 2
    assert revised["revisions"][0]["status"] == "rejected"
    assert revised["revisions"][0]["item_digest"] == old_digest
    assert revised["revisions"][1]["item_digest"] == old_digest

    edited = workflow.store.replace_draft_items(
        actor_user_id=workflow.estimator_id,
        quotation_id=submitted["quotation_id"],
        items=_items("120.00"),
        expected_version=5,
        request_id="replace-revision-002",
    )
    assert edited["revisions"][0]["subtotal"] == "125.30"
    assert edited["revisions"][1]["subtotal"] == "150.30"
    assert edited["revisions"][0]["item_digest"] == old_digest
    assert edited["revisions"][1]["item_digest"] != old_digest


def test_optimistic_concurrency_and_payload_bound_idempotency(workflow):
    quote = _new_quote(workflow, "IDEMP")
    first = workflow.store.replace_draft_items(
        actor_user_id=workflow.estimator_id,
        quotation_id=quote["quotation_id"],
        items=_items(),
        expected_version=1,
        request_id="replace-idempotent-001",
    )
    replay = workflow.store.replace_draft_items(
        actor_user_id=workflow.estimator_id,
        quotation_id=quote["quotation_id"],
        items=_items(),
        expected_version=1,
        request_id="replace-idempotent-001",
    )
    assert replay == first
    assert len(workflow.store.get_audit_events(
        aggregate_type="quotation", aggregate_id=quote["quotation_id"]
    )) == 2

    with pytest.raises(WorkflowConflictError, match="different workflow"):
        workflow.store.replace_draft_items(
            actor_user_id=workflow.estimator_id,
            quotation_id=quote["quotation_id"],
            items=_items("999.00"),
            expected_version=1,
            request_id="replace-idempotent-001",
        )
    with pytest.raises(WorkflowConflictError, match="stale quotation"):
        workflow.store.submit_quotation(
            actor_user_id=workflow.estimator_id,
            quotation_id=quote["quotation_id"],
            expected_version=1,
            request_id="submit-stale-version-001",
        )


@pytest.mark.parametrize(
    "item, message",
    [
        ({"description": "Valid item", "unit": "no", "quantity": 0,
          "unit_price": 1}, "greater than zero"),
        ({"description": "Valid item", "unit": "no", "quantity": 1,
          "unit_price": "NaN"}, "finite"),
        ({"description": "x", "unit": "no", "quantity": 1,
          "unit_price": 1}, "at least 3"),
    ],
)
def test_invalid_commercial_values_never_reach_the_database(
    workflow, item, message,
):
    suffix = message.replace(" ", "-").upper()
    quote = _new_quote(workflow, suffix)
    with pytest.raises(WorkflowValidationError, match=message):
        workflow.store.replace_draft_items(
            actor_user_id=workflow.estimator_id,
            quotation_id=quote["quotation_id"],
            items=[item],
            expected_version=1,
            request_id=f"invalid-item-{suffix}",
        )
    unchanged = workflow.store.get_quotation(quote["quotation_id"])
    assert unchanged["row_version"] == 1
    assert unchanged["revisions"][0]["items"] == []


def test_integrity_check_detects_audit_tampering(workflow):
    quote = _new_quote(workflow, "TAMPER")
    assert workflow.store.verify_integrity()["valid"]
    with sqlite3.connect(workflow.database) as conn:
        conn.execute(
            "UPDATE audit_events SET payload_json = ? WHERE aggregate_id = ?",
            ('{"tampered":true}', quote["quotation_id"]),
        )
        conn.commit()
    with pytest.raises(WorkflowIntegrityError, match="hash is invalid"):
        workflow.store.verify_integrity()


def test_bootstrap_is_replay_safe_and_refuses_an_unrelated_database(tmp_path):
    database = tmp_path / "bootstrap.db"
    store = SQLiteWorkflowStore(database)
    store.initialize()
    first = store.bootstrap_admin(
        email="admin@example.com",
        display_name="Workflow Admin",
        request_id="bootstrap-replay-001",
    )
    assert store.initialize()["initialized"]
    replay = store.bootstrap_admin(
        email="admin@example.com",
        display_name="Workflow Admin",
        request_id="bootstrap-replay-001",
    )
    assert replay == first
    with pytest.raises(WorkflowConflictError, match="different bootstrap"):
        store.bootstrap_admin(
            email="other@example.com",
            display_name="Other Admin",
            request_id="bootstrap-replay-001",
        )

    unrelated = tmp_path / "unrelated.db"
    with sqlite3.connect(unrelated) as conn:
        conn.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(WorkflowConfigurationError, match="unrelated"):
        SQLiteWorkflowStore(unrelated).initialize()


def test_archived_project_is_frozen_against_new_revisions(workflow):
    submitted = _prepare_submitted(workflow, "ARCHIVE")
    approved = workflow.store.decide_quotation(
        actor_user_id=workflow.approver_id,
        quotation_id=submitted["quotation_id"],
        decision="approved",
        expected_version=3,
        request_id="approve-before-archive-001",
    )
    project = workflow.store.archive_project(
        actor_user_id=workflow.admin_id,
        project_id=workflow.project_id,
        expected_version=1,
        request_id="archive-project-001",
    )
    assert project["status"] == "archived"
    with pytest.raises(WorkflowStateError, match="archived project"):
        workflow.store.create_revision(
            actor_user_id=workflow.estimator_id,
            quotation_id=approved["quotation_id"],
            expected_version=4,
            request_id="revision-after-archive-001",
        )


def test_workflow_readiness_never_claims_production_writes(monkeypatch):
    monkeypatch.setattr("config.WORKFLOW_DB_PATH", "")
    report = workflow_readiness()
    assert report["workflow_version"] == "4.1.0-foundation"
    assert report["deployment_mode"] == "local_only"
    assert not report["production_ready"]
    assert not report["mutations_exposed"]
    assert not report["storage"]["configured"]
    assert any(
        "intentionally local-only" in item
        for item in report["deployment_constraints"]
    )
