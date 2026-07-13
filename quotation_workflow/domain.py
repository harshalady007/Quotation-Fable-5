"""Domain types and deterministic validation for the V4 workflow.

All commercial arithmetic uses :class:`decimal.Decimal`.  The database stores
canonical decimal strings, which prevents binary floating-point drift from
changing quotation totals or audit hashes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4


WORKFLOW_VERSION = "4.1.0-foundation"
WORKFLOW_SCHEMA_VERSION = 1
GENESIS_AUDIT_HASH = "0" * 64

MONEY_QUANTUM = Decimal("0.01")
QUANTITY_QUANTUM = Decimal("0.0001")
MAX_LINE_ITEMS = 500
MAX_PRICING_SNAPSHOT_BYTES = 100_000
MAX_QUANTITY = Decimal("1000000000")
MAX_UNIT_PRICE = Decimal("1000000000000")

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{7,127}$")


class WorkflowError(RuntimeError):
    """Base class for expected workflow failures."""


class WorkflowValidationError(WorkflowError):
    """A command contains invalid or incomplete data."""


class WorkflowNotFoundError(WorkflowError):
    """A requested workflow aggregate does not exist."""


class WorkflowAuthorizationError(WorkflowError):
    """An actor is not allowed to perform a workflow command."""


class WorkflowStateError(WorkflowError):
    """A command is invalid for the aggregate's current state."""


class WorkflowConflictError(WorkflowError):
    """Optimistic concurrency or idempotency detected a conflict."""


class WorkflowIntegrityError(WorkflowError):
    """Persistent workflow state or its audit chain failed verification."""


class WorkflowConfigurationError(WorkflowError):
    """Workflow persistence or its runtime is not safely configured."""


class UserRole(str, Enum):
    ESTIMATOR = "estimator"
    APPROVER = "approver"
    ADMIN = "admin"


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class QuotationStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalDecision(str, Enum):
    APPROVE = "approved"
    REJECT = "rejected"


def new_id(prefix: str) -> str:
    """Return an opaque, sortable-enough identifier with a readable prefix."""
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> str:
    """Return a canonical UTC timestamp used by records and audit hashes."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def require_text(value: Any, field: str, *, min_length: int = 1,
                 max_length: int = 500) -> str:
    """Normalize user text without silently accepting empty placeholders."""
    text = " ".join(str(value or "").strip().split())
    if len(text) < min_length:
        raise WorkflowValidationError(
            f"{field} must contain at least {min_length} character(s)."
        )
    if len(text) > max_length:
        raise WorkflowValidationError(
            f"{field} must contain at most {max_length} characters."
        )
    return text


def optional_text(value: Any, field: str, *, max_length: int = 2000) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > max_length:
        raise WorkflowValidationError(
            f"{field} must contain at most {max_length} characters."
        )
    return text


def normalize_email(value: Any) -> str:
    email = require_text(value, "email", max_length=254).casefold()
    if not _EMAIL_RE.fullmatch(email):
        raise WorkflowValidationError("email must be a valid address.")
    return email


def normalize_currency(value: Any) -> str:
    currency = require_text(value, "currency", min_length=3, max_length=3).upper()
    if not _CURRENCY_RE.fullmatch(currency):
        raise WorkflowValidationError("currency must be a three-letter ISO code.")
    return currency


def normalize_request_id(value: Any) -> str:
    request_id = str(value or "").strip()
    if not _REQUEST_ID_RE.fullmatch(request_id):
        raise WorkflowValidationError(
            "request_id must be 8-128 URL-safe characters and begin with "
            "a letter or number."
        )
    return request_id


def normalize_role(value: UserRole | str) -> UserRole:
    try:
        return value if isinstance(value, UserRole) else UserRole(str(value))
    except ValueError as exc:
        raise WorkflowValidationError(
            f"role must be one of: {', '.join(role.value for role in UserRole)}."
        ) from exc


def normalize_decision(value: ApprovalDecision | str) -> ApprovalDecision:
    try:
        return (
            value if isinstance(value, ApprovalDecision)
            else ApprovalDecision(str(value))
        )
    except ValueError as exc:
        raise WorkflowValidationError(
            "decision must be either 'approved' or 'rejected'."
        ) from exc


def decimal_value(value: Any, field: str, *, quantum: Decimal,
                  allow_zero: bool, maximum: Decimal) -> Decimal:
    """Parse and quantize a finite decimal using explicit half-up rounding."""
    if isinstance(value, bool) or value in (None, ""):
        raise WorkflowValidationError(f"{field} must be a decimal number.")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise WorkflowValidationError(f"{field} must be a decimal number.") from exc
    if not parsed.is_finite():
        raise WorkflowValidationError(f"{field} must be finite.")
    if parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "zero or greater" if allow_zero else "greater than zero"
        raise WorkflowValidationError(f"{field} must be {qualifier}.")
    if parsed > maximum:
        raise WorkflowValidationError(
            f"{field} must not exceed {format(maximum, 'f')}."
        )
    try:
        return parsed.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise WorkflowValidationError(
            f"{field} cannot be represented at the required precision."
        ) from exc


def money_value(value: Any, field: str = "unit_price") -> Decimal:
    return decimal_value(
        value, field, quantum=MONEY_QUANTUM, allow_zero=True,
        maximum=MAX_UNIT_PRICE,
    )


def quantity_value(value: Any) -> Decimal:
    return decimal_value(
        value, "quantity", quantum=QUANTITY_QUANTUM, allow_zero=False,
        maximum=MAX_QUANTITY,
    )


def decimal_text(value: Decimal, *, quantum: Decimal) -> str:
    """Return a fixed-scale representation suitable for persistence/hashing."""
    return format(value.quantize(quantum, rounding=ROUND_HALF_UP), "f")


def canonical_json(value: Any) -> str:
    """Serialize a JSON value deterministically and reject NaN/unknown types."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError(f"value is not valid JSON: {exc}") from exc


def normalize_pricing_snapshot(value: Any) -> str | None:
    if value in (None, {}):
        return None
    if not isinstance(value, Mapping):
        raise WorkflowValidationError("pricing_snapshot must be a JSON object.")
    rendered = canonical_json(dict(value))
    if len(rendered.encode("utf-8")) > MAX_PRICING_SNAPSHOT_BYTES:
        raise WorkflowValidationError(
            "pricing_snapshot exceeds the 100KB per-line audit limit."
        )
    return rendered


def normalize_line_items(
    items: Iterable[Mapping[str, Any]],
    *,
    id_factory: Callable[[str], str] = new_id,
) -> tuple[list[dict[str, Any]], str, str]:
    """Validate line items and return canonical records, subtotal and digest."""
    if isinstance(items, (str, bytes, Mapping)):
        raise WorkflowValidationError("items must be a list of line-item objects.")
    raw_items = list(items)
    if not raw_items:
        raise WorkflowValidationError("a quotation requires at least one line item.")
    if len(raw_items) > MAX_LINE_ITEMS:
        raise WorkflowValidationError(
            f"a quotation may contain at most {MAX_LINE_ITEMS} line items."
        )

    normalized: list[dict[str, Any]] = []
    subtotal = Decimal("0.00")
    digest_items = []
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, Mapping):
            raise WorkflowValidationError(
                f"line item {index} must be a JSON object."
            )
        description = require_text(
            raw.get("description"),
            f"items[{index}].description",
            min_length=3,
            max_length=2000,
        )
        unit = require_text(
            raw.get("unit"), f"items[{index}].unit", max_length=30
        ).casefold()
        quantity = quantity_value(raw.get("quantity"))
        unit_price = money_value(
            raw.get("unit_price"), f"items[{index}].unit_price"
        )
        line_total = (quantity * unit_price).quantize(
            MONEY_QUANTUM, rounding=ROUND_HALF_UP
        )
        subtotal += line_total
        snapshot = normalize_pricing_snapshot(raw.get("pricing_snapshot"))
        item = {
            "item_id": id_factory("item"),
            "line_number": index,
            "description": description,
            "unit": unit,
            "quantity": decimal_text(quantity, quantum=QUANTITY_QUANTUM),
            "unit_price": decimal_text(unit_price, quantum=MONEY_QUANTUM),
            "line_total": decimal_text(line_total, quantum=MONEY_QUANTUM),
            "pricing_snapshot_json": snapshot,
        }
        normalized.append(item)
        digest_items.append({
            key: item[key]
            for key in (
                "line_number", "description", "unit", "quantity",
                "unit_price", "line_total", "pricing_snapshot_json",
            )
        })
    subtotal_text = decimal_text(subtotal, quantum=MONEY_QUANTUM)
    digest = sha256(canonical_json(digest_items).encode("utf-8")).hexdigest()
    return normalized, subtotal_text, digest


def audit_event_hash(*, previous_hash: str, event_id: str,
                     aggregate_type: str, aggregate_id: str, action: str,
                     actor_user_id: str, occurred_at: str,
                     payload_json: str) -> str:
    """Hash one audit event, chaining it to every event that preceded it."""
    components = (
        previous_hash,
        event_id,
        aggregate_type,
        aggregate_id,
        action,
        actor_user_id,
        occurred_at,
        payload_json,
    )
    return sha256("\x1f".join(components).encode("utf-8")).hexdigest()
