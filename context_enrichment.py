"""Auditable V3 workflow for collecting missing pricing context.

The historical workbook remains immutable. This module exports a review
template keyed by stable record IDs and imports only explicitly approved,
source-backed values into the versioned correction manifest. A dataset
fingerprint prevents a stale spreadsheet from being applied after the data or
pricing code changed.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from context_readiness import CONTEXT_FIELDS, normalize_quotation_date
from data_corrections import (SCHEMA_VERSION, DataCorrectionError,
                              load_correction_manifest,
                              validate_correction_manifest)
from family_readiness import dataset_fingerprint
from pricing_context import normalize_pricing_context
from pricing_dataset import load_pricing_dataset
from production_pricing import SUPPORTED_INPUT_FAMILIES
from similarity_search import SimilaritySearcher


REVIEW_SCHEMA_VERSION = 1
REVIEW_STATUSES = {"pending", "approved", "rejected"}
REVIEW_FIELD_TO_SET_FIELD = {
    "quantity": "quantity",
    "supplier": "supplier",
    "location": "location",
    "quotation_date": "date",
}
SET_FIELD_TO_CONTEXT_FIELD = {
    value: key for key, value in REVIEW_FIELD_TO_SET_FIELD.items()
}
EMPTY_CONTEXT_TEXT = {"", "nan", "none", "null", "n/a", "na", "unknown"}

CONTEXT_REVIEW_COLUMNS = [
    "review_schema_version",
    "dataset_fingerprint",
    "record_id",
    "source_group",
    "source_revision",
    "source",
    "family",
    "subtype",
    "material",
    "scope",
    "project",
    "client",
    "contractor",
    "description",
    "unit",
    "rate",
    "pricing_eligible",
    "data_quality_flags",
    "fields_needing_review",
    "current_quantity",
    "current_supplier",
    "current_location",
    "current_quotation_date",
    "review_status",
    "reviewed_quantity",
    "reviewed_supplier",
    "reviewed_location",
    "reviewed_quotation_date",
    "allow_override",
    "evidence_reference",
    "review_reason",
    "reviewed_by",
    "reviewed_at",
]


class ContextEnrichmentError(RuntimeError):
    """Raised when a context review cannot be imported safely."""


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _cell_text(value) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    # Exported spreadsheet text that could be interpreted as a formula is
    # prefixed with an apostrophe. Remove only that exact safety prefix.
    if len(text) > 1 and text[0] == "'":
        unescaped = text[1:].lstrip()
        if unescaped and unescaped[0] in "=+-@":
            text = text[1:]
    return text


def _normalize_context_value(field: str, value, *, strict: bool = False):
    if _is_missing(value):
        if strict:
            raise ContextEnrichmentError(f"{field} cannot be blank.")
        return None
    if field == "quantity":
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ContextEnrichmentError("quantity must be numeric.") from exc
            return None
        if not math.isfinite(number) or number <= 0:
            if strict:
                raise ContextEnrichmentError("quantity must be a positive number.")
            return None
        return int(number) if number.is_integer() else number
    if field == "quotation_date":
        try:
            return normalize_quotation_date(value, strict=strict)
        except ValueError as exc:
            raise ContextEnrichmentError(str(exc)) from exc
    if field not in {"supplier", "location"}:
        raise ContextEnrichmentError(f"Unsupported context field: {field}.")
    normalized = normalize_pricing_context({field: value}).get(field)
    if not normalized or normalized in EMPTY_CONTEXT_TEXT:
        if strict:
            raise ContextEnrichmentError(
                f"{field} must be a specific value, not a placeholder."
            )
        return None
    return normalized


def _selected_fields(fields: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = tuple(fields or CONTEXT_FIELDS)
    unknown = set(selected) - set(CONTEXT_FIELDS)
    if unknown:
        raise ContextEnrichmentError(
            "Unknown context field(s): " + ", ".join(sorted(unknown)) + "."
        )
    if not selected:
        raise ContextEnrichmentError("At least one context field is required.")
    return selected


def build_context_review_queue(
    data_path: str | None = None,
    corrections_path: str | None = None,
    family: str | None = None,
    fields: list[str] | tuple[str, ...] | None = None,
    *,
    include_quarantined: bool = False,
    missing_only: bool = True,
) -> tuple[list[dict], dict]:
    """Build a stable, prioritized context-review queue and compact profile."""
    selected = _selected_fields(fields)
    if family and family not in SUPPORTED_INPUT_FAMILIES:
        raise ContextEnrichmentError(f"Unsupported product family: {family}.")

    dataset = load_pricing_dataset(data_path, corrections_path)
    fingerprint = dataset_fingerprint(dataset)
    searcher = SimilaritySearcher(dataset)
    scoped: list[tuple[int, pd.Series, dict]] = []
    excluded_quarantined = 0

    for position, (_, row) in enumerate(dataset.iterrows()):
        attrs = searcher.item_attrs[position]
        item_family = attrs.get("item_type")
        if item_family not in SUPPORTED_INPUT_FAMILIES:
            continue
        if family and item_family != family:
            continue
        flags = list(row.get("data_quality_flags") or [])
        if "superseded_revision" in flags:
            continue
        if not include_quarantined and not bool(row.get("pricing_eligible")):
            excluded_quarantined += 1
            continue
        scoped.append((position, row, attrs))

    field_profile = {}
    for field in selected:
        populated = [
            _normalize_context_value(field, row.get(field)) is not None
            for _, row, _ in scoped
        ]
        populated_groups = {
            str(row.get("source_group"))
            for is_populated, (_, row, _) in zip(populated, scoped)
            if is_populated and str(row.get("source_group") or "").strip()
        }
        field_profile[field] = {
            "rows": len(scoped),
            "populated_rows": int(sum(populated)),
            "missing_rows": int(len(scoped) - sum(populated)),
            "coverage": round(sum(populated) / len(scoped), 4) if scoped else 0.0,
            "populated_quote_groups": len(populated_groups),
        }

    family_priority = {
        "bench": 0,
        "litter bin": 1,
        "bike rack": 2,
        "bollard": 3,
        "recycle bin": 4,
        "planter": 5,
    }
    queue = []
    for _, row, attrs in scoped:
        missing = [
            field for field in selected
            if _normalize_context_value(field, row.get(field)) is None
        ]
        if missing_only and not missing:
            continue
        item = {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "dataset_fingerprint": fingerprint,
            "record_id": str(row.get("record_id") or ""),
            "source_group": str(row.get("source_group") or ""),
            "source_revision": int(row.get("source_revision") or 0),
            "source": row.get("source"),
            "family": attrs.get("item_type"),
            "subtype": attrs.get("subtype"),
            "material": attrs.get("material"),
            "scope": attrs.get("scope"),
            "project": row.get("project"),
            "client": row.get("client"),
            "contractor": row.get("contractor"),
            "description": row.get("full_description"),
            "unit": row.get("unit_norm") or None,
            "rate": float(row.get("rate")),
            "pricing_eligible": bool(row.get("pricing_eligible")),
            "data_quality_flags": list(row.get("data_quality_flags") or []),
            "fields_needing_review": missing,
            "current_quantity": _normalize_context_value(
                "quantity", row.get("quantity")
            ),
            "current_supplier": _normalize_context_value(
                "supplier", row.get("supplier")
            ),
            "current_location": _normalize_context_value(
                "location", row.get("location")
            ),
            "current_quotation_date": _normalize_context_value(
                "quotation_date", row.get("quotation_date")
            ),
            "review_status": "pending",
            "reviewed_quantity": "",
            "reviewed_supplier": "",
            "reviewed_location": "",
            "reviewed_quotation_date": "",
            "allow_override": "",
            "evidence_reference": "",
            "review_reason": "",
            "reviewed_by": "",
            "reviewed_at": "",
            "_priority": (
                -len(missing),
                family_priority.get(attrs.get("item_type"), 99),
                str(row.get("source_group") or ""),
                str(row.get("record_id") or ""),
            ),
        }
        queue.append(item)

    queue.sort(key=lambda item: item["_priority"])
    for item in queue:
        item.pop("_priority", None)
    summary = {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "family": family,
        "selected_fields": list(selected),
        "scope_rows": len(scoped),
        "scope_quote_groups": len({
            str(row.get("source_group")) for _, row, _ in scoped
            if str(row.get("source_group") or "").strip()
        }),
        "queue_rows": len(queue),
        "excluded_quarantined_rows": excluded_quarantined,
        "missing_only": missing_only,
        "fields": field_profile,
    }
    return queue, summary


def _spreadsheet_safe(value):
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _review_frame(rows: list[dict]) -> pd.DataFrame:
    rendered = []
    for item in rows:
        row = {column: item.get(column, "") for column in CONTEXT_REVIEW_COLUMNS}
        for field in ("data_quality_flags", "fields_needing_review"):
            values = row.get(field) or []
            if not isinstance(values, str):
                row[field] = "; ".join(str(value) for value in values)
        row = {key: _spreadsheet_safe(value) for key, value in row.items()}
        rendered.append(row)
    return pd.DataFrame(rendered, columns=CONTEXT_REVIEW_COLUMNS)


def write_context_review(
    path: str | Path,
    rows: list[dict],
    summary: dict,
    *,
    file_format: str | None = None,
) -> Path:
    """Write a CSV, XLSX or JSON estimator template."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    format_name = (file_format or output.suffix.lstrip(".")).lower()
    if format_name not in {"csv", "xlsx", "json"}:
        raise ContextEnrichmentError("Review format must be csv, xlsx or json.")
    frame = _review_frame(rows)
    if format_name == "csv":
        frame.to_csv(output, index=False)
    elif format_name == "json":
        payload = {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "dataset_fingerprint": summary.get("dataset_fingerprint"),
            "summary": _json_safe(summary),
            "rows": _json_safe(rows),
        }
        output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        instructions = pd.DataFrame([
            ("1", "Review only against the original quotation/PDF."),
            ("2", "Fill one or more reviewed_* fields; never guess."),
            ("3", "Set review_status to approved only after verification."),
            ("4", "Provide evidence_reference, reason, reviewer and review date."),
            ("5", "Use allow_override=yes only to replace an existing value."),
            ("6", "Import refuses stale fingerprints and changed record IDs."),
        ], columns=["Step", "Instruction"])
        summary_rows = []
        for field, profile in (summary.get("fields") or {}).items():
            summary_rows.append({"field": field, **profile})
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Context Review", index=False)
            instructions.to_excel(writer, sheet_name="Instructions", index=False)
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name="Coverage Summary", index=False
            )
            sheet = writer.book["Context Review"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    return output


def _read_review_rows(path: str | Path) -> list[dict]:
    review_path = Path(path)
    if not review_path.exists():
        raise ContextEnrichmentError(f"Context review file not found: {review_path}.")
    suffix = review_path.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(review_path.read_text(encoding="utf-8"))
            rows = payload.get("rows") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ContextEnrichmentError("JSON review must contain a rows list.")
            return rows
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(
                review_path, sheet_name="Context Review", dtype=object,
                keep_default_na=False,
            )
        else:
            frame = pd.read_csv(review_path, dtype=object, keep_default_na=False)
    except ContextEnrichmentError:
        raise
    except Exception as exc:
        raise ContextEnrichmentError(
            f"Could not read context review {review_path}: {exc}"
        ) from exc
    return frame.to_dict(orient="records")


def _yes(value) -> bool:
    return _cell_text(value).lower() in {"1", "true", "yes", "y"}


def _same_context_value(field: str, left, right) -> bool:
    return (
        _normalize_context_value(field, left)
        == _normalize_context_value(field, right)
    )


def merge_context_reviews(
    review_path: str | Path,
    data_path: str | None = None,
    corrections_path: str | None = None,
) -> tuple[dict, dict]:
    """Validate approved review rows and merge them into schema-2 corrections."""
    dataset = load_pricing_dataset(data_path, corrections_path)
    current_fingerprint = dataset_fingerprint(dataset)
    rows = _read_review_rows(review_path)
    if not rows:
        raise ContextEnrichmentError("Context review contains no rows.")

    required_columns = {
        "review_schema_version", "dataset_fingerprint", "record_id",
        "source_group", "review_status", "allow_override",
        "reviewed_quantity", "reviewed_supplier", "reviewed_location",
        "reviewed_quotation_date", "evidence_reference", "review_reason",
        "reviewed_by", "reviewed_at",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ContextEnrichmentError(
            "Context review is missing columns: "
            + ", ".join(sorted(missing_columns)) + "."
        )

    versions = {_cell_text(row.get("review_schema_version")) for row in rows}
    if versions != {str(REVIEW_SCHEMA_VERSION)}:
        raise ContextEnrichmentError(
            f"Context review schema must be {REVIEW_SCHEMA_VERSION}."
        )
    fingerprints = {_cell_text(row.get("dataset_fingerprint")) for row in rows}
    if fingerprints != {current_fingerprint}:
        raise ContextEnrichmentError(
            "Context review is stale or belongs to a different dataset. "
            "Export a new review template before importing."
        )

    by_record_id = {str(value): index for index, value in dataset["record_id"].items()}
    seen_review_ids: set[str] = set()
    approved_records = []
    skipped = Counter()
    field_counts = Counter()

    for number, review_row in enumerate(rows, start=2):
        record_id = _cell_text(review_row.get("record_id"))
        if not record_id:
            raise ContextEnrichmentError(f"Review row {number} has no record_id.")
        if record_id in seen_review_ids:
            raise ContextEnrichmentError(
                f"Review contains duplicate record_id {record_id}."
            )
        seen_review_ids.add(record_id)
        row_index = by_record_id.get(record_id)
        if row_index is None:
            raise ContextEnrichmentError(
                f"Review record_id {record_id} no longer exists in the dataset."
            )
        current = dataset.loc[row_index]
        expected_group = str(current.get("source_group") or "")
        if _cell_text(review_row.get("source_group")) != expected_group:
            raise ContextEnrichmentError(
                f"Review record {record_id} has a changed source_group."
            )

        status = _cell_text(review_row.get("review_status")).lower() or "pending"
        if status not in REVIEW_STATUSES:
            raise ContextEnrichmentError(
                f"Review record {record_id} has invalid status {status!r}."
            )
        if status != "approved":
            skipped[status] += 1
            continue

        set_fields = {}
        for context_field, set_field in REVIEW_FIELD_TO_SET_FIELD.items():
            reviewed = review_row.get(f"reviewed_{context_field}")
            if _is_missing(reviewed):
                continue
            value = _normalize_context_value(context_field, reviewed, strict=True)
            current_value = current.get(context_field)
            if _same_context_value(context_field, current_value, value):
                continue
            if (_normalize_context_value(context_field, current_value) is not None
                    and not _yes(review_row.get("allow_override"))):
                raise ContextEnrichmentError(
                    f"Review record {record_id} would overwrite existing "
                    f"{context_field}; set allow_override=yes after verifying it."
                )
            set_fields[set_field] = value
            field_counts[context_field] += 1

        if not set_fields:
            raise ContextEnrichmentError(
                f"Approved review record {record_id} contains no new context values."
            )
        evidence = _cell_text(review_row.get("evidence_reference"))
        reason = _cell_text(review_row.get("review_reason"))
        reviewer = _cell_text(review_row.get("reviewed_by"))
        reviewed_at_raw = _cell_text(review_row.get("reviewed_at"))
        if not evidence or not reason or not reviewer or not reviewed_at_raw:
            raise ContextEnrichmentError(
                f"Approved review record {record_id} requires evidence_reference, "
                "review_reason, reviewed_by and reviewed_at."
            )
        try:
            reviewed_at = normalize_quotation_date(reviewed_at_raw, strict=True)
        except ValueError as exc:
            raise ContextEnrichmentError(
                f"Review record {record_id} has an invalid reviewed_at date."
            ) from exc
        field_reviews = {
            field: {
                "evidence_reference": evidence,
                "reason": reason,
                "reviewed_by": reviewer,
                "reviewed_at": reviewed_at,
            }
            for field in set_fields
        }
        approved_records.append({
            "record_id": record_id,
            "status": "approved",
            "reason": reason,
            "reviewed_by": reviewer,
            "set": set_fields,
            "field_reviews": field_reviews,
        })

    manifest_path = (
        config.CORRECTIONS_PATH if corrections_path is None else corrections_path
    )
    manifest = copy.deepcopy(load_correction_manifest(manifest_path))
    existing_by_id = {
        str(record["record_id"]): position
        for position, record in enumerate(manifest["records"])
    }
    new_records = 0
    merged_records = 0
    for incoming in approved_records:
        record_id = incoming["record_id"]
        position = existing_by_id.get(record_id)
        if position is None:
            manifest["records"].append(incoming)
            existing_by_id[record_id] = len(manifest["records"]) - 1
            new_records += 1
            continue
        existing = manifest["records"][position]
        if existing.get("status", "proposed") != "approved":
            raise ContextEnrichmentError(
                f"Record {record_id} already has a non-approved correction. "
                "Resolve it before importing context so unrelated proposed "
                "fields are not activated accidentally."
            )
        merged = copy.deepcopy(existing)
        merged_set = dict(merged.get("set") or {})
        for set_field, value in incoming["set"].items():
            context_field = SET_FIELD_TO_CONTEXT_FIELD[set_field]
            if (set_field in merged_set
                    and not _same_context_value(
                        context_field, merged_set[set_field], value
                    )):
                raise ContextEnrichmentError(
                    f"Record {record_id} already has a conflicting approved "
                    f"{context_field}; the importer will not overwrite it."
                )
            merged_set[set_field] = value
        merged_reviews = dict(merged.get("field_reviews") or {})
        for set_field, metadata in incoming["field_reviews"].items():
            if set_field in merged_reviews and merged_reviews[set_field] != metadata:
                raise ContextEnrichmentError(
                    f"Record {record_id} already has different audit evidence "
                    f"for {SET_FIELD_TO_CONTEXT_FIELD[set_field]}."
                )
            merged_reviews[set_field] = metadata
        merged["set"] = merged_set
        merged["field_reviews"] = merged_reviews
        manifest["records"][position] = merged
        merged_records += 1

    manifest["schema_version"] = SCHEMA_VERSION
    try:
        validate_correction_manifest(manifest)
    except DataCorrectionError as exc:
        raise ContextEnrichmentError(
            f"Merged correction manifest is unsafe: {exc}"
        ) from exc
    summary = {
        "dataset_fingerprint": current_fingerprint,
        "review_rows": len(rows),
        "approved_review_rows": len(approved_records),
        "new_correction_records": new_records,
        "merged_correction_records": merged_records,
        "skipped_pending_rows": skipped["pending"],
        "skipped_rejected_rows": skipped["rejected"],
        "approved_fields": dict(sorted(field_counts.items())),
        "manifest_records": len(manifest["records"]),
        "manifest_schema_version": manifest["schema_version"],
    }
    return manifest, summary


def write_correction_manifest(path: str | Path, manifest: dict) -> Path:
    """Atomically write a validated correction manifest."""
    validate_correction_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2) + "\n"
    temporary_name = None
    target_mode = (output.stat().st_mode & 0o777) if output.exists() else 0o644
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.chmod(temporary_name, target_mode)
        Path(temporary_name).replace(output)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return output
