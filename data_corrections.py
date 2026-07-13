"""Versioned, reviewable corrections for historical quotation records.

The source workbook remains immutable.  Corrections are keyed by the stable
``record_id`` created during cleaning and are applied only when their status
is ``approved``.  Proposed corrections can therefore travel through review
without affecting production pricing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from cleaner import compose_search_text, normalize_unit, source_lineage
from context_readiness import normalize_quotation_date
from pricing_context import normalize_pricing_context


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, SCHEMA_VERSION}
ALLOWED_SET_FIELDS = {
    "unit", "rate", "source", "date", "category", "section", "location",
    "supplier", "quantity", "remarks", "quotation_number", "project",
    "client", "contractor", "currency",
}
ALLOWED_ATTRIBUTE_FIELDS = {
    "item_type", "subtype", "material", "finish", "grade", "scope",
    "civil_works", "mobility", "capacity_l", "compartments", "length_mm",
    "width_mm", "height_mm", "diameter_mm", "thickness_mm", "features",
}
ALLOWED_STATUSES = {"proposed", "approved", "rejected"}
CONTEXT_SET_FIELDS = {"quantity", "supplier", "location", "date"}
FIELD_REVIEW_KEYS = {
    "evidence_reference", "reason", "reviewed_by", "reviewed_at",
}
_EMPTY_CONTEXT_TEXT = {"", "nan", "none", "null", "n/a", "na", "unknown"}
_EMPTY_AUDIT_TEXT = _EMPTY_CONTEXT_TEXT | {"tbd", "todo", "pending"}


class DataCorrectionError(Exception):
    """Raised when the correction manifest is malformed or unsafe."""


def _empty_manifest() -> dict:
    return {"schema_version": SCHEMA_VERSION, "records": []}


def validate_correction_manifest(manifest: dict) -> dict:
    """Strictly validate and return a correction-manifest object.

    Schema 1 remains readable for the pre-V3 correction history. New
    manifests use schema 2, which requires field-level evidence for approved
    quantity, supplier, location and quotation-date changes.
    """
    if not isinstance(manifest, dict):
        raise DataCorrectionError("Correction manifest must be an object.")
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise DataCorrectionError(
            "Correction manifest schema_version must be one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    records = manifest.get("records")
    if not isinstance(records, list):
        raise DataCorrectionError("Correction manifest 'records' must be a list.")

    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DataCorrectionError(f"Correction #{index + 1} must be an object.")
        record_id = str(record.get("record_id") or "").strip()
        if not record_id:
            raise DataCorrectionError(f"Correction #{index + 1} has no record_id.")
        if record_id in seen:
            raise DataCorrectionError(f"Duplicate correction record_id: {record_id}.")
        seen.add(record_id)
        status = record.get("status", "proposed")
        if status not in ALLOWED_STATUSES:
            raise DataCorrectionError(
                f"Correction {record_id} has unsupported status {status!r}."
            )
        if not str(record.get("reason") or "").strip():
            raise DataCorrectionError(f"Correction {record_id} requires a reason.")
        if status == "approved" and not str(record.get("reviewed_by") or "").strip():
            raise DataCorrectionError(
                f"Approved correction {record_id} requires reviewed_by."
            )
        set_fields = record.get("set") or {}
        attributes = record.get("attributes") or {}
        field_reviews = record.get("field_reviews") or {}
        if not isinstance(set_fields, dict) or not isinstance(attributes, dict):
            raise DataCorrectionError(
                f"Correction {record_id} set/attributes values must be objects."
            )
        if not isinstance(field_reviews, dict):
            raise DataCorrectionError(
                f"Correction {record_id} field_reviews must be an object."
            )
        if ("exclude_from_pricing" in record
                and not isinstance(record["exclude_from_pricing"], bool)):
            raise DataCorrectionError(
                f"Correction {record_id} exclude_from_pricing must be true/false."
            )
        unknown = set(set_fields) - ALLOWED_SET_FIELDS
        unknown_attrs = set(attributes) - ALLOWED_ATTRIBUTE_FIELDS
        if unknown or unknown_attrs:
            fields = sorted(unknown | unknown_attrs)
            raise DataCorrectionError(
                f"Correction {record_id} contains unsupported fields: {fields}."
            )
        unknown_reviews = set(field_reviews) - CONTEXT_SET_FIELDS
        orphan_reviews = set(field_reviews) - set(set_fields)
        if unknown_reviews or orphan_reviews:
            fields = sorted(unknown_reviews | orphan_reviews)
            raise DataCorrectionError(
                f"Correction {record_id} has invalid field_reviews: {fields}."
            )
        for field, review in field_reviews.items():
            if not isinstance(review, dict):
                raise DataCorrectionError(
                    f"Correction {record_id} review for {field} must be an object."
                )
            unknown_metadata = set(review) - FIELD_REVIEW_KEYS
            if unknown_metadata:
                raise DataCorrectionError(
                    f"Correction {record_id} review for {field} contains "
                    f"unsupported metadata: {sorted(unknown_metadata)}."
                )
            for key in FIELD_REVIEW_KEYS:
                value = str(review.get(key) or "").strip()
                if not value or value.lower() in _EMPTY_AUDIT_TEXT:
                    raise DataCorrectionError(
                        f"Correction {record_id} review for {field} requires a "
                        f"specific {key}."
                    )
            try:
                normalize_quotation_date(review["reviewed_at"], strict=True)
            except ValueError as exc:
                raise DataCorrectionError(
                    f"Correction {record_id} review for {field} has an invalid "
                    "reviewed_at date."
                ) from exc
        if schema_version >= 2 and status == "approved":
            missing_reviews = (set(set_fields) & CONTEXT_SET_FIELDS) - set(
                field_reviews
            )
            if missing_reviews:
                raise DataCorrectionError(
                    f"Approved correction {record_id} requires field-level "
                    "evidence for: " + ", ".join(sorted(missing_reviews)) + "."
                )
            if "quantity" in set_fields:
                try:
                    quantity = float(set_fields["quantity"])
                except (TypeError, ValueError) as exc:
                    raise DataCorrectionError(
                        f"Correction {record_id} quantity must be numeric."
                    ) from exc
                if not math.isfinite(quantity) or quantity <= 0:
                    raise DataCorrectionError(
                        f"Correction {record_id} quantity must be positive."
                    )
            for field in ("supplier", "location"):
                if field not in set_fields:
                    continue
                value = normalize_pricing_context({field: set_fields[field]}).get(
                    field
                )
                if not value or value in _EMPTY_CONTEXT_TEXT:
                    raise DataCorrectionError(
                        f"Correction {record_id} {field} must be a specific, "
                        "non-placeholder value."
                    )
            if "date" in set_fields:
                try:
                    normalize_quotation_date(set_fields["date"], strict=True)
                except ValueError as exc:
                    raise DataCorrectionError(
                        f"Correction {record_id} date must be a valid quotation date."
                    ) from exc
    return manifest


def load_correction_manifest(path: str | Path | None) -> dict:
    """Load and strictly validate a correction manifest."""
    if not path:
        return _empty_manifest()
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise DataCorrectionError(f"Correction manifest not found: {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataCorrectionError(
            f"Could not read correction manifest {manifest_path}: {exc}"
        ) from exc
    return validate_correction_manifest(manifest)


def apply_data_corrections(df: pd.DataFrame, path: str | Path | None) -> pd.DataFrame:
    """Apply approved corrections and attach an audit summary to ``df.attrs``."""
    out = df.copy()
    manifest = load_correction_manifest(path)
    out["attribute_overrides"] = [{} for _ in range(len(out))]
    out["correction_ids"] = [[] for _ in range(len(out))]
    out["manual_pricing_exclusion"] = False

    stats = {
        "manifest_schema_version": manifest["schema_version"],
        "records": len(manifest["records"]),
        "proposed": 0,
        "approved": 0,
        "rejected": 0,
        "applied": 0,
        "unmatched_approved": 0,
    }

    duplicate_ids = out.loc[out["record_id"].duplicated(False), "record_id"]
    if not duplicate_ids.empty:
        raise DataCorrectionError(
            "Dataset contains duplicate stable record_id values: "
            + ", ".join(sorted(set(duplicate_ids.astype(str))))
            + ". Corrections cannot be applied ambiguously."
        )
    by_id = {str(value): index for index, value in out["record_id"].items()}
    for correction in manifest["records"]:
        status = correction.get("status", "proposed")
        stats[status] += 1
        if status != "approved":
            continue
        record_id = str(correction["record_id"])
        row_index = by_id.get(record_id)
        if row_index is None:
            stats["unmatched_approved"] += 1
            continue

        set_fields = dict(correction.get("set") or {})
        if "rate" in set_fields:
            try:
                set_fields["rate"] = float(set_fields["rate"])
            except (TypeError, ValueError) as exc:
                raise DataCorrectionError(
                    f"Correction {record_id} rate must be numeric."
                ) from exc
            if not math.isfinite(set_fields["rate"]) or set_fields["rate"] <= 0:
                raise DataCorrectionError(
                    f"Correction {record_id} rate must be positive."
                )
        if "quantity" in set_fields:
            value = set_fields["quantity"]
            if value in (None, ""):
                set_fields["quantity"] = None
            else:
                try:
                    set_fields["quantity"] = float(value)
                except (TypeError, ValueError) as exc:
                    raise DataCorrectionError(
                        f"Correction {record_id} quantity must be numeric."
                    ) from exc
                if (not math.isfinite(set_fields["quantity"])
                        or set_fields["quantity"] <= 0):
                    raise DataCorrectionError(
                        f"Correction {record_id} quantity must be positive."
                    )
        for field in ("supplier", "location"):
            if field not in set_fields:
                continue
            normalized = normalize_pricing_context({field: set_fields[field]}).get(
                field
            )
            if not normalized or normalized in _EMPTY_CONTEXT_TEXT:
                raise DataCorrectionError(
                    f"Correction {record_id} {field} must be a specific, "
                    "non-placeholder value."
                )
            set_fields[field] = normalized
        if "date" in set_fields:
            raw_date = set_fields["date"]
            quotation_date = normalize_quotation_date(raw_date)
            if raw_date not in (None, "") and quotation_date is None:
                raise DataCorrectionError(
                    f"Correction {record_id} date must be a valid quotation date."
                )
            set_fields["date"] = quotation_date
        for field, value in set_fields.items():
            out.at[row_index, field] = value
        if "unit" in set_fields:
            out.at[row_index, "unit_norm"] = normalize_unit(set_fields["unit"])
        if "source" in set_fields:
            source_group, source_revision = source_lineage(set_fields["source"])
            out.at[row_index, "source_group"] = source_group
            out.at[row_index, "source_revision"] = source_revision
        if "date" in set_fields:
            out.at[row_index, "quotation_date"] = set_fields["date"]

        try:
            overrides = normalize_pricing_context(
                dict(correction.get("attributes") or {})
            )
        except ValueError as exc:
            raise DataCorrectionError(
                f"Correction {record_id} has invalid attribute values: {exc}"
            ) from exc
        out.at[row_index, "attribute_overrides"] = overrides
        out.at[row_index, "manual_pricing_exclusion"] = bool(
            correction.get("exclude_from_pricing", False)
        )
        out.at[row_index, "correction_ids"] = [record_id]
        stats["applied"] += 1

    if stats["unmatched_approved"]:
        raise DataCorrectionError(
            f"{stats['unmatched_approved']} approved correction(s) no longer match "
            "the loaded dataset. Regenerate the review queue before pricing."
        )

    if stats["applied"]:
        out["search_text"] = out.apply(compose_search_text, axis=1)

    out.attrs.update(df.attrs)
    out.attrs["corrections"] = stats
    return out


def correction_summary(df: pd.DataFrame) -> dict:
    """Return the manifest/application summary attached to a dataset."""
    return dict(df.attrs.get("corrections") or {
        "manifest_schema_version": SCHEMA_VERSION,
        "records": 0,
        "proposed": 0,
        "approved": 0,
        "rejected": 0,
        "applied": 0,
        "unmatched_approved": 0,
    })
