"""Evidence gates for V3 quantity, supplier, date and location adjustments.

Context fields are accepted and audited before they are allowed to change a
price.  A field becomes *ready for modelling* only when the historical data
has enough coverage, independent quotation groups and variation.  It still
requires an explicit allow-list change after quotation-lineage-held-out
calibration; readiness alone never activates an adjustment in production.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

import pandas as pd

import config
from production_pricing import (APPROVED_AUTO_FAMILIES,
                                FAMILY_REQUIRED_FIELDS,
                                SUPPORTED_INPUT_FAMILIES)


CONTEXT_READINESS_SCHEMA_VERSION = 1
CONTEXT_FIELDS = ("quantity", "supplier", "quotation_date", "location")
COHORT_NUMERIC_FIELDS = (
    "capacity_l", "compartments", "length_mm", "width_mm", "height_mm",
    "diameter_mm", "thickness_mm",
)
COHORT_TEXT_FIELDS = ("civil_works", "mobility", "finish", "grade")

# V3 starts in shadow mode. Adjustment IDs are family-scoped (for example,
# ``bench.quantity``); evidence for one family can never activate another.
# Add an ID only after its calibrated adjustment and base family both pass
# their independent quotation-lineage release gates.
APPROVED_CONTEXT_ADJUSTMENTS: set[str] = set()
IMPLEMENTED_CONTEXT_ADJUSTMENTS: set[str] = set()

_ORDINAL_SUFFIX = re.compile(r"(?i)(\d{1,2})(st|nd|rd|th)\b")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMPTY_TEXT = {"", "nan", "none", "null", "n/a", "na", "unknown"}


class ContextReadinessError(RuntimeError):
    """Raised when a contextual adjustment is configured unsafely."""


def context_adjustment_id(family: str, field: str) -> str:
    """Return the canonical family-scoped context-adjustment ID."""
    return f"{family}.{field}"


def _validate_adjustment_ids(values: set[str]) -> set[str]:
    valid = {
        context_adjustment_id(family, field)
        for family in SUPPORTED_INPUT_FAMILIES
        for field in CONTEXT_FIELDS
    }
    invalid = set(values) - valid
    if invalid:
        raise ContextReadinessError(
            "Unsupported context adjustment ID(s): "
            + ", ".join(sorted(invalid))
        )
    return set(values)


def normalize_quotation_date(value, *, strict: bool = False) -> str | None:
    """Return an ISO date from the workbook's mixed human-readable formats."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    text = _ORDINAL_SUFFIX.sub(r"\1", text)
    try:
        if _ISO_DATE.fullmatch(text):
            return date.fromisoformat(text).isoformat()
        parsed = pd.to_datetime(text, errors="raise", dayfirst=True)
    except (TypeError, ValueError, OverflowError) as exc:
        if strict:
            raise ValueError(
                "quotation_date must be a valid date such as 2026-07-13."
            ) from exc
        return None
    return parsed.date().isoformat()


def _usable_text(value) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).strip().lower().split())
    return None if text in _EMPTY_TEXT else text


def build_context_evidence_frame(
    dataset: pd.DataFrame,
    item_attrs: list[dict] | None,
) -> pd.DataFrame:
    """Return eligible rows with conservative exact-product cohort IDs."""
    frame = dataset.reset_index(drop=True).copy()
    if item_attrs is not None and len(item_attrs) != len(frame):
        raise ValueError("item_attrs must align one-to-one with the pricing dataset.")
    frame["_family"] = [
        (item_attrs[index] or {}).get("item_type")
        if item_attrs is not None else None
        for index in range(len(frame))
    ]
    cohort_signatures = []
    for index, row in frame.iterrows():
        attrs = (item_attrs[index] or {}) if item_attrs is not None else {}
        family = attrs.get("item_type")
        required = FAMILY_REQUIRED_FIELDS.get(family, ())
        required_values = [attrs.get(field) for field in required]
        unit = _usable_text(row.get("unit_norm"))
        scope = _usable_text(attrs.get("scope"))
        material = _usable_text(attrs.get("material"))
        valid_required = True
        for value in required_values:
            try:
                valid_required = valid_required and math.isfinite(float(value))
                valid_required = valid_required and float(value) > 0
            except (TypeError, ValueError):
                valid_required = False
        if (family not in SUPPORTED_INPUT_FAMILIES or not unit or not scope
                or not material or not valid_required):
            cohort_signatures.append(None)
            continue
        numeric_signature = []
        for field in COHORT_NUMERIC_FIELDS:
            value = attrs.get(field)
            try:
                formatted = (
                    format(float(value), ".8g")
                    if value is not None and math.isfinite(float(value))
                    else ""
                )
            except (TypeError, ValueError):
                formatted = ""
            numeric_signature.append(f"{field}={formatted}")
        text_signature = [
            f"{field}={_usable_text(attrs.get(field)) or ''}"
            for field in COHORT_TEXT_FIELDS
        ]
        raw_features = attrs.get("features") or []
        if isinstance(raw_features, str):
            raw_features = [raw_features]
        features = sorted({
            value for value in (_usable_text(item) for item in raw_features)
            if value
        })
        cohort_signatures.append("\x1f".join([
            family,
            _usable_text(attrs.get("subtype")) or "",
            material,
            unit,
            scope,
            *text_signature,
            *numeric_signature,
            "features=" + "|".join(features),
        ]))
    frame["_cohort"] = cohort_signatures
    if "pricing_eligible" in frame:
        frame = frame[frame["pricing_eligible"].fillna(False).astype(bool)].copy()
    frame["_source_group"] = frame.get(
        "source_group", pd.Series("", index=frame.index)
    ).map(_usable_text)
    return frame


def _field_evidence(
    frame: pd.DataFrame,
    field: str,
    adjustment_id: str | None = None,
) -> dict:
    rows = int(len(frame))
    groups_total = int(frame["_source_group"].dropna().nunique()) if rows else 0

    if field == "quantity":
        values = pd.to_numeric(
            frame.get(field, pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        usable = values.where(values > 0)
    elif field == "quotation_date":
        source = frame.get(field)
        if source is None:
            source = frame.get("date", pd.Series(index=frame.index, dtype=object))
        usable = source.map(normalize_quotation_date)
    else:
        source = frame.get(field, pd.Series(index=frame.index, dtype=object))
        usable = source.map(_usable_text)

    mask = usable.notna()
    populated = int(mask.sum())
    coverage = populated / rows if rows else 0.0
    groups = int(frame.loc[mask, "_source_group"].dropna().nunique())
    distinct = int(usable[mask].nunique())
    blocking_reasons: list[str] = []
    details: dict = {}

    cohort_rows = pd.DataFrame({
        "cohort": frame.loc[mask, "_cohort"],
        "group": frame.loc[mask, "_source_group"],
        "value": usable[mask],
    }).dropna()
    matched_cohorts = 0
    if not cohort_rows.empty:
        for _, cohort in cohort_rows.groupby("cohort"):
            if cohort["group"].nunique() < 2 or cohort["value"].nunique() < 2:
                continue
            if field == "quotation_date":
                dates = pd.to_datetime(cohort["value"], errors="coerce").dropna()
                span = int((dates.max() - dates.min()).days) if len(dates) > 1 else 0
                if span < config.V3_DATE_MIN_COHORT_SPAN_DAYS:
                    continue
            matched_cohorts += 1
    details["matched_product_cohorts"] = matched_cohorts
    details["minimum_matched_product_cohorts"] = (
        config.V3_CONTEXT_MIN_MATCHED_COHORTS
    )
    collection_targets = {
        "additional_populated_rows_for_coverage": max(
            0,
            math.ceil(rows * config.V3_CONTEXT_MIN_COVERAGE) - populated,
        ),
        "additional_populated_quote_groups": max(
            0, config.V3_CONTEXT_MIN_QUOTE_GROUPS - groups
        ),
        "additional_matched_product_cohorts": max(
            0, config.V3_CONTEXT_MIN_MATCHED_COHORTS - matched_cohorts
        ),
    }

    if coverage < config.V3_CONTEXT_MIN_COVERAGE:
        blocking_reasons.append(
            f"usable coverage {coverage:.1%} is below "
            f"{config.V3_CONTEXT_MIN_COVERAGE:.0%}"
        )
    if groups < config.V3_CONTEXT_MIN_QUOTE_GROUPS:
        blocking_reasons.append(
            f"only {groups} independent quotation groups contain the field; "
            f"at least {config.V3_CONTEXT_MIN_QUOTE_GROUPS} are required"
        )
    if matched_cohorts < config.V3_CONTEXT_MIN_MATCHED_COHORTS:
        blocking_reasons.append(
            f"only {matched_cohorts} matched product cohorts vary this field; "
            f"at least {config.V3_CONTEXT_MIN_MATCHED_COHORTS} are required"
        )

    if field == "quotation_date":
        parsed = pd.to_datetime(usable[mask], errors="coerce")
        span_days = int((parsed.max() - parsed.min()).days) if len(parsed) > 1 else 0
        details["span_days"] = span_days
        details["minimum_span_days"] = config.V3_DATE_MIN_SPAN_DAYS
        details["minimum_matched_cohort_span_days"] = (
            config.V3_DATE_MIN_COHORT_SPAN_DAYS
        )
        collection_targets["additional_history_span_days"] = max(
            0, config.V3_DATE_MIN_SPAN_DAYS - span_days
        )
        if span_days < config.V3_DATE_MIN_SPAN_DAYS:
            blocking_reasons.append(
                f"history spans only {span_days} days; at least "
                f"{config.V3_DATE_MIN_SPAN_DAYS} days are required"
            )
    elif field == "quantity":
        collection_targets["additional_distinct_values"] = max(
            0, config.V3_CONTEXT_MIN_DISTINCT_VALUES - distinct
        )
        if distinct < config.V3_CONTEXT_MIN_DISTINCT_VALUES:
            blocking_reasons.append(
                f"only {distinct} distinct positive quantities are available; "
                f"at least {config.V3_CONTEXT_MIN_DISTINCT_VALUES} are required"
            )
    else:
        supported_levels = 0
        if populated:
            counted = pd.DataFrame({
                "value": usable[mask],
                "group": frame.loc[mask, "_source_group"],
            }).dropna()
            if not counted.empty:
                per_level = counted.groupby("value")["group"].nunique()
                supported_levels = int(
                    (per_level >= config.V3_CONTEXT_MIN_GROUPS_PER_LEVEL).sum()
                )
        details["supported_levels"] = supported_levels
        details["minimum_groups_per_level"] = config.V3_CONTEXT_MIN_GROUPS_PER_LEVEL
        collection_targets["additional_supported_levels"] = max(
            0, 2 - supported_levels
        )
        if supported_levels < 2:
            blocking_reasons.append(
                "fewer than two values have enough independent quotation support"
            )

    ready = not blocking_reasons
    approved = bool(
        adjustment_id and adjustment_id in APPROVED_CONTEXT_ADJUSTMENTS
    )
    return {
        "field": field,
        "status": (
            "production" if approved and ready
            else "ready_for_modeling" if ready
            else "needs_data"
        ),
        "approved_for_price_adjustment": approved,
        "rows": rows,
        "independent_quote_groups": groups_total,
        "populated_rows": populated,
        "populated_quote_groups": groups,
        "coverage": round(coverage, 4),
        "distinct_values": distinct,
        "blocking_reasons": blocking_reasons,
        "collection_targets": collection_targets,
        **details,
    }


def build_context_readiness(dataset: pd.DataFrame,
                            item_attrs: list[dict] | None = None) -> dict:
    """Profile V3 adjustment evidence overall and by supported family."""
    approved_adjustments = _validate_adjustment_ids(
        APPROVED_CONTEXT_ADJUSTMENTS
    )
    implemented_adjustments = _validate_adjustment_ids(
        IMPLEMENTED_CONTEXT_ADJUSTMENTS
    )
    unimplemented = approved_adjustments - implemented_adjustments
    if unimplemented:
        raise ContextReadinessError(
            "Context adjustment(s) approved without an implementation: "
            + ", ".join(sorted(unimplemented))
        )
    premature = {
        adjustment_id for adjustment_id in approved_adjustments
        if adjustment_id.rsplit(".", 1)[0] not in APPROVED_AUTO_FAMILIES
    }
    if premature:
        raise ContextReadinessError(
            "Context adjustment(s) approved before their base family: "
            + ", ".join(sorted(premature))
        )
    frame = build_context_evidence_frame(dataset, item_attrs)
    overall = {field: _field_evidence(frame, field) for field in CONTEXT_FIELDS}
    families = {}
    for family in sorted(SUPPORTED_INPUT_FAMILIES):
        family_frame = frame[frame["_family"] == family]
        families[family] = {
            field: _field_evidence(
                family_frame, field, context_adjustment_id(family, field)
            )
            for field in CONTEXT_FIELDS
        }
    return {
        "schema_version": CONTEXT_READINESS_SCHEMA_VERSION,
        "mode": "shadow",
        "approved_adjustments": sorted(approved_adjustments),
        "implemented_adjustments": sorted(implemented_adjustments),
        "overall": overall,
        "families": families,
    }


def context_adjustment_warnings(input_attrs: dict, readiness: dict) -> list[str]:
    """Explain why explicitly supplied V3 fields did not alter the price."""
    explicit = set(input_attrs.get("context_fields") or [])
    family = input_attrs.get("item_type")
    family_profile = (readiness.get("families") or {}).get(family, {})
    overall = readiness.get("overall") or {}
    labels = {
        "quantity": "Quantity",
        "supplier": "Supplier",
        "quotation_date": "Quotation date",
        "location": "Location",
    }
    warnings = []
    approved = set(readiness.get("approved_adjustments") or [])
    for field in CONTEXT_FIELDS:
        adjustment_id = context_adjustment_id(family, field)
        if field not in explicit or adjustment_id in approved:
            continue
        evidence = family_profile.get(field) or overall.get(field) or {}
        reasons = evidence.get("blocking_reasons") or [
            "the adjustment has not passed quotation-lineage-held-out calibration"
        ]
        warnings.append(
            f"{labels[field]} was recorded but did not change the price: "
            + "; ".join(reasons)
            + "."
        )
    return warnings


def compact_context_readiness(readiness: dict) -> dict:
    """Return the health-check subset without the per-family detail."""
    return {
        "mode": readiness.get("mode", "shadow"),
        "approved_adjustments": list(readiness.get("approved_adjustments") or []),
        "implemented_adjustments": list(
            readiness.get("implemented_adjustments") or []
        ),
        "fields": {
            field: {
                "status": item.get("status"),
                "coverage": item.get("coverage"),
                "populated_quote_groups": item.get("populated_quote_groups"),
                "collection_targets": dict(item.get("collection_targets") or {}),
                "blocking_reasons": list(item.get("blocking_reasons") or []),
            }
            for field, item in (readiness.get("overall") or {}).items()
        },
    }
