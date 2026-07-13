"""Normalize and apply user-confirmed pricing context.

Free text remains useful for discovery, but explicit fields are authoritative
for unit, family and commercial scope.  This avoids dangerous guesses such as
reading ``ITEM NO J`` as a per-item pricing basis.
"""

from __future__ import annotations

import math

from attribute_extractor import detect_scope, extract_attributes
from cleaner import normalize_text, normalize_unit
from context_readiness import normalize_quotation_date


NUMERIC_FIELDS = {
    "quantity",
    "capacity_l",
    "compartments",
    "length_mm",
    "width_mm",
    "height_mm",
    "diameter_mm",
    "thickness_mm",
}


def _positive_number(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a number, received {value!r}.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Pricing dimensions, capacity and quantity must be positive.")
    return number


def normalize_pricing_context(context: dict | None) -> dict:
    """Return a canonical context dictionary, dropping empty values."""
    if not context:
        return {}
    raw = dict(context)
    out = {}

    family = raw.get("product_family") or raw.get("item_type")
    if family:
        detected = extract_attributes(normalize_text(str(family))).get("item_type")
        out["item_type"] = detected or normalize_text(str(family))

    if raw.get("subtype"):
        out["subtype"] = normalize_text(str(raw["subtype"]))

    if raw.get("unit"):
        out["unit_hint"] = normalize_unit(raw["unit"])

    if raw.get("scope"):
        normalized = normalize_text(str(raw["scope"]))
        out["scope"] = detect_scope(normalized) or normalized

    if raw.get("material"):
        material = extract_attributes(normalize_text(str(raw["material"]))).get("material")
        out["material"] = material or normalize_text(str(raw["material"]))

    for field in ("finish", "grade", "location", "supplier"):
        if raw.get(field):
            out[field] = normalize_text(str(raw[field]))

    quotation_date = raw.get("quotation_date") or raw.get("date")
    if quotation_date:
        out["quotation_date"] = normalize_quotation_date(
            quotation_date, strict=True
        )

    if raw.get("mobility"):
        mobility = normalize_text(str(raw["mobility"]))
        if mobility not in {"fixed", "movable", "removable"}:
            raise ValueError("mobility must be fixed, movable or removable.")
        out["mobility"] = mobility

    civil = raw.get("civil_works")
    if civil is not None and civil != "":
        if isinstance(civil, bool):
            out["civil_works"] = "included" if civil else "excluded"
        else:
            value = normalize_text(str(civil))
            if value in {"included", "include", "yes", "with civil works"}:
                out["civil_works"] = "included"
            elif value in {"excluded", "exclude", "no", "without civil works"}:
                out["civil_works"] = "excluded"
            else:
                raise ValueError("civil_works must be included/excluded or true/false.")

    for field in NUMERIC_FIELDS:
        value = _positive_number(raw.get(field))
        if value is not None:
            if field == "compartments" and not value.is_integer():
                raise ValueError("compartments must be a positive whole number.")
            if field == "compartments":
                value = int(value)
            out[field] = value

    features = raw.get("features")
    if features:
        if isinstance(features, str):
            features = [part.strip() for part in features.split(",")]
        out["features"] = sorted({normalize_text(str(v)) for v in features if str(v).strip()})

    return out


def apply_pricing_context(attrs: dict, context: dict | None) -> dict:
    """Overlay explicit context on extracted attributes and recompute proxies."""
    result = dict(attrs)
    normalized = normalize_pricing_context(context)
    result.update(normalized)
    result["context_fields"] = sorted(normalized)

    length = result.get("length_mm")
    width = result.get("width_mm")
    height = result.get("height_mm")
    if length and width:
        result["footprint_mm2"] = length * width
    if length and width and height:
        result["envelope_mm3"] = length * width * height

    sizes = list(result.get("sizes_mm") or [])
    sizes.extend(v for v in (length, width, height, result.get("diameter_mm")) if v)
    result["sizes_mm"] = sorted(set(float(v) for v in sizes))
    result["max_size_mm"] = result["sizes_mm"][-1] if result["sizes_mm"] else None
    if length:
        result["size_proxy"] = length
        result["size_proxy_kind"] = "length"
    elif result.get("max_size_mm"):
        result["size_proxy"] = result["max_size_mm"]
        result["size_proxy_kind"] = "max"
    return result
