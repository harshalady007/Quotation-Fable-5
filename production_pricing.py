"""Deterministic, guarded comparable pricing for Production V1."""

from __future__ import annotations

import math
from typing import Iterable

import config


SUPPORTED_INPUT_FAMILIES = {
    "planter",
    "litter bin",
    "bench",
    "bollard",
    "bike rack",
    "recycle bin",
}

# Only families that have cleared grouped, source-held-out validation are
# allowed to return an automatic price. The others remain searchable pilots
# and explicitly abstain until their subtype data is curated.
APPROVED_AUTO_FAMILIES = {"planter"}

# Specifications that materially define price for each supported family.
# They may be extracted from text or supplied explicitly by the user.
FAMILY_REQUIRED_FIELDS = {
    "planter": ("length_mm", "width_mm", "height_mm"),
    "litter bin": ("capacity_l",),
    "recycle bin": ("capacity_l",),
    "bench": ("length_mm",),
    "bollard": ("diameter_mm", "height_mm"),
    "bike rack": ("length_mm",),
}


def _weighted_median(items: list[tuple[float, float]]) -> float:
    ordered = sorted(items, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    threshold = total / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _weighted_quantile(items: list[tuple[float, float]], quantile: float) -> float:
    ordered = sorted(items, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    threshold = total * quantile
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _numeric_distance(input_attrs: dict, match: dict, fields: Iterable[str]) -> float:
    """Log-ratio distance, where 0 means equal and log(2) means 2x apart."""
    distance = 0.0
    compared = 0
    match_attrs = match.get("attributes") or {}
    for field in fields:
        a = input_attrs.get(field)
        b = match_attrs.get(field)
        if not a or not b:
            continue
        distance += abs(math.log(float(b) / float(a)))
        compared += 1
    return distance / compared if compared else 0.0


def _manual_decision(reasons: list[str], comparables: list,
                     indicative_price: float | None = None) -> dict:
    return {
        "status": "manual_review",
        "predicted_unit_price": None,
        "indicative_price": round(indicative_price, 2) if indicative_price else None,
        "price_interval": None,
        "confidence": "Manual review",
        "pricing_method": "insufficient validated comparable evidence",
        "review_reasons": reasons,
        "comparables": comparables,
    }


def price_from_comparables(input_attrs: dict, candidates: list[dict]) -> dict:
    """Return an automatic price only when production evidence gates pass."""
    reasons: list[str] = []
    family = input_attrs.get("item_type")
    unit = input_attrs.get("unit_hint")
    scope = input_attrs.get("scope")

    if not family:
        reasons.append("Product family is missing or was not recognized.")
    elif family not in SUPPORTED_INPUT_FAMILIES:
        reasons.append(
            f"Product family '{family}' is not yet approved for automatic pricing."
        )
    elif family not in APPROVED_AUTO_FAMILIES:
        reasons.append(
            f"Product family '{family}' has not yet passed the grouped "
            "holdout accuracy gate; comparables are shown for manual review."
        )
    if not unit:
        reasons.append("Unit of measure must be explicitly confirmed.")
    if not scope:
        reasons.append("Commercial scope must be explicitly confirmed.")
    if not input_attrs.get("material"):
        reasons.append("Primary material must be explicitly confirmed or recognized.")
    if scope == "supply and install" and not input_attrs.get("civil_works"):
        reasons.append(
            "Civil-works inclusion must be confirmed for installation pricing."
        )

    required = FAMILY_REQUIRED_FIELDS.get(family, ())
    missing_specs = [field for field in required if not input_attrs.get(field)]
    if missing_specs:
        reasons.append(
            "Missing price-defining specification(s): " + ", ".join(missing_specs) + "."
        )

    if reasons:
        return _manual_decision(reasons, [])

    # Hard eligibility filters. No cross-unit, cross-family or quarantined
    # rates are allowed to set an automatic price.
    eligible = [
        m for m in candidates
        if m.get("pricing_eligible", True)
        and m.get("item_type") == family
        and (m.get("unit_norm") or "") == unit
        and m.get("scope") == scope
        and m.get("rate")
    ]

    civil = input_attrs.get("civil_works")
    if civil and scope == "supply and install":
        eligible = [
            m for m in eligible
            if (m.get("attributes") or {}).get("civil_works") == civil
        ]

    material = input_attrs.get("material")
    if material:
        same_material = [m for m in eligible if m.get("material") == material]
        if len(same_material) >= config.PRODUCTION_MIN_COMPARABLES:
            eligible = same_material
        else:
            reasons.append(
                f"Fewer than {config.PRODUCTION_MIN_COMPARABLES} validated "
                f"'{material}' comparables exist on the same unit and scope basis."
            )

    # Every automatic comparable must state the same family-critical numeric
    # fields and be within a 2x specification range of the input.
    spec_eligible = []
    for match in eligible:
        match_attrs = match.get("attributes") or {}
        if any(not match_attrs.get(field) for field in required):
            continue
        if any(
            max(float(input_attrs[field]), float(match_attrs[field]))
            / min(float(input_attrs[field]), float(match_attrs[field])) > 2.0
            for field in required
        ):
            continue
        spec_eligible.append(match)
    eligible = spec_eligible

    eligible.sort(key=lambda m: m.get("similarity_score", 0), reverse=True)
    eligible = eligible[:config.PRICING_MAX_MATCHES]

    if len(eligible) < config.PRODUCTION_MIN_COMPARABLES:
        reasons.append(
            f"Only {len(eligible)} validated same-family, same-unit, same-scope "
            f"comparables remain; at least {config.PRODUCTION_MIN_COMPARABLES} are required."
        )
    best_score = eligible[0]["similarity_score"] if eligible else 0.0
    if eligible and best_score < config.PRODUCTION_MIN_BEST_SCORE:
        reasons.append(
            f"Best validated comparable score is {best_score:.2f}; minimum is "
            f"{config.PRODUCTION_MIN_BEST_SCORE:.2f}."
        )

    weighted_rates = []
    for match in eligible:
        numeric_distance = _numeric_distance(input_attrs, match, required)
        weight = max(float(match.get("similarity_score") or 0.01), 0.01) ** 2
        weight *= math.exp(-numeric_distance)
        match["production_weight"] = round(weight, 6)
        weighted_rates.append((float(match["rate"]), weight))

    indicative = _weighted_median(weighted_rates) if weighted_rates else None
    rates = [value for value, _ in weighted_rates]
    rate_ratio = max(rates) / min(rates) if rates and min(rates) > 0 else float("inf")
    if rates and rate_ratio > config.PRODUCTION_MAX_RATE_RATIO:
        reasons.append(
            f"Comparable rates vary by {rate_ratio:.2f}x, above the safe "
            f"{config.PRODUCTION_MAX_RATE_RATIO:.2f}x limit."
        )

    if reasons:
        return _manual_decision(reasons, eligible, indicative)

    estimate = _weighted_median(weighted_rates)
    low = _weighted_quantile(weighted_rates, 0.20)
    high = _weighted_quantile(weighted_rates, 0.80)
    confidence = (
        "High"
        if len(eligible) >= 5 and best_score >= 0.65 and rate_ratio <= 1.35
        else "Medium"
    )
    return {
        "status": "priced",
        "predicted_unit_price": round(estimate, 2),
        "indicative_price": round(estimate, 2),
        "price_interval": {"low": round(low, 2), "high": round(high, 2)},
        "confidence": confidence,
        "pricing_method": "guarded similarity-weighted median",
        "review_reasons": [],
        "comparables": eligible,
    }
