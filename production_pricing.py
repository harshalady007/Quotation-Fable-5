"""Deterministic comparable pricing with universal estimate fallbacks.

Strict release gates remain available for offline validation, but the request
path always returns the best numeric estimate supported by quality-eligible
historical rates.  Low-evidence estimates carry an explicit evidence tier,
wide interval and low confidence instead of silently claiming validation.
"""

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

# Product families for which the request-time API must return an estimate.
# This is deliberately separate from APPROVED_AUTO_FAMILIES: the latter is an
# evidence/accuracy approval list, while this set is the user-facing operating
# policy.  A family can therefore return a low-confidence estimate without
# being represented as having passed the historical release gate.
ESTIMATE_ENABLED_FAMILIES = set(SUPPORTED_INPUT_FAMILIES)

# Families that have cleared quotation-lineage-held-out validation. This is
# evidence metadata, not the request-time estimate switch: the universal
# policy may return a visibly lower-confidence estimate for an unapproved
# family without claiming that the accuracy gate passed.
APPROVED_AUTO_FAMILIES: set[str] = set()

# Bump whenever extraction, eligibility or pricing logic changes.  The value
# is included in the readiness fingerprint so the API cannot present an old
# scorecard as current after a code-only model change.
PRICING_ENGINE_VERSION = "3.3.0-universal-estimate"
PRICING_POLICY = "always_estimate"

BENCH_CRITICAL_FEATURES = {
    "armrest",
    "backrest",
    "perforated",
    "wood accent",
}

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

FAMILY_CONTRACTS = {
    "planter": {
        "required_fields": FAMILY_REQUIRED_FIELDS["planter"],
        "recommended_fields": ("subtype", "thickness_mm", "features"),
        "known_subtypes": ("standalone planter", "integrated seating"),
    },
    "bench": {
        "required_fields": FAMILY_REQUIRED_FIELDS["bench"],
        "recommended_fields": ("subtype", "width_mm", "height_mm", "features"),
        "known_subtypes": ("backless bench", "bench with backrest", "tree bench",
                           "sculptural bench", "shaped bench", "linear bench"),
    },
    "litter bin": {
        "required_fields": FAMILY_REQUIRED_FIELDS["litter bin"],
        "recommended_fields": ("subtype", "compartments", "mobility"),
        "known_subtypes": ("pedal bin", "multi-stream bin", "wall-mounted bin",
                           "mobile bin"),
    },
    "recycle bin": {
        "required_fields": FAMILY_REQUIRED_FIELDS["recycle bin"],
        "recommended_fields": ("subtype", "compartments", "mobility"),
        "known_subtypes": ("multi-stream bin", "wall-mounted bin", "mobile bin"),
    },
    "bollard": {
        "required_fields": FAMILY_REQUIRED_FIELDS["bollard"],
        "recommended_fields": ("subtype", "mobility", "thickness_mm"),
        "known_subtypes": ("fixed bollard", "removable bollard",
                           "retractable bollard", "flexible bollard"),
    },
    "bike rack": {
        "required_fields": FAMILY_REQUIRED_FIELDS["bike rack"],
        "recommended_fields": ("subtype", "width_mm", "height_mm"),
        "known_subtypes": ("hoop rack", "multi-bike rack", "wall-mounted rack"),
    },
}


def family_contract(family: str) -> dict | None:
    contract = FAMILY_CONTRACTS.get(family)
    if not contract:
        return None
    return {
        "family": family,
        "automatic_pricing": family in ESTIMATE_ENABLED_FAMILIES,
        "estimate_enabled": family in ESTIMATE_ENABLED_FAMILIES,
        "release_gate_approved": family in APPROVED_AUTO_FAMILIES,
        "required_fields": list(contract["required_fields"]),
        "recommended_fields": list(contract["recommended_fields"]),
        "known_subtypes": list(contract["known_subtypes"]),
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


def _abstained_decision(reasons: list[str], comparables: list,
                        indicative_price: float | None = None) -> dict:
    return {
        "status": "abstained",
        "predicted_unit_price": None,
        "indicative_price": round(indicative_price, 2) if indicative_price else None,
        "price_interval": None,
        "confidence": "Not priced",
        "pricing_method": "insufficient validated comparable evidence",
        "review_reasons": reasons,
        "comparables": comparables,
    }


def _strict_price_from_comparables(
    input_attrs: dict,
    candidates: list[dict],
    approved_families: set[str],
) -> dict:
    """Apply the original release-gated comparable decision.

    This strict path is still used by offline holdout evaluation.  The public
    request path wraps it with a quality-eligible fallback so a valid request
    receives a numeric estimate even when one of these gates is not met.
    """
    reasons: list[str] = []
    family = input_attrs.get("item_type")
    unit = input_attrs.get("unit_hint")
    scope = input_attrs.get("scope")

    active_families = set(approved_families)
    if not family:
        reasons.append("Product family is missing or was not recognized.")
    elif family not in SUPPORTED_INPUT_FAMILIES:
        reasons.append(
            f"Product family '{family}' is not yet approved for automatic pricing."
        )
    elif family not in active_families:
        reasons.append(
            f"Product family '{family}' has not yet passed the quotation-lineage "
            "holdout accuracy gate."
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
        return _abstained_decision(reasons, [])

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

    subtype = input_attrs.get("subtype")
    if subtype:
        same_subtype = [
            m for m in eligible
            if (m.get("subtype") or (m.get("attributes") or {}).get("subtype")) == subtype
        ]
        if len(same_subtype) >= config.PRODUCTION_MIN_COMPARABLES:
            eligible = same_subtype
        else:
            reasons.append(
                f"Fewer than {config.PRODUCTION_MIN_COMPARABLES} validated "
                f"'{subtype}' subtype comparables exist on the same unit and scope basis."
            )

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

    # Bench add-ons are commercial scope, not decorative text.  A plain
    # concrete seat must not be anchored to concrete-plus-timber benches, and
    # a one-off perforated or armrest design must abstain when its own cohort
    # is too sparse.
    if family == "bench":
        input_features = set(input_attrs.get("features") or [])
        for feature in sorted(BENCH_CRITICAL_FEATURES):
            matching = [
                m for m in eligible
                if (feature in set((m.get("attributes") or {}).get("features") or []))
                == (feature in input_features)
            ]
            if (feature in input_features
                    and len(matching) < config.PRODUCTION_MIN_COMPARABLES):
                reasons.append(
                    f"Fewer than {config.PRODUCTION_MIN_COMPARABLES} validated "
                    f"bench comparables share the '{feature}' feature."
                )
            elif len(matching) >= config.PRODUCTION_MIN_COMPARABLES:
                eligible = matching

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

    eligible = _independent_matches(eligible)[:config.PRICING_MAX_MATCHES]

    if len(eligible) < config.PRODUCTION_MIN_COMPARABLES:
        reasons.append(
            f"Only {len(eligible)} independent validated same-family, same-unit, "
            f"same-scope quotation lineages remain; at least "
            f"{config.PRODUCTION_MIN_COMPARABLES} are required."
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
        return _abstained_decision(reasons, eligible, indicative)

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


def _scope_adjusted_rate(rate: float, source_scope: str | None,
                         target_scope: str | None) -> float:
    """Convert a rate between the two supported commercial scope groups."""
    supply = {"supply only", "supply and delivery"}
    install = {"supply and install"}
    value = float(rate)
    if not source_scope or not target_scope:
        return value
    if source_scope in supply and target_scope in install:
        return value * config.SCOPE_INSTALL_UPLIFT
    if source_scope in install and target_scope in supply:
        return value / config.SCOPE_INSTALL_UPLIFT
    return value


def _dedupe_text(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = " ".join(str(value).split())
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _best_group_value(matches: list[dict], field: str) -> str | None:
    """Choose one coherent unit/scope using strongest aggregate evidence."""
    groups: dict[str, list[dict]] = {}
    for match in matches:
        value = match.get(field)
        if value:
            groups.setdefault(str(value), []).append(match)
    if not groups:
        return None

    def score(value: str) -> tuple[float, float, int, str]:
        ranked = sorted(
            groups[value],
            key=lambda item: float(item.get("similarity_score") or 0.0),
            reverse=True,
        )[:config.PRICING_MAX_MATCHES]
        scores = [float(item.get("similarity_score") or 0.0) for item in ranked]
        return (sum(scores), max(scores, default=0.0), len(ranked), value)

    return max(groups, key=score)


def _families_compatible(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return left == right or {left, right} <= {"litter bin", "recycle bin"}


def _independent_matches(matches: list[dict]) -> list[dict]:
    """Use at most one rate from each quotation lineage when possible."""
    result = []
    seen = set()
    for match in sorted(
        matches,
        key=lambda item: float(item.get("similarity_score") or 0.0),
        reverse=True,
    ):
        lineage = match.get("source_group")
        key = ("source", lineage) if lineage else (
            "record", match.get("record_id") or match.get("rank") or id(match)
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(match)
    return result


def _universal_estimate(
    input_attrs: dict,
    candidates: list[dict],
    strict_reasons: list[str],
) -> dict:
    """Return the best coherent estimate from quality-eligible evidence.

    The fallback never mixes unit bases and never uses quarantined rates.  It
    progressively relaxes product-family and commercial-scope similarity,
    records exactly which tier was used, and widens the interval as evidence
    becomes weaker.
    """
    validated = [
        match for match in candidates
        if match.get("pricing_eligible", True)
        and match.get("rate") is not None
        and float(match["rate"]) > 0
        and match.get("unit_norm")
    ]
    if not validated:
        raise ValueError(
            "No quality-eligible historical rates are available to estimate "
            "this request."
        )

    family = input_attrs.get("item_type")
    requested_unit = input_attrs.get("unit_hint")
    requested_scope = input_attrs.get("scope")
    warnings = list(strict_reasons)

    same_family = [
        match for match in validated if match.get("item_type") == family
    ] if family else []
    compatible_family = [
        match for match in validated
        if _families_compatible(family, match.get("item_type"))
    ] if family else []

    if requested_unit:
        unit_pool = [
            match for match in validated
            if match.get("unit_norm") == requested_unit
        ]
        estimated_unit = requested_unit
    else:
        unit_source = same_family or compatible_family or validated
        estimated_unit = _best_group_value(unit_source, "unit_norm")
        unit_pool = [
            match for match in validated
            if match.get("unit_norm") == estimated_unit
        ]
        warnings.append(
            f"Unit was not supplied; the estimate uses the strongest "
            f"historical unit basis '{estimated_unit}'."
        )

    if not unit_pool:
        unit_source = same_family or compatible_family or validated
        estimated_unit = _best_group_value(unit_source, "unit_norm")
        unit_pool = [
            match for match in validated
            if match.get("unit_norm") == estimated_unit
        ]
        warnings.append(
            f"No quality-eligible candidates use requested unit "
            f"'{requested_unit}'; the returned estimate is instead on the "
            f"'{estimated_unit}' basis."
        )

    exact_family_pool = [
        match for match in unit_pool if match.get("item_type") == family
    ] if family else []
    compatible_pool = [
        match for match in unit_pool
        if _families_compatible(family, match.get("item_type"))
    ] if family else []
    if exact_family_pool:
        pool = exact_family_pool
        tier = "same_family_unit"
    elif compatible_pool:
        pool = compatible_pool
        tier = "compatible_family_unit"
        warnings.append(
            "No exact-family validated rate remained; a compatible bin-family "
            "cohort was used."
        )
    else:
        pool = unit_pool
        tier = "same_unit_cross_family"
        warnings.append(
            "No exact-family validated rate remained on this unit basis; the "
            "estimate uses the most similar quality-eligible products sharing "
            "the same unit."
        )

    if requested_scope:
        exact_scope = [
            match for match in pool if match.get("scope") == requested_scope
        ]
        known_scope = [match for match in pool if match.get("scope")]
        if len(exact_scope) >= config.PRODUCTION_MIN_COMPARABLES:
            pool = exact_scope
            tier += "_scope"
        elif known_scope:
            pool = known_scope
            tier += "_scope_adjusted"
            warnings.append(
                "Too few exact-scope rates were available; historical rates "
                "were normalized with the configured 20% installation rule."
            )
        else:
            tier += "_scope_unknown"
            warnings.append(
                "Historical commercial scope is missing; no scope conversion "
                "could be applied."
            )
        estimated_scope = requested_scope
    else:
        estimated_scope = _best_group_value(pool, "scope")
        if estimated_scope:
            pool = [
                match for match in pool
                if match.get("scope") == estimated_scope
            ]
        tier += "_scope_inferred"
        warnings.append(
            "Commercial scope was not supplied; the estimate uses the "
            f"strongest historical scope basis '{estimated_scope or 'unknown'}'."
        )

    subtype = input_attrs.get("subtype")
    if subtype:
        same_subtype = [
            match for match in pool
            if (match.get("subtype")
                or (match.get("attributes") or {}).get("subtype")) == subtype
        ]
        if len(same_subtype) >= 2:
            pool = same_subtype
            tier += "_subtype"

    material = input_attrs.get("material")
    if material:
        same_material = [
            match for match in pool if match.get("material") == material
        ]
        if len(same_material) >= 2:
            pool = same_material
            tier += "_material"

    required = FAMILY_REQUIRED_FIELDS.get(family, ())
    known_required = [field for field in required if input_attrs.get(field)]
    if known_required:
        close_specs = []
        for match in pool:
            attrs = match.get("attributes") or {}
            compared = [field for field in known_required if attrs.get(field)]
            if compared and all(
                max(float(input_attrs[field]), float(attrs[field]))
                / min(float(input_attrs[field]), float(attrs[field])) <= 4.0
                for field in compared
            ):
                close_specs.append(match)
        if close_specs:
            pool = close_specs
            tier += "_spec_near"

    selected = _independent_matches(pool)[:config.PRICING_MAX_MATCHES]
    if not selected:
        raise ValueError(
            "No independent quality-eligible historical rates are available "
            "to estimate this request."
        )

    weighted_rates: list[tuple[float, float]] = []
    comparables = []
    for match in selected:
        comparable = dict(match)
        effective_rate = _scope_adjusted_rate(
            float(match["rate"]), match.get("scope"), estimated_scope
        )
        numeric_distance = _numeric_distance(
            input_attrs, match, known_required
        )
        weight = max(float(match.get("similarity_score") or 0.01), 0.01) ** 2
        weight *= math.exp(-numeric_distance)
        comparable["scope_adjusted_rate"] = round(effective_rate, 2)
        comparable["production_weight"] = round(weight, 6)
        weighted_rates.append((effective_rate, weight))
        comparables.append(comparable)

    estimate = _weighted_median(weighted_rates)
    observed_low = _weighted_quantile(weighted_rates, 0.20)
    observed_high = _weighted_quantile(weighted_rates, 0.80)
    best_score = max(
        float(match.get("similarity_score") or 0.0) for match in comparables
    )
    rates = [rate for rate, _ in weighted_rates]
    rate_ratio = max(rates) / min(rates) if min(rates) > 0 else float("inf")

    strong_tier = tier.startswith("same_family_unit")
    confidence = (
        "Low"
        if strong_tier and len(comparables) >= 2 and best_score >= 0.30
        else "Very low"
    )
    spread = 0.25 if confidence == "Low" else 0.50
    low = min(observed_low, estimate * (1.0 - spread))
    high = max(observed_high, estimate * (1.0 + spread))
    if rate_ratio > config.PRODUCTION_MAX_RATE_RATIO:
        warnings.append(
            f"Selected comparable rates span {rate_ratio:.2f}x; the interval "
            "has been widened and confidence remains low."
        )

    warnings.append(
        f"Estimate fallback tier: {tier}; {len(comparables)} independent "
        "quality-eligible quotation lineage(s) set the price."
    )
    return {
        "status": "priced",
        "predicted_unit_price": round(estimate, 2),
        "indicative_price": round(estimate, 2),
        "price_interval": {
            "low": round(max(low, 0.01), 2),
            "high": round(high, 2),
        },
        "confidence": confidence,
        "pricing_method": f"universal validated-comparable fallback ({tier})",
        "review_reasons": _dedupe_text(warnings),
        "comparables": comparables,
        "evidence_tier": tier,
        "estimated_unit": estimated_unit,
        "estimated_scope": estimated_scope,
        "release_gate_approved": family in APPROVED_AUTO_FAMILIES,
        "fallback_used": True,
    }


def price_from_comparables(
    input_attrs: dict,
    candidates: list[dict],
    approved_families: set[str] | None = None,
    *,
    always_estimate: bool = True,
) -> dict:
    """Price from historical comparables under strict or universal policy.

    ``always_estimate=False`` preserves the abstaining decision used by
    offline accuracy evaluation.  The default request policy returns a
    numeric estimate for every valid request while retaining the strict
    reasons as evidence warnings.
    """
    release_approved = (
        set(APPROVED_AUTO_FAMILIES)
        if approved_families is None else set(approved_families)
    )
    strict_active = (
        set(SUPPORTED_INPUT_FAMILIES)
        if always_estimate and approved_families is None
        else release_approved
    )
    strict = _strict_price_from_comparables(
        input_attrs, candidates, strict_active
    )
    if not always_estimate:
        return strict

    family = input_attrs.get("item_type")
    release_warning = []
    if family in SUPPORTED_INPUT_FAMILIES and family not in release_approved:
        release_warning.append(
            f"Product family '{family}' has not passed the historical "
            "quotation-lineage accuracy gate; this numeric result is an "
            "estimate under the always-estimate policy."
        )

    if strict["status"] == "priced":
        strict["review_reasons"] = _dedupe_text(
            [*strict.get("review_reasons", []), *release_warning]
        )
        if release_warning:
            strict["confidence"] = "Low"
            estimate = float(strict["predicted_unit_price"])
            interval = strict.get("price_interval") or {}
            strict["price_interval"] = {
                "low": round(min(
                    float(interval.get("low", estimate)), estimate * 0.75
                ), 2),
                "high": round(max(
                    float(interval.get("high", estimate)), estimate * 1.25
                ), 2),
            }
        strict.update({
            "evidence_tier": "strict_comparables",
            "estimated_unit": input_attrs.get("unit_hint"),
            "estimated_scope": input_attrs.get("scope"),
            "release_gate_approved": family in release_approved,
            "fallback_used": False,
        })
        return strict

    return _universal_estimate(
        input_attrs,
        candidates,
        [*strict.get("review_reasons", []), *release_warning],
    )
