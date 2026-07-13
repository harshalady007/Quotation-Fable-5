"""Quotation-lineage-held-out V2 readiness for every supported family."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from pricing_dataset import load_pricing_dataset
from production_pricing import (APPROVED_AUTO_FAMILIES, FAMILY_REQUIRED_FIELDS,
                                PRICING_ENGINE_VERSION,
                                SUPPORTED_INPUT_FAMILIES, family_contract,
                                price_from_comparables)
from similarity_search import SimilaritySearcher


READINESS_SCHEMA_VERSION = 2


def dataset_fingerprint(dataset: pd.DataFrame) -> str:
    """Fingerprint fields that can change comparable eligibility or price."""
    parts = []
    for _, row in dataset.sort_values("record_id").iterrows():
        parts.append("\x1f".join((
            str(row.get("record_id") or ""),
            str(row.get("source") or ""),
            str(row.get("source_group") or ""),
            str(row.get("source_revision") or ""),
            str(row.get("unit_norm") or ""),
            format(float(row.get("rate")), ".12g"),
            "1" if bool(row.get("pricing_eligible", True)) else "0",
            str(row.get("category") or ""),
            str(row.get("search_text") or ""),
            json.dumps(row.get("attribute_overrides") or {}, sort_keys=True),
        )))
    payload = PRICING_ENGINE_VERSION + "\n" + "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _context_for(row, attrs: dict) -> dict:
    context = {
        "product_family": attrs.get("item_type"),
        "subtype": attrs.get("subtype"),
        "unit": row.get("unit_norm"),
        "scope": attrs.get("scope"),
        "material": attrs.get("material"),
        "civil_works": attrs.get("civil_works"),
        "mobility": attrs.get("mobility"),
        "compartments": attrs.get("compartments"),
    }
    for field in FAMILY_REQUIRED_FIELDS.get(attrs.get("item_type"), ()):
        context[field] = attrs.get(field)
    return {key: value for key, value in context.items()
            if value not in (None, "", [])}


def missing_target_fields(row, attrs: dict) -> list[str]:
    family = attrs.get("item_type")
    fields = []
    if not row.get("unit_norm"):
        fields.append("unit")
    if not attrs.get("scope"):
        fields.append("scope")
    if not attrs.get("material"):
        fields.append("material")
    for field in FAMILY_REQUIRED_FIELDS.get(family, ()):
        if not attrs.get(field):
            fields.append(field)
    if attrs.get("scope") == "supply and install" and not attrs.get("civil_works"):
        fields.append("civil_works")
    return fields


def release_gate(metrics: dict) -> tuple[bool, list[str]]:
    """Evaluate the non-negotiable V2 family release thresholds."""
    failures = []
    if metrics.get("priced_holdouts", 0) < config.PRODUCTION_GATE_MIN_CASES:
        failures.append(
            f"needs at least {config.PRODUCTION_GATE_MIN_CASES} priced holdouts"
        )
    if metrics.get("priced_quote_groups", 0) < config.V2_GATE_MIN_QUOTE_GROUPS:
        failures.append(
            f"needs at least {config.V2_GATE_MIN_QUOTE_GROUPS} independent "
            "priced quotation groups"
        )
    if metrics.get("automatic_coverage", 0) < config.V2_GATE_MIN_COVERAGE:
        failures.append(
            f"coverage below {config.V2_GATE_MIN_COVERAGE:.0%}"
        )
    within_20 = metrics.get("within_20")
    if within_20 is None or within_20 < config.PRODUCTION_GATE_WITHIN_20:
        failures.append(
            f"within ±20% below {config.PRODUCTION_GATE_WITHIN_20:.0%}"
        )
    median_ape = metrics.get("median_ape")
    if median_ape is None or median_ape > config.V2_GATE_MAX_MEDIAN_APE:
        failures.append(
            f"median APE above {config.V2_GATE_MAX_MEDIAN_APE:.0%}"
        )
    p90_ape = metrics.get("p90_ape")
    if p90_ape is None or p90_ape > config.V2_GATE_MAX_P90_APE:
        failures.append(
            f"p90 APE above {config.V2_GATE_MAX_P90_APE:.0%}"
        )
    if metrics.get("factor2_errors", 0) > config.V2_GATE_MAX_FACTOR2_ERRORS:
        failures.append("contains factor-of-two errors")
    return not failures, failures


def evaluate_family_readiness(data_path: str | None = None,
                              corrections_path: str | None = None) -> dict:
    """Evaluate families with the target quotation and revisions held out."""
    dataset = load_pricing_dataset(data_path, corrections_path)
    searcher = SimilaritySearcher(dataset)
    families = sorted(SUPPORTED_INPUT_FAMILIES)
    stats = {
        family: {
            "rows": 0,
            "quality_eligible_rows": 0,
            "complete_targets": 0,
            "source_groups": set(),
            "priced": [],
            "missing_fields": Counter(),
            "refusals": Counter(),
        }
        for family in families
    }

    for i, row in dataset.iterrows():
        attrs = searcher.item_attrs[i]
        family = attrs.get("item_type")
        if family not in stats:
            continue
        item = stats[family]
        item["rows"] += 1
        if bool(row.get("pricing_eligible")):
            item["quality_eligible_rows"] += 1
        missing = missing_target_fields(row, attrs)
        if missing:
            item["missing_fields"].update(missing)
            continue
        if not bool(row.get("pricing_eligible")):
            continue

        item["complete_targets"] += 1
        if row.get("source_group"):
            item["source_groups"].add(str(row.get("source_group")))
        search = searcher.search(
            row["full_description"],
            top_k=5,
            pricing_context=_context_for(row, attrs),
        )
        candidates = [
            match for match in search["candidate_matches"]
            if match.get("source_group") != row.get("source_group")
        ]
        decision = price_from_comparables(
            search["input_attributes"], candidates, approved_families={family}
        )
        if decision["status"] != "priced":
            item["refusals"].update(decision["review_reasons"])
            continue
        actual = float(row["rate"])
        predicted = float(decision["predicted_unit_price"])
        ratio = predicted / actual
        item["priced"].append({
            "actual": actual,
            "predicted": predicted,
            "ape": abs(ratio - 1.0),
            "ratio": ratio,
            "source_group": row.get("source_group"),
        })

    family_results = []
    for family in families:
        item = stats[family]
        results = pd.DataFrame(item["priced"])
        priced = int(len(results))
        complete = int(item["complete_targets"])
        metrics = {
            "family": family,
            "status": "not_ready",
            "approved_for_automatic_pricing": family in APPROVED_AUTO_FAMILIES,
            "rows": int(item["rows"]),
            "quality_eligible_rows": int(item["quality_eligible_rows"]),
            "complete_targets": complete,
            "independent_sources": len(item["source_groups"]),
            "independent_quote_groups": len(item["source_groups"]),
            "priced_holdouts": priced,
            "priced_quote_groups": int(results["source_group"].nunique())
            if priced else 0,
            "automatic_coverage": round(priced / complete, 4) if complete else 0.0,
            "within_20": None,
            "within_50": None,
            "median_ape": None,
            "p90_ape": None,
            "factor2_errors": 0,
            "missing_fields": dict(item["missing_fields"].most_common()),
            "top_refusals": [
                {"reason": reason, "count": count}
                for reason, count in item["refusals"].most_common(5)
            ],
            "contract": family_contract(family),
        }
        if priced:
            metrics.update({
                "within_20": round(float((results["ape"] <= 0.20).mean()), 4),
                "within_50": round(float((results["ape"] <= 0.50).mean()), 4),
                "median_ape": round(float(results["ape"].median()), 4),
                "p90_ape": round(float(results["ape"].quantile(0.90)), 4),
                "factor2_errors": int(
                    ((results["ratio"] < 0.50) | (results["ratio"] > 2.0)).sum()
                ),
            })
        passed, failures = release_gate(metrics)
        metrics["release_gate_passed"] = passed
        metrics["release_gate_failures"] = failures
        if passed and family in APPROVED_AUTO_FAMILIES:
            metrics["status"] = "production"
        elif passed:
            metrics["status"] = "ready_for_approval"
        elif complete < config.PRODUCTION_GATE_MIN_CASES:
            metrics["status"] = "needs_data"
        else:
            metrics["status"] = "shadow"
        family_results.append(metrics)

    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "pricing_version": PRICING_ENGINE_VERSION,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "dataset_rows": int(len(dataset)),
        "approved_auto_families": sorted(APPROVED_AUTO_FAMILIES),
        "gate": {
            "minimum_priced_holdouts": config.PRODUCTION_GATE_MIN_CASES,
            "minimum_priced_quote_groups": config.V2_GATE_MIN_QUOTE_GROUPS,
            "minimum_automatic_coverage": config.V2_GATE_MIN_COVERAGE,
            "minimum_within_20": config.PRODUCTION_GATE_WITHIN_20,
            "maximum_median_ape": config.V2_GATE_MAX_MEDIAN_APE,
            "maximum_p90_ape": config.V2_GATE_MAX_P90_APE,
            "maximum_factor2_errors": config.V2_GATE_MAX_FACTOR2_ERRORS,
        },
        "families": family_results,
    }


def load_readiness_snapshot(path: str | Path | None = None) -> dict:
    snapshot_path = Path(path or config.FAMILY_READINESS_PATH)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load family readiness snapshot: {exc}") from exc
    if snapshot.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError(
            f"Readiness snapshot schema_version must be {READINESS_SCHEMA_VERSION}."
        )
    return snapshot
