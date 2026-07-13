#!/usr/bin/env python3
"""Grouped quotation-lineage holdout evaluation for the production gate.

Each target is priced after removing every comparable from the target's full
quotation family, including R1/R2 PDF revisions. This prevents revised or
near-duplicate prices from leaking into both evidence and evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pricing_dataset import load_pricing_dataset
from production_pricing import (APPROVED_AUTO_FAMILIES,
                                FAMILY_REQUIRED_FIELDS,
                                price_from_comparables)
from similarity_search import SimilaritySearcher


def _context_for(row, attrs: dict) -> dict:
    context = {
        "product_family": attrs.get("item_type"),
        "unit": row.get("unit_norm"),
        "scope": attrs.get("scope"),
        "material": attrs.get("material"),
        "civil_works": attrs.get("civil_works"),
    }
    for field in FAMILY_REQUIRED_FIELDS.get(attrs.get("item_type"), ()):
        context[field] = attrs.get(field)
    return {key: value for key, value in context.items() if value not in (None, "")}


def evaluate(data_path: str) -> dict:
    dataset = load_pricing_dataset(data_path)
    searcher = SimilaritySearcher(dataset)
    eligible_targets = 0
    decisions = []
    refusal_reasons: dict[str, int] = {}

    for i, row in dataset.iterrows():
        attrs = searcher.item_attrs[i]
        family = attrs.get("item_type")
        if not row.get("pricing_eligible") or family not in APPROVED_AUTO_FAMILIES:
            continue
        required = FAMILY_REQUIRED_FIELDS[family]
        required_values = [row.get("unit_norm"), attrs.get("scope"),
                           attrs.get("material"), *(attrs.get(f) for f in required)]
        if any(value in (None, "") for value in required_values):
            continue
        if attrs.get("scope") == "supply and install" and not attrs.get("civil_works"):
            continue
        eligible_targets += 1

        search = searcher.search(
            row["full_description"],
            top_k=5,
            pricing_context=_context_for(row, attrs),
        )
        # Group holdout: remove the entire quotation lineage, including all
        # PDF revisions, not only the exact source filename or target row.
        candidates = [m for m in search["candidate_matches"]
                      if m.get("source_group") != row.get("source_group")]
        decision = price_from_comparables(search["input_attributes"], candidates)
        if decision["status"] != "priced":
            for reason in decision["review_reasons"]:
                refusal_reasons[reason] = refusal_reasons.get(reason, 0) + 1
            continue

        actual = float(row["rate"])
        predicted = float(decision["predicted_unit_price"])
        ratio = predicted / actual
        decisions.append({
            "family": family,
            "source_group": row.get("source_group"),
            "actual": actual,
            "predicted": predicted,
            "ape": abs(ratio - 1.0),
            "ratio": ratio,
        })

    results = pd.DataFrame(decisions)
    auto_cases = int(len(results))
    metrics = {
        "dataset_rows": int(len(dataset)),
        "eligible_targets": int(eligible_targets),
        "automatic_cases": auto_cases,
        "automatic_quote_groups": int(results["source_group"].nunique())
        if auto_cases else 0,
        "automatic_coverage": round(auto_cases / eligible_targets, 4)
        if eligible_targets else 0.0,
        "approved_families": sorted(APPROVED_AUTO_FAMILIES),
        "refusal_reasons": dict(sorted(
            refusal_reasons.items(), key=lambda item: item[1], reverse=True
        )),
    }
    if auto_cases:
        metrics.update({
            "within_20": round(float((results["ape"] <= 0.20).mean()), 4),
            "within_50": round(float((results["ape"] <= 0.50).mean()), 4),
            "median_ape": round(float(results["ape"].median()), 4),
            "p90_ape": round(float(results["ape"].quantile(0.90)), 4),
            "outside_factor_2": round(float(
                ((results["ratio"] < 0.50) | (results["ratio"] > 2.0)).mean()
            ), 4),
        })
        per_family = {}
        for family, group in results.groupby("family"):
            per_family[family] = {
                "cases": int(len(group)),
                "within_20": round(float((group["ape"] <= 0.20).mean()), 4),
                "median_ape": round(float(group["ape"].median()), 4),
            }
        metrics["per_family"] = per_family
    else:
        metrics.update({"within_20": 0.0, "within_50": 0.0,
                        "median_ape": None, "p90_ape": None,
                        "outside_factor_2": None, "per_family": {}})

    # An empty allow-list is an intentional fail-safe operating mode. CI
    # should pass when automatic pricing is disabled, while still rejecting
    # any enabled family that lacks independent validation evidence.
    metrics["gate_passed"] = bool(
        not APPROVED_AUTO_FAMILIES
        or (
            metrics["automatic_cases"] >= config.PRODUCTION_GATE_MIN_CASES
            and metrics["automatic_quote_groups"] >= config.V2_GATE_MIN_QUOTE_GROUPS
            and metrics["within_20"] >= config.PRODUCTION_GATE_WITHIN_20
        )
    )
    metrics["automatic_pricing_enabled"] = bool(APPROVED_AUTO_FAMILIES)
    metrics["gate"] = {
        "minimum_cases": config.PRODUCTION_GATE_MIN_CASES,
        "minimum_quote_groups": config.V2_GATE_MIN_QUOTE_GROUPS,
        "minimum_within_20": config.PRODUCTION_GATE_WITHIN_20,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(args.data)
    print(json.dumps(metrics, indent=2))
    return 1 if args.enforce_gate and not metrics["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
