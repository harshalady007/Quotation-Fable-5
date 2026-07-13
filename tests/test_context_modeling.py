"""Leakage-safe V3 context shadow-model tests."""

from __future__ import annotations

import math
import json
from datetime import date, timedelta

import pandas as pd
import pytest

import context_modeling
from context_modeling import (ContextModelEvaluationError,
                              evaluate_context_adjustment,
                              load_context_model_snapshot)
from context_readiness import (build_context_evidence_frame,
                               build_context_readiness)


def _context_history(kind: str) -> tuple[pd.DataFrame, list[dict]]:
    groups = 20
    quantities = [1, 2, 5, 10]
    start = date(2024, 1, 1)
    rows = []
    attrs = []
    for group_index in range(groups):
        quantity = quantities[group_index % len(quantities)]
        quote_date = start + timedelta(days=group_index * 30)
        supplier = "supplier a" if group_index % 2 else "supplier b"
        location = "dubai" if group_index % 2 else "abu dhabi"
        for length, base_rate in ((1200, 1000), (1500, 1300), (1800, 1600)):
            if kind == "quantity":
                factor = quantity ** -0.18
            elif kind == "date":
                factor = math.exp(
                    0.22 * ((quote_date - start).days / 365.25)
                )
            elif kind == "supplier":
                factor = 0.90 if supplier == "supplier a" else 1.10
            else:
                factor = 1.0
            rows.append({
                "source_group": f"q-{group_index:02d}",
                "pricing_eligible": True,
                "unit_norm": "no",
                "rate": base_rate * factor,
                "quantity": quantity,
                "supplier": supplier,
                "location": location,
                "quotation_date": quote_date.isoformat(),
            })
            attrs.append({
                "item_type": "bench",
                "subtype": "backless bench",
                "material": "mild steel",
                "scope": "supply only",
                "length_mm": length,
                "features": [],
            })
    return pd.DataFrame(rows), attrs


def _evaluate(kind: str, field: str) -> dict:
    dataset, attrs = _context_history(kind)
    readiness = build_context_readiness(dataset, attrs)
    evidence_frame = build_context_evidence_frame(dataset, attrs)
    return evaluate_context_adjustment(
        evidence_frame,
        "bench",
        field,
        readiness["families"]["bench"][field],
    )


def test_quantity_shadow_model_beats_group_held_out_cohort_baseline():
    result = _evaluate("quantity", "quantity")
    assert result["evaluation_mode"] == (
        "rolling_origin_and_quotation_lineage_holdout"
    )
    assert result["statistical_gate_passed"]
    assert result["model_metrics"]["quote_groups"] >= 8
    assert result["model_metrics"]["cohorts"] == 3
    assert result["model_metrics"]["within_20"] == 1.0
    assert result["median_ape_improvement"] > 0.10
    assert result["clamp_share"] == 0.0


def test_date_shadow_model_uses_only_earlier_training_quotations():
    result = _evaluate("date", "quotation_date")
    assert result["evaluation_mode"] == (
        "rolling_origin_and_quotation_lineage_holdout"
    )
    assert result["statistical_gate_passed"]
    assert result["temporal_order_violations"] == 0
    assert result["evaluation_coverage"] >= 0.35
    assert any(
        item["reason"] == "too few independent training quotation groups"
        for item in result["top_skip_reasons"]
    )


def test_context_model_is_rejected_when_it_adds_no_incremental_value():
    result = _evaluate("none", "quantity")
    assert not result["statistical_gate_passed"]
    assert result["evaluation_status"] == "rejected_by_shadow_gate"
    assert any(
        "improvement over baseline" in reason
        for reason in result["statistical_gate_failures"]
    )


def test_categorical_shadow_model_requires_supported_held_out_levels():
    result = _evaluate("supplier", "supplier")
    assert result["statistical_gate_passed"]
    assert result["model_metrics"]["cases"] >= 25
    assert result["improved_case_share"] == 1.0


def test_exact_product_cohorts_separate_known_feature_differences():
    dataset = pd.DataFrame([
        {
            "source_group": "q-1", "pricing_eligible": True,
            "unit_norm": "no", "rate": 1000, "quantity": 1,
            "quotation_date": "2026-01-01",
        },
        {
            "source_group": "q-2", "pricing_eligible": True,
            "unit_norm": "no", "rate": 1200, "quantity": 5,
            "quotation_date": "2026-06-01",
        },
    ])
    shared = {
        "item_type": "bench", "subtype": "backless bench",
        "material": "mild steel", "scope": "supply only",
        "length_mm": 1800,
    }
    evidence = build_context_evidence_frame(dataset, [
        {**shared, "features": []},
        {**shared, "features": ["perforated"]},
    ])
    assert evidence["_cohort"].nunique() == 2


def test_accuracy_metrics_give_each_quotation_equal_total_weight():
    predictions = [
        {
            "source_group": "long-quote", "cohort": f"c-{index}",
            "actual": 100.0, "predicted": 100.0,
        }
        for index in range(10)
    ]
    predictions.append({
        "source_group": "short-quote", "cohort": "c-bad",
        "actual": 100.0, "predicted": 300.0,
    })
    metrics = context_modeling._prediction_metrics(predictions, "predicted")
    assert metrics["within_20"] == 0.5
    assert metrics["factor2_errors"] == 1


def test_model_snapshot_requires_every_family_field_exactly_once(tmp_path):
    path = tmp_path / "model-readiness.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "algorithm_version": "cohort-residual-ridge-v1",
        "dataset_fingerprint": "fingerprint",
        "adjustments": [],
    }), encoding="utf-8")
    with pytest.raises(ContextModelEvaluationError, match="exactly once"):
        load_context_model_snapshot(path)
