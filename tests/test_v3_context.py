"""V3 contextual-data contracts and evidence-gate tests."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import context_readiness
from context_readiness import (build_context_readiness,
                               context_adjustment_warnings,
                               normalize_quotation_date)
from data_loader import load_dataset
from pricing_context import normalize_pricing_context


def test_mixed_quotation_dates_normalize_without_day_month_inversion():
    assert normalize_quotation_date("July 30th 2025") == "2025-07-30"
    assert normalize_quotation_date("27th Jan 2026") == "2026-01-27"
    assert normalize_quotation_date("2026-07-13") == "2026-07-13"
    assert normalize_quotation_date(pd.Timestamp("2026-06-20")) == "2026-06-20"
    assert normalize_quotation_date("not a date") is None
    with pytest.raises(ValueError, match="quotation_date"):
        normalize_quotation_date("not a date", strict=True)


def test_summary_sheet_context_joins_by_source_without_entering_product_text(tmp_path):
    path = tmp_path / "quotations.xlsx"
    products = pd.DataFrame([{
        "Source File": "Q-100.pdf",
        "Quotation Date": None,
        "Scope of Work": "Supply Only",
        "Item Name": "Bench",
        "Description": "Mild steel bench L1800mm",
        "Unit": "Nos",
        "Unit Price": 1200,
    }])
    summary = pd.DataFrame([{
        "Source File": "q-100.pdf",
        "Quotation Number": "Q-100",
        "Quotation Date": "13 July 2026",
        "Project": "Test Park",
        "Client": "Test Client",
        "Contractor": "Test Contractor",
        "Currency": "AED",
    }])
    with pd.ExcelWriter(path) as writer:
        products.to_excel(writer, sheet_name="Products", index=False)
        summary.to_excel(writer, sheet_name="Quotation Summary", index=False)

    loaded = load_dataset(path)
    assert loaded.loc[0, "date"] == "13 July 2026"
    assert loaded.loc[0, "project"] == "Test Park"
    assert loaded.loc[0, "client"] == "Test Client"
    assert loaded.loc[0, "contractor"] == "Test Contractor"
    assert loaded.loc[0, "currency"] == "AED"
    assert loaded.attrs["summary_context"]["matched_product_rows"] == 1


def _context_dataset() -> tuple[pd.DataFrame, list[dict]]:
    start = date(2025, 1, 1)
    rows = []
    attrs = []
    for index in range(12):
        rows.append({
            "source_group": f"q-{index}",
            "pricing_eligible": True,
            "quantity": (1, 5, 10)[(index // 3) % 3],
            "supplier": ("supplier a", "supplier b")[index % 2],
            "location": ("dubai", "abu dhabi")[index % 2],
            "quotation_date": (start + timedelta(days=index * 30)).isoformat(),
        })
        attrs.append({
            "item_type": "bench",
            "subtype": "backless bench",
            "material": "mild steel",
            "scope": "supply only",
            "length_mm": float(1500 + (index % 3) * 300),
        })
        rows[-1]["unit_norm"] = "no"
    return pd.DataFrame(rows), attrs


def test_context_fields_need_independent_coverage_and_remain_shadow_only():
    dataset, attrs = _context_dataset()
    report = build_context_readiness(dataset, attrs)
    for field in ("quantity", "supplier", "quotation_date", "location"):
        evidence = report["families"]["bench"][field]
        assert evidence["status"] == "ready_for_modeling"
        assert not evidence["approved_for_price_adjustment"]
    assert report["approved_adjustments"] == []


def test_missing_context_data_blocks_adjustments_and_explains_the_refusal():
    dataset = pd.DataFrame([
        {"source_group": "q-1", "pricing_eligible": True,
         "quantity": None, "supplier": None, "location": None,
         "quotation_date": "2026-01-01"},
    ])
    report = build_context_readiness(dataset, [{
        "item_type": "bench", "material": "mild steel",
        "scope": "supply only", "length_mm": 1800,
    }])
    assert report["families"]["bench"]["quantity"]["status"] == "needs_data"
    warnings = context_adjustment_warnings({
        "item_type": "bench",
        "quantity": 10,
        "context_fields": ["quantity"],
    }, report)
    assert len(warnings) == 1
    assert "did not change the price" in warnings[0]
    assert "coverage" in warnings[0]
    targets = report["families"]["bench"]["quantity"]["collection_targets"]
    assert targets["additional_populated_rows_for_coverage"] == 1
    assert targets["additional_populated_quote_groups"] == 12
    assert targets["additional_matched_product_cohorts"] == 3


def test_explicit_v3_context_is_normalized_but_not_implicitly_approved():
    context = normalize_pricing_context({
        "product_family": "bench",
        "supplier": "  ACME  Industries ",
        "location": " Abu Dhabi ",
        "quotation_date": "13 July 2026",
        "quantity": 20,
    })
    assert context["supplier"] == "acme industries"
    assert context["location"] == "abu dhabi"
    assert context["quotation_date"] == "2026-07-13"
    assert context["quantity"] == 20


def test_context_adjustment_cannot_be_approved_without_implementation(monkeypatch):
    dataset, attrs = _context_dataset()
    monkeypatch.setattr(
        context_readiness, "APPROVED_CONTEXT_ADJUSTMENTS", {"bench.quantity"}
    )
    monkeypatch.setattr(
        context_readiness, "IMPLEMENTED_CONTEXT_ADJUSTMENTS", set()
    )
    with pytest.raises(RuntimeError, match="approved without an implementation"):
        build_context_readiness(dataset, attrs)


def test_context_adjustment_cannot_precede_its_base_family(monkeypatch):
    dataset, attrs = _context_dataset()
    monkeypatch.setattr(
        context_readiness, "APPROVED_CONTEXT_ADJUSTMENTS", {"bench.quantity"}
    )
    monkeypatch.setattr(
        context_readiness, "IMPLEMENTED_CONTEXT_ADJUSTMENTS", {"bench.quantity"}
    )
    monkeypatch.setattr(context_readiness, "APPROVED_AUTO_FAMILIES", set())
    with pytest.raises(RuntimeError, match="before their base family"):
        build_context_readiness(dataset, attrs)
