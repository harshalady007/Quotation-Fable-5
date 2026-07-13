"""Production safety-gate and deterministic pricing tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cleaner import (clean_dataset, normalize_search_text, normalize_text,
                     source_lineage)
from data_corrections import DataCorrectionError, apply_data_corrections
from data_quality import annotate_data_quality
from family_readiness import release_gate
from attribute_extractor import extract_attributes
from pricing_context import apply_pricing_context
from production_pricing import price_from_comparables


def comparable(rate, score, length=1000, eligible=True):
    return {
        "rank": int(rate),
        "rate": float(rate),
        "similarity_score": score,
        "item_type": "planter",
        "unit_norm": "no",
        "scope": "supply only",
        "material": "mild steel",
        "pricing_eligible": eligible,
        "attributes": {
            "length_mm": float(length),
            "width_mm": 500.0,
            "height_mm": 500.0,
            "civil_works": None,
        },
    }


def planter_attrs(**overrides):
    attrs = {
        "item_type": "planter",
        "unit_hint": "no",
        "scope": "supply only",
        "material": "mild steel",
        "length_mm": 1000.0,
        "width_mm": 500.0,
        "height_mm": 500.0,
    }
    attrs.update(overrides)
    return attrs


def test_bare_item_number_is_not_a_unit():
    attrs = extract_attributes(normalize_text("ITEM NO J - GRP planter"))
    assert attrs["unit_hint"] is None
    explicit = extract_attributes(normalize_text("GRP planter priced per no"))
    assert explicit["unit_hint"] == "no"


def test_explicit_context_overrides_free_text():
    extracted = extract_attributes(normalize_text(
        "mild steel waste bin, supply and install"
    ))
    result = apply_pricing_context(extracted, {
        "product_family": "litter bin",
        "unit": "Nos",
        "scope": "supply only",
        "material": "HDPE",
        "capacity_l": 120,
    })
    assert result["unit_hint"] == "no"
    assert result["scope"] == "supply only"
    assert result["material"] == "hdpe"
    assert result["capacity_l"] == 120


def test_boilerplate_removed_from_search_evidence():
    cleaned = normalize_search_text(
        'Bin Total Value in AED AED7,129.50 Brand & Origin "BLUESTREAM", "MADE IN UAE"'
    )
    assert "7,129" not in cleaned
    assert "bluestream" not in cleaned
    assert "made in uae" not in cleaned


def test_quality_gate_quarantines_missing_unit_and_price_conflict():
    frame = pd.DataFrame([
        {"clean_text": "same bin", "unit_norm": "no", "rate": 100,
         "source": "a", "date": "2026-01-01"},
        {"clean_text": "same bin", "unit_norm": "no", "rate": 250,
         "source": "b", "date": "2026-01-02"},
        {"clean_text": "other bin", "unit_norm": "", "rate": 100,
         "source": "c", "date": "2026-01-03"},
    ])
    checked = annotate_data_quality(frame)
    assert not checked.loc[0, "pricing_eligible"]
    assert not checked.loc[1, "pricing_eligible"]
    assert not checked.loc[2, "pricing_eligible"]
    assert "conflicting_duplicate_rate" in checked.loc[0, "data_quality_flags"]
    assert "missing_unit" in checked.loc[2, "data_quality_flags"]


def test_source_lineage_groups_revisions_and_quarantines_old_prices():
    assert source_lineage("BS-QT-22-20-15074 - R2.pdf") == (
        "bs-qt-22-20-15074", 2
    )
    assert source_lineage("BS-QT-22-20-15074.pdf") == (
        "bs-qt-22-20-15074", 0
    )
    assert source_lineage("BS-QT-22-20-15444 - R2_1_1.pdf") == (
        "bs-qt-22-20-15444", 2
    )
    assert source_lineage("BS-QT-22-20-14175_2B & 3B.pdf")[0] == (
        "bs-qt-22-20-14175"
    )
    frame = pd.DataFrame([
        {"source": "Q-100.pdf", "source_group": "q-100",
         "source_revision": 0, "unit_norm": "no", "rate": 100,
         "clean_text": "bench a", "date": "2026-01-01"},
        {"source": "Q-100-R2.pdf", "source_group": "q-100",
         "source_revision": 2, "unit_norm": "no", "rate": 120,
         "clean_text": "bench a revised", "date": "2026-01-02"},
    ])
    checked = annotate_data_quality(frame)
    assert "superseded_revision" in checked.loc[0, "data_quality_flags"]
    assert not checked.loc[0, "pricing_eligible"]
    assert checked.loc[1, "pricing_eligible"]


def test_cleaning_preserves_same_rate_from_independent_sources():
    base = {"item_name": "Bench", "description": "Mild steel bench",
            "unit": "Nos", "quantity": None, "rate": 100, "amount": None,
            "category": "Supply Only", "section": None, "location": None,
            "remarks": None, "date": "2026-01-01"}
    cleaned = clean_dataset(pd.DataFrame([
        {**base, "source": "q-1.pdf"},
        {**base, "source": "q-2.pdf"},
    ]))
    assert len(cleaned) == 2
    assert cleaned["source_group"].nunique() == 2


def test_production_price_uses_validated_comparables():
    decision = price_from_comparables(planter_attrs(), [
        comparable(100, 0.90), comparable(110, 0.80),
        comparable(120, 0.70), comparable(500, 0.95, eligible=False),
    ], approved_families={"planter"})
    assert decision["status"] == "priced"
    assert decision["predicted_unit_price"] == 110.0
    assert len(decision["comparables"]) == 3


def test_strict_evaluator_abstains_when_evidence_or_specs_are_missing():
    missing = price_from_comparables(planter_attrs(length_mm=None), [
        comparable(100, 0.9), comparable(110, 0.8), comparable(120, 0.7)
    ], approved_families={"planter"}, always_estimate=False)
    assert missing["status"] == "abstained"
    assert missing["predicted_unit_price"] is None

    sparse = price_from_comparables(planter_attrs(), [
        comparable(100, 0.9), comparable(110, 0.8)
    ], approved_families={"planter"}, always_estimate=False)
    assert sparse["status"] == "abstained"
    assert any("at least 3" in reason for reason in sparse["review_reasons"])


def test_duplicate_item_name_is_not_repeated():
    base = {"description": "Bench made of aluminium", "item_name": "Bench",
            "unit": "Nos", "quantity": None, "rate": 100, "amount": None,
            "category": "Supply Only", "section": None, "location": None,
            "remarks": None, "source": "q.pdf", "date": "2026-01-01"}
    cleaned = clean_dataset(pd.DataFrame([base]))
    assert cleaned.iloc[0]["full_description"] == "Bench made of aluminium"


def test_approved_correction_overlay_is_audited(tmp_path):
    base = {"description": "Bench made of aluminium L1800mm", "item_name": "Bench",
            "unit": None, "quantity": None, "rate": 100, "amount": None,
            "category": "Supply Only", "section": None, "location": None,
            "remarks": None, "source": "q.pdf", "date": "2026-01-01"}
    cleaned = clean_dataset(pd.DataFrame([base]))
    record_id = cleaned.iloc[0]["record_id"]
    manifest = {
        "schema_version": 1,
        "records": [{
            "record_id": record_id,
            "status": "approved",
            "reason": "Estimator confirmed the quotation unit and subtype.",
            "reviewed_by": "estimator@example.com",
            "set": {"unit": "Nos", "category": "Supply and Delivery"},
            "attributes": {"subtype": "backless bench"},
        }],
    }
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    corrected = apply_data_corrections(cleaned, path)
    assert corrected.iloc[0]["unit_norm"] == "no"
    assert corrected.iloc[0]["attribute_overrides"] == {"subtype": "backless bench"}
    assert corrected.iloc[0]["correction_ids"] == [record_id]
    assert "supply and delivery" in corrected.iloc[0]["search_text"]
    assert corrected.attrs["corrections"]["applied"] == 1


def test_proposed_correction_never_changes_production_data(tmp_path):
    base = {"description": "Mild steel bench L1800mm", "item_name": "Bench",
            "unit": None, "quantity": None, "rate": 100, "amount": None,
            "category": "Supply Only", "section": None, "location": None,
            "remarks": None, "source": "q.pdf", "date": "2026-01-01"}
    cleaned = clean_dataset(pd.DataFrame([base]))
    manifest = {
        "schema_version": 1,
        "records": [{
            "record_id": cleaned.iloc[0]["record_id"],
            "status": "proposed",
            "reason": "Unit needs estimator confirmation.",
            "set": {"unit": "Nos"},
        }],
    }
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    corrected = apply_data_corrections(cleaned, path)
    assert corrected.iloc[0]["unit_norm"] == ""
    assert corrected.attrs["corrections"]["applied"] == 0


def test_unmatched_approved_correction_stops_pricing(tmp_path):
    base = {"description": "Mild steel bench L1800mm", "item_name": "Bench",
            "unit": "Nos", "quantity": None, "rate": 100, "amount": None,
            "category": "Supply Only", "section": None, "location": None,
            "remarks": None, "source": "q.pdf", "date": "2026-01-01"}
    cleaned = clean_dataset(pd.DataFrame([base]))
    manifest = {
        "schema_version": 1,
        "records": [{
            "record_id": "record-that-no-longer-exists",
            "status": "approved",
            "reason": "Previously reviewed row.",
            "reviewed_by": "estimator@example.com",
            "set": {"unit": "Nos"},
        }],
    }
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataCorrectionError, match="no longer match"):
        apply_data_corrections(cleaned, path)


def test_v2_release_gate_requires_accuracy_coverage_and_no_large_errors():
    good = {
        "priced_holdouts": 30, "automatic_coverage": 0.40,
        "priced_quote_groups": 6,
        "within_20": 0.85, "median_ape": 0.10, "p90_ape": 0.30,
        "factor2_errors": 0,
    }
    assert release_gate(good) == (True, [])
    bad = {**good, "factor2_errors": 1, "automatic_coverage": 0.20}
    passed, failures = release_gate(bad)
    assert not passed
    assert any("coverage" in reason for reason in failures)
    assert any("factor-of-two" in reason for reason in failures)


def test_production_never_mixes_known_subtypes():
    attrs = planter_attrs(subtype="integrated seating")
    matches = [
        {**comparable(100, 0.90), "subtype": "integrated seating"},
        {**comparable(110, 0.80), "subtype": "integrated seating"},
        {**comparable(120, 0.70), "subtype": "integrated seating"},
        {**comparable(20, 0.99), "subtype": "standalone planter"},
    ]
    decision = price_from_comparables(
        attrs, matches, approved_families={"planter"}
    )
    assert decision["status"] == "priced"
    assert decision["predicted_unit_price"] == 110.0
    assert all(m["subtype"] == "integrated seating"
               for m in decision["comparables"])


def test_default_policy_prices_unapproved_family_with_warning():
    decision = price_from_comparables(planter_attrs(), [
        comparable(100, 0.9), comparable(110, 0.8), comparable(120, 0.7)
    ])
    assert decision["status"] == "priced"
    assert decision["predicted_unit_price"] == 110.0
    assert decision["confidence"] == "Low"
    assert decision["price_interval"] == {"low": 82.5, "high": 137.5}
    assert not decision["release_gate_approved"]
    assert any("has not passed" in reason
               for reason in decision["review_reasons"])


def test_universal_policy_prices_sparse_or_incomplete_evidence():
    missing = price_from_comparables(planter_attrs(length_mm=None), [
        comparable(100, 0.9), comparable(110, 0.8), comparable(120, 0.7)
    ])
    assert missing["status"] == "priced"
    assert missing["predicted_unit_price"] == 110.0
    assert missing["fallback_used"]
    assert missing["confidence"] in {"Low", "Very low"}
    assert missing["price_interval"]["low"] < 110 < missing["price_interval"]["high"]

    sparse = price_from_comparables(planter_attrs(), [
        comparable(100, 0.9), comparable(110, 0.8)
    ])
    assert sparse["status"] == "priced"
    assert sparse["predicted_unit_price"] in {100.0, 110.0}
    assert sparse["fallback_used"]


def test_universal_policy_excludes_quarantined_rates_and_keeps_unit_basis():
    wrong_unit = comparable(5000, 0.99)
    wrong_unit["unit_norm"] = "m"
    quarantined = comparable(9000, 1.0, eligible=False)
    decision = price_from_comparables(planter_attrs(), [
        wrong_unit, quarantined, comparable(100, 0.8), comparable(120, 0.7)
    ])
    assert decision["status"] == "priced"
    assert decision["estimated_unit"] == "no"
    assert decision["predicted_unit_price"] in {100.0, 120.0}
    assert all(item["pricing_eligible"] for item in decision["comparables"])
    assert all(item["unit_norm"] == "no" for item in decision["comparables"])


def test_bench_special_feature_requires_its_own_comparable_cohort():
    attrs = {
        "item_type": "bench", "unit_hint": "no", "scope": "supply only",
        "material": "precast", "length_mm": 1800.0,
        "features": ["perforated"],
    }
    matches = [{
        "rate": rate, "similarity_score": 0.9, "item_type": "bench",
        "unit_norm": "no", "scope": "supply only", "material": "precast",
        "pricing_eligible": True,
        "attributes": {"length_mm": 1800.0, "features": []},
    } for rate in (2000, 2100, 2200)]
    decision = price_from_comparables(
        attrs, matches, approved_families={"bench"}, always_estimate=False
    )
    assert decision["status"] == "abstained"
    assert any("perforated" in reason for reason in decision["review_reasons"])


def test_real_engine_returns_numeric_estimate_for_every_supported_family():
    import config
    from pricing_engine import get_engine

    if not Path(config.DATA_PATH).exists():
        pytest.skip("real dataset not present")
    engine = get_engine()
    cases = [
        ("planter", "Mild steel planter 1000 x 500 x 500mm", {
            "length_mm": 1000, "width_mm": 500, "height_mm": 500,
        }),
        ("litter bin", "Stainless steel litter bin 60 litre", {
            "capacity_l": 60,
        }),
        ("recycle bin", "Mild steel recycle bin 120 litre", {
            "capacity_l": 120,
        }),
        ("bench", "Precast bench 1800mm long", {"length_mm": 1800}),
        ("bollard", "Stainless steel bollard 150mm dia x 900mm high", {
            "diameter_mm": 150, "height_mm": 900,
        }),
        ("bike rack", "Mild steel bike rack 1000mm long", {
            "length_mm": 1000,
        }),
    ]
    for family, description, specs in cases:
        result = engine.predict_price(description, pricing_context={
            "product_family": family,
            "unit": "no",
            "scope": "supply only",
            "material": "mild steel",
            **specs,
        })
        assert result["status"] == "priced", family
        assert result["predicted_unit_price"] > 0, family
        assert result["price_interval"]["low"] > 0, family
        assert result["price_interval"]["high"] >= result["predicted_unit_price"], family

    unknown = engine.predict_price("Completely custom fabricated item")
    assert unknown["status"] == "priced"
    assert unknown["predicted_unit_price"] > 0
