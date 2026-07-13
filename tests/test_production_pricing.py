"""Production safety-gate and deterministic pricing tests."""

import pandas as pd

from cleaner import clean_dataset, normalize_search_text, normalize_text
from data_quality import annotate_data_quality
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


def test_production_price_uses_validated_comparables():
    decision = price_from_comparables(planter_attrs(), [
        comparable(100, 0.90), comparable(110, 0.80),
        comparable(120, 0.70), comparable(500, 0.95, eligible=False),
    ])
    assert decision["status"] == "priced"
    assert decision["predicted_unit_price"] == 110.0
    assert len(decision["comparables"]) == 3


def test_production_abstains_when_evidence_or_specs_are_missing():
    missing = price_from_comparables(planter_attrs(length_mm=None), [
        comparable(100, 0.9), comparable(110, 0.8), comparable(120, 0.7)
    ])
    assert missing["status"] == "manual_review"
    assert missing["predicted_unit_price"] is None

    sparse = price_from_comparables(planter_attrs(), [
        comparable(100, 0.9), comparable(110, 0.8)
    ])
    assert sparse["status"] == "manual_review"
    assert any("at least 3" in reason for reason in sparse["review_reasons"])


def test_duplicate_item_name_is_not_repeated():
    base = {"description": "Bench made of aluminium", "item_name": "Bench",
            "unit": "Nos", "quantity": None, "rate": 100, "amount": None,
            "category": "Supply Only", "section": None, "location": None,
            "remarks": None, "source": "q.pdf", "date": "2026-01-01"}
    cleaned = clean_dataset(pd.DataFrame([base]))
    assert cleaned.iloc[0]["full_description"] == "Bench made of aluminium"
