"""Tests for cleaning, attribute extraction and similarity search."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attribute_extractor import compare_attributes, extract_attributes
from cleaner import clean_dataset, normalize_text, normalize_unit
from similarity_search import SearchError, SimilaritySearcher


def make_df(rows):
    base = {"description": None, "item_name": None, "unit": None,
            "quantity": None, "rate": None, "amount": None, "category": None,
            "section": None, "location": None, "remarks": None,
            "source": None, "date": None}
    return pd.DataFrame([{**base, **r} for r in rows])


SAMPLE = make_df([
    {"description": "Supply and install stainless steel handrail 50mm dia, "
                    "brushed finish", "unit": "LM", "rate": 450.0,
     "category": "Metalwork"},
    {"description": "Supply only aluminium handrail 40mm dia powder coated",
     "unit": "LM", "rate": 210.0, "category": "Metalwork"},
    {"description": "Stainless steel plate 3mm thick", "unit": "Nos",
     "rate": 120.0, "category": "Metalwork"},
    {"description": "Painting works to internal walls", "unit": "m2",
     "rate": 18.0, "category": "Painting"},
    {"description": "Supply & installation of mild steel bollard 220mm dia "
                    "x 1000mm H, galvanized", "unit": "Nos", "rate": 950.0},
    # Rows that cleaning must drop:
    {"description": "", "rate": 100.0},                      # no description
    {"description": "Ghost item with no price", "rate": None},
    {"description": "Negative rate item", "rate": -5.0},
])


def test_normalize_text_preserves_specs():
    s = normalize_text("Supply & Install 50 mm DIA. S.S-316 handrail,\nBrushed finish")
    assert "50mm" in s
    assert "dia" in s
    assert "brushed" in s
    assert "supply and install" in s


def test_normalize_unit():
    assert normalize_unit("Nos") == "no"
    assert normalize_unit("LM") == "m"
    assert normalize_unit("SQM") == "m2"
    assert normalize_unit(None) == ""


def test_clean_dataset_drops_bad_rows():
    cleaned = clean_dataset(SAMPLE)
    assert len(cleaned) == 5  # the 3 bad rows are gone
    assert (cleaned["rate"] > 0).all()
    assert (cleaned["clean_text"].str.len() >= 3).all()
    assert "search_text" in cleaned.columns


def test_attribute_extraction():
    attrs = extract_attributes(normalize_text(
        "Supply and install 50mm diameter stainless steel handrail "
        "with brushed finish"))
    assert attrs["material"] == "stainless steel"
    assert attrs["diameter_mm"] == 50.0
    assert attrs["finish"] == "brushed"
    assert attrs["scope"] == "supply and install"
    assert attrs["category"] == "metalwork"


def test_compare_attributes_flags_material_mismatch():
    a = extract_attributes(normalize_text("stainless steel handrail 50mm dia"))
    b = extract_attributes(normalize_text("aluminium handrail 50mm dia"))
    cmp = compare_attributes(a, b)
    assert any("material" in m for m in cmp["mismatched"])
    assert any("diameter_mm" in m for m in cmp["matched"])


def test_search_prefers_true_match_over_keyword_overlap():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    out = searcher.search(
        "Supply and install 50mm diameter stainless steel handrail with "
        "brushed finish", top_k=3)
    top = out["matches"][0]
    # The stainless steel 50mm handrail must beat the aluminium handrail
    # and the stainless steel plate.
    assert "stainless steel handrail" in top["clean_description"]
    assert top["rate"] == 450.0
    assert top["similarity_score"] > out["matches"][1]["similarity_score"]
    scores = {m["clean_description"]: m["similarity_score"] for m in out["matches"]}
    for desc, score in scores.items():
        if "aluminium" in desc or "plate" in desc:
            assert score < top["similarity_score"]


def test_search_result_shape():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    out = searcher.search("mild steel bollard 220mm dia galvanized", top_k=2)
    m = out["matches"][0]
    for key in ("rank", "description", "unit", "rate", "similarity_score",
                "text_similarity", "attribute_score", "matched_attributes",
                "mismatched_attributes", "explanation"):
        assert key in m
    assert "bollard" in m["clean_description"]


def test_scope_defaults_to_supply_and_install():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    # No scope in the input -> assumed supply and install.
    out = searcher.search("stainless steel handrail 50mm dia brushed", top_k=2)
    assert out["input_attributes"]["scope"] == "supply and install"
    assert out["input_attributes"].get("scope_assumed") is True
    # Explicit scope is respected, not overridden.
    out = searcher.search("supply only aluminium handrail 40mm dia", top_k=2)
    assert out["input_attributes"]["scope"] == "supply only"
    assert "scope_assumed" not in out["input_attributes"]


def test_empty_query_rejected():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    with pytest.raises(SearchError):
        searcher.search("   ")


def test_price_independent_of_top_k(monkeypatch):
    """The predicted price must not change with how many matches are shown."""
    import config
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")  # deterministic fallback
    from pricing_engine import select_pricing_matches
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    from deepseek_pricing import fallback_prediction

    def price_at(top_k):
        out = searcher.search("stainless steel handrail 50mm dia brushed",
                              top_k=max(top_k, config.PRICING_MAX_MATCHES))
        pricing = select_pricing_matches(out["matches"])
        return fallback_prediction(pricing, out["weak_matches"], "no key")[
            "predicted_unit_price"]

    prices = {price_at(k) for k in (1, 3, 5, 10)}
    assert len(prices) == 1, f"price varies with top_k: {prices}"


def test_real_dataset_if_available():
    import config
    from data_loader import load_dataset
    if not Path(config.DATA_PATH).exists():
        pytest.skip("real dataset not present")
    cleaned = clean_dataset(load_dataset(config.DATA_PATH))
    assert len(cleaned) > 100
    searcher = SimilaritySearcher(cleaned)
    out = searcher.search("stainless steel bollard 220mm dia", top_k=5)
    assert len(out["matches"]) == 5
    assert "bollard" in out["matches"][0]["clean_description"]
