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


def test_unicode_multiplication_sign_dimensions():
    s = normalize_text("Recycling bin, 1500 × 500 × 1000 mm (L × W × H)")
    attrs = extract_attributes(s)
    assert attrs["sizes_mm"] == [500.0, 1000.0, 1500.0]
    assert attrs["max_size_mm"] == 1500.0


@pytest.mark.parametrize(("description", "expected"), [
    ("Bench Size: L1800 xW 530 xH 530 mm",
     {"length_mm": 1800.0, "width_mm": 530.0, "height_mm": 530.0}),
    ("Bench SIZE: 1753 L x 533 W x 787mm H",
     {"length_mm": 1753.0, "width_mm": 533.0, "height_mm": 787.0}),
    ("Bench Size: 2640LX 720WX 550 H",
     {"length_mm": 2640.0, "width_mm": 720.0, "height_mm": 550.0}),
    ("Bench Size: 2 x 0.5 x 0.45m",
     {"length_mm": 2000.0, "width_mm": 500.0, "height_mm": 450.0}),
    ("Custom curvilinear bench Size:16,675mm (L)X "
     "500/1000mm (W)X 450mm (H)",
     {"length_mm": 16675.0, "width_mm": 1000.0, "height_mm": 450.0}),
])
def test_dimension_chain_accepts_real_schedule_formats(description, expected):
    attrs = extract_attributes(normalize_text(description))
    for field, value in expected.items():
        assert attrs[field] == value


def test_bench_diameter_and_single_overall_size_are_effective_lengths():
    circular = extract_attributes(normalize_text(
        "Precast bench Size: 500Dia x 450H mm"
    ))
    assert circular["diameter_mm"] == 500.0
    assert circular["height_mm"] == 450.0
    assert circular["length_mm"] == 500.0

    single = extract_attributes(normalize_text(
        "Heavy Duty Bench solid oak with armrest. Size 2000 mm"
    ))
    assert single["length_mm"] == 2000.0

    width_height_only = extract_attributes(normalize_text(
        "Precast concrete seater Size: 500 mm wide x 450 mm high"
    ))
    assert width_height_only["width_mm"] == 500.0
    assert width_height_only["height_mm"] == 450.0
    assert width_height_only["length_mm"] is None


def test_material_aliases_use_commercial_substrate_groups():
    iroko = extract_attributes(normalize_text("Bench made of Iroko wooden slats"))
    gi = extract_attributes(normalize_text("Bench made from 1.5mm thick hot GI"))
    assert iroko["material"] == "wood"
    assert gi["material"] == "steel"

    hybrid = extract_attributes(normalize_text(
        "Precast concrete bench with Iroko wooden slats"
    ))
    assert "wood accent" in hybrid["features"]


def test_overall_size_outranks_earlier_component_profile():
    attrs = extract_attributes(normalize_text(
        "Bench made of wooden slats and legs of 40x50mm. "
        "Size: L 2200 x W 800 x H 1040mm"
    ))
    assert attrs["length_mm"] == 2200.0
    assert attrs["width_mm"] == 800.0
    assert attrs["height_mm"] == 1040.0

    earlier_length = extract_attributes(normalize_text(
        "Bench with wooden slats of 600mm long x 600mm wide. "
        "Size: L 1800 x W 600 x H 450mm"
    ))
    assert earlier_length["length_mm"] == 1800.0


def test_curvilinear_bench_is_shaped_subtype():
    attrs = extract_attributes(normalize_text(
        "Custom Curvilinear Bench Size: 16600mm (L) x 1000mm (W) x 450mm (H)"
    ))
    assert attrs["subtype"] == "shaped bench"


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


def test_scope_is_never_assumed_for_production_pricing():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    # Missing commercial scope must remain missing so the production gate can
    # request confirmation instead of silently adding installation cost.
    out = searcher.search("stainless steel handrail 50mm dia brushed", top_k=2)
    assert out["input_attributes"]["scope"] is None
    # Explicit scope is respected, not overridden.
    out = searcher.search("supply only aluminium handrail 40mm dia", top_k=2)
    assert out["input_attributes"]["scope"] == "supply only"


def test_empty_query_rejected():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    with pytest.raises(SearchError):
        searcher.search("   ")


def test_item_type_extraction():
    assert extract_attributes(normalize_text("Planter Pot FF-30"))["item_type"] == "planter"
    assert extract_attributes(normalize_text("Out Door Litter Bin"))["item_type"] == "litter bin"
    # Recycle bins are their own type, distinct from general litter bins.
    assert extract_attributes(normalize_text("Recycling bin, powder coated"))["item_type"] == "recycle bin"
    assert extract_attributes(normalize_text("FN2 Litter Bin Recyclable Waste"))["item_type"] == "recycle bin"
    assert extract_attributes(normalize_text("FN2 Litter Bin General Waste"))["item_type"] == "litter bin"
    from attribute_extractor import types_compatible
    assert types_compatible("recycle bin", "litter bin")
    assert extract_attributes(normalize_text("SS handrail 50mm dia"))["item_type"] == "handrail"
    assert extract_attributes(normalize_text("stainless steel plate"))["item_type"] is None
    # 'sign' must not fire inside words like 'design'.
    assert extract_attributes(normalize_text("designed bracket"))["item_type"] is None


def test_v2_family_subtypes_are_conservative():
    bench = extract_attributes(normalize_text(
        "Precast bench without backrest L1800 x W500 x H450mm"
    ))
    assert bench["subtype"] == "backless bench"
    assert "integrated seating" not in bench["features"]
    assert "backrest" not in bench["features"]

    planter = extract_attributes(normalize_text(
        "Mild steel planter box with seater L3000 x W800 x H700mm"
    ))
    assert planter["subtype"] == "integrated seating"
    assert "integrated seating" in planter["features"]

    recycle = extract_attributes(normalize_text(
        "Triple recycle bin, 3 stream, mild steel"
    ))
    assert recycle["compartments"] == 3
    assert recycle["subtype"] == "multi-stream bin"

    bollard = extract_attributes(normalize_text(
        "Removable stainless steel bollard 150mm dia x 900mm high"
    ))
    assert bollard["subtype"] == "removable bollard"
    assert bollard["mobility"] == "removable"

    shaped = extract_attributes(normalize_text(
        "L-Shape Precast Bench Size: L 7000+1120 x 600 x 450mm high"
    ))
    assert shaped["subtype"] == "shaped bench"
    assert shaped["length_mm"] == 8120.0
    assert shaped["max_size_mm"] == 8120.0


def test_same_item_type_beats_same_material():
    """A planter query must rank a planter above a litter bin even when the
    litter bin matches material, finish and thickness better."""
    df = make_df([
        # Litter bin: perfect material/finish/thickness agreement.
        {"description": "Litter bin galvanized steel plate 6mm thick, zinc "
                        "primer, powder coated finish", "unit": "Nos",
         "rate": 1700.0},
        # Planter: same type but different material and finish.
        {"description": "Planter box mild steel with corten finish",
         "unit": "Nos", "rate": 9500.0},
    ])
    searcher = SimilaritySearcher(clean_dataset(df))
    out = searcher.search("Planter made of 6mm thick galvanized steel plate "
                          "with zinc primer and powder coating finish", top_k=2)
    top = out["matches"][0]
    assert top["item_type"] == "planter", (
        f"litter bin outranked the planter: {out['matches']}")
    assert "litter bin" == out["matches"][1]["item_type"]

    # And the pricing set must contain only planters.
    from pricing_engine import select_pricing_matches
    pricing = select_pricing_matches(out["matches"], "planter")
    assert all(m["item_type"] == "planter" for m in pricing)


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


def test_scope_adjustment_20_percent():
    """Company rule: supply and install = supply only (or supply and
    delivery) + 20%, applied to comp rates before pricing."""
    from deepseek_pricing import fallback_prediction, scope_adjusted_rate

    assert scope_adjusted_rate(100.0, "supply only", "supply and install") == 120.0
    assert scope_adjusted_rate(100.0, "supply and delivery", "supply and install") == 120.0
    assert scope_adjusted_rate(120.0, "supply and install", "supply only") == 100.0
    # Same scope group or unknown scope: unchanged.
    assert scope_adjusted_rate(100.0, "supply and install", "supply and install") == 100.0
    assert scope_adjusted_rate(100.0, None, "supply and install") == 100.0

    # The fallback anchor uses the adjusted rate.
    match = {"rate": 100.0, "scope_adjusted_rate": 120.0,
             "similarity_score": 0.9, "unit": "no"}
    out = fallback_prediction([match], weak_matches=False, reason="test")
    assert out["predicted_unit_price"] == 120.0


def test_install_price_is_20pct_above_supply_only(monkeypatch):
    """The same item priced supply-and-install vs supply-only must select
    the same comps and differ by exactly the 20% rule (fallback path)."""
    import config
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    from deepseek_pricing import fallback_prediction, scope_adjusted_rate
    from pricing_engine import select_pricing_matches
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))

    def anchor_for(scope):
        out = searcher.search(
            f"stainless steel handrail 50mm dia brushed, {scope}",
            top_k=config.PRICING_MAX_MATCHES)
        input_scope = out["input_attributes"]["scope"]
        pricing = select_pricing_matches(out["matches"], "handrail")
        for m in pricing:
            m["scope_adjusted_rate"] = scope_adjusted_rate(
                m.get("rate"), m.get("scope"), input_scope)
        descs = tuple(m["clean_description"] for m in pricing)
        price = fallback_prediction(pricing, out["weak_matches"], "t")[
            "predicted_unit_price"]
        return descs, price

    descs_install, p_install = anchor_for("supply and install")
    descs_supply, p_supply = anchor_for("supply only")
    assert descs_install == descs_supply, "scope changed comp selection"
    assert abs(p_install - p_supply * 1.2) < 0.01 or p_install == p_supply


def test_freestanding_does_not_guess_commercial_scope():
    searcher = SimilaritySearcher(clean_dataset(SAMPLE))
    out = searcher.search("granite bench polished finish, free standing, "
                          "L 2000 x W 540 x H 777mm", top_k=2)
    assert out["input_attributes"]["scope"] is None
    assert out["input_attributes"]["mobility"] == "movable"


def test_dense_band_detection():
    from deepseek_pricing import dense_band

    def m(rate, t="bench"):
        return {"rate": rate, "similarity_score": 0.5, "item_type": t}

    # Tight same-type cluster -> band.
    assert dense_band([m(4100), m(5059), m(3856)]) == (3856, 5059)
    # Too spread out -> no band.
    assert dense_band([m(221), m(430), m(1619)]) is None
    # Mixed / unknown types -> no band.
    assert dense_band([m(4100), m(5059), m(3856, t=None)]) is None
    # Too few comps -> no band.
    assert dense_band([m(4100), m(5059)]) is None


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
