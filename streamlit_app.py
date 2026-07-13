"""Streamlit UI for the quotation pricing bot.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import config
from cleaner import EmptyDatasetError
from context_modeling import (ContextModelEvaluationError,
                              load_context_model_snapshot)
from context_readiness import ContextReadinessError
from data_corrections import DataCorrectionError
from data_loader import DataLoadError
from family_readiness import load_readiness_snapshot
from production_pricing import family_contract
from similarity_search import SearchError

st.set_page_config(page_title="Quotation Pricing Bot", page_icon="💰", layout="wide")
st.title("💰 Quotation Pricing Bot")
st.caption(
    "Enter a new item or service description. The bot finds the most similar "
    "historical quotation items, compares pricing attributes like an "
    "estimator, and predicts a unit price."
)
st.info(
    "Numeric estimates are enabled for every supported family. Confidence, "
    "evidence tier and price intervals show when historical validation is weak."
)


@st.cache_resource(show_spinner="Loading and indexing the quotation dataset...")
def load_engine():
    from pricing_engine import PricingEngine
    return PricingEngine()


try:
    engine = load_engine()
except (DataLoadError, EmptyDatasetError, DataCorrectionError,
        ContextReadinessError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info(
        "Check that the Excel file exists and set QUOTATION_DATA_PATH if it "
        "lives somewhere else. Current path: " + config.DATA_PATH
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of top matches", 1, 15, 5)
    st.divider()
    st.subheader("Dataset")
    st.write(f"**File:** `{engine.data_path}`")
    st.write(f"**Sheet:** {engine.sheet}")
    st.write(f"**Usable rows:** {len(engine.dataset)}")
    st.write(f"**Pricing eligible:** {engine.data_quality['pricing_eligible_rows']}")
    st.caption("Unsafe or incomplete records remain visible but are quarantined from pricing.")
    st.divider()
    st.subheader("V3 context adjustments")
    approved_context = engine.context_readiness.get("approved_adjustments", [])
    st.write("**Approved:** " + (", ".join(approved_context) or "none"))
    for field, evidence in engine.context_readiness.get("overall", {}).items():
        st.caption(f"{field}: {evidence['status']} ({evidence['coverage']:.0%} coverage)")

description = st.text_area(
    "Item / service description",
    placeholder="e.g. Supply and install 50mm diameter stainless steel "
                "handrail with brushed finish",
    height=100,
)

st.subheader("Confirmed pricing details")
f1, f2, f3, f4 = st.columns(4)
family = f1.selectbox("Product family (recommended)", ["", "planter", "litter bin",
                                            "recycle bin", "bench", "bollard",
                                            "bike rack"])
unit = f2.selectbox("Unit (recommended)", ["", "no", "m", "m2", "set"])
scope = f3.selectbox("Commercial scope (recommended)", ["", "supply only",
                                              "supply and delivery",
                                              "supply and install"])
civil = f4.selectbox("Civil works", ["", "excluded", "included"])
contract = family_contract(family) if family else None
subtype = st.selectbox(
    "Product subtype",
    [""] + list((contract or {}).get("known_subtypes", [])),
)
try:
    readiness = load_readiness_snapshot()
    readiness_item = next(
        (item for item in readiness.get("families", []) if item["family"] == family),
        None,
    )
except ValueError:
    readiness_item = None
if readiness_item:
    if readiness_item["status"] == "production":
        st.success(
            f"{family} is production approved: "
            f"{readiness_item['within_20']:.1%} within ±20% over "
            f"{readiness_item['priced_holdouts']} held-out cases."
        )
    else:
        st.info(
            f"{family} returns numeric estimates; historical validation remains "
            "in V3 shadow. "
            + "; ".join(readiness_item.get("release_gate_failures", []))
        )
if family:
    try:
        model_snapshot = load_context_model_snapshot()
        family_models = [
            item for item in model_snapshot.get("adjustments", [])
            if item.get("family") == family
        ]
        ready_models = [
            item["field"] for item in family_models
            if item.get("status") == "ready_for_approval"
        ]
        blocked_models = sum(
            item.get("status") not in {"ready_for_approval", "production"}
            for item in family_models
        )
        st.caption(
            "Offline context models ready for approval: "
            f"{', '.join(ready_models) if ready_models else 'none'}; "
            f"{blocked_models} blocked by evidence."
        )
    except ContextModelEvaluationError:
        st.caption("Offline context-model readiness snapshot is unavailable.")
f5, f6, f7, f8 = st.columns(4)
material = f5.text_input("Primary material (recommended)")
quantity = f6.number_input("Quantity", min_value=0.0, value=0.0)
capacity_l = f7.number_input("Capacity (litres)", min_value=0.0, value=0.0)
compartments = f8.number_input("Compartments / streams", min_value=0, value=0)
f9, f10, f11, f12 = st.columns(4)
mobility = f9.selectbox("Fixing / mobility", ["", "fixed", "movable", "removable"])
diameter_mm = f10.number_input("Diameter (mm)", min_value=0.0, value=0.0)
length_mm = f11.number_input("Length (mm)", min_value=0.0, value=0.0)
width_mm = f12.number_input("Width (mm)", min_value=0.0, value=0.0)
height_mm = st.number_input("Height (mm)", min_value=0.0, value=0.0)
f13, f14, f15 = st.columns(3)
supplier = f13.text_input("Supplier", help="Recorded for V3 evidence; not yet a price adjustment.")
location = f14.text_input("Project location")
quotation_date = f15.date_input("Quotation date", value=None)

if st.button("Predict price", type="primary"):
    if not description.strip():
        st.warning("Please enter a description first.")
        st.stop()
    try:
        with st.spinner("Searching history and estimating price..."):
            context = {
                "product_family": family, "subtype": subtype,
                "unit": unit, "scope": scope,
                "civil_works": civil, "material": material,
                "supplier": supplier, "location": location,
                "quotation_date": quotation_date.isoformat() if quotation_date else None,
                "quantity": quantity or None, "capacity_l": capacity_l or None,
                "compartments": compartments or None, "mobility": mobility,
                "diameter_mm": diameter_mm or None, "length_mm": length_mm or None,
                "width_mm": width_mm or None, "height_mm": height_mm or None,
            }
            result = engine.predict_price(description, top_k=top_k,
                                          pricing_context=context)
    except (ValueError, SearchError) as exc:
        st.error(str(exc))
        st.stop()

    # ---- Headline result ----
    st.info("Price estimate issued")
    c1, c2, c3, c4 = st.columns(4)
    price = result["predicted_unit_price"]
    c1.metric("Estimated unit price",
              f"{price:,.2f} {result['currency']}" if price is not None else "n/a")
    c2.metric("Unit", result["unit"] or "unknown")
    c3.metric("Confidence", result["confidence"])
    c4.metric("Price source", result.get("price_source", "Comparable engine"))

    for w in result["warnings"]:
        st.warning(w)

    st.subheader("Estimator reasoning")
    st.write(result["reasoning"] or "_No reasoning returned._")
    if result.get("statistical_anchor") is not None:
        st.write(
            f"**Statistical anchor:** {result['statistical_anchor']:,.2f} "
            f"{result['currency']} ({result.get('anchor_method', '')} over the "
            f"{len(result.get('pricing_matches_used', []))} strongest matches — "
            "the price is always computed from these, regardless of how many "
            "matches are displayed)"
        )
    if result["price_basis"]:
        st.write(f"**Price basis:** {result['price_basis']}")
    if result["adjustments"]:
        st.write("**Adjustments considered:**")
        for a in result["adjustments"]:
            st.write(f"- {a}")

    # ---- Input attributes ----
    with st.expander("Extracted input attributes", expanded=False):
        attrs = {k: v for k, v in result["input_attributes"].items()
                 if v not in (None, [])}
        st.json(attrs or {"note": "no attributes detected"})

    # ---- Matches table ----
    st.subheader(f"Top {len(result['matches'])} similar historical items")
    rows = []
    for m in result["matches"]:
        rows.append({
            "Rank": m["rank"],
            "Priced on": "✓" if m.get("used_for_pricing") else "—",
            "Similarity": f"{m['similarity_score']:.2f}",
            "Text sim": f"{m['text_similarity']:.2f}",
            "Attr score": f"{m['attribute_score']:.2f}",
            "Description": m["description"][:300],
            "Unit": m["unit"] or "",
            "Quantity": m["quantity"] or "",
            "Rate": m["rate"],
            "Rate (scope-adj)": m.get("scope_adjusted_rate") or m["rate"],
            "Amount": m["amount"] or "",
            "Category / scope": m["category"] or "",
            "Source file": m.get("source") or "—",
            "Date": m.get("date") or "—",
            "Matched attributes": "; ".join(m["matched_attributes"]) or "—",
            "Mismatched attributes": "; ".join(m["mismatched_attributes"]) or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- Per-match comparison detail ----
    st.subheader("Input vs match comparison")
    for m in result["matches"]:
        rate = f"{m['rate']:,.2f}" if m["rate"] is not None else "n/a"
        with st.expander(
            f"#{m['rank']} — {m['description'].splitlines()[0][:80]} "
            f"(rate {rate}, similarity {m['similarity_score']:.2f})"
        ):
            st.write(f"**Why it matched:** {m['explanation']}")
            cc1, cc2, cc3 = st.columns(3)
            cc1.write("**Matched**")
            cc1.write("\n".join(f"- {x}" for x in m["matched_attributes"]) or "—")
            cc2.write("**Mismatched**")
            cc2.write("\n".join(f"- {x}" for x in m["mismatched_attributes"]) or "—")
            cc3.write("**Not stated in item**")
            cc3.write("\n".join(f"- {x}" for x in m["missing_attributes"]) or "—")
            if m["differences"]:
                st.write("**Price-relevant differences:** " + "; ".join(m["differences"]))
            st.text(m["description"])
