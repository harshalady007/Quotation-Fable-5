"""Streamlit UI for the quotation pricing bot.

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import config
from cleaner import EmptyDatasetError
from data_loader import DataLoadError
from similarity_search import SearchError

st.set_page_config(page_title="Quotation Pricing Bot", page_icon="💰", layout="wide")
st.title("💰 Quotation Pricing Bot")
st.caption(
    "Enter a new item or service description. The bot finds the most similar "
    "historical quotation items, compares pricing attributes like an "
    "estimator, and predicts a unit price."
)


@st.cache_resource(show_spinner="Loading and indexing the quotation dataset...")
def load_engine():
    from pricing_engine import PricingEngine
    return PricingEngine()


try:
    engine = load_engine()
except (DataLoadError, EmptyDatasetError) as exc:
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
    if not config.DEEPSEEK_API_KEY:
        st.warning(
            "DEEPSEEK_API_KEY is not set — predictions will use the "
            "statistical fallback (median/weighted average of matches). "
            "See the README for how to set the key."
        )

description = st.text_area(
    "Item / service description",
    placeholder="e.g. Supply and install 50mm diameter stainless steel "
                "handrail with brushed finish",
    height=100,
)

if st.button("Predict price", type="primary"):
    if not description.strip():
        st.warning("Please enter a description first.")
        st.stop()
    try:
        with st.spinner("Searching history and estimating price..."):
            result = engine.predict_price(description, top_k=top_k)
    except (ValueError, SearchError) as exc:
        st.error(str(exc))
        st.stop()

    # ---- Headline result ----
    c1, c2, c3, c4 = st.columns(4)
    price = result["predicted_unit_price"]
    c1.metric("Predicted unit price",
              f"{price:,.2f} {result['currency']}" if price is not None else "n/a")
    c2.metric("Unit", result["unit"] or "unknown")
    c3.metric("Confidence", result["confidence"])
    c4.metric("Price source",
              "Statistical fallback" if result["fallback_used"] else "DeepSeek AI")

    for w in result["warnings"]:
        st.warning(w)

    st.subheader("Estimator reasoning")
    st.write(result["reasoning"] or "_No reasoning returned._")
    if result.get("statistical_anchor") is not None:
        st.write(
            f"**Statistical anchor:** {result['statistical_anchor']:,.2f} "
            f"{result['currency']} (similarity-weighted median of the "
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
