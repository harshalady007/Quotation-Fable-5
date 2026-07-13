"""Hybrid similarity search: TF-IDF text similarity + attribute scoring."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from attribute_extractor import (compare_attributes, detect_scope,
                                 extract_attributes, types_compatible)
from cleaner import normalize_search_text, normalize_text, normalize_unit
from pricing_context import apply_pricing_context


class SearchError(Exception):
    pass


class SimilaritySearcher:
    """Indexes a cleaned dataset once, then answers search() queries."""

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise SearchError("Cannot build search index: dataset is empty.")
        if "search_text" not in df.columns:
            raise SearchError("Dataset missing 'search_text'; run cleaner first.")
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])
        # Pre-extract attributes for every dataset row (one-time cost).
        # The row's actual unit column is authoritative over any unit word
        # found inside the description text.
        self.item_attrs = []
        for i, text in enumerate(self.df["clean_text"]):
            attrs = extract_attributes(text)
            overrides = self.df.iloc[i].get("attribute_overrides")
            if isinstance(overrides, dict) and overrides:
                attrs = apply_pricing_context(attrs, overrides)
            unit_norm = self.df.iloc[i].get("unit_norm")
            if unit_norm:
                attrs["unit_hint"] = unit_norm
            # The scope of work usually lives in its own column (mapped to
            # category), not in the item description.
            category_text = normalize_text(self.df.iloc[i].get("category"))
            category_attrs = extract_attributes(category_text)
            if not attrs.get("scope"):
                attrs["scope"] = detect_scope(category_text)
            if not attrs.get("civil_works"):
                attrs["civil_works"] = category_attrs.get("civil_works")
            self.item_attrs.append(attrs)

    def search(self, query: str, top_k: int = 5,
               pricing_context: dict | None = None) -> dict:
        """Return top_k matches with combined text+attribute scoring."""
        if not query or not str(query).strip():
            raise SearchError("Input description is empty.")
        top_k = max(1, min(int(top_k), 20))

        clean_query = normalize_search_text(query)
        input_attrs = apply_pricing_context(
            extract_attributes(clean_query), pricing_context
        )

        query_vec = self.vectorizer.transform([clean_query])
        text_sims = cosine_similarity(query_vec, self.matrix).ravel()

        # Rescore a candidate pool much larger than top_k so attribute
        # matching can promote items that raw text similarity underrates
        # (terse descriptions score badly on TF-IDF even when they are the
        # best pricing comparables).
        input_type = input_attrs.get("item_type")
        input_unit = input_attrs.get("unit_hint")
        pool = min(len(self.df), max(top_k * 10, 200))
        ranked_indices = [int(idx) for idx in text_sims.argsort()[::-1]]
        candidates = ranked_indices[:pool]
        selected_indices = set(candidates)
        reserve_indices: set[int] = set()

        def append_reserve(predicate) -> None:
            added = 0
            seen_lineages = set()
            for idx in ranked_indices:
                row = self.df.iloc[idx]
                if (not bool(row.get("pricing_eligible", True))
                        or not _safe_num(row.get("rate"))
                        or not row.get("unit_norm")
                        or not predicate(idx, row)):
                    continue
                lineage = row.get("source_group") or row.get("record_id") or idx
                if lineage in seen_lineages:
                    continue
                seen_lineages.add(lineage)
                reserve_indices.add(idx)
                if idx not in selected_indices:
                    candidates.append(idx)
                    selected_indices.add(idx)
                added += 1
                if added >= config.PRICING_MAX_MATCHES:
                    break

        # Keep a small reserve outside the text-only top-200 window. This
        # guarantees coherent same-family/unit evidence when it exists and at
        # least some quality-eligible evidence for the universal fallback.
        if input_type and input_unit:
            append_reserve(lambda idx, row: (
                self.item_attrs[idx].get("item_type") == input_type
                and row.get("unit_norm") == input_unit
            ))
        if input_type:
            append_reserve(lambda idx, row: (
                self.item_attrs[idx].get("item_type") == input_type
            ))
        if input_unit:
            append_reserve(lambda idx, row: row.get("unit_norm") == input_unit)
        append_reserve(lambda idx, row: True)

        results = []
        for idx in candidates:
            comparison = compare_attributes(input_attrs, self.item_attrs[idx])
            text_score = float(text_sims[idx])
            final = (config.TEXT_WEIGHT * text_score
                     + config.ATTR_WEIGHT * comparison["attribute_score"])
            # Hard penalty when both sides have a recognized item type and
            # they differ: a litter bin must never outrank a real planter
            # for a planter query just because material/finish agree.
            item_type = self.item_attrs[idx].get("item_type")
            if (input_type and item_type and input_type != item_type
                    and not types_compatible(input_type, item_type)):
                final *= config.TYPE_MISMATCH_PENALTY
            input_subtype = input_attrs.get("subtype")
            item_subtype = self.item_attrs[idx].get("subtype")
            if input_subtype and item_subtype and input_subtype != item_subtype:
                final *= config.SUBTYPE_MISMATCH_PENALTY
            # Different size class (e.g. 50mm frame member vs 2.6m planter)
            # is nearly as disqualifying as a different item type — but only
            # for same-type, per-item products: for per-metre/per-m2 rates
            # the stated sizes are profiles, not product scale.
            in_size = input_attrs.get("max_size_mm")
            item_size = self.item_attrs[idx].get("max_size_mm")
            linear_units = ("m", "m2", "m3")
            if (in_size and item_size and input_type and item_type == input_type
                    and input_attrs.get("unit_hint") not in linear_units
                    and (self.item_attrs[idx].get("unit_hint") or "") not in linear_units):
                ratio = max(in_size, item_size) / max(min(in_size, item_size), 1.0)
                if ratio > config.SIZE_MISMATCH_RATIO:
                    final *= config.SIZE_MISMATCH_PENALTY
            # A product with integrated seating is a different product from
            # one without (planter vs planter-with-bench combo).
            in_seat = "integrated seating" in (input_attrs.get("features") or [])
            item_seat = "integrated seating" in (self.item_attrs[idx].get("features") or [])
            if in_seat != item_seat:
                final *= config.SEATING_MISMATCH_PENALTY
            row = self.df.iloc[idx]
            results.append({
                "_dataset_index": idx,
                "item_type": item_type,
                "subtype": item_subtype,
                "scope": self.item_attrs[idx].get("scope"),
                "material": self.item_attrs[idx].get("material"),
                "max_size_mm": self.item_attrs[idx].get("max_size_mm"),
                "size_proxy": self.item_attrs[idx].get("size_proxy"),
                "size_proxy_kind": self.item_attrs[idx].get("size_proxy_kind"),
                "attributes": dict(self.item_attrs[idx]),
                "description": row["full_description"],
                "record_id": _safe_str(row.get("record_id")),
                "clean_description": row["clean_text"],
                "unit": row.get("unit") if pd.notna(row.get("unit")) else None,
                "unit_norm": row.get("unit_norm") or None,
                "quantity": _safe_num(row.get("quantity")),
                "rate": _safe_num(row.get("rate")),
                "amount": _safe_num(row.get("amount")),
                "category": _safe_str(row.get("category")) or _safe_str(row.get("section")),
                "source": _safe_str(row.get("source")),
                "source_group": _safe_str(row.get("source_group")),
                "source_revision": _safe_num(row.get("source_revision")),
                "date": _safe_str(row.get("date")),
                "quotation_date": _safe_str(row.get("quotation_date")),
                "location": _safe_str(row.get("location")),
                "supplier": _safe_str(row.get("supplier")),
                "project": _safe_str(row.get("project")),
                "client": _safe_str(row.get("client")),
                "contractor": _safe_str(row.get("contractor")),
                "pricing_eligible": bool(row.get("pricing_eligible", True)),
                "data_quality_flags": list(row.get("data_quality_flags") or []),
                "correction_ids": list(row.get("correction_ids") or []),
                "similarity_score": round(final, 4),
                "text_similarity": round(text_score, 4),
                "attribute_score": comparison["attribute_score"],
                "matched_attributes": comparison["matched"],
                "mismatched_attributes": comparison["mismatched"],
                "missing_attributes": comparison["missing"],
                "differences": comparison["differences"],
                "explanation": _explain(text_score, comparison),
            })

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        candidate_results = results[:config.PRICING_CANDIDATE_RESULTS]
        retained_indices = {
            item["_dataset_index"] for item in candidate_results
        }
        candidate_results.extend(
            item for item in results
            if item["_dataset_index"] in reserve_indices
            and item["_dataset_index"] not in retained_indices
        )
        for item in candidate_results:
            item.pop("_dataset_index", None)

        for rank, r in enumerate(candidate_results, 1):
            r["rank"] = rank
        display_results = candidate_results[:top_k]

        best = candidate_results[0]["similarity_score"] if candidate_results else 0.0
        return {
            "input_clean": clean_query,
            "input_attributes": input_attrs,
            "matches": display_results,
            "candidate_matches": candidate_results,
            "weak_matches": best < config.WEAK_MATCH_THRESHOLD,
            "best_score": best,
        }


def _safe_num(v):
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_str(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    return s or None


def _explain(text_score: float, comparison: dict) -> str:
    parts = []
    if text_score >= 0.5:
        parts.append("strong text overlap with the input")
    elif text_score >= 0.25:
        parts.append("moderate text overlap with the input")
    else:
        parts.append("weak text overlap with the input")
    if comparison["matched"]:
        parts.append("matches on " + ", ".join(m.split(":")[0] for m in comparison["matched"]))
    if comparison["mismatched"]:
        parts.append("differs on " + ", ".join(m.split(":")[0] for m in comparison["mismatched"]))
    if comparison["missing"]:
        parts.append("does not state " + ", ".join(comparison["missing"]))
    return "; ".join(parts)
