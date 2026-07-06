"""Hybrid similarity search: TF-IDF text similarity + attribute scoring."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from attribute_extractor import compare_attributes, extract_attributes
from cleaner import normalize_text, normalize_unit


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
        self.item_attrs = [extract_attributes(t) for t in self.df["clean_text"]]

    def search(self, query: str, top_k: int = 5) -> dict:
        """Return top_k matches with combined text+attribute scoring."""
        if not query or not str(query).strip():
            raise SearchError("Input description is empty.")
        top_k = max(1, min(int(top_k), 20))

        clean_query = normalize_text(query)
        input_attrs = extract_attributes(clean_query)
        # Estimating default: when the input does not state a work scope,
        # assume supply and installation (dataset items keep their own scope).
        if input_attrs.get("scope") is None:
            input_attrs["scope"] = "supply and install"
            input_attrs["scope_assumed"] = True

        query_vec = self.vectorizer.transform([clean_query])
        text_sims = cosine_similarity(query_vec, self.matrix).ravel()

        # Rescore a candidate pool larger than top_k so attribute matching
        # can promote items that raw text similarity underrates.
        pool = min(len(self.df), max(top_k * 8, 40))
        candidates = text_sims.argsort()[::-1][:pool]

        results = []
        for idx in candidates:
            comparison = compare_attributes(input_attrs, self.item_attrs[idx])
            text_score = float(text_sims[idx])
            final = (config.TEXT_WEIGHT * text_score
                     + config.ATTR_WEIGHT * comparison["attribute_score"])
            row = self.df.iloc[idx]
            results.append({
                "description": row["full_description"],
                "clean_description": row["clean_text"],
                "unit": row.get("unit") if pd.notna(row.get("unit")) else None,
                "unit_norm": row.get("unit_norm") or None,
                "quantity": _safe_num(row.get("quantity")),
                "rate": _safe_num(row.get("rate")),
                "amount": _safe_num(row.get("amount")),
                "category": _safe_str(row.get("category")) or _safe_str(row.get("section")),
                "source": _safe_str(row.get("source")),
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
        results = results[:top_k]
        for rank, r in enumerate(results, 1):
            r["rank"] = rank

        best = results[0]["similarity_score"] if results else 0.0
        return {
            "input_clean": clean_query,
            "input_attributes": input_attrs,
            "matches": results,
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
