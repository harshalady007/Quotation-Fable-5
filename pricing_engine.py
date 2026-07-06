"""Ties loading, cleaning, search and DeepSeek pricing into one API."""

import logging
from functools import lru_cache

import config
from cleaner import clean_dataset
from data_loader import load_dataset
from deepseek_pricing import predict_price_with_deepseek, weighted_median_rate
from similarity_search import SimilaritySearcher

logger = logging.getLogger(__name__)


def select_pricing_matches(matches: list) -> list:
    """Pick the fixed set of matches the price is computed from.

    Independent of how many matches the user displays: takes matches whose
    score is within PRICING_RELATIVE_CUTOFF of the best score, capped at
    PRICING_MAX_MATCHES, with at least PRICING_MIN_MATCHES.
    """
    if not matches:
        return []
    best = matches[0]["similarity_score"]
    cutoff = best * config.PRICING_RELATIVE_CUTOFF
    selected = [m for m in matches
                if m["similarity_score"] >= cutoff][:config.PRICING_MAX_MATCHES]
    if len(selected) < config.PRICING_MIN_MATCHES:
        selected = matches[:config.PRICING_MIN_MATCHES]
    return selected


class PricingEngine:
    def __init__(self, data_path: str | None = None):
        self.data_path = data_path or config.DATA_PATH
        raw = load_dataset(self.data_path)          # raises DataLoadError
        self.dataset = clean_dataset(raw)           # raises EmptyDatasetError
        self.column_mapping = raw.attrs.get("column_mapping", {})
        self.sheet = raw.attrs.get("sheet", "")
        self.searcher = SimilaritySearcher(self.dataset)
        logger.info("PricingEngine ready: %d usable rows from sheet %r",
                    len(self.dataset), self.sheet)

    def predict_price(self, input_description: str, top_k: int = 5) -> dict:
        """Full pipeline: search -> compare -> DeepSeek (or fallback) -> result."""
        if not input_description or not str(input_description).strip():
            raise ValueError("Please enter an item or service description.")

        # Retrieve enough candidates for both the display table (top_k) and
        # the pricing set, which is always chosen by the same fixed rule so
        # the predicted price does not depend on top_k.
        pool_size = max(int(top_k), config.PRICING_MAX_MATCHES)
        search = self.searcher.search(input_description, top_k=pool_size)
        pool = search["matches"]
        pricing_matches = select_pricing_matches(pool)
        pricing_ranks = {m["rank"] for m in pricing_matches}
        matches = pool[:max(int(top_k), 1)]
        for m in matches:
            m["used_for_pricing"] = m["rank"] in pricing_ranks

        warnings = []
        if search["input_attributes"].get("scope_assumed"):
            warnings.append(
                "No work scope stated in the input; assumed 'supply and "
                "install' by default. Mention e.g. 'supply only' to override."
            )
        if search["weak_matches"]:
            warnings.append(
                f"Best match similarity is only {search['best_score']:.2f}; "
                "no strong historical match was found. Treat the prediction "
                "with caution."
            )

        prediction = predict_price_with_deepseek(
            input_description, search["input_attributes"], pricing_matches,
            search["weak_matches"],
        )
        anchor = weighted_median_rate(pricing_matches)

        return {
            "input_description": input_description,
            "input_attributes": search["input_attributes"],
            "statistical_anchor": round(anchor, 2) if anchor is not None else None,
            "pricing_matches_used": sorted(pricing_ranks),
            "predicted_unit_price": prediction["predicted_unit_price"],
            "currency": prediction["currency"],
            "unit": prediction["unit"],
            "confidence": prediction["confidence"],
            "reasoning": prediction["reasoning"],
            "price_basis": prediction.get("price_basis", ""),
            "adjustments": prediction.get("adjustments", []),
            "fallback_used": prediction.get("fallback_used", False),
            "matches": matches,
            "weak_matches": search["weak_matches"],
            "warnings": warnings + prediction.get("warnings", []),
        }


@lru_cache(maxsize=1)
def get_engine(data_path: str | None = None) -> PricingEngine:
    """Cached engine so the dataset is indexed once per process."""
    return PricingEngine(data_path)
