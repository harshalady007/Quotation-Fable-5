"""Ties loading, cleaning, search and DeepSeek pricing into one API."""

import logging
from functools import lru_cache

import config
from cleaner import clean_dataset
from data_loader import load_dataset
from deepseek_pricing import predict_price_with_deepseek, weighted_median_rate
from similarity_search import SimilaritySearcher

logger = logging.getLogger(__name__)


def select_pricing_matches(matches: list, input_type: str | None = None,
                           input_unit: str | None = None) -> list:
    """Pick the fixed set of matches the price is computed from.

    Independent of how many matches the user displays: takes matches whose
    score is within PRICING_RELATIVE_CUTOFF of the best score, capped at
    PRICING_MAX_MATCHES, with at least PRICING_MIN_MATCHES. Eligibility is
    narrowed in order of importance:
      1. same item type (or a compatible cousin type) when any exist —
         a planter is priced from planters, never from litter bins;
      2. same unit basis when any exist — a per-m2 input is never priced
         from per-item rates while per-m2 rates are available.
    """
    from attribute_extractor import types_compatible

    if not matches:
        return []
    pool = matches
    if input_type:
        same_type = [m for m in pool if m.get("item_type") == input_type]
        if not same_type:
            same_type = [m for m in pool
                         if types_compatible(input_type, m.get("item_type"))]
        if same_type:
            pool = same_type
    if input_unit:
        same_unit = [m for m in pool
                     if (m.get("unit_norm") or "") == input_unit]
        if same_unit:
            pool = same_unit
    best = pool[0]["similarity_score"]
    cutoff = best * config.PRICING_RELATIVE_CUTOFF
    selected = [m for m in pool
                if m["similarity_score"] >= cutoff][:config.PRICING_MAX_MATCHES]
    # One strong comparable beats several weak ones: only fall back to the
    # raw top of the pool when the cutoff selected nothing at all.
    if not selected:
        selected = pool[:config.PRICING_MIN_MATCHES]
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
        input_type = search["input_attributes"].get("item_type")
        input_unit = search["input_attributes"].get("unit_hint")
        pricing_matches = select_pricing_matches(pool, input_type, input_unit)
        pricing_ranks = {m["rank"] for m in pricing_matches}
        matches = pool[:max(int(top_k), 1)]
        for m in matches:
            m["used_for_pricing"] = m["rank"] in pricing_ranks

        warnings = []
        if input_unit and not any((m.get("unit_norm") or "") == input_unit
                                  for m in pricing_matches):
            units_found = sorted({m.get("unit_norm") or "unknown"
                                  for m in pricing_matches})
            warnings.append(
                f"The input is priced per '{input_unit}' but no historical "
                f"match uses that unit (matches are per {', '.join(units_found)}). "
                "Rates are NOT directly comparable — treat this estimate as "
                "indicative only."
            )
        if input_type and not any(m.get("item_type") == input_type
                                  for m in pricing_matches):
            warnings.append(
                f"No historical items of type '{input_type}' were found in "
                "the dataset; the price is based on the closest other items "
                "and should be treated with extra caution."
            )
        input_size = search["input_attributes"].get("max_size_mm")
        if input_size and pricing_matches:
            comp_sizes = [m for m in pricing_matches
                          if any("overall size" in d for d in m["differences"])]
            if len(comp_sizes) == len(pricing_matches):
                warnings.append(
                    "All historical matches are a very different overall size "
                    "from the input item; the rate has been adjusted for size "
                    "but should be verified."
                )
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
