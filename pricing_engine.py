"""Production orchestration for guarded historical-comparable pricing."""

import logging
from functools import lru_cache

import config
from context_readiness import (build_context_readiness,
                               context_adjustment_warnings)
from data_corrections import correction_summary
from data_quality import quality_summary
from pricing_dataset import load_pricing_dataset
from production_pricing import PRICING_ENGINE_VERSION, price_from_comparables
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
        self.dataset = load_pricing_dataset(self.data_path)
        self.data_quality = quality_summary(self.dataset)
        self.data_quality["corrections"] = correction_summary(self.dataset)
        self.column_mapping = self.dataset.attrs.get("column_mapping", {})
        self.summary_context = self.dataset.attrs.get("summary_context", {})
        self.sheet = self.dataset.attrs.get("sheet", "")
        self.searcher = SimilaritySearcher(self.dataset)
        self.context_readiness = build_context_readiness(
            self.dataset, self.searcher.item_attrs
        )
        logger.info("PricingEngine ready: %d usable rows from sheet %r",
                    len(self.dataset), self.sheet)

    def predict_price(self, input_description: str, top_k: int = 5,
                      pricing_context: dict | None = None) -> dict:
        """Production pipeline: retrieve -> validate -> price or abstain.

        ``pricing_context`` contains user-confirmed structured fields.  Free
        text remains supported for backward compatibility, but missing
        price-defining information produces a manual-review decision instead
        of a fabricated number.
        """
        if not input_description or not str(input_description).strip():
            raise ValueError("Please enter an item or service description.")

        search = self.searcher.search(
            input_description,
            top_k=max(1, int(top_k)),
            pricing_context=pricing_context,
        )
        candidates = search["candidate_matches"]
        decision = price_from_comparables(search["input_attributes"], candidates)
        pricing_matches = decision["comparables"]
        pricing_ranks = {m["rank"] for m in pricing_matches}
        matches = search["matches"]
        for match in candidates:
            match["used_for_pricing"] = match["rank"] in pricing_ranks
            # Production comparables are exact-scope only; no hidden 20%
            # conversion is applied.
            match["scope_adjusted_rate"] = match.get("rate")

        warnings = list(decision["review_reasons"])
        warnings.extend(context_adjustment_warnings(
            search["input_attributes"], self.context_readiness
        ))
        quarantined_displayed = [m for m in matches if not m.get("pricing_eligible", True)]
        if quarantined_displayed:
            warnings.append(
                f"{len(quarantined_displayed)} displayed historical match(es) "
                "are quarantined from automatic pricing because of data-quality defects."
            )

        if decision["status"] == "priced":
            reasoning = (
                f"Automatic price based on {len(pricing_matches)} validated "
                "same-family, same-unit and same-scope historical comparables. "
                "A robust weighted median limits the influence of individual rates."
            )
        else:
            reasoning = (
                "No automatic price was issued because the evidence failed one "
                "or more production safety gates. Review the reasons and confirm "
                "the missing specifications or historical comparables."
            )

        return {
            "pricing_version": PRICING_ENGINE_VERSION,
            "status": decision["status"],
            "input_description": input_description,
            "input_attributes": search["input_attributes"],
            "statistical_anchor": decision.get("indicative_price"),
            "anchor_method": decision["pricing_method"],
            "pricing_matches_used": sorted(pricing_ranks),
            "predicted_unit_price": decision["predicted_unit_price"],
            "indicative_price": decision.get("indicative_price"),
            "price_interval": decision.get("price_interval"),
            "currency": config.DEFAULT_CURRENCY,
            "unit": search["input_attributes"].get("unit_hint") or "unknown",
            "confidence": decision["confidence"],
            "reasoning": reasoning,
            "price_basis": decision["pricing_method"],
            "adjustments": [],
            "context_adjustments": {
                "mode": self.context_readiness.get("mode", "shadow"),
                "applied": [],
                "factor": 1.0,
            },
            "fallback_used": False,
            "price_source": (
                "Production comparable engine" if decision["status"] == "priced"
                else "Historical comparable evidence (manual review)"
            ),
            "review_reasons": decision["review_reasons"],
            "matches": matches,
            "weak_matches": search["weak_matches"],
            "warnings": warnings,
            "data_quality": self.data_quality,
        }


@lru_cache(maxsize=1)
def get_engine(data_path: str | None = None) -> PricingEngine:
    """Cached engine so the dataset is indexed once per process."""
    return PricingEngine(data_path)
