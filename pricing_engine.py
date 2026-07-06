"""Ties loading, cleaning, search and DeepSeek pricing into one API."""

import logging
from functools import lru_cache

import config
from cleaner import clean_dataset
from data_loader import load_dataset
from deepseek_pricing import predict_price_with_deepseek
from similarity_search import SimilaritySearcher

logger = logging.getLogger(__name__)


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

        search = self.searcher.search(input_description, top_k=top_k)
        matches = search["matches"]

        warnings = []
        if search["weak_matches"]:
            warnings.append(
                f"Best match similarity is only {search['best_score']:.2f}; "
                "no strong historical match was found. Treat the prediction "
                "with caution."
            )

        prediction = predict_price_with_deepseek(
            input_description, search["input_attributes"], matches,
            search["weak_matches"],
        )

        return {
            "input_description": input_description,
            "input_attributes": search["input_attributes"],
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
