"""Build an estimator review queue without guessing historical values."""

from __future__ import annotations

from data_quality import BLOCKING_FLAGS
from family_readiness import missing_target_fields
from pricing_dataset import load_pricing_dataset
from production_pricing import FAMILY_CONTRACTS, SUPPORTED_INPUT_FAMILIES
from similarity_search import SimilaritySearcher


def build_correction_queue(data_path: str | None = None,
                           corrections_path: str | None = None,
                           family: str | None = None,
                           blocking_only: bool = False) -> list[dict]:
    """Return rows requiring human confirmation, ordered by production value."""
    dataset = load_pricing_dataset(data_path, corrections_path)
    searcher = SimilaritySearcher(dataset)
    queue = []
    family_priority = {
        "bench": 0,
        "litter bin": 1,
        "bike rack": 2,
        "bollard": 3,
        "recycle bin": 4,
        "planter": 5,
    }
    for i, row in dataset.iterrows():
        attrs = searcher.item_attrs[i]
        item_family = attrs.get("item_type")
        if item_family not in SUPPORTED_INPUT_FAMILIES:
            continue
        if family and item_family != family:
            continue
        quality_issues = list(row.get("data_quality_flags") or [])
        # Superseded PDFs are intentionally quarantined by source lineage;
        # they do not need estimator correction and would otherwise bury the
        # genuinely actionable records in the review queue.
        if "superseded_revision" in quality_issues:
            continue
        required = missing_target_fields(row, attrs)
        recommended = [
            field for field in FAMILY_CONTRACTS[item_family]["recommended_fields"]
            if attrs.get(field) in (None, "")
        ]
        blocking_quality = [issue for issue in quality_issues
                            if issue in BLOCKING_FLAGS]
        blocking = bool(blocking_quality or required)
        if blocking_only and not blocking:
            continue
        if not blocking and not recommended:
            continue
        queue.append({
            "record_id": row.get("record_id"),
            "family": item_family,
            "subtype": attrs.get("subtype"),
            "source": row.get("source"),
            "date": row.get("date"),
            "description": row.get("full_description"),
            "unit": row.get("unit_norm") or None,
            "rate": float(row.get("rate")),
            "quality_issues": quality_issues,
            "missing_required_fields": required,
            "missing_recommended_fields": recommended,
            "blocking": blocking,
            "correction_ids": list(row.get("correction_ids") or []),
            "_priority": (
                0 if quality_issues else 1,
                family_priority.get(item_family, 99),
                str(row.get("source") or ""),
                str(row.get("record_id") or ""),
            ),
        })
    queue.sort(key=lambda item: item["_priority"])
    for item in queue:
        item.pop("_priority", None)
    return queue
