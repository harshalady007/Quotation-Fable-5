"""Single construction path for the production pricing dataset."""

from __future__ import annotations

import config
from cleaner import clean_dataset
from data_corrections import apply_data_corrections
from data_loader import load_dataset
from data_quality import annotate_data_quality


def load_pricing_dataset(data_path: str | None = None,
                         corrections_path: str | None = None):
    """Load, clean, correct and quality-annotate quotation records."""
    raw = load_dataset(data_path or config.DATA_PATH)
    metadata = {
        "column_mapping": raw.attrs.get("column_mapping", {}),
        "sheet": raw.attrs.get("sheet", ""),
        "summary_context": raw.attrs.get("summary_context", {}),
    }
    dataset = clean_dataset(raw)
    dataset = apply_data_corrections(
        dataset,
        config.CORRECTIONS_PATH if corrections_path is None else corrections_path,
    )
    corrections = dict(dataset.attrs.get("corrections") or {})
    dataset = annotate_data_quality(dataset)
    dataset.attrs.update(metadata)
    dataset.attrs["corrections"] = corrections
    return dataset
