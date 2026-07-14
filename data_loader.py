"""Load the quotation Excel file and map its columns to standard fields.

The loader does not assume exact column names: it scores every sheet's
columns against known synonyms and picks the sheet that maps best.
Internal fields:
    description, item_name, unit, quantity, rate, amount,
    category, section, location, remarks, source, date
"""

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def image_key(source, item, desc) -> str:
    """Stable key linking a product row to its extracted image file
    (data/images/<key>.png), derived from the row's identifying text."""
    joined = "|".join(
        re.sub(r"\s+", " ", str(x if x is not None else "").strip().lower())
        for x in (source, item, desc)
    )
    return hashlib.sha1(joined.encode()).hexdigest()[:12]


class DataLoadError(Exception):
    """Raised when the Excel file cannot be loaded into a usable dataset."""


# Internal field -> lowercase column-name fragments that indicate it.
# Order matters: earlier fields claim columns first.
COLUMN_SYNONYMS = {
    "description": ["description", "item description", "service description",
                    "details", "particulars", "work description"],
    "item_name": ["item name", "item", "product name", "product", "service name"],
    "rate": ["unit price", "unit rate", "rate", "price per", "unit cost", "price"],
    "amount": ["amount", "total price", "total value", "line total", "total"],
    "quantity": ["quantity", "qty", "no of units"],
    "unit": ["unit", "uom", "unit of measure"],
    "category": ["category", "trade", "package", "discipline", "scope of work", "scope"],
    "section": ["section", "bill", "group", "division"],
    "location": ["location", "area", "zone", "site"],
    "remarks": ["remarks", "notes", "comment"],
    "source": ["source file", "source", "file name", "page"],
    "date": ["quotation date", "date"],
}


def _map_columns(columns) -> dict:
    """Return {internal_field: actual_column_name} for one sheet."""
    mapping = {}
    taken = set()
    for field, synonyms in COLUMN_SYNONYMS.items():
        for syn in synonyms:
            for col in columns:
                if col in taken:
                    continue
                name = str(col).strip().lower()
                if name == syn or syn in name:
                    mapping[field] = col
                    taken.add(col)
                    break
            if field in mapping:
                break
    return mapping


def _sheet_score(mapping: dict) -> int:
    """Score a sheet's usefulness: it needs a description and some price info."""
    score = 0
    if "description" in mapping or "item_name" in mapping:
        score += 2
    if "rate" in mapping:
        score += 2
    elif "amount" in mapping and "quantity" in mapping:
        score += 1
    score += len(mapping)
    return score


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Excel file and return a DataFrame with standard columns.

    Raises DataLoadError with a user-friendly message on any failure.
    """
    p = Path(path)
    if not p.exists():
        raise DataLoadError(
            f"Excel file not found: {path}. "
            "Set QUOTATION_DATA_PATH to the correct location."
        )
    try:
        xl = pd.ExcelFile(p)
    except Exception as exc:
        raise DataLoadError(f"Could not read Excel file {path}: {exc}") from exc

    if not xl.sheet_names:
        raise DataLoadError(f"No sheets found in {path}.")

    best = None  # (score, sheet_name, df, mapping)
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception as exc:
            logger.warning("Skipping unreadable sheet %r: %s", sheet, exc)
            continue
        if df.empty:
            continue
        mapping = _map_columns(df.columns)
        score = _sheet_score(mapping)
        if best is None or score > best[0]:
            best = (score, sheet, df, mapping)

    if best is None:
        raise DataLoadError(f"No readable, non-empty sheets in {path}.")

    score, sheet, df, mapping = best
    if "description" not in mapping and "item_name" not in mapping:
        raise DataLoadError(
            f"Could not find a description-like column in {path}. "
            f"Columns seen: {list(df.columns)}"
        )
    if "rate" not in mapping and not ("amount" in mapping and "quantity" in mapping):
        raise DataLoadError(
            f"Could not find price information (rate, or amount+quantity) in {path}. "
            f"Columns seen: {list(df.columns)}"
        )

    logger.info("Using sheet %r with column mapping: %s", sheet, mapping)

    out = pd.DataFrame(index=df.index)
    for field in COLUMN_SYNONYMS:
        out[field] = df[mapping[field]] if field in mapping else None

    # Numeric coercion at the boundary; invalid values become NaN.
    for num_field in ("rate", "amount", "quantity"):
        out[num_field] = pd.to_numeric(out[num_field], errors="coerce")

    # Derive rate = amount / quantity where rate is missing and quantity valid.
    need_rate = out["rate"].isna() & out["amount"].notna() & (out["quantity"] > 0)
    out.loc[need_rate, "rate"] = out.loc[need_rate, "amount"] / out.loc[need_rate, "quantity"]

    out["image_key"] = [
        image_key(s, i, d)
        for s, i, d in zip(out["source"], out["item_name"], out["description"])
    ]

    out.attrs["sheet"] = sheet
    out.attrs["column_mapping"] = {k: str(v) for k, v in mapping.items()}
    return out
