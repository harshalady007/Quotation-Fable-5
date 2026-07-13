"""Load the quotation Excel file and map its columns to standard fields.

The loader does not assume exact column names: it scores every sheet's
columns against known synonyms and picks the sheet that maps best.
Internal fields:
    description, item_name, unit, quantity, rate, amount,
    category, section, location, supplier, remarks, source, date,
    quotation_number, project, client, contractor, currency
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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
    "supplier": ["supplier", "vendor", "manufacturer"],
    "remarks": ["remarks", "notes", "comment"],
    "source": ["source file", "source", "file name", "page"],
    "date": ["quotation date", "date"],
}

# Quote-level context often lives on a separate summary sheet.  These fields
# are joined by source filename for audit/readiness purposes; they are never
# placed in the product search text or allowed to adjust a price implicitly.
SUMMARY_COLUMN_SYNONYMS = {
    "source": ["source file", "source", "file name"],
    "quotation_number": ["quotation number", "quote number", "quotation no"],
    "date": ["quotation date", "date"],
    "project": ["project", "project name"],
    "client": ["client", "customer"],
    "contractor": ["contractor", "main contractor"],
    "supplier": ["supplier", "vendor", "manufacturer"],
    "currency": ["currency"],
}


def _map_columns_for(columns, synonyms_by_field: dict) -> dict:
    """Return {internal_field: actual_column_name} for one sheet."""
    mapping = {}
    taken = set()
    for field, synonyms in synonyms_by_field.items():
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


def _map_columns(columns) -> dict:
    return _map_columns_for(columns, COLUMN_SYNONYMS)


def _source_key(value) -> str:
    """Normalize a source filename for a safe product-to-summary join."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\s+", " ", text).strip().lower()


def _nonempty_values(series: pd.Series) -> list:
    values = []
    seen = set()
    for value in series:
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        if text not in seen:
            seen.add(text)
            values.append(value)
    return values


def _load_summary_context(
    xl: pd.ExcelFile, product_sheet: str
) -> tuple[pd.DataFrame, dict]:
    """Load non-conflicting quote-level fields from the best summary sheet."""
    best = None
    for sheet in xl.sheet_names:
        if sheet == product_sheet:
            continue
        try:
            frame = xl.parse(sheet)
        except Exception as exc:
            logger.warning("Skipping unreadable context sheet %r: %s", sheet, exc)
            continue
        if frame.empty:
            continue
        mapping = _map_columns_for(frame.columns, SUMMARY_COLUMN_SYNONYMS)
        if "source" not in mapping or len(mapping) < 2:
            continue
        score = len(mapping)
        if best is None or score > best[0]:
            best = (score, sheet, frame, mapping)

    if best is None:
        return pd.DataFrame(), {"sheet": None, "mapping": {}, "conflicts": {}}

    _, sheet, frame, mapping = best
    normalized = pd.DataFrame(index=frame.index)
    normalized["_source_key"] = frame[mapping["source"]].map(_source_key)
    for field in SUMMARY_COLUMN_SYNONYMS:
        if field == "source":
            continue
        normalized[field] = frame[mapping[field]] if field in mapping else None
    normalized = normalized[normalized["_source_key"].ne("")]

    rows = []
    conflicts = {field: 0 for field in SUMMARY_COLUMN_SYNONYMS if field != "source"}
    for source_key, group in normalized.groupby("_source_key", sort=False):
        item = {"_source_key": source_key}
        for field in conflicts:
            values = _nonempty_values(group[field])
            if len(values) > 1:
                conflicts[field] += 1
                item[field] = None
            else:
                item[field] = values[0] if values else None
        rows.append(item)
    conflicts = {field: count for field, count in conflicts.items() if count}
    return pd.DataFrame(rows), {
        "sheet": sheet,
        "mapping": {key: str(value) for key, value in mapping.items()},
        "source_rows": int(len(normalized)),
        "source_keys": int(normalized["_source_key"].nunique()),
        "conflicts": conflicts,
    }


def _missing_text(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(
        {"", "nan", "none", "null"}
    )


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

    summary, summary_metadata = _load_summary_context(xl, sheet)
    summary_fields = (
        "quotation_number", "project", "client", "contractor", "supplier",
        "currency",
    )
    for field in summary_fields:
        if field not in out:
            out[field] = None
    if not summary.empty:
        out["_source_key"] = out["source"].map(_source_key)
        summary_keys = set(summary["_source_key"])
        summary_metadata["matched_product_rows"] = int(
            out["_source_key"].isin(summary_keys).sum()
        )
        summary_metadata["unmatched_product_rows"] = int(
            (~out["_source_key"].isin(summary_keys)).sum()
        )
        summary = summary.rename(columns={
            field: f"_summary_{field}"
            for field in summary.columns if field != "_source_key"
        })
        out = out.merge(summary, on="_source_key", how="left", sort=False)
        if "_summary_date" in out:
            out["date"] = out["date"].astype(object)
            out.loc[_missing_text(out["date"]), "date"] = out["_summary_date"]
        for field in summary_fields:
            summary_field = f"_summary_{field}"
            if summary_field in out:
                out[field] = out[field].astype(object)
                out.loc[_missing_text(out[field]), field] = out[summary_field]
        out = out.drop(
            columns=[column for column in out if column.startswith("_summary_")]
            + ["_source_key"]
        )
    else:
        summary_metadata["matched_product_rows"] = 0
        summary_metadata["unmatched_product_rows"] = int(len(out))

    # Numeric coercion at the boundary; invalid values become NaN.
    for num_field in ("rate", "amount", "quantity"):
        out[num_field] = pd.to_numeric(out[num_field], errors="coerce")

    # Derive rate = amount / quantity where rate is missing and quantity valid.
    need_rate = out["rate"].isna() & out["amount"].notna() & (out["quantity"] > 0)
    out.loc[need_rate, "rate"] = out.loc[need_rate, "amount"] / out.loc[need_rate, "quantity"]

    out.attrs["sheet"] = sheet
    out.attrs["column_mapping"] = {k: str(v) for k, v in mapping.items()}
    out.attrs["summary_context"] = summary_metadata
    return out
