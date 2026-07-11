"""Clean the loaded dataset and normalize description text."""

import re

import pandas as pd


class EmptyDatasetError(Exception):
    """Raised when cleaning leaves no usable rows."""


# Unit spelling variants -> canonical unit.
UNIT_MAP = {
    "nos": "no", "no": "no", "no.": "no", "nr": "no", "each": "no", "ea": "no",
    "item": "no", "pcs": "no", "pc": "no", "unit": "no",
    "set": "set", "sets": "set", "pair": "set",
    "m": "m", "mtr": "m", "meter": "m", "metre": "m",
    "lm": "m", "rm": "m", "rmt": "m", "lin.m": "m",
    "running metre": "m", "linear metre": "m", "running meter": "m",
    "m2": "m2", "sqm": "m2", "sq.m": "m2", "sq m": "m2", "m²": "m2",
    "m3": "m3", "cum": "m3", "cu.m": "m3", "m³": "m3",
    "kg": "kg", "kgs": "kg", "ton": "ton", "tonne": "ton", "mt": "ton",
    "ls": "ls", "lumpsum": "ls", "lump sum": "ls", "lot": "ls",
    "day": "day", "days": "day", "hour": "hr", "hr": "hr", "hrs": "hr",
}

# Term standardization applied to lowercase text (regex -> replacement).
# Preserves dimensions, grades, materials and finishes.
_NORMALIZE_PATTERNS = [
    (re.compile(r"\bstainless[\s\-]*steel\b"), "stainless steel"),
    (re.compile(r"\bss[\s\-]*(304|316l?|201|430)\b"), r"ss\1"),
    (re.compile(r"\b(304|316l?|201|430)[\s\-]*ss\b"), r"ss\1"),
    (re.compile(r"\bmild[\s\-]*steel\b|\bms\b(?=\s*(?:tube|pipe|plate|sheet|post|frame|steel)?)"), "mild steel"),
    (re.compile(r"\bgalvani[sz]ed\b|\bgi\b"), "galvanized"),
    (re.compile(r"\balu(?:minium|minum)\b"), "aluminium"),
    (re.compile(r"\bpowder[\s\-]*coat(?:ed|ing)?\b"), "powder coated"),
    (re.compile(r"\bdia(?:meter)?\.?\b"), "dia"),
    (re.compile(r"\bthk\.?\b|\bthick(?:ness)?\b"), "thick"),
    (re.compile(r"(\d)\s*(?:mm|millimeter[s]?)\b"), r"\1mm"),
    (re.compile(r"(\d)\s*(?:cm)\b"), r"\1cm"),
    (re.compile(r"(\d(?:\.\d+)?)\s*(?:m|meter[s]?|metre[s]?)\b"), r"\1m"),
    (re.compile(r"\bsq\.?\s*m\b|\bm²\b|\bsqm\b"), "m2"),
    (re.compile(r"\bcu\.?\s*m\b|\bm³\b|\bcum\b"), "m3"),
    # "900 (W) x 380 (D) x 850 (H) mm" -> "900 x 380 x 850 mm"
    (re.compile(r"\(\s*[lwhd]\s*\)"), " "),
    (re.compile(r"\bsupply\s*(?:&|and|\+)\s*install(?:ation)?\b"), "supply and install"),
    (re.compile(r"\bsupply\s*(?:&|and|\+)\s*deliver(?:y)?\b"), "supply and delivery"),
    (re.compile(r"\bsupply\s+only\b"), "supply only"),
    (re.compile(r"\bexcluding\s+civil\s*wor?ks?\b|\bwithout\s+civil\s*wor?ks?\b"),
     "without civil works"),
]

# Characters that carry no pricing meaning. Keep digits, letters, units,
# x (dimensions), . , / - * ( ) % " and degree symbols.
_JUNK_CHARS = re.compile(r"[^\w\s\.\,\/\-\*\(\)%x\"°²³&+:]")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_unit(unit) -> str:
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return ""
    key = str(unit).strip().lower()
    return UNIT_MAP.get(key, key)


def normalize_text(text) -> str:
    """Lowercase, de-noise and standardize a description for searching.

    Keeps dimensions (50mm, 1200x600mm), grades (ss316, m20), materials,
    finishes and scope wording.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).replace("\r", "\n")
    s = s.replace("\n", " ")
    s = s.lower()
    s = _JUNK_CHARS.sub(" ", s)
    for pattern, repl in _NORMALIZE_PATTERNS:
        s = pattern.sub(repl, s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and add clean_text / search_text / unit_norm columns."""
    df = df.copy()

    # Combined raw description (item name + description when both exist).
    def combine(row):
        parts = []
        for f in ("item_name", "description"):
            v = row.get(f)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                v = str(v).strip()
                if v and v.lower() not in ("nan", "none"):
                    # Avoid "Bollard\nBollard ..." duplication.
                    if not parts or v.lower() not in parts[0].lower():
                        parts.append(v)
        return "\n".join(parts)

    df["full_description"] = df.apply(combine, axis=1)
    df["clean_text"] = df["full_description"].map(normalize_text)

    # Rows must have a usable description and a valid positive rate.
    df = df[df["clean_text"].str.len() >= 3]
    df = df[pd.to_numeric(df["rate"], errors="coerce").notna()]
    df = df[df["rate"] > 0]

    df["unit_norm"] = df["unit"].map(normalize_unit)

    # Searchable text combines every useful text field.
    extra = []
    for field in ("category", "section", "location", "remarks"):
        extra.append(df[field].map(normalize_text))
    df["search_text"] = df["clean_text"]
    for col in extra:
        df["search_text"] = (df["search_text"] + " " + col.fillna("")).str.strip()

    # Duplicates: same cleaned text, unit and rate -> keep first.
    df = df.drop_duplicates(subset=["clean_text", "unit_norm", "rate"], keep="first")
    df = df.reset_index(drop=True)

    if df.empty:
        raise EmptyDatasetError(
            "No usable rows left after cleaning: every row lacked a "
            "description or a valid positive rate."
        )
    return df
