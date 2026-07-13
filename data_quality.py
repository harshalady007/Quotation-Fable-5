"""Data-quality gates for historical quotation rows.

Rows are never silently deleted here.  They remain searchable and visible to
an estimator, but records with blocking defects are excluded from automatic
price construction.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


CONFLICT_RATE_RATIO = 2.0

# These defects make a rate unsafe as an automatic comparable.  Other flags
# may be surfaced as warnings while still allowing the record to be used.
BLOCKING_FLAGS = {
    "missing_unit",
    "conflicting_duplicate_rate",
}


def annotate_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags and a boolean ``pricing_eligible`` column.

    An exact normalized description/unit pair with a >=2x price spread is
    treated as contradictory evidence.  Both records are quarantined because
    choosing either one automatically would merely hide the source problem.
    """
    out = df.copy()
    flags: dict[int, list[str]] = {int(i): [] for i in out.index}

    for i, row in out.iterrows():
        if not str(row.get("unit_norm") or "").strip():
            flags[int(i)].append("missing_unit")
        if not str(row.get("source") or "").strip():
            flags[int(i)].append("missing_source")
        if not str(row.get("date") or "").strip():
            flags[int(i)].append("missing_date")

    grouped = out.groupby(["clean_text", "unit_norm"], dropna=False)
    for _, group in grouped:
        rates = pd.to_numeric(group["rate"], errors="coerce").dropna()
        if len(rates) < 2 or rates.min() <= 0:
            continue
        if rates.max() / rates.min() >= CONFLICT_RATE_RATIO:
            for i in group.index:
                flags[int(i)].append("conflicting_duplicate_rate")

    out["data_quality_flags"] = [sorted(set(flags[int(i)])) for i in out.index]
    out["pricing_eligible"] = out["data_quality_flags"].map(
        lambda values: not any(flag in BLOCKING_FLAGS for flag in values)
    )
    return out


def quality_summary(df: pd.DataFrame) -> dict:
    """Return compact health metrics for API health checks and diagnostics."""
    counts: Counter[str] = Counter()
    if "data_quality_flags" in df:
        for values in df["data_quality_flags"]:
            counts.update(values or [])
    eligible = int(df.get("pricing_eligible", pd.Series(False, index=df.index)).sum())
    return {
        "rows": int(len(df)),
        "pricing_eligible_rows": eligible,
        "quarantined_rows": int(len(df) - eligible),
        "flag_counts": dict(sorted(counts.items())),
    }
