"""Predict a unit price with the DeepSeek API, with a statistical fallback.

The API key is read from the DEEPSEEK_API_KEY environment variable and is
never logged or returned in any response.
"""

import json
import logging
import re

import requests

import config
from attribute_extractor import INSTALL_SCOPES, SUPPLY_SCOPES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior quantity surveyor / quotation estimator. You are given "
    "a new item description with extracted attributes, and the most similar "
    "historical quotation items with their rates. Compare them like a real "
    "estimator: adjust the price up for better material, larger size, thicker "
    "section, better finish, higher grade, or installation included; adjust "
    "down for simpler material, smaller size, lower grade, supply-only scope, "
    "or simpler finish. Anchor on the strongest matches, not a blind average. "
    "Be deterministic: the same input must always produce the same price. "
    "Start from the given statistical anchor and apply explicit, quantified "
    "adjustments only for attribute differences that are actually stated. "
    "You may go outside the historical rate range only when the matches are "
    "sparse or clearly dissimilar products - state the multiplier you "
    "applied. "
    "COMPONENT RULE: when only a component differs (e.g. granite legs, an "
    "SS316 frame, different slats), adjust ONLY that component's share of "
    "cost (typically 10-30% of the item); NEVER apply whole-item material "
    "multipliers for component-level differences, and never stack several "
    "component multipliers into a large combined factor. "
    "Typical cost relativities to apply: stainless steel fabrication is "
    "roughly 2.5-3x mild or galvanized steel; SS316 is ~1.15x SS304; "
    "COMPANY RULE: supply and installation is exactly 20% more expensive "
    "than supply only or supply and delivery - scope-adjusted rates given "
    "to you already include this conversion, so do NOT apply it again; "
    "linear items "
    "scale roughly with length/height and structural complexity; area items "
    "scale with buildup thickness and finish quality; a much larger overall "
    "size means proportionally more material. When the input and the matches "
    "state different dimensions, scale the price for the size difference "
    "(roughly proportional to material content) - two items differing only "
    "in length must NOT get the same price, even inside an established band. "
    "Never compare per-metre or per-m2 rates directly with per-item rates: "
    "if the unit bases differ, say so and lower confidence to Low. "
    "Respond ONLY with a JSON object with exactly these keys: "
    "predicted_unit_price (number), currency (string), unit (string), "
    "confidence (one of High/Medium/Low), reasoning (string), "
    "price_basis (string), adjustments (array of strings), "
    "warnings (array of strings)."
)


def scope_adjusted_rate(rate, item_scope, input_scope):
    """Convert a historical rate to the input's scope basis.

    Company rule: supply and installation is SCOPE_INSTALL_UPLIFT (20%)
    more expensive than supply only / supply and delivery.
    """
    if not rate or not input_scope or not item_scope:
        return rate
    if input_scope in INSTALL_SCOPES and item_scope in SUPPLY_SCOPES:
        return round(rate * config.SCOPE_INSTALL_UPLIFT, 2)
    if input_scope in SUPPLY_SCOPES and item_scope in INSTALL_SCOPES:
        return round(rate / config.SCOPE_INSTALL_UPLIFT, 2)
    return rate


def _effective_rate(m):
    return m.get("scope_adjusted_rate") or m.get("rate")


def dense_band(matches) -> tuple | None:
    """(min, max) of scope-adjusted rates when >=3 same-type comps sit in a
    tight band (max <= 2.2x min) — the product class price is established
    and predictions should stay inside it. None otherwise."""
    rates = [_effective_rate(m) for m in matches if _effective_rate(m)]
    types = {m.get("item_type") for m in matches}
    if (len(rates) >= 3 and len(types) == 1 and None not in types
            and max(rates) <= 2.2 * min(rates)):
        return (min(rates), max(rates))
    return None


def size_interpolated_anchor(matches, input_attrs) -> float | None:
    """Interpolate the rate along the size ladder of comparable items.

    An estimator prices a 500mm bin BETWEEN the 385mm comp and the 1015mm
    comp — never by anchoring on the biggest match and scaling up. Uses
    same-material comps when the input's material is known and at least
    two of them state sizes; falls back to all comps, then to None.
    """
    in_proxy = input_attrs.get("size_proxy")
    in_kind = input_attrs.get("size_proxy_kind")
    if not in_proxy:
        return None

    def points(require_material: bool):
        material = input_attrs.get("material")
        pts = []
        for m in matches:
            rate = _effective_rate(m)
            if not rate or not m.get("size_proxy"):
                continue
            if m.get("size_proxy_kind") != in_kind:
                continue
            if require_material and material and m.get("material") != material:
                continue
            pts.append((float(m["size_proxy"]), float(rate)))
        return sorted(pts)

    pts = points(require_material=True)
    if len({x for x, _ in pts}) < 2:
        pts = points(require_material=False)
    if len({x for x, _ in pts}) < 2:
        return None

    lower = [p for p in pts if p[0] <= in_proxy]
    upper = [p for p in pts if p[0] >= in_proxy]
    if lower and upper:
        x1, y1 = lower[-1]
        x2, y2 = upper[0]
        if x2 == x1:
            return round((y1 + y2) / 2, 2)
        return round(y1 + (in_proxy - x1) / (x2 - x1) * (y2 - y1), 2)
    # Outside the ladder: scale the nearest edge comp sublinearly, capped.
    x, y = pts[-1] if lower else pts[0]
    factor = min(max((in_proxy / x) ** 0.8, 0.5), 1.6)
    return round(y * factor, 2)


def compute_anchor(matches, input_attrs) -> tuple:
    """(anchor, method) — size interpolation when possible, else the
    similarity-weighted median."""
    anchor = size_interpolated_anchor(matches, input_attrs or {})
    if anchor is not None:
        return anchor, "size-interpolated across comparable items"
    return weighted_median_rate(matches), "similarity-weighted median"


def weighted_median_rate(matches) -> float | None:
    """Similarity-weighted median of the matches' scope-adjusted rates
    (deterministic)."""
    priced = sorted((m for m in matches if _effective_rate(m)),
                    key=_effective_rate)
    if not priced:
        return None
    total = sum(max(m["similarity_score"], 0.01) for m in priced)
    cum = 0.0
    for m in priced:
        cum += max(m["similarity_score"], 0.01)
        if cum >= total / 2:
            return float(_effective_rate(m))
    return float(_effective_rate(priced[-1]))


def _build_user_prompt(description, input_attrs, matches, weak_matches) -> str:
    lines = [
        "NEW ITEM TO PRICE:",
        description.strip(),
        "",
        "EXTRACTED INPUT ATTRIBUTES:",
        json.dumps({k: v for k, v in input_attrs.items() if v not in (None, [])},
                   ensure_ascii=False),
        "",
        f"INPUT UNIT BASIS: {input_attrs.get('unit_hint') or 'not stated'}",
        f"HISTORICAL CURRENCY: {config.DEFAULT_CURRENCY}",
        "",
        "TOP SIMILAR HISTORICAL ITEMS:",
    ]
    for m in matches:
        adj = m.get("scope_adjusted_rate")
        rate_line = (f"Rate: {m['rate']} (scope-adjusted to {adj} on the "
                     f"input's scope basis per the 20% installation rule)"
                     if adj and adj != m["rate"] else f"Rate: {m['rate']}")
        lines += [
            f"--- Match {m['rank']} (similarity {m['similarity_score']:.2f}, "
            f"text {m['text_similarity']:.2f}, attributes {m['attribute_score']:.2f}) ---",
            f"Description: {m['description']}",
            f"Unit: {m['unit'] or 'unknown'} | Quantity: {m['quantity'] or 'n/a'} | "
            f"{rate_line} | Amount: {m['amount'] or 'n/a'}",
            f"Scope of work: {m.get('scope') or 'unknown'}",
            f"Category/Scope: {m['category'] or 'unknown'}",
            f"Matched attributes: {', '.join(m['matched_attributes']) or 'none'}",
            f"Mismatched attributes: {', '.join(m['mismatched_attributes']) or 'none'}",
            f"Attributes not stated: {', '.join(m['missing_attributes']) or 'none'}",
        ]
        if m["differences"]:
            lines.append("Price-relevant differences: " + "; ".join(m["differences"]))
    rates = [_effective_rate(m) for m in matches if _effective_rate(m)]
    anchor, anchor_method = compute_anchor(matches, input_attrs)
    if rates and anchor is not None:
        sized = sorted((m for m in matches
                        if m.get("size_proxy") and _effective_rate(m)),
                       key=lambda m: m["size_proxy"])
        if len(sized) >= 2:
            table = "; ".join(
                f"{m['size_proxy']:g}mm -> {_effective_rate(m):,.2f}"
                for m in sized)
            lines += ["", f"SIZE/RATE LADDER of these matches (scope-adjusted): {table}",
                      f"The input item's size on this ladder: "
                      f"{input_attrs.get('size_proxy'):g}mm"
                      if input_attrs.get("size_proxy") else ""]
        lines += [
            "",
            f"STATISTICAL ANCHOR ({anchor_method}): "
            f"{anchor:,.2f} {config.DEFAULT_CURRENCY} - start here. Do NOT "
            "anchor on the largest or most expensive match; interpolate "
            "along the size ladder.",
            f"HISTORICAL RATE RANGE of these matches: {min(rates):,.2f} to "
            f"{max(rates):,.2f} {config.DEFAULT_CURRENCY}",
            "Start from the anchor and apply quantified adjustments for the "
            "stated attribute differences (material, grade, size, scope, "
            "finish); going outside the historical range is allowed when "
            "the differences clearly justify it.",
        ]
        band = dense_band(matches)
        if band:
            lines += [
                f"PRICE BAND ESTABLISHED: {len(matches)} historical items of "
                f"the same type price this product class between "
                f"{band[0]:,.2f} and {band[1]:,.2f} {config.DEFAULT_CURRENCY} "
                "(scope-adjusted). Your prediction MUST stay within this "
                "band: component-level differences (legs, frame, slats, "
                "finish) move the price WITHIN the band, not outside it.",
            ]
    if weak_matches:
        lines += ["", "WARNING: All matches are weak. State this in warnings "
                      "and lower your confidence accordingly."]
    lines += ["", "Decide whether the new item should be priced similar, higher, "
                  "or lower than these historical items, and return the JSON."]
    return "\n".join(lines)


def _parse_json_response(content: str) -> dict:
    """Parse DeepSeek's reply; tolerate markdown fences and stray text."""
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence:
        content = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", content, re.DOTALL)
        if brace:
            content = brace.group(0)
    data = json.loads(content)
    price = float(data["predicted_unit_price"])
    if price < 0:
        raise ValueError("negative predicted price")
    return {
        "predicted_unit_price": round(price, 2),
        "currency": str(data.get("currency") or config.DEFAULT_CURRENCY),
        "unit": str(data.get("unit") or "unknown"),
        "confidence": str(data.get("confidence") or "Low"),
        "reasoning": str(data.get("reasoning") or ""),
        "price_basis": str(data.get("price_basis") or ""),
        "adjustments": [str(a) for a in data.get("adjustments") or []],
        "warnings": [str(w) for w in data.get("warnings") or []],
    }


def fallback_prediction(matches, weak_matches: bool, reason: str,
                        input_attrs: dict | None = None) -> dict:
    """Deterministic estimate when DeepSeek is unavailable: size-interpolated
    anchor when sizes are stated, else similarity-weighted median."""
    priced = [m for m in matches if _effective_rate(m)]
    if not priced:
        return {
            "predicted_unit_price": None,
            "currency": config.DEFAULT_CURRENCY,
            "unit": "unknown",
            "confidence": "Low",
            "reasoning": "No historical matches with a valid rate were found.",
            "price_basis": "none",
            "adjustments": [],
            "warnings": [reason, "No priced matches available."],
            "fallback_used": True,
        }
    rates = [_effective_rate(m) for m in priced]
    # Deterministic and identical no matter how many matches are displayed.
    anchor, anchor_method = compute_anchor(priced, input_attrs or {})
    estimate = round(anchor, 2)
    units = [m["unit"] for m in priced if m["unit"]]
    return {
        "predicted_unit_price": estimate,
        "currency": config.DEFAULT_CURRENCY,
        "unit": units[0] if units else "unknown",
        "confidence": "Low" if weak_matches else "Medium",
        "reasoning": (
            f"Statistical fallback ({anchor_method}) over the {len(priced)} "
            f"strongest historical rates, scope-adjusted to the input's "
            f"work scope ({min(rates):,.2f} to {max(rates):,.2f}). {reason}"
        ),
        "price_basis": f"Top {len(priced)} historical matches (statistical, no LLM).",
        "adjustments": [],
        "warnings": [reason] + (["Match quality is weak; treat this estimate "
                                 "with caution."] if weak_matches else []),
        "fallback_used": True,
    }


def predict_price_with_deepseek(description: str, input_attrs: dict,
                                matches: list, weak_matches: bool) -> dict:
    """Call DeepSeek; on any failure return the statistical fallback."""
    api_key = config.DEEPSEEK_API_KEY
    if not api_key:
        return fallback_prediction(
            matches, weak_matches,
            "DEEPSEEK_API_KEY is not set. Set it in your environment to enable "
            "AI price estimation (see README).",
            input_attrs,
        )

    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(
                description, input_attrs, matches, weak_matches)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            f"{config.DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=config.DEEPSEEK_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as exc:
        logger.warning("DeepSeek API request failed: %s", type(exc).__name__)
        return fallback_prediction(matches, weak_matches,
                                   f"DeepSeek API request failed ({type(exc).__name__}).",
                                   input_attrs)
    except (KeyError, IndexError, ValueError):
        return fallback_prediction(matches, weak_matches,
                                   "DeepSeek API returned an unexpected response shape.",
                                   input_attrs)

    try:
        result = _parse_json_response(content)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return fallback_prediction(matches, weak_matches,
                                   "DeepSeek returned invalid JSON.",
                                   input_attrs)
    result["fallback_used"] = False

    # Clamp the LLM's price to the pricing set's historical range so a
    # drifting response can never produce an implausible number.
    rates = [_effective_rate(m) for m in matches if _effective_rate(m)]
    if rates and result["predicted_unit_price"] is not None:
        band = dense_band(matches)
        if band:
            # Established product-class band: hold the prediction close.
            lo = round(band[0] * 0.75, 2)
            hi = round(band[1] * 1.25, 2)
        else:
            lo = round(min(rates) * config.PRICE_CLAMP_LOW, 2)
            hi = round(max(rates) * config.PRICE_CLAMP_HIGH, 2)
        p = result["predicted_unit_price"]
        if p < lo or p > hi:
            result["predicted_unit_price"] = min(max(p, lo), hi)
            result["warnings"].append(
                f"Model suggested {p:,.2f}, outside the historical evidence "
                f"range; clamped to {result['predicted_unit_price']:,.2f} "
                f"(allowed {lo:,.2f} to {hi:,.2f})."
            )

    if weak_matches and "Low" not in result["confidence"]:
        result["warnings"].append("Similarity matches were weak; verify manually.")
    return result
