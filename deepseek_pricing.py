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
    "You may go outside the historical rate range when the differences "
    "clearly justify it - state the multiplier you applied. "
    "Typical cost relativities to apply: stainless steel fabrication is "
    "roughly 2.5-3x mild or galvanized steel; SS316 is ~1.15x SS304; "
    "COMPANY RULE: supply and installation is exactly 20% more expensive "
    "than supply only or supply and delivery - scope-adjusted rates given "
    "to you already include this conversion, so do NOT apply it again; "
    "linear items "
    "scale roughly with length/height and structural complexity; area items "
    "scale with buildup thickness and finish quality; a much larger overall "
    "size means proportionally more material. "
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
    anchor = weighted_median_rate(matches)
    if rates and anchor is not None:
        lines += [
            "",
            f"CLOSEST MATCH RATE (highest similarity, Match 1, scope-adjusted): "
            f"{_effective_rate(matches[0]):,.2f} {config.DEFAULT_CURRENCY} - anchor "
            "primarily on this item; use the others as corroboration.",
            f"STATISTICAL ANCHOR (similarity-weighted median of these matches): "
            f"{anchor:,.2f} {config.DEFAULT_CURRENCY}",
            f"HISTORICAL RATE RANGE of these matches: {min(rates):,.2f} to "
            f"{max(rates):,.2f} {config.DEFAULT_CURRENCY}",
            "Start from the anchor and apply quantified adjustments for the "
            "stated attribute differences (material, grade, size, scope, "
            "finish); going outside the historical range is allowed when "
            "the differences clearly justify it.",
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


def fallback_prediction(matches, weak_matches: bool, reason: str) -> dict:
    """Similarity-weighted median-style estimate when DeepSeek is unavailable."""
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
    # Similarity-weighted median: deterministic and robust to outliers, and
    # identical no matter how many matches the user chose to display.
    estimate = round(weighted_median_rate(priced), 2)
    units = [m["unit"] for m in priced if m["unit"]]
    return {
        "predicted_unit_price": estimate,
        "currency": config.DEFAULT_CURRENCY,
        "unit": units[0] if units else "unknown",
        "confidence": "Low" if weak_matches else "Medium",
        "reasoning": (
            f"Statistical fallback: similarity-weighted median of the "
            f"{len(priced)} strongest historical rates, scope-adjusted to "
            f"the input's work scope ({min(rates):,.2f} to {max(rates):,.2f}). "
            f"{reason}"
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
                                   f"DeepSeek API request failed ({type(exc).__name__}).")
    except (KeyError, IndexError, ValueError):
        return fallback_prediction(matches, weak_matches,
                                   "DeepSeek API returned an unexpected response shape.")

    try:
        result = _parse_json_response(content)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return fallback_prediction(matches, weak_matches,
                                   "DeepSeek returned invalid JSON.")
    result["fallback_used"] = False

    # Clamp the LLM's price to the pricing set's historical range so a
    # drifting response can never produce an implausible number.
    rates = [_effective_rate(m) for m in matches if _effective_rate(m)]
    if rates and result["predicted_unit_price"] is not None:
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
