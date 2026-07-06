"""Extract structured pricing attributes from item descriptions and
compare the attributes of an input item against a dataset item.

All extraction runs on text already normalized by cleaner.normalize_text.
"""

import re

MATERIALS = [
    "stainless steel", "mild steel", "carbon steel", "galvanized", "aluminium",
    "concrete", "uhpc", "precast", "timber", "wood", "hardwood", "glass",
    "gypsum", "pvc", "hdpe", "upvc", "copper", "brass", "bronze", "cast iron",
    "iron", "granite", "marble", "ceramic", "porcelain", "rubber", "epdm",
    "frp", "grp", "acrylic", "polycarbonate", "fabric", "steel",
]

FINISHES = [
    "brushed", "polished", "painted", "powder coated", "galvanized",
    "anodized", "anodised", "laminated", "matte", "gloss", "epoxy coated",
    "epoxy", "wiredrawing", "sandblasted", "textured", "zinc rich primer",
    "hot dip", "chrome", "satin", "mirror",
]

GRADES = re.compile(
    r"\b(ss ?304|ss ?316l?|ss ?201|304|316l|316|grade ?[a-z0-9]+|m\d{2}\b|"
    r"fire rated|acoustic|waterproof|uv stabilized|marine grade|c\d{2}/\d{2})\b"
)

SCOPES = [
    ("supply and install", ["supply and install", "supply, install",
                            "supply & install", "installation included"]),
    ("supply and delivery", ["supply and delivery", "supply & delivery"]),
    ("supply only", ["supply only"]),
    ("install only", ["install only", "installation only", "fixing only"]),
    ("labour only", ["labour only", "labor only"]),
    ("testing and commissioning", ["testing and commissioning", "t&c"]),
]

CATEGORIES = [
    ("metalwork", ["handrail", "balustrade", "railing", "bollard", "gate",
                   "steel structure", "metal", "bike rack", "grating", "ladder"]),
    ("street furniture", ["bench", "litter bin", "planter", "picnic", "table",
                          "shade structure", "pergola", "gazebo", "seat",
                          "drinking fountain", "cycle stand", "signage", "sign"]),
    ("civil", ["concrete", "excavation", "foundation", "paving", "kerb",
               "blockwork", "screed", "civil works"]),
    ("electrical", ["cable", "light", "lighting", "socket", "switch", "db",
                    "conduit", "electrical"]),
    ("plumbing", ["pipe", "drainage", "valve", "sanitary", "plumbing",
                  "water supply"]),
    ("hvac", ["duct", "ac unit", "chiller", "ventilation", "hvac", "fcu", "ahu"]),
    ("joinery", ["joinery", "cabinet", "wardrobe", "door", "counter", "shelf"]),
    ("flooring", ["flooring", "tile", "carpet", "vinyl", "rubber flooring",
                  "epdm", "sports flooring"]),
    ("ceiling", ["ceiling", "gypsum board", "false ceiling"]),
    ("painting", ["paint", "painting", "coating works"]),
    ("landscape", ["landscape", "irrigation", "tree", "planting", "turf",
                   "playground", "play equipment", "swing", "slide", "outdoor gym",
                   "fitness equipment"]),
]

LOCATIONS = [
    "indoor", "outdoor", "external", "internal", "facade", "façade", "roof",
    "bathroom", "kitchen", "plant room", "basement", "podium", "park",
    "playground", "beach", "corniche",
]

UNIT_WORDS = re.compile(
    r"\b(per\s+)?(no|nos|each|item|set|pair|lm|rm|rmt|m2|sqm|m3|cum|kg|ton|"
    r"day|hour|hr|ls|lump sum|running metre|linear metre|metre|meter)\b"
)

_NUM = r"(\d+(?:\.\d+)?)"
DIA_RE = re.compile(rf"(?:{_NUM}\s*mm\s*dia|dia\.?\s*{_NUM}\s*mm|dia\.?\s*{_NUM}|"
                    rf"d\s?{_NUM}(?=\s?\*|\s?mm)|{_NUM}mm\s+dia)")
THICK_RE = re.compile(rf"{_NUM}\s*mm\s*thick|thick(?:ness)?[:\s]*{_NUM}\s*mm|"
                      rf"\*\s*{_NUM}\s*mm(?:\s|$)")
DIMS_RE = re.compile(
    rf"(?:l\s*)?{_NUM}\s*(?:mm)?\s*[x\*]\s*(?:w\s*)?{_NUM}\s*(?:mm)?"
    rf"(?:\s*[x\*]\s*(?:h\s*)?{_NUM}\s*(?:mm)?)?"
)
LEN_RE = re.compile(rf"\b(?:l|length)[\s:.]+{_NUM}\s*(mm|m)\b|{_NUM}\s*(m|mm)\s+(?:long|length)")
HEIGHT_RE = re.compile(rf"{_NUM}\s*mm\s*h\b|\bh[\s:.]+{_NUM}\s*mm|height[:\s]*{_NUM}")
SIZE_TOKEN_RE = re.compile(rf"{_NUM}\s*mm\b")


def _first_number(match) -> float | None:
    for g in match.groups():
        if g is not None:
            try:
                return float(g)
            except ValueError:
                continue
    return None


def extract_attributes(text: str) -> dict:
    """Extract a pricing-attribute dictionary from normalized text.

    Missing attributes are None (or [] for list attributes).
    """
    t = text or ""
    attrs = {
        "material": None, "finish": None, "grade": None, "scope": None,
        "category": None, "location": None, "unit_hint": None, "brand": None,
        "diameter_mm": None, "thickness_mm": None, "length_mm": None,
        "height_mm": None, "dimensions": None, "sizes_mm": [],
    }

    for m in MATERIALS:
        if m in t:
            # Prefer the most specific: "stainless steel" beats "steel".
            attrs["material"] = m
            break

    finishes = [f for f in FINISHES if f in t]
    attrs["finish"] = finishes[0] if finishes else None
    attrs["finishes_all"] = finishes

    g = GRADES.search(t)
    attrs["grade"] = g.group(1).replace(" ", "") if g else None

    for scope, keys in SCOPES:
        if any(k in t for k in keys):
            attrs["scope"] = scope
            break

    for cat, keys in CATEGORIES:
        if any(k in t for k in keys):
            attrs["category"] = cat
            break

    for loc in LOCATIONS:
        if loc in t:
            attrs["location"] = "facade" if loc == "façade" else loc
            break

    u = UNIT_WORDS.search(t)
    if u:
        attrs["unit_hint"] = u.group(2)

    b = re.search(r"brand\s*(?:&\s*origin)?\s*[\":]*\s*\"?([a-z0-9 \-]{2,30})\"?", t)
    if b:
        attrs["brand"] = b.group(1).strip()

    d = DIA_RE.search(t)
    if d:
        attrs["diameter_mm"] = _first_number(d)

    th = THICK_RE.search(t)
    if th:
        attrs["thickness_mm"] = _first_number(th)

    ln = LEN_RE.search(t)
    if ln:
        val = _first_number(ln)
        if val is not None:
            unit = next((g for g in ln.groups() if g in ("m", "mm")), "mm")
            attrs["length_mm"] = val * 1000 if unit == "m" else val

    h = HEIGHT_RE.search(t)
    if h:
        attrs["height_mm"] = _first_number(h)

    dims = DIMS_RE.search(t)
    if dims:
        attrs["dimensions"] = dims.group(0).strip()

    attrs["sizes_mm"] = sorted({float(x) for x in SIZE_TOKEN_RE.findall(t)})
    return attrs


def _close(a: float, b: float, tolerance: float = 0.25) -> bool:
    """True when two numeric sizes are within `tolerance` relative difference."""
    if not a or not b:
        return False
    return abs(a - b) / max(a, b) <= tolerance


def compare_attributes(input_attrs: dict, item_attrs: dict) -> dict:
    """Compare input vs dataset item attributes.

    Returns matched / mismatched / missing lists, price-relevant differences,
    and a 0..1 attribute score.
    """
    matched, mismatched, missing, differences = [], [], [], []
    score, weight_total = 0.0, 0.0

    def judge(name, weight, equal, describe_mismatch=None):
        nonlocal score, weight_total
        a, b = input_attrs.get(name), item_attrs.get(name)
        if a is None:
            return  # input doesn't specify it -> not scored
        weight_total += weight
        if b is None:
            missing.append(name)
            score += weight * 0.35  # unknown is better than a contradiction
            return
        if equal(a, b):
            matched.append(f"{name}: {b}")
            score += weight
        else:
            mismatched.append(f"{name}: input={a} vs item={b}")
            if describe_mismatch:
                differences.append(describe_mismatch(a, b))

    judge("material", 3.0, lambda a, b: a == b or a in b or b in a,
          lambda a, b: f"different material ({b} instead of {a})")
    judge("category", 2.0, lambda a, b: a == b)
    judge("scope", 2.0, lambda a, b: a == b,
          lambda a, b: f"different work scope ({b} instead of {a})")
    judge("finish", 1.5, lambda a, b: a == b or a in (item_attrs.get("finishes_all") or []),
          lambda a, b: f"different finish ({b} instead of {a})")
    judge("grade", 1.5, lambda a, b: a == b,
          lambda a, b: f"different grade/spec ({b} instead of {a})")
    judge("diameter_mm", 2.0, _close,
          lambda a, b: f"different diameter ({b:g}mm instead of {a:g}mm)")
    judge("thickness_mm", 1.0, _close,
          lambda a, b: f"different thickness ({b:g}mm instead of {a:g}mm)")
    judge("length_mm", 0.75, _close)
    judge("height_mm", 0.75, _close)
    judge("unit_hint", 1.0, lambda a, b: a == b,
          lambda a, b: f"different unit basis ({b} instead of {a})")
    judge("location", 0.5, lambda a, b: a == b)

    attr_score = (score / weight_total) if weight_total > 0 else 0.5
    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "differences": differences,
        "attribute_score": round(attr_score, 4),
    }
