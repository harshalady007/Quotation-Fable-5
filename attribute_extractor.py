"""Extract structured pricing attributes from item descriptions and
compare the attributes of an input item against a dataset item.

All extraction runs on text already normalized by cleaner.normalize_text.
"""

import re

from cleaner import normalize_unit

# What the item fundamentally IS. Checked in order; first hit wins, so more
# specific phrases come before generic ones. Matching a wrong item type is
# penalized harder than any other attribute.
ITEM_TYPES = [
    # Recycle bin before litter bin: "FN2 Litter Bin Recyclable Waste"
    # must type as a recycle bin, not a general litter bin.
    ("recycle bin", ["recycle bin", "recycled bin", "recycling bin",
                     "recyclable waste", "recycling station",
                     "recycle station"]),
    ("litter bin", ["litter bin", "waste bin", "trash bin", "garbage bin",
                    "dust bin", "dustbin", "trash can", "waste bins",
                    "general waste", "pedal bin", "bin"]),
    ("planter", ["planter pot", "planter box", "planter", "flower pot",
                 "flower box", "plant pot", "plant box"]),
    ("bench", ["bench"]),
    ("bollard", ["bollard"]),
    ("bike rack", ["bike rack", "cycle stand", "bicycle stand", "cycle rack",
                   "bike stand", "bicycle rack"]),
    ("shade structure", ["shade structure", "pergola", "gazebo", "shade sail",
                         "canopy"]),
    ("trellis", ["trellis"]),
    ("hatch", ["manhole hatch", "man hole hatch", "access hatch", "hatch",
               "manhole cover", "access cover", "manhole", "man hole"]),
    ("handrail", ["handrail", "hand rail"]),
    ("balustrade", ["balustrade", "guardrail", "railing"]),
    ("fence", ["fence", "fencing"]),
    ("gate", ["gate"]),
    ("signage", ["signage", "sign board", "wayfinding", "sign"]),
    ("table", ["picnic table", "table"]),
    ("drinking fountain", ["drinking fountain", "water fountain"]),
    ("tree grate", ["tree grate", "tree grille", "tree guard"]),
    ("play equipment", ["play equipment", "playground equipment", "swing",
                        "slide", "seesaw", "climbing frame", "play unit"]),
    ("outdoor gym", ["outdoor gym", "fitness equipment", "gym equipment",
                     "exercise equipment"]),
    ("shelter", ["bus shelter", "shelter", "kiosk"]),
    ("flagpole", ["flagpole", "flag pole"]),
    ("bowl", ["concrete bowl", "bowl"]),
    ("ladder", ["ladder"]),
    ("grating", ["grating", "grille"]),
    ("decking", ["decking", "deck"]),
]

# Item types that are close enough cousins to price from each other when no
# exact-type match exists (no hard score penalty between them).
COMPATIBLE_TYPES = [
    {"handrail", "balustrade"},
    {"recycle bin", "litter bin"},
]


def types_compatible(a, b) -> bool:
    if not a or not b or a == b:
        return False
    return any(a in group and b in group for group in COMPATIBLE_TYPES)


# Add-on features that transform an item's price (a planter WITH integrated
# seating is a different product from a plain planter box).
ADDON_FEATURES = [
    ("integrated seating", ["seater", "seating", "seat", "bench"]),
    ("lighting", ["led", "lighting", "illuminated", "light fitting",
                  "spotlight", "floodlight", "solar light"]),
    ("irrigation", ["irrigation", "drain fitting", "drainage"]),
    ("water feature", ["water feature", "fountain"]),
    ("cladding", ["cladded", "cladding"]),
]

_FEATURE_PATTERNS = [
    (name, re.compile(r"\b" + re.escape(syn) + r"\b"))
    for name, syns in ADDON_FEATURES for syn in syns
]

_ITEM_TYPE_PATTERNS = [
    (canonical, re.compile(r"\b" + re.escape(syn) + r"\b"))
    for canonical, syns in ITEM_TYPES for syn in syns
]

MATERIALS = [
    "stainless steel", "mild steel", "carbon steel", "galvanized", "aluminium",
    "composite bamboo", "bamboo", "corten",
    "concrete", "uhpc", "precast", "timber", "wood", "hardwood", "glass",
    "gypsum", "pvc", "hdpe", "upvc", "copper", "brass", "bronze", "cast iron",
    "iron", "granite", "marble", "ceramic", "porcelain", "rubber", "epdm",
    "frp", "grp", "acrylic", "polycarbonate", "fabric", "steel",
]

# Word-boundary patterns so substrings never false-match ("fabricated"
# must not read as material "fabric", "waterproof" is not location "roof").
_MATERIAL_PATTERNS = [(m, re.compile(r"\b" + re.escape(m) + r"\b"))
                      for m in MATERIALS]

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
                   "steel structure", "metalwork", "bike rack", "grating",
                   "ladder"]),
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

_UNIT_TOKENS = (r"no|nos|each|item|set|pair|lm|rm|rmt|m2|sqm|m3|cum|kg|ton|"
                r"day|hour|hr|ls|lump sum|running metre|linear metre|"
                r"metre|meter|m")
# An explicit "per <unit>" states the pricing basis and always wins over a
# bare unit word that may just be counting parts ("11 no.s of cables").
PER_UNIT_RE = re.compile(rf"\bper\s+({_UNIT_TOKENS})\b")
UNIT_WORDS = re.compile(rf"\b({_UNIT_TOKENS})\b")

_NUM = r"(\d+(?:\.\d+)?)"
DIA_RE = re.compile(rf"(?:{_NUM}\s*mm\s*dia|dia\.?\s*{_NUM}\s*mm|dia\.?\s*{_NUM}|"
                    rf"\bd\s?{_NUM}(?=\s?[\*x]|\s?mm)|{_NUM}mm\s+dia)")
THICK_RE = re.compile(rf"{_NUM}\s*mm\s*thick|thick(?:ness)?[:\s]*{_NUM}\s*mm|"
                      rf"\*\s*{_NUM}\s*mm(?:\s|$)")
# Dimension chains: "1200x600mm", "l 17770 x w 3050 x h 800mm",
# "d3063 x h800mm" — any of l/w/h/d may prefix each number.
_DIM_PREFIX = r"(?:\b[lwhd][\s:.]*)?"
DIMS_RE = re.compile(
    rf"{_DIM_PREFIX}{_NUM}\s*(?:mm)?\s*[x\*]\s*{_DIM_PREFIX}{_NUM}\s*(?:mm)?"
    rf"(?:\s*[x\*]\s*{_DIM_PREFIX}{_NUM}\s*(?:mm)?)?"
)
LEN_RE = re.compile(rf"\b(?:l|length)[\s:.]+{_NUM}\s*(mm|m)\b|{_NUM}\s*(m|mm)\s+(?:long|length)")
HEIGHT_RE = re.compile(rf"{_NUM}\s*mm\s*h\b|\bh[\s:.]+{_NUM}\s*mm|height[:\s]*{_NUM}")
SIZE_TOKEN_RE = re.compile(rf"{_NUM}\s*mm\b")


# Scopes whose rates are convertible into each other via the company's
# 20% installation rule — a comp in one group is still a valid pricing
# comparable for an input in the other, after rate conversion.
INSTALL_SCOPES = {"supply and install", "install only"}
SUPPLY_SCOPES = {"supply only", "supply and delivery"}


def detect_scope(text: str) -> str | None:
    """Detect the work scope in normalized text (also usable on the
    dataset's scope-of-work column)."""
    if not text:
        return None
    for scope, keys in SCOPES:
        if any(k in text for k in keys):
            return scope
    return None


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
        "item_type": None,
        "material": None, "finish": None, "grade": None, "scope": None,
        "category": None, "location": None, "unit_hint": None, "brand": None,
        "diameter_mm": None, "thickness_mm": None, "length_mm": None,
        "height_mm": None, "dimensions": None, "sizes_mm": [],
    }

    for canonical, pattern in _ITEM_TYPE_PATTERNS:
        if pattern.search(t):
            attrs["item_type"] = canonical
            break

    attrs["features"] = sorted({name for name, pattern in _FEATURE_PATTERNS
                                if pattern.search(t)})

    # Primary material = the one mentioned EARLIEST in the text ("mild steel
    # posts ... stainless steel cables" is a mild steel item). Overlapping
    # names resolve naturally: "stainless steel" starts before its "steel".
    best_pos = None
    for m, pattern in _MATERIAL_PATTERNS:
        hit = pattern.search(t)
        if hit and (best_pos is None or hit.start() < best_pos):
            best_pos = hit.start()
            attrs["material"] = m

    finishes = [f for f in FINISHES
                if re.search(r"\b" + re.escape(f) + r"\b", t)]
    attrs["finish"] = finishes[0] if finishes else None
    attrs["finishes_all"] = finishes

    g = GRADES.search(t)
    attrs["grade"] = g.group(1).replace(" ", "") if g else None

    attrs["scope"] = detect_scope(t)

    for cat, keys in CATEGORIES:
        if any(k in t for k in keys):
            attrs["category"] = cat
            break

    for loc in LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", t):
            attrs["location"] = "facade" if loc == "façade" else loc
            break

    u = PER_UNIT_RE.search(t) or UNIT_WORDS.search(t)
    if u:
        attrs["unit_hint"] = normalize_unit(u.group(1))

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

    sizes = {float(x) for x in SIZE_TOKEN_RE.findall(t)}
    # Numbers inside a dimension chain ("l 17770 x w 3050 x h 800mm") share
    # the trailing unit — without this, only the last "800mm" is seen and a
    # 17.7m item looks smaller than a 2.6m one.
    if dims:
        for g in dims.groups():
            if g is not None:
                try:
                    sizes.add(float(g))
                except ValueError:
                    pass
    attrs["sizes_mm"] = sorted(sizes)
    # Characteristic overall size: the largest stated dimension. Lets the
    # comparison flag "same item type but a much bigger/smaller one".
    attrs["max_size_mm"] = attrs["sizes_mm"][-1] if attrs["sizes_mm"] else None
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

    # Item type is the single most price-defining attribute: a planter must
    # never be priced from litter bins just because material/finish agree.
    # Compatible cousins (handrail/balustrade, trellis/shade structure) get
    # partial credit instead of a mismatch.
    a, b = input_attrs.get("item_type"), item_attrs.get("item_type")
    if a is not None:
        weight_total += 5.0
        if b is None:
            missing.append("item_type")
            score += 5.0 * 0.35
        elif a == b:
            matched.append(f"item_type: {b}")
            score += 5.0
        elif types_compatible(a, b):
            matched.append(f"item_type: {b} (related to {a})")
            score += 5.0 * 0.6
        else:
            mismatched.append(f"item_type: input={a} vs item={b}")
            differences.append(f"different item type ({b} instead of {a})")

    # Add-on features: an item carrying a costly extra the input lacks
    # (e.g. integrated seating) is a poor pricing comparable.
    feats_in = set(input_attrs.get("features") or [])
    feats_item = set(item_attrs.get("features") or [])
    for feat in sorted(feats_in | feats_item):
        w = 2.0 if feat == "integrated seating" else 1.0
        weight_total += w
        if feat in feats_in and feat in feats_item:
            matched.append(f"feature: {feat}")
            score += w
        elif feat in feats_item:
            mismatched.append(f"feature: item includes {feat}, input does not")
            differences.append(
                f"match includes {feat} which the input does not have "
                "(its rate covers more than the input item)")
        else:
            mismatched.append(f"feature: input includes {feat}, item does not")
            differences.append(
                f"input includes {feat} which the match does not have")
            score += w * 0.25
    judge("material", 3.0, lambda a, b: a == b or a in b or b in a,
          lambda a, b: f"different material ({b} instead of {a})")
    judge("category", 2.0, lambda a, b: a == b)
    # Scopes within the convertible supply/install groups count as matched:
    # the 20% rule converts their rates, so a supply-only comp is a valid
    # comparable for an install input (and selecting the same comps for
    # both scopes is what keeps install = supply + 20% consistent).
    _convertible = INSTALL_SCOPES | SUPPLY_SCOPES
    judge("scope", 2.0,
          lambda a, b: a == b or (a in _convertible and b in _convertible),
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
    # Overall size only means "product scale" for per-item products; for
    # per-metre/per-m2 rates the stated sizes are profiles, not scale.
    linear_units = ("m", "m2", "m3")
    if (input_attrs.get("unit_hint") not in linear_units
            and item_attrs.get("unit_hint") not in linear_units):
        judge("max_size_mm", 1.5, lambda a, b: _close(a, b, 0.25),
              lambda a, b: f"very different overall size ({b:g}mm vs {a:g}mm "
                           "largest dimension)")
    judge("location", 0.5, lambda a, b: a == b)

    attr_score = (score / weight_total) if weight_total > 0 else 0.5
    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "differences": differences,
        "attribute_score": round(attr_score, 4),
    }
