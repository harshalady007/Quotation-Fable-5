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
    # Components must be recognized before their parent product. A planter
    # extension or a metal edge beside a balustrade is not a complete item.
    ("planter component", ["planter side wall extension",
                           "planter sidewall extension",
                           "structure for planter", "planter structure"]),
    ("planter", ["planter pot", "planter box", "planter", "flower pot",
                 "flower box", "plant pot", "plant box"]),
    ("metal component", ["metal edge at balustrade", "metal edge"]),
    ("raw material", ["raw material supply", "raw material"]),
    ("cladding", ["cladding with coping", "metal cladding", "cladding"]),
    ("sun lounger", ["sun lounger", "sunbed", "sun bed", "pool lounger",
                     "wet lounger", "lounger"]),
    # Table precedes chair/bench because picnic sets commonly mention both.
    ("table", ["picnic table", "coffee table", "table set", "table"]),
    ("bench", ["bench", "seater", "feature seating"]),
    ("chair", ["lifeguard chair", "chair"]),
    ("cabinet", ["towel cabinet", "cabinet"]),
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
    # Generic "bench"/"seat" terms are handled contextually below. Treating
    # them as add-ons marked every standalone bench as integrated seating.
    ("integrated seating", ["integrated seating", "integrated seat",
                            "integrated bench", "with seater", "with seating",
                            "seating support"]),
    ("lighting", ["led", "lighting", "illuminated", "light fitting",
                  "spotlight", "floodlight", "solar light"]),
    ("irrigation", ["irrigation", "drain fitting", "drainage"]),
    ("water feature", ["water feature", "fountain"]),
    ("cladding", ["cladded", "cladding"]),
    ("backrest", ["backrest", "back rest"]),
    ("armrest", ["armrest", "arm rest"]),
    ("perforated", ["perforated", "perforation", "perforations"]),
    ("pedal", ["pedal"]),
    ("liner", ["liner", "inner bin"]),
    ("wheels", ["wheel", "wheels", "wheeled"]),
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
    # Galvanized is a finish/coating, not a substrate.
    "stainless steel", "mild steel", "carbon steel", "aluminium",
    "composite bamboo", "bamboo", "corten",
    # Common quotation wording that still describes an existing canonical
    # substrate.  Keep the alias in the matcher, then normalize below so
    # commercial groups are not fragmented into "iroko" versus "wood" or
    # "hot gi" versus "steel".
    "hot galvanized", "hot gi", "galvanized iron", "galvanised iron",
    "iroko", "wooden", "wpc",
    "concrete", "uhpc", "precast", "timber", "wood", "hardwood", "glass",
    "gypsum", "pvc", "hdpe", "upvc", "copper", "brass", "bronze", "cast iron",
    "iron", "granite", "marble", "ceramic", "porcelain", "rubber", "epdm",
    "frp", "grp", "acrylic", "polycarbonate", "fabric", "steel",
]

MATERIAL_ALIASES = {
    "hot galvanized": "steel",
    "hot gi": "steel",
    "galvanized iron": "steel",
    "galvanised iron": "steel",
    "iroko": "wood",
    "wooden": "wood",
    "wpc": "wood",
}

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
# Only explicit pricing-basis language is accepted. Bare tokens are unsafe:
# "ITEM NO J", "table set" and "2 m long" do not mean the quotation unit is
# respectively no/set/metre.
PER_UNIT_RE = re.compile(rf"\bper\s+(?:\d+(?:\.\d+)?\s+)?({_UNIT_TOKENS})\b")
UNIT_LABEL_RE = re.compile(rf"\b(?:unit|uom|unit of measure)\s*[:=\-]\s*({_UNIT_TOKENS})\b")

_NUM = r"(\d+(?:\.\d+)?)"
_NUM_NC = r"\d+(?:\.\d+)?"
_DIM_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
DIA_RE = re.compile(rf"(?:{_NUM}\s*mm\s*dia|dia\.?\s*{_NUM}\s*mm|dia\.?\s*{_NUM}|"
                    rf"\bd\s?{_NUM}(?=\s?[\*x]|\s?mm)|{_NUM}mm\s+dia)")
THICK_RE = re.compile(rf"{_NUM}\s*mm\s*thick|thick(?:ness)?[:\s]*{_NUM}\s*mm|"
                      rf"\*\s*{_NUM}\s*mm(?:\s|$)")
# Dimension chains in the source workbook use both prefix and suffix axes:
# ``L1800 xW530 xH530``, ``1753 L x 533 W x 787mm H`` and plain
# ``1500x400x450mm`` are all common.  Match complete components first and
# interpret their labels in ``_extract_dimension_chain`` rather than relying
# on capture-group position.
_DIM_AXIS = r"(?:length|width|wide|height|high|diameter|dia|[lwhd])"
_DIM_LABEL = rf"(?:{_DIM_AXIS}|\(\s*{_DIM_AXIS}\s*\))"
_DIM_VALUE = rf"{_DIM_NUM}(?:\s*/\s*{_DIM_NUM})?"
_DIM_COMPONENT = (
    rf"(?:{_DIM_LABEL}\s*[:.]?\s*)?{_DIM_VALUE}\s*(?:mm|m)?\s*"
    rf"(?:{_DIM_LABEL})?"
)
DIM_CHAIN_RE = re.compile(
    rf"(?P<first>{_DIM_COMPONENT})\s*[x\*]\s*"
    rf"(?P<second>{_DIM_COMPONENT})"
    rf"(?:\s*[x\*]\s*(?P<third>{_DIM_COMPONENT}))?"
)
DIM_COMPONENT_RE = re.compile(
    rf"^\s*(?:(?P<prefix>{_DIM_LABEL})\s*[:.]?\s*)?"
    rf"(?P<value>{_DIM_VALUE})\s*(?P<unit>mm|m)?\s*"
    rf"(?P<suffix>{_DIM_LABEL})?\s*$"
)
LEN_RE = re.compile(rf"\b(?:l|length)[\s:.]+{_NUM}\s*(mm|m)\b|{_NUM}\s*(m|mm)\s+(?:long|length)")
# Developed lengths such as "L 7000+1120 x 600 x 450mm" occur on shaped
# benches. Treating only the last 1120mm as length creates a catastrophic size
# error, so the explicit segments are summed.
SUM_LENGTH_RE = re.compile(
    r"\b(?:l|length)[\s:.]*(\d+(?:\.\d+)?(?:\s*\+\s*\d+(?:\.\d+)?)+)"
    r"\s*(?:mm)?\s*[x\*]"
)
HEIGHT_RE = re.compile(rf"{_NUM}\s*mm\s*h\b|\bh[\s:.]+{_NUM}\s*mm|height[:\s]*{_NUM}")
SIZE_TOKEN_RE = re.compile(rf"{_NUM}\s*mm\b")
CAPACITY_L_RE = re.compile(
    rf"\b{_NUM}\s*(?:l|ltr|ltrs|litre|litres|liter|liters)\b"
)
CAPACITY_M3_RE = re.compile(rf"\b{_NUM}\s*(?:m3|cbm|cum)\b")
COMPARTMENT_RE = re.compile(
    r"\b(?:(single|double|dual|triple|quadruple)|([1-9]\d*))\s+compartment[s]?\b"
)
STREAM_RE = re.compile(
    r"\b(?:(single|double|dual|triple|quadruple)|([1-9]\d*))\s+(?:waste\s+)?stream[s]?\b"
)


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


def detect_subtype(text: str, attrs: dict) -> str | None:
    """Extract a conservative, family-specific functional subtype.

    Subtypes are emitted only from explicit wording or a defining feature;
    material remains a separate price driver.  Unknown is safer than forcing
    a record into the wrong commercial class.
    """
    family = attrs.get("item_type")
    features = set(attrs.get("features") or [])
    compartments = attrs.get("compartments")

    if family == "planter":
        return "integrated seating" if "integrated seating" in features else "standalone planter"
    if family == "bench":
        if re.search(r"\btree\s+pit\b|\btree\s+bench\b|\baround\s+(?:a\s+)?tree\b", text):
            return "tree bench"
        if re.search(r"\b(?:galet|barrel|monolith)\b|\bfeature\s+seating\b", text):
            return "sculptural bench"
        if re.search(r"\b(?:l|u|s)[\s\-]*shape(?:d)?\b|\btriangular\b|"
                     r"\b(?:curved|curvilinear|circular|semicircular|"
                     r"semi[\s\-]*circular)\b|\bbench\s+arch\b", text):
            return "shaped bench"
        if re.search(r"\blinear\s+bench\b", text):
            return "linear bench"
        if re.search(r"\bwithout\s+back(?:rest)?\b|\bbackless\b", text):
            return "backless bench"
        if "backrest" in features or re.search(r"\bwith\s+back(?:rest)?\b", text):
            return "bench with backrest"
        return None
    if family in {"litter bin", "recycle bin"}:
        if compartments and compartments > 1:
            return "multi-stream bin"
        if "pedal" in features:
            return "pedal bin"
        if re.search(r"\bwall[\s\-]*mounted\b", text):
            return "wall-mounted bin"
        if "wheels" in features or re.search(r"\bwheeled\b", text):
            return "mobile bin"
        return None
    if family == "bollard":
        if re.search(r"\bremovable\b|\bdemountable\b", text):
            return "removable bollard"
        if re.search(r"\bretractable\b|\btelescopic\b", text):
            return "retractable bollard"
        if re.search(r"\bflexible\b", text):
            return "flexible bollard"
        if attrs.get("mobility") == "fixed":
            return "fixed bollard"
        return None
    if family == "bike rack":
        if re.search(r"\bwall[\s\-]*mounted\b|\bvertical\s+(?:bike|cycle)", text):
            return "wall-mounted rack"
        if re.search(r"\bwave\s+rack\b|\bspiral\s+rack\b", text):
            return "multi-bike rack"
        if re.search(r"\bsheffield\b|\bhoop\b|\bu[\s\-]*rack\b", text):
            return "hoop rack"
        return None
    return None


def _first_number(match) -> float | None:
    for g in match.groups():
        if g is not None:
            try:
                return float(g)
            except ValueError:
                continue
    return None


def _dimension_axis(value: str | None) -> str | None:
    """Return a canonical axis for a dimension label."""
    if not value:
        return None
    value = value.strip().strip("()").strip()
    return {
        "l": "length_mm", "length": "length_mm",
        "w": "width_mm", "wide": "width_mm", "width": "width_mm",
        "h": "height_mm", "high": "height_mm", "height": "height_mm",
        "d": "diameter_mm", "dia": "diameter_mm",
        "diameter": "diameter_mm",
    }.get(value)


def _extract_dimension_chain(text: str) -> tuple[re.Match | None, dict, list[float]]:
    """Parse the strongest 2/3-part dimension chain without guessing axes.

    Explicit L/W/H/D labels win.  Unlabelled values use conventional
    L x W x H order.  A trailing unit applies to preceding unitless values,
    which correctly interprets both ``2 x .5 x .45m`` and
    ``2000 x 500 x 450mm``.
    """
    matches = list(DIM_CHAIN_RE.finditer(text))
    if not matches:
        return None, {}, []

    def match_priority(candidate: re.Match) -> tuple:
        # Product dimensions are commonly preceded by Size/Dimensions.  This
        # must outrank an earlier component profile such as a bench leg's
        # 40x50mm section.  Explicit axes and a 3-part chain are the next
        # strongest evidence; later occurrence is the final tie-breaker.
        before = text[max(0, candidate.start() - 32):candidate.start()]
        size_context = bool(re.search(
            r"\b(?:overall\s+)?(?:size|dimensions?)\s*[:=\-]?\s*$", before
        ))
        explicit_axes = 0
        component_count = 0
        for name in ("first", "second", "third"):
            raw = candidate.group(name)
            if raw is None:
                continue
            component_count += 1
            parsed = DIM_COMPONENT_RE.fullmatch(raw)
            if parsed and (parsed.group("prefix") or parsed.group("suffix")):
                explicit_axes += 1
        return size_context, explicit_axes, component_count, candidate.start()

    match = max(matches, key=match_priority)

    parts = []
    for name in ("first", "second", "third"):
        raw = match.group(name)
        if raw is None:
            continue
        parsed = DIM_COMPONENT_RE.fullmatch(raw)
        if not parsed:  # Defensive: the outer and inner patterns must agree.
            return None, {}, []
        parts.append({
            "value": max(
                float(value.replace(",", ""))
                for value in parsed.group("value").split("/")
            ),
            "unit": parsed.group("unit"),
            "axis": _dimension_axis(
                parsed.group("prefix") or parsed.group("suffix")
            ),
        })

    shared_unit = next(
        (part["unit"] for part in reversed(parts) if part["unit"]), "mm"
    )
    for part in parts:
        unit = part["unit"] or shared_unit
        if unit == "m":
            part["value"] *= 1000

    dimensions = {}
    conventional_axes = ("length_mm", "width_mm", "height_mm")
    claimed = {part["axis"] for part in parts if part["axis"]}
    for position, part in enumerate(parts):
        axis = part["axis"]
        if axis is None:
            preferred = conventional_axes[position]
            if preferred not in claimed and preferred not in dimensions:
                axis = preferred
            else:
                axis = next(
                    (candidate for candidate in conventional_axes
                     if candidate not in claimed and candidate not in dimensions),
                    None,
                )
        if axis and axis not in dimensions:
            dimensions[axis] = part["value"]
    return match, dimensions, [part["value"] for part in parts]


def extract_attributes(text: str) -> dict:
    """Extract a pricing-attribute dictionary from normalized text.

    Missing attributes are None (or [] for list attributes).
    """
    t = text or ""
    attrs = {
        "item_type": None, "subtype": None,
        "material": None, "finish": None, "grade": None, "scope": None,
        "category": None, "location": None, "unit_hint": None, "brand": None,
        "diameter_mm": None, "thickness_mm": None, "length_mm": None,
        "width_mm": None, "depth_mm": None, "height_mm": None,
        "capacity_l": None, "compartments": None, "mobility": None,
        "civil_works": None, "dimensions": None, "sizes_mm": [],
    }

    for canonical, pattern in _ITEM_TYPE_PATTERNS:
        if pattern.search(t):
            attrs["item_type"] = canonical
            break

    features = {name for name, pattern in _FEATURE_PATTERNS if pattern.search(t)}
    if re.search(r"\bwithout\s+back(?:rest)?\b|\bbackless\b", t):
        features.discard("backrest")
    if (attrs["item_type"] == "planter"
            and re.search(r"\b(?:seater|seating|seat|bench)\b", t)):
        features.add("integrated seating")
    attrs["features"] = sorted(features)

    # Primary material = the one mentioned EARLIEST in the text ("mild steel
    # posts ... stainless steel cables" is a mild steel item). Overlapping
    # names resolve naturally: "stainless steel" starts before its "steel".
    best_pos = None
    materials_all = []
    for m, pattern in _MATERIAL_PATTERNS:
        hit = pattern.search(t)
        if hit:
            materials_all.append((hit.start(), m))
        if hit and (best_pos is None or hit.start() < best_pos):
            best_pos = hit.start()
            attrs["material"] = MATERIAL_ALIASES.get(m, m)
    attrs["materials_all"] = list(dict.fromkeys(
        MATERIAL_ALIASES.get(m, m) for _, m in sorted(materials_all)
    ))
    wood_materials = {"wood", "timber", "hardwood"}
    if (attrs["item_type"] == "bench"
            and attrs["material"] not in wood_materials
            and wood_materials.intersection(attrs["materials_all"])):
        features.add("wood accent")
        attrs["features"] = sorted(features)

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

    u = PER_UNIT_RE.search(t) or UNIT_LABEL_RE.search(t)
    if u:
        attrs["unit_hint"] = normalize_unit(u.group(1))

    if re.search(r"\bwithout\s+civil\s+works\b|\bexcluding\s+civil\s+works\b", t):
        attrs["civil_works"] = "excluded"
    elif re.search(r"\bincluding\s+civil\s+works\b|\bwith\s+civil\s+works\b", t):
        attrs["civil_works"] = "included"

    if re.search(r"\bremovable\b|\bdemountable\b", t):
        attrs["mobility"] = "removable"
    elif re.search(r"\bmovable\b|\bmobile\b|\bportable\b|\bfree[\s\-]*standing\b", t):
        attrs["mobility"] = "movable"
    elif re.search(r"\bfixed\b|\bbase[\s\-]*plated\b|\bembedded\b", t):
        attrs["mobility"] = "fixed"

    cap = CAPACITY_L_RE.search(t)
    if cap:
        attrs["capacity_l"] = _first_number(cap)
    else:
        cap_m3 = CAPACITY_M3_RE.search(t)
        if cap_m3:
            value = _first_number(cap_m3)
            attrs["capacity_l"] = value * 1000 if value is not None else None

    compartments = COMPARTMENT_RE.search(t) or STREAM_RE.search(t)
    if compartments:
        word, number = compartments.groups()
        mapping = {"single": 1, "double": 2, "dual": 2, "triple": 3,
                   "quadruple": 4}
        attrs["compartments"] = mapping.get(word, int(number) if number else None)

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

    summed_length = SUM_LENGTH_RE.search(t)
    if summed_length:
        attrs["length_mm"] = sum(
            float(part.strip()) for part in summed_length.group(1).split("+")
        )

    h = HEIGHT_RE.search(t)
    if h:
        attrs["height_mm"] = _first_number(h)

    dims, parsed_dimensions, dim_values = _extract_dimension_chain(t)
    if dims:
        attrs["dimensions"] = dims.group(0).strip()
        before_dims = t[max(0, dims.start() - 32):dims.start()]
        overall_dimensions = bool(re.search(
            r"\b(?:overall\s+)?(?:size|dimensions?)\s*[:=\-]?\s*$",
            before_dims,
        ))
        for field, value in parsed_dimensions.items():
            # A labelled Size/Dimensions chain is authoritative over an
            # earlier component statement such as "slats 600mm long".  A
            # developed L=a+b length remains the one intentional exception.
            if (attrs.get(field) is None or overall_dimensions) and not (
                field == "length_mm" and summed_length
            ):
                attrs[field] = value

    # For a circular bench the diameter is its price-defining overall span.
    # Expose that span through the bench contract's length field, while
    # retaining diameter_mm so the geometry remains explicit.
    if (attrs["item_type"] == "bench" and attrs["length_mm"] is None
            and attrs["diameter_mm"]):
        attrs["length_mm"] = attrs["diameter_mm"]

    # Some bench schedules state a single unambiguous overall size (for
    # example "Heavy Duty Bench ... Size 2000 mm").  Do not apply this rule
    # to other product families or when width/height wording is present.
    if (attrs["item_type"] == "bench" and attrs["length_mm"] is None
            and not any(attrs.get(field) for field in (
                "width_mm", "height_mm", "diameter_mm"
            ))):
        single_size = re.search(r"\bsize\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*mm\b", t)
        if single_size:
            attrs["length_mm"] = float(single_size.group(1))

    sizes = {float(x) for x in SIZE_TOKEN_RE.findall(t)}
    # Numbers inside a dimension chain ("l 17770 x w 3050 x h 800mm") share
    # the trailing unit — without this, only the last "800mm" is seen and a
    # 17.7m item looks smaller than a 2.6m one.
    if dims:
        sizes.update(dim_values)
    sizes.update(
        float(value) for value in (
            attrs.get("length_mm"), attrs.get("width_mm"), attrs.get("height_mm"),
            attrs.get("diameter_mm"),
        ) if value
    )
    attrs["sizes_mm"] = sorted(sizes)
    # Characteristic overall size: the largest stated dimension. Lets the
    # comparison flag "same item type but a much bigger/smaller one".
    attrs["max_size_mm"] = attrs["sizes_mm"][-1] if attrs["sizes_mm"] else None
    if attrs["length_mm"] and attrs["width_mm"]:
        attrs["footprint_mm2"] = attrs["length_mm"] * attrs["width_mm"]
    else:
        attrs["footprint_mm2"] = None
    if attrs["footprint_mm2"] and attrs["height_mm"]:
        attrs["envelope_mm3"] = attrs["footprint_mm2"] * attrs["height_mm"]
    else:
        attrs["envelope_mm3"] = None
    # Size proxy for price interpolation: prefer the stated length (the
    # dominant cost driver), fall back to the largest dimension.
    if attrs["length_mm"] and attrs["length_mm"] >= 100:
        attrs["size_proxy"] = attrs["length_mm"]
        attrs["size_proxy_kind"] = "length"
    elif attrs["max_size_mm"] and attrs["max_size_mm"] >= 100:
        attrs["size_proxy"] = attrs["max_size_mm"]
        attrs["size_proxy_kind"] = "max"
    else:
        attrs["size_proxy"] = None
        attrs["size_proxy_kind"] = None
    attrs["subtype"] = detect_subtype(t, attrs)
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

    judge("subtype", 3.0, lambda a, b: a == b,
          lambda a, b: f"different product subtype ({b} instead of {a})")

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
    # Generic "steel" must not receive full material credit against stainless
    # or mild steel; those price very differently.
    judge("material", 3.0, lambda a, b: a == b,
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
    judge("capacity_l", 2.5, lambda a, b: _close(a, b, 0.20),
          lambda a, b: f"different capacity ({b:g}L instead of {a:g}L)")
    judge("compartments", 1.5, lambda a, b: a == b,
          lambda a, b: f"different compartment count ({b} instead of {a})")
    judge("mobility", 1.0, lambda a, b: a == b,
          lambda a, b: f"different fixing/mobility ({b} instead of {a})")
    judge("civil_works", 2.0, lambda a, b: a == b,
          lambda a, b: f"different civil-works scope ({b} instead of {a})")
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
