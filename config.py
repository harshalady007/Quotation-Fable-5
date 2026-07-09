"""Central configuration for the pricing bot.

Override any of these with environment variables so the project works both
in this repo (data/quotation_items.xlsx) and on a local Windows machine
(e.g. C:\\Harshal\\Quotation Dataset\\...\\quotation_items.xlsx).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the historical quotation Excel file.
DATA_PATH = os.environ.get(
    "QUOTATION_DATA_PATH",
    str(PROJECT_ROOT / "data" / "quotation_items.xlsx"),
)

# DeepSeek API settings. The key MUST come from the environment.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.environ.get("DEEPSEEK_TIMEOUT", "60"))

# Default currency used when the dataset does not state one per row.
# The attached dataset's Quotation Summary sheet is 100% AED.
DEFAULT_CURRENCY = os.environ.get("DEFAULT_CURRENCY", "AED")

# Similarity search weights: final = TEXT_WEIGHT*text + ATTR_WEIGHT*attributes
TEXT_WEIGHT = 0.55
ATTR_WEIGHT = 0.45

# Matches below this final score are considered weak.
WEAK_MATCH_THRESHOLD = 0.35

# When both the input and a dataset item have a recognized item type
# (planter, litter bin, bench, ...) and they differ, the item's final score
# is multiplied by this penalty so same-type items always outrank it.
TYPE_MISMATCH_PENALTY = 0.45

# When two per-item products of the SAME type state sizes whose largest
# dimensions differ by more than this ratio, the item is a different size
# class (a 50mm frame member is not a comparable for a 2.6m planter) and
# its score is penalized. Never applied to per-metre/per-m2 items, whose
# stated sizes are profiles, not product scale.
SIZE_MISMATCH_RATIO = 4.0
SIZE_MISMATCH_PENALTY = 0.6

# An item that includes integrated seating when the input does not (or
# vice versa) is priced for a different product; its score is penalized.
SEATING_MISMATCH_PENALTY = 0.7

# Company rule: supply and installation is 20% more expensive than supply
# only / supply and delivery. Historical rates are converted to the input's
# scope basis before the anchor and clamp are computed.
SCOPE_INSTALL_UPLIFT = float(os.environ.get("SCOPE_INSTALL_UPLIFT", "1.20"))

# Pricing is always computed from a fixed set of the strongest matches,
# independent of how many matches the user displays (top_k). A match joins
# the pricing set when its score is within PRICING_RELATIVE_CUTOFF of the
# best score; the set has between MIN and MAX members.
PRICING_MIN_MATCHES = 3
PRICING_MAX_MATCHES = 5
PRICING_RELATIVE_CUTOFF = 0.85

# The predicted price is clamped to this factor of the pricing set's
# historical rate range. Wide enough to allow justified spec adjustments
# (e.g. SS316 vs mild steel comps can legitimately be ~3x), while still
# catching order-of-magnitude drift.
PRICE_CLAMP_LOW = 0.4    # x lowest rate in pricing set
PRICE_CLAMP_HIGH = 3.0   # x highest rate in pricing set
