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
