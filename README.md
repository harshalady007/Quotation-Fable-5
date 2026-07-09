# Quotation Pricing Bot

A local Python pricing assistant for quotation/estimation work. You enter a
new item or service description; the bot searches your historical quotation
Excel dataset, finds the most similar items, compares them attribute by
attribute (material, size, diameter, thickness, finish, grade, scope, unit,
category) like a real estimator, and predicts a unit price using the
DeepSeek API — with a statistical fallback when the API is unavailable.

It does **not** match by text alone: a hybrid score combines TF-IDF text
similarity with structured attribute matching, so a "50mm dia stainless
steel handrail, brushed, supply and install" prefers stainless steel items
with the same diameter, finish and scope over an aluminium handrail or a
generic steel plate.

## Project structure

```text
├── streamlit_app.py        # Streamlit web interface (run locally / Streamlit Cloud)
├── api/index.py            # FastAPI JSON API (for Vercel / serverless hosting)
├── api/requirements.txt    # Lean dependencies for the serverless function
├── vercel.json             # Vercel routing + function config
├── config.py               # Paths, API settings, weights (env-overridable)
├── data_loader.py          # Excel loading + fuzzy column detection
├── cleaner.py              # Row cleaning, text/unit normalization, search_text
├── attribute_extractor.py  # Material/size/finish/grade/scope/... extraction + comparison
├── similarity_search.py    # TF-IDF + cosine + attribute rescoring
├── deepseek_pricing.py     # DeepSeek estimator prompt + JSON parsing + fallback
├── pricing_engine.py       # Orchestrates everything: predict_price()
├── data/quotation_items.xlsx   # Historical quotation dataset
├── requirements.txt
└── tests/test_similarity.py
```

## Requirements

- Python 3.10+ (developed and tested on 3.11)
- Dependencies in `requirements.txt` (pandas, openpyxl, scikit-learn,
  streamlit, requests, pytest)

## Setup (Windows, cmd)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set DEEPSEEK_API_KEY=your_api_key_here
streamlit run streamlit_app.py
```

### PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY="your_api_key_here"
streamlit run streamlit_app.py
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=your_api_key_here
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.

## Configuring the Excel file path

By default the bot reads `data/quotation_items.xlsx` inside the project.
To point it at another file (e.g. your original location), set
`QUOTATION_DATA_PATH` before starting:

```bash
:: Windows cmd
set QUOTATION_DATA_PATH=C:\Harshal\Quotation Dataset\Turner & Townsend International Limited\BS-QT-22-20_output\quotation_items.xlsx
```

```powershell
# PowerShell
$env:QUOTATION_DATA_PATH="C:\Harshal\Quotation Dataset\Turner & Townsend International Limited\BS-QT-22-20_output\quotation_items.xlsx"
```

Other optional environment variables: `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`
(default `deepseek-chat`), `DEEPSEEK_TIMEOUT`, `DEFAULT_CURRENCY`
(default `AED`, matching the dataset).

## DeepSeek API key

Get a key from https://platform.deepseek.com and set `DEEPSEEK_API_KEY` as
shown above. The key is only read from the environment — it is never
hardcoded, printed, or included in results. **Without the key the app still
works**: it falls back to a blend of the similarity-weighted average and
median of the top historical rates, clearly labelled "Statistical fallback"
with Medium/Low confidence.

## Using the app

1. Enter a description, e.g.
   `Supply and install 50mm diameter stainless steel handrail with brushed finish`
2. Pick the number of top matches in the sidebar (default 5).
3. Click **Predict price**.

You get: predicted unit price + currency, unit, confidence, estimator
reasoning, price basis and adjustments, your input's extracted attributes,
a ranked table of similar historical items (similarity / text / attribute
scores, unit, quantity, rate, amount, category, matched and mismatched
attributes), a per-match comparison, and warnings when matches are weak.

Good sample inputs for the included dataset (street-furniture heavy):

- `Stainless steel bollard 220mm dia x 1200mm high, supply only`
- `Precast concrete bench with polished finish, 1800mm long`
- `Litter bin 610 x 590 x 910mm mild steel powder coated with GI liner`

## Programmatic use

```python
from pricing_engine import PricingEngine
engine = PricingEngine()  # or PricingEngine("path/to/file.xlsx")
result = engine.predict_price("stainless steel bollard 220mm dia", top_k=5)
print(result["predicted_unit_price"], result["currency"], result["confidence"])
```

## Deploying

### Streamlit UI → Streamlit Community Cloud (recommended for the web app)

**Vercel cannot host Streamlit** — its Python runtime only runs serverless
functions exporting an ASGI/WSGI `app`, while Streamlit is a long-running
websocket server. Deploy the UI to [Streamlit Community Cloud](https://share.streamlit.io)
(free) instead:

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Pick this repository and branch; the main file `streamlit_app.py` is
   detected automatically.
3. Under **Advanced settings → Secrets**, add
   `DEEPSEEK_API_KEY = "your_api_key_here"`.
4. Deploy — the dataset ships inside the repo, so nothing else is needed.

### JSON API → Vercel

The repo includes `api/index.py` (FastAPI, exports `app`) and `vercel.json`,
which is what Vercel's Python runtime expects. Deploying this repo to Vercel
serves a JSON API (no UI):

- `GET /health` — dataset row count and API-key status
- `POST /predict` — body `{"description": "...", "top_k": 5}` returns the
  full prediction result
- `GET /docs` — interactive Swagger UI

Set `DEEPSEEK_API_KEY` in the Vercel project settings
(**Settings → Environment Variables**), then redeploy.

Notes:
- `api/requirements.txt` keeps the function lean (no streamlit/pytest), but
  the pandas + scikit-learn + numpy/scipy stack is heavy; if the build hits
  Vercel's 250 MB unzipped function limit, the API needs a bigger plan or a
  different host (Render / Railway / Fly.io run it without size issues).
- The first request after a cold start loads and indexes the Excel file, so
  it is slower than subsequent ones.

## Running the tests

```bash
python -m pytest tests/ -v
```

Nine tests cover text/unit normalization, row cleaning, attribute
extraction, mismatch detection, ranking quality (true match beats
keyword-only overlap), result shape, empty-input rejection, and a
smoke test against the real dataset.

## Example output (fallback mode, no API key)

```text
Predicted unit price: 1,062.25 AED   Unit: Nos   Confidence: Medium
Source: Statistical fallback
Reasoning: Statistical fallback: blend of the similarity-weighted average
and the median of the top 5 historical rates. DEEPSEEK_API_KEY is not set...
```

With the API key set, DeepSeek returns structured reasoning explaining
which historical items anchored the price and what was adjusted up or down
(finish, diameter, scope, grade, etc.).

## How column detection works

The loader does not assume exact column names. It scores every sheet
against synonym lists (description / item / unit / qty / rate / unit price /
amount / category / section / location / remarks / source) and picks the
best sheet. If `rate` is absent but `amount` and `quantity` exist, it
computes `rate = amount / quantity` (skipping zero/invalid quantities).
For the included file it selects the **Products** sheet: Description,
Item Name, Unit Price→rate, Unit, Scope of Work→category, Remarks,
Source File.

## How the price is kept stable and reliable

- The **item type** (planter, litter bin, bench, bollard, handrail, ...) is
  extracted from both the input and every dataset item. A different item
  type gets its score cut by more than half, so a litter bin can never
  outrank a real planter for a planter query just because material and
  finish agree — and the pricing set is restricted to same-type items
  whenever any exist. If the dataset has no items of the input's type, a
  warning says the price comes from other item types.
- The price is **always computed from a fixed pricing set**: the strongest
  matches (within 80% of the best similarity score, up to 5 items) — completely
  independent of how many matches you choose to *display*. Changing "top
  matches" changes the table, never the price. The table marks the pricing
  set with a ✓ in the "Priced on" column. One strong comparable is preferred
  over several weak ones.
- The pricing set is **unit-aware**: a per-m² or per-metre input is never
  priced from per-item rates while same-unit rates exist, and a loud warning
  appears when the dataset has no rates in the input's unit at all.
- A deterministic **statistical anchor** (similarity-weighted median of the
  pricing set) is computed and given to DeepSeek, which must start from it
  and justify any adjustment from stated attribute differences.
- DeepSeek runs at **temperature 0** with quantified estimator guidance
  (stainless ≈ 2.5–3× mild steel, installation +15–35%, size scaling), and
  its answer is **clamped** to 0.4×min – 3×max of the pricing set's
  historical rates — wide enough for justified spec adjustments, tight
  enough to catch order-of-magnitude drift; a clamped price is flagged.
- The no-API-key fallback returns the anchor itself, so it is exactly
  reproducible.

## Known limitations and assumptions

- The included dataset is mostly street furniture/landscape items priced
  "per Nos" in AED; queries for unrelated trades (e.g. HVAC ducting) will
  produce weak matches, and the app warns you when that happens.
- The dataset has no quantity/amount columns, so quantity shows as n/a;
  the loader still supports those fields for other Excel files.
- If the input does not state a work scope, **"supply and install" is
  assumed by default** (a warning says so); write "supply only" etc. to
  override.
- Attribute extraction is regex/keyword based — unusual phrasing may miss
  an attribute (it is then simply not scored, never guessed).
- When a description mentions several materials (e.g. mild steel body with
  stainless steel fixings), the extractor picks the first/most specific one.
- Prices are predicted from your historical data only; no market indices
  or inflation adjustment.
- DeepSeek responses are validated; invalid JSON or a failed call always
  degrades to the statistical fallback rather than crashing.
