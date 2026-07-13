# Quotation Pricing Bot — Production V1

A guarded quotation-pricing assistant built from historical Excel data. The
system extracts product attributes, retrieves comparable quotation items and
either:

- returns a deterministic automatic price when the evidence passes every
  production gate; or
- returns `manual_review` with exact reasons when the data is insufficient.

The system never forces a price. DeepSeek is no longer responsible for the
final number; automatic prices come from validated, same-family, same-unit and
same-scope historical comparables.

## Current production scope

Only **planters** currently pass the grouped source-holdout accuracy gate and
are enabled for automatic pricing. Litter bins, recycle bins, benches,
bollards and bike racks remain searchable but return manual review until their
subtype data passes the same gate.

Current grouped holdout result for automatically priced planter cases:

- 35 automatic cases
- 85.7% within ±20%
- 10.2% median absolute percentage error
- zero errors outside a factor of two

Run the evaluation yourself; the result is derived from the repository data,
not hardcoded:

```bash
python scripts/evaluate_production.py --enforce-gate
```

## Safety architecture

1. Excel rows are cleaned without losing source traceability.
2. Missing-unit and contradictory exact-description rates are quarantined.
3. User-confirmed structured fields override free-text guesses.
4. Candidate retrieval is independent of how many matches the UI displays.
5. Pricing requires exact product family, unit and commercial scope.
6. Family-critical dimensions/capacity and material must be present.
7. At least three validated comparables are required.
8. Weak similarity or excessive rate dispersion triggers manual review.
9. The final price is a robust similarity-weighted median.

Quarantined rows remain visible to estimators but cannot set an automatic
price.

## Project structure

```text
├── api/index.py                  # FastAPI/Vercel entrypoint
├── api/ui.html                   # Structured browser UI
├── streamlit_app.py              # Structured Streamlit UI
├── pricing_engine.py             # Production orchestration
├── production_pricing.py         # Comparable gates, pricing and abstention
├── pricing_context.py            # User-confirmed field normalization
├── data_quality.py               # Quarantine rules and health summary
├── similarity_search.py          # Fixed-pool hybrid retrieval
├── attribute_extractor.py        # Family/specification extraction
├── cleaner.py                    # Text, boilerplate and unit cleaning
├── data_loader.py                # Excel loading and column detection
├── scripts/evaluate_production.py# Grouped holdout evaluation
├── data/quotation_items.xlsx
└── tests/
```

`deepseek_pricing.py` remains in the repository for legacy reference but is
not used to calculate Production V1 prices.

## Setup

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api.index:app --reload
```

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

The default dataset is `data/quotation_items.xlsx`. Override it with:

```bash
export QUOTATION_DATA_PATH=/path/to/quotation_items.xlsx
```

## API

`POST /predict`

```json
{
  "description": "Mild steel planter with integrated seating, 9460 x 4900 x 800mm, supply and install excluding civil works",
  "top_k": 5,
  "product_family": "planter",
  "unit": "no",
  "scope": "supply and install",
  "material": "mild steel",
  "civil_works": "excluded",
  "length_mm": 9460,
  "width_mm": 4900,
  "height_mm": 800,
  "quantity": 10
}
```

Successful automatic decision:

```json
{
  "status": "priced",
  "predicted_unit_price": 30956.0,
  "currency": "AED",
  "confidence": "Medium",
  "price_source": "Production comparable engine",
  "price_interval": {"low": 28656.0, "high": 42655.0},
  "review_reasons": []
}
```

Safe refusal:

```json
{
  "status": "manual_review",
  "predicted_unit_price": null,
  "confidence": "Manual review",
  "review_reasons": [
    "Only 2 validated same-family, same-unit, same-scope comparables remain; at least 3 are required."
  ]
}
```

`GET /health` reports usable rows, quarantined rows, defect counts and the
active pricing mode.

## Tests and validation

```bash
pytest -q
python scripts/evaluate_production.py --enforce-gate
```

Unit tests cover cleaning, attribute extraction, explicit-context overrides,
quarantine rules, deterministic comparable pricing and refusal behavior.

The evaluation is grouped by source quotation: when an item is tested, every
row from its source PDF is removed from the comparable pool. This prevents
same-quotation leakage and is the release gate for enabling a product family.

## Adding another automatic family

1. Correct missing units and suspicious rates for that family.
2. Add family-specific required fields and subtype attributes.
3. Add structured UI/API inputs for those fields.
4. Run grouped holdout evaluation.
5. Enable the family in `APPROVED_AUTO_FAMILIES` only after it achieves at
   least 80% within ±20% over at least 25 automatic holdout cases.

Until then, the family remains available for comparable discovery and manual
estimator review.
