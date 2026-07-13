# Quotation Pricing Bot — V2 Safety Rebuild

A guarded quotation-pricing assistant built from historical Excel data. The
system extracts product attributes, retrieves comparable quotation items and
either:

- returns a deterministic automatic price when the evidence passes every
  production gate; or
- returns `manual_review` with exact reasons when the data is insufficient.

The system never forces a price. DeepSeek is no longer responsible for the
final number; automatic prices come from validated, same-family, same-unit and
same-scope historical comparables.

V2 shadow infrastructure is active: versioned estimator corrections,
family-specific subtype contracts, a correction review queue and full
quotation-lineage holdout scoring across every supported family. PDF
revisions are grouped under one quotation ID and superseded revisions are
quarantined from pricing evidence.

## Current production scope

Automatic pricing is currently **paused for every family**. The earlier
planter result treated R1/R2 copies of the same quotation as independent
evidence; the corrected lineage-held-out evaluation produces no qualifying
planter holdouts. Keeping planter enabled would therefore claim validation
that the data does not support.

The application remains usable for guarded comparable discovery and explicit
manual review. A family is re-enabled only after it passes all accuracy,
coverage, large-error and independent-quotation gates.

Run the evaluation yourself; the result is derived from the repository data,
not hardcoded:

```bash
python scripts/evaluate_production.py --enforce-gate
python scripts/evaluate_families.py --enforce-production --check-snapshot
```

## Safety architecture

1. Excel rows are cleaned without losing source traceability.
2. R1/R2 revisions share one quotation lineage; superseded prices are blocked.
3. Missing-unit and contradictory exact-description rates are quarantined.
4. User-confirmed structured fields override free-text guesses.
5. Candidate retrieval is independent of how many matches the UI displays.
6. Pricing requires exact product family, unit and commercial scope.
7. Family-critical dimensions/capacity and material must be present.
8. At least three validated comparables are required.
9. Weak similarity or excessive rate dispersion triggers manual review.
10. The final price is a robust similarity-weighted median.

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
├── data_corrections.py           # Approved estimator correction overlay
├── pricing_dataset.py            # One load/clean/correct/quality pipeline
├── family_readiness.py           # V2 quotation-lineage-held-out scorecard
├── correction_queue.py           # Stable review queue for missing fields
├── similarity_search.py          # Fixed-pool hybrid retrieval
├── attribute_extractor.py        # Family/specification extraction
├── cleaner.py                    # Text, boilerplate and unit cleaning
├── data_loader.py                # Excel loading and column detection
├── scripts/evaluate_production.py# Quotation-lineage holdout evaluation
├── scripts/evaluate_families.py  # All-family V2 shadow evaluation
├── scripts/export_correction_queue.py
├── data/pricing_corrections.json # Versioned estimator decisions
├── data/family_readiness.json    # Audited readiness snapshot
├── data/quotation_items.xlsx
└── tests/
```

`deepseek_pricing.py` remains in the repository for legacy reference but is
not used to calculate guarded pricing decisions.

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

Current safe decision while no family is approved:

```json
{
  "status": "manual_review",
  "predicted_unit_price": null,
  "currency": "AED",
  "confidence": "Manual review",
  "price_source": "Historical comparable evidence (manual review)",
  "price_interval": null,
  "review_reasons": [
    "Product family 'planter' has not yet passed the quotation-lineage holdout accuracy gate; comparables are shown for manual review."
  ]
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

`GET /readiness` returns the current V2 family scorecard, release-gate
failures, required fields and known subtypes. `snapshot_current` is false if
the dataset/corrections changed without regenerating the audited snapshot.

## V2 correction workflow

The source workbook is immutable. Export a review queue, confirm values with
an estimator, then add only reviewed decisions to the correction manifest:

```bash
python scripts/export_correction_queue.py --family bench --blocking-only \
  --format csv --output bench-review.csv
```

Every manifest record is keyed by a stable `record_id`. Proposed records have
no production effect. An approved record requires a reason and `reviewed_by`;
it can correct source fields, override structured attributes or explicitly
exclude a row from automatic pricing.

## Tests and validation

```bash
pytest -q
python scripts/evaluate_production.py --enforce-gate
```

Unit tests cover cleaning, attribute extraction, explicit-context overrides,
quarantine rules, deterministic comparable pricing and refusal behavior.

The evaluation is grouped by quotation ID: when an item is tested, every row
from the quotation and all of its PDF revisions are removed from the
comparable pool. This prevents revision leakage and is the release gate for
enabling a product family.

## Adding another automatic family

1. Correct missing units and suspicious rates for that family.
2. Add family-specific required fields and subtype attributes.
3. Add structured UI/API inputs for those fields.
4. Run quotation-lineage holdout evaluation.
5. Enable the family in `APPROVED_AUTO_FAMILIES` only after it achieves at
   least 80% within ±20% over at least 25 automatic holdout cases from at
   least five independent quotation groups, with no factor-of-two errors.

Until then, the family remains available for comparable discovery and manual
estimator review.
