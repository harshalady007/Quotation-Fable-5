# Quotation Pricing Bot — V4 Workflow Foundation

A deterministic quotation-pricing assistant built from historical Excel data.
The system extracts product attributes, retrieves comparable quotation items
and returns a numeric unit-price estimate for every valid request. Each result
includes an evidence tier, confidence, price interval and exact warnings when
the available history is sparse or incomplete.

DeepSeek is no longer responsible for the final number. Estimates use only
quality-eligible historical rates, never mix unit bases, and prefer validated
same-family, same-unit and same-scope comparables. When strict evidence is not
available, the engine uses a deterministic fallback and labels the additional
uncertainty instead of returning `manual_review`.

V2 safety infrastructure remains active: versioned estimator corrections,
family-specific subtype contracts, a correction review queue and full
quotation-lineage holdout scoring across every supported family. PDF
revisions are grouped under one quotation ID and superseded revisions are
quarantined from pricing evidence.

V3 shadow infrastructure adds a context data contract for quantity, supplier,
quotation date and project location. Quote-level project/client/contractor and
currency metadata are safely joined from the workbook's `Quotation Summary`
sheet. Context fields are recorded but cannot alter a price until they have
enough matched-product evidence, pass lineage-held-out calibration and are
explicitly approved.

V3.1 adds the operational evidence pipeline: a spreadsheet review queue keyed
by stable row IDs, dataset-fingerprint protection against stale imports,
field-level reviewer/source metadata, atomic correction-manifest writes and a
dry-run-first importer. Reviewed context is never read from filenames or
guessed from descriptions.

V3.2 adds an offline-only model evaluator for every family/context pair. It
compares a fixed-hyperparameter context model against an exact-product cohort
median with the complete quotation lineage held out. Quotation-date models
and every other context model use only strictly earlier training evidence. The
evaluator persists metrics, not fitted models, and cannot change a
request-time price.

V4.1 starts the complete quotation workflow without weakening the V3 pricing
controls. It adds transactional users, projects, quotations, immutable
revisions, four-eyes approvals, exact decimal totals, optimistic concurrency,
payload-bound idempotency and a tamper-evident audit chain. The workflow core
uses SQLite for explicit local development only. No workflow mutation route
is exposed, and the Vercel filesystem is never treated as
durable quotation storage.

## Current production scope

Numeric estimates are enabled for **every supported family**. None of the
families currently passes the independent quotation-lineage accuracy gate, so
the API does not claim that these estimates are validated production prices.
The earlier planter result treated R1/R2 copies of the same quotation as
independent evidence; the corrected holdout evaluation still records that
limitation.

Request-time pricing and offline validation are intentionally separate. The
request path always produces the best available deterministic estimate. The
readiness scorecard continues to report the real accuracy, coverage,
large-error and independent-quotation results without converting a user policy
decision into a false validation claim.

On the current 772-row lineage-held-out coverage audit, the universal policy
returns a positive estimate for 772/772 rows and preserves all 727 stated unit
bases. Accuracy is still weak: 29.53% of estimates are within ±20%, median
absolute percentage error is 41.5%, and 246 cases miss by more than a factor
of two. Consequently, no family is release-gate approved and request-time
results remain `Low` or `Very low` confidence until the dataset improves.

No V3 context adjustment is currently approved. In the present workbook,
quantity, supplier and location have no usable line-level values. Quotation
dates are fully normalized, but the stricter exact-specification cohort check
finds only one qualifying cohort overall; the evidence gate requires at least
three. All 24 family/context model candidates are therefore blocked by data.
The result is a recorded field plus an explanatory warning; unsupported
context fields still use a neutral `1.0` multiplier.

The V4 workflow is currently at the **foundation** stage. The deployed API may
report its capabilities through `GET /workflow/readiness`, but workflow writes
are intentionally local-only. Cloud authentication, project membership and a
serverless workflow database are out of scope. PDF/Excel document generation
and the local estimator workspace follow in later V4 work.

Run the evaluation yourself; the result is derived from the repository data,
not hardcoded:

```bash
python scripts/evaluate_production.py --enforce-gate
python scripts/evaluate_families.py --enforce-production --check-snapshot
python scripts/evaluate_context_models.py --enforce-production --check-snapshot
```

## Safety architecture

1. Excel rows are cleaned without losing source traceability.
2. R1/R2 revisions share one quotation lineage; superseded prices are blocked.
3. Missing-unit and contradictory exact-description rates are quarantined.
4. User-confirmed structured fields override free-text guesses.
5. Candidate retrieval is independent of how many matches the UI displays.
6. Pricing requires exact product family, unit and commercial scope.
7. Family-critical dimensions/capacity and material must be present.
8. Strict pricing prefers at least three validated comparables; otherwise the
   fallback uses the strongest coherent quality-eligible evidence.
9. Weak similarity or excessive rate dispersion lowers confidence and widens
   the returned price interval.
10. Every estimate is a robust similarity-weighted median.
11. V3 context effects require minimum coverage, independent quotation groups,
    multiple matched product cohorts and explicit production approval.
12. V4 submitted quotation revisions are immutable and use decimal money.
13. The creator/submitter cannot approve their own quotation revision.
14. Every mutation requires an expected version and payload-bound idempotency
    key, and is appended to a hash-chained audit log in the same transaction.

Quarantined rows remain visible to estimators but cannot set an automatic
price.

## Project structure

```text
├── api/index.py                  # FastAPI/Vercel entrypoint
├── api/ui.html                   # Structured browser UI
├── streamlit_app.py              # Structured Streamlit UI
├── pricing_engine.py             # Production orchestration
├── production_pricing.py         # Strict gates and universal estimate fallback
├── pricing_context.py            # User-confirmed field normalization
├── context_readiness.py          # V3 context evidence and activation gates
├── context_enrichment.py         # Audited context export/import workflow
├── context_modeling.py           # Group/time-held-out shadow evaluation
├── quotation_workflow/
│   ├── domain.py                 # V4 validation, money and audit primitives
│   ├── store.py                  # Transactional SQLite workflow repository
│   ├── readiness.py              # Safe capability/blocker reporting
│   └── schema.sql                # V4 schema version 1
├── data_quality.py               # Quarantine rules and health summary
├── data_corrections.py           # Approved estimator correction overlay
├── pricing_dataset.py            # One load/clean/correct/quality pipeline
├── family_readiness.py           # V3 family/context readiness scorecard
├── correction_queue.py           # Stable review queue for missing fields
├── similarity_search.py          # Fixed-pool hybrid retrieval
├── attribute_extractor.py        # Family/specification extraction
├── cleaner.py                    # Text, boilerplate and unit cleaning
├── data_loader.py                # Excel loading and column detection
├── scripts/evaluate_production.py# Quotation-lineage holdout evaluation
├── scripts/evaluate_families.py  # V3 family/context shadow evaluation
├── scripts/export_correction_queue.py
├── scripts/export_context_review.py
├── scripts/import_context_reviews.py
├── scripts/evaluate_context_models.py
├── scripts/init_workflow_db.py   # Explicit local V4 store bootstrap
├── data/pricing_corrections.json # Versioned estimator decisions
├── data/family_readiness.json    # Audited readiness snapshot
├── data/context_model_readiness.json
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

### Local V4 workflow store

Initialize a local database explicitly; there is intentionally no default
runtime database:

```bash
python scripts/init_workflow_db.py \
  --database runtime/quotation-workflow.db \
  --bootstrap-email admin@example.com \
  --bootstrap-name "Workflow Admin" \
  --request-id bootstrap-admin-v4-001
export QUOTATION_WORKFLOW_DB_PATH=runtime/quotation-workflow.db
```

The initializer is idempotent when the same request ID and payload are used.
Database, WAL and shared-memory files are ignored by Git because they contain
commercial quotation state and user identities. SQLite is suitable for local
workflow development and tests; do not point the Vercel deployment at a local
SQLite path.

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
  "quantity": 10,
  "supplier": "Example supplier",
  "location": "Dubai",
  "quotation_date": "2026-07-13"
}
```

Estimate response while no family is historically release-approved:

```json
{
  "pricing_policy": "always_estimate",
  "status": "priced",
  "predicted_unit_price": 11000.00,
  "currency": "AED",
  "confidence": "Low",
  "price_source": "Universal validated-comparable estimate",
  "price_interval": {"low": 8250.00, "high": 13750.00},
  "evidence_tier": "same_family_unit_scope_material_spec_near",
  "estimate_warnings": [
    "Product family 'planter' has not passed the historical quotation-lineage accuracy gate; this numeric result is an estimate under the always-estimate policy."
  ]
}
```

Sparse-evidence response:

```json
{
  "status": "priced",
  "predicted_unit_price": 3650.00,
  "confidence": "Very low",
  "price_interval": {"low": 1825.00, "high": 5475.00},
  "estimate_warnings": [
    "Estimate fallback tier: same_unit_cross_family_scope_adjusted; 2 independent quality-eligible quotation lineages set the price."
  ]
}
```

`GET /health` reports usable rows, quarantined rows, defect counts and the
active pricing mode.

`GET /readiness` returns the current V3 family scorecard, release-gate
failures, required fields, known subtypes and per-family contextual-evidence
profiles plus the offline context-model scorecard. `snapshot_current` and
`context_model_snapshot_current` become false if the dataset/corrections
changed without regenerating the corresponding audited snapshots.

`GET /workflow/readiness` returns the V4.1 workflow version, implemented
transaction guarantees, configured-store integrity status and local-only
deployment constraints. It is read-only. V4.1 deliberately exposes no
create/edit/submit/approve HTTP endpoints; those commands are used only by
local workflow tools against an explicitly configured SQLite store.

## V4 quotation workflow

The workflow state machine is deliberately small and auditable:

- `draft` revisions may have their line-item set replaced by their owner or an
  administrator;
- `submitted` revisions are locked and await an independent approver;
- an approver may mark the revision `approved` or `rejected`, but never their
  own revision;
- an approved or rejected quotation may create a new draft revision by
  copying the prior immutable item snapshot.

Every command carries `expected_version` to prevent lost updates and a
`request_id` bound to the normalized command payload. Retrying the same
request returns its original result; reusing the key for different data is a
conflict. Line totals and subtotals use fixed-scale `Decimal` values. Audit
events are chained by SHA-256 and written in the same transaction as the
business change; `verify_integrity()` checks SQLite, foreign keys, current
state alignment, line totals, revision digests and the complete audit chain.

Remaining V4 delivery:

1. **V4.1 (current):** workflow domain, SQLite repository, revisions,
   approvals, concurrency, idempotency, audit verification and read-only
   readiness.
2. **Local workspace:** estimator quotation editor, approval controls and
   revision comparison using the local SQLite store.
3. **Documents:** approved PDF rendering, Excel import/export, attachments and
   source-page evidence.
4. **Hardening:** local backup/restore, retention and release
   hardening before V5 monitoring/continuous ingestion begins.

## Estimator correction workflow

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

## V3 context-enrichment workflow

Export the highest-value current-revision rows to an Excel template:

```bash
python scripts/export_context_review.py --family bench --format xlsx \
  --output bench-context-review.xlsx
```

An estimator must verify values against the original quotation, fill one or
more `reviewed_*` fields, set `review_status` to `approved`, and provide an
evidence reference, reason, reviewer and review date. Quantity is line-level;
supplier, location and quotation date may repeat across rows only when the
source document confirms that they apply to those rows.

Validate the completed file without changing the manifest:

```bash
python scripts/import_context_reviews.py bench-context-review.xlsx
```

After reviewing the dry-run summary, apply it atomically:

```bash
python scripts/import_context_reviews.py bench-context-review.xlsx --write
```

The importer refuses stale dataset fingerprints, missing record IDs,
placeholder values, undocumented approvals, implicit overwrites, conflicting
approved values and merges that would accidentally activate an unrelated
proposed correction. Schema-2 correction records keep evidence metadata per
context field.

## V3 shadow-model evaluation

Regenerate the deterministic offline scorecard after reviewed context data or
pricing code changes:

```bash
python scripts/evaluate_families.py \
  --output data/family_readiness.json --quiet
python scripts/evaluate_context_models.py \
  --output data/context_model_readiness.json --quiet
```

Each candidate adjustment is scoped as `family.field`, such as
`bench.quantity`. For every field, the evaluator removes the target quotation,
all its PDF revisions and every quotation on or after the target date.
Duplicate line evidence is collapsed to one median observation per
quotation/cohort/context value; training and evaluation metrics give each
quotation equal total weight so a long quotation cannot dominate the result.

The candidate predicts a bounded factor on top of the held-out exact-product
cohort median. It must independently satisfy minimum cases, quotation groups,
cohorts and evaluation coverage; achieve at least 80% within ±20%; stay below
15% median APE and 40% p90 APE with no factor-of-two errors; improve median
APE by at least two percentage points; improve at least 55% of cases; and
avoid depending on the factor clamp. The base family must separately pass its
production pricing gate. No hyperparameter search is performed on evaluation
data.

Passing this scorecard still does not activate pricing. Activation requires a
family-scoped implementation and explicit allow-list approval, and the base
family must already be approved.

## Tests and validation

```bash
pytest -q
python scripts/evaluate_production.py --enforce-gate
python scripts/evaluate_families.py --enforce-production --check-snapshot --quiet
python scripts/evaluate_context_models.py --enforce-production --check-snapshot --quiet
```

Unit tests cover cleaning, attribute extraction, explicit-context overrides,
quarantine rules, deterministic comparable pricing, refusal behavior,
family-scoped activation safety, quotation-balanced metrics, leakage-safe
rolling context-model evaluation, transactional quotation lifecycle,
four-eyes approval, immutable revisions, decimal totals, idempotency,
optimistic concurrency and audit tamper detection.

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
