"""FastAPI entrypoint for serverless deployment (e.g. Vercel).

Vercel's Python runtime requires a module exporting a top-level `app`
(ASGI). The Streamlit UI (streamlit_app.py) cannot run on Vercel; this
API exposes the same pricing engine as JSON endpoints instead.

Run locally with:  uvicorn api.index:app --reload
"""

import sys
from pathlib import Path

# Make the project root importable when Vercel runs this file from api/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cleaner import EmptyDatasetError
from context_readiness import ContextReadinessError, compact_context_readiness
from context_modeling import (ContextModelEvaluationError,
                              compact_context_model_readiness,
                              load_context_model_snapshot)
from data_corrections import DataCorrectionError
from data_loader import DataLoadError
from family_readiness import dataset_fingerprint, load_readiness_snapshot
from production_pricing import (APPROVED_AUTO_FAMILIES,
                                ESTIMATE_ENABLED_FAMILIES,
                                PRICING_ENGINE_VERSION, PRICING_POLICY)
from quotation_workflow.domain import WORKFLOW_VERSION
from quotation_workflow.readiness import (compact_workflow_readiness,
                                           workflow_readiness as
                                           build_workflow_readiness)
from similarity_search import SearchError

app = FastAPI(
    title="Quotation Pricing Bot API",
    description="Deterministic unit-price estimates from quality-eligible "
                "historical quotation evidence, with explicit uncertainty.",
    version=PRICING_ENGINE_VERSION,
)

_engine = None
_engine_error = None


def get_engine():
    """Lazy singleton so the dataset is loaded/indexed once per instance."""
    global _engine, _engine_error
    if _engine is None and _engine_error is None:
        try:
            from pricing_engine import PricingEngine
            _engine = PricingEngine()
        except (DataLoadError, EmptyDatasetError, DataCorrectionError,
                ContextReadinessError) as exc:
            _engine_error = str(exc)
    if _engine is None:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {_engine_error}")
    return _engine


class PredictRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=2000,
                             description="Item or service description to price")
    top_k: int = Field(5, ge=1, le=15, description="Number of similar items to use")
    product_family: str | None = Field(None, max_length=100)
    subtype: str | None = Field(None, max_length=100)
    unit: str | None = Field(None, max_length=30)
    scope: str | None = Field(None, max_length=100)
    material: str | None = Field(None, max_length=100)
    supplier: str | None = Field(None, max_length=200)
    location: str | None = Field(None, max_length=200)
    quotation_date: str | None = Field(None, max_length=50)
    civil_works: bool | str | None = None
    quantity: float | None = Field(None, gt=0)
    capacity_l: float | None = Field(None, gt=0)
    compartments: int | None = Field(None, gt=0, le=20)
    length_mm: float | None = Field(None, gt=0)
    width_mm: float | None = Field(None, gt=0)
    height_mm: float | None = Field(None, gt=0)
    diameter_mm: float | None = Field(None, gt=0)
    thickness_mm: float | None = Field(None, gt=0)
    mobility: str | None = Field(None, max_length=30)
    features: list[str] | None = None

    def pricing_context(self) -> dict:
        fields = (
            "product_family", "subtype", "unit", "scope", "material",
            "supplier", "location", "quotation_date",
            "civil_works", "quantity", "capacity_l", "compartments",
            "length_mm", "width_mm", "height_mm", "diameter_mm",
            "thickness_mm", "mobility", "features",
        )
        return {name: getattr(self, name) for name in fields
                if getattr(self, name) not in (None, "", [])}


_UI_PATH = Path(__file__).resolve().parent / "ui.html"


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the single-file web UI."""
    try:
        return _UI_PATH.read_text(encoding="utf-8")
    except OSError:
        return HTMLResponse(
            "<h1>Quotation Pricing Bot API</h1>"
            "<p>UI file missing. Use <a href='/docs'>/docs</a> to call the "
            "API directly.</p>", status_code=200)


@app.get("/api")
def api_info():
    return {
        "service": "Quotation Pricing Bot API",
        "pricing_version": PRICING_ENGINE_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "pricing_policy": PRICING_POLICY,
        "endpoints": {
            "POST /predict": "numeric unit-price estimate for every valid request",
            "GET /health": "dataset and quality-gate status",
            "GET /readiness": "V3 family and context shadow scorecard",
            "GET /workflow/readiness": (
                "V4 workflow capability and safety status"
            ),
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    engine = get_engine()
    try:
        context_model_snapshot = load_context_model_snapshot()
        context_models = compact_context_model_readiness(
            context_model_snapshot
        )
        context_models["snapshot_current"] = (
            context_model_snapshot.get("dataset_fingerprint")
            == dataset_fingerprint(engine.dataset)
        )
    except ContextModelEvaluationError as exc:
        context_models = {"mode": "unavailable", "error": str(exc)}
    return {
        "status": "ok",
        "dataset_rows": len(engine.dataset),
        "sheet": engine.sheet,
        "summary_context": engine.summary_context,
        "pricing_mode": "universal deterministic estimates",
        "pricing_policy": PRICING_POLICY,
        "manual_review_enabled": False,
        "estimate_enabled_families": sorted(ESTIMATE_ENABLED_FAMILIES),
        "approved_auto_families": sorted(APPROVED_AUTO_FAMILIES),
        "release_gate_approved_families": sorted(APPROVED_AUTO_FAMILIES),
        "pricing_version": PRICING_ENGINE_VERSION,
        "data_quality": engine.data_quality,
        "context_adjustments": compact_context_readiness(
            engine.context_readiness
        ),
        "context_model_evaluation": context_models,
        "quotation_workflow": compact_workflow_readiness(),
    }


@app.get("/workflow/readiness")
def workflow_readiness():
    """Return V4 capabilities/blockers without enabling workflow writes."""
    return build_workflow_readiness()


@app.get("/readiness")
def readiness():
    """Return the audited V3 snapshot; never changes production allow-lists."""
    engine = get_engine()
    try:
        snapshot = load_readiness_snapshot()
        context_models = load_context_model_snapshot()
    except (ValueError, ContextModelEvaluationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    current = dataset_fingerprint(engine.dataset)
    return {
        **snapshot,
        "snapshot_current": snapshot.get("dataset_fingerprint") == current,
        "current_dataset_fingerprint": current,
        "context_model_readiness": context_models,
        "context_model_snapshot_current": (
            context_models.get("dataset_fingerprint") == current
        ),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    engine = get_engine()
    try:
        return engine.predict_price(
            req.description,
            top_k=req.top_k,
            pricing_context=req.pricing_context(),
        )
    except (ValueError, SearchError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
