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

import config
from cleaner import EmptyDatasetError
from data_loader import DataLoadError
from similarity_search import SearchError

app = FastAPI(
    title="Quotation Pricing Bot API",
    description="Predict a unit price for a new item from historical "
                "quotation data using similarity search + DeepSeek.",
    version="1.0.0",
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
        except (DataLoadError, EmptyDatasetError) as exc:
            _engine_error = str(exc)
    if _engine is None:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {_engine_error}")
    return _engine


class PredictRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=2000,
                             description="Item or service description to price")
    top_k: int = Field(5, ge=1, le=15, description="Number of similar items to use")


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
        "endpoints": {"POST /predict": "predict a unit price",
                      "GET /health": "dataset and API-key status"},
        "docs": "/docs",
    }


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "dataset_rows": len(engine.dataset),
        "sheet": engine.sheet,
        "deepseek_configured": bool(config.DEEPSEEK_API_KEY),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    engine = get_engine()
    try:
        return engine.predict_price(req.description, top_k=req.top_k)
    except (ValueError, SearchError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
