# ============================================================
# api/main.py
# Kroniq Regime Radar — FastAPI
#
# Mode: LIVE
#   - Data downloaded through today at startup
#   - Model trained on fixed 2015-2020 window (OOS integrity)
#   - Data does NOT auto-refresh after startup
#   - Refresh requires app restart
#
# Endpoints:
#   GET  /regime          — latest regime from today's data
#   GET  /regime/history  — backtested walk-forward 2021-2024
#   POST /regime/explain  — rule-based explanation template
#   GET  /health          — readiness check
#
# Week 6 — April 13 2026
# ============================================================

import os
import logging
from datetime import date
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kroniq_regime_radar_v5 import (
    download_prices,
    download_credit_spreads,
    build_features,
    prepare_static_model,
    get_current_regime,
    walk_forward,
    ROUTE,
    TRAIN_START,
    TRAIN_END,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── Config ───────────────────────────────────────────────────
# Dev:  CORS_ORIGINS=*  (default)
# Prod: CORS_ORIGINS=https://kroniq.finance
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Model Cache ──────────────────────────────────────────────
model_cache: dict = {}
# Keys after successful startup:
#   model_bundle — output of prepare_static_model()
#   features     — full DataFrame (TRAIN_START → today)
#   wf_labels    — backtested walk-forward labels 2021-2024
#   ready        — bool
#   mode         — "live"
#   started_at   — ISO date string of when data was loaded

# ── Pydantic Response Models ─────────────────────────────────
class PosteriorItem(BaseModel):
    state_id:    int
    regime:      str
    probability: float

class RegimeResponse(BaseModel):
    regime:             str
    confidence:         float
    allocation:         float
    signal_driver_hint: str       # rule-based heuristic, not feature attribution
    all_posteriors:     list[PosteriorItem]
    as_of:              str
    as_of_note:         str
    model_train_end:    str
    mode:               str

class HistoryItem(BaseModel):
    date:       str
    regime:     str
    allocation: float

class HistoryResponse(BaseModel):
    days_returned: int
    source:        str
    period:        str
    history:       list[HistoryItem]

class ExplainRequest(BaseModel):
    regime: str = Field(..., example="Macro")

class ExplainResponse(BaseModel):
    regime:              str
    summary:             str
    drivers:             list[str]
    allocation:          str
    historical_analogue: str
    explanation_type:    str       # "rule-based template"
    model:               str

class HealthResponse(BaseModel):
    status:     str
    ready:      bool
    mode:       Optional[str] = None
    model:      Optional[str] = None
    trained_on: Optional[int] = None
    train_end:  Optional[str] = None
    as_of:      Optional[str] = None
    started_at: Optional[str] = None   # when live data was loaded at startup

# ── Explanations ─────────────────────────────────────────────
EXPLANATIONS = {
    "Crisis": {
        "summary":             "Acute market stress. VIX > 48, credit spreads > 7.8%. Full cash.",
        "drivers":             ["VIX spike above 30", "Credit spread OAS > 6.5%", "Cross-asset stress confirmed"],
        "allocation":          "0% SPY — full cash",
        "historical_analogue": "March 2020 COVID crash",
    },
    "Macro": {
        "summary":             "Chronic macro stress. Negative equity returns, elevated credit spreads.",
        "drivers":             ["Negative SPY return", "VIX elevated above 18", "Credit spread widening"],
        "allocation":          "30% SPY — defensive",
        "historical_analogue": "2022 Federal Reserve rate shock",
    },
    "Low-Vol": {
        "summary":             "Extreme calm. VIX compressed, tight credit, low volatility clustering.",
        "drivers":             ["VIX below 15", "Low volatility clustering", "Tight credit conditions"],
        "allocation":          "100% SPY — fully invested",
        "historical_analogue": "2017 low volatility regime",
    },
    "Bull": {
        "summary":             "Strong bull market. Highest return state, positive cross-asset momentum.",
        "drivers":             ["Highest annualised return", "Moderate VIX", "Positive equity momentum"],
        "allocation":          "100% SPY — fully invested",
        "historical_analogue": "2021 post-COVID recovery",
    },
    "Neutral": {
        "summary":             "Transitional conditions. Mixed signals, residual state.",
        "drivers":             ["No dominant signal", "Transitional cross-asset environment"],
        "allocation":          "70% SPY — moderate",
        "historical_analogue": "Periods between major regime transitions",
    },
}

# ── Startup ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load data and model at startup. Clear cache on shutdown.
    LIVE mode: downloads data through today.
    Data does not auto-refresh — restart required for fresh data.
    """
    model_cache["ready"] = False
    today = date.today().isoformat()

    try:
        log.info("=" * 55)
        log.info("Kroniq API startup — LIVE mode")
        log.info(f"Downloading data through {today}")
        log.info("Model will be trained on fixed 2015-2020 window")
        log.info("=" * 55)

        prices   = download_prices(TRAIN_START, today)
        cs       = download_credit_spreads(TRAIN_START, today)
        features = build_features(prices, cs)

        log.info(
            f"Features: {features.index[0].date()} "
            f"→ {features.index[-1].date()}"
        )

        bundle = prepare_static_model(features, TRAIN_END)

        log.info("Pre-computing walk-forward history (2021-2024)...")
        features_oos = features[features.index <= "2024-12-31"]
        n_tr         = len(features_oos[features_oos.index <= TRAIN_END])
        wf_labels    = walk_forward(features_oos, n_tr)

        model_cache["model_bundle"] = bundle
        model_cache["features"]     = features
        model_cache["wf_labels"]    = wf_labels
        model_cache["ready"]        = True
        model_cache["mode"]         = "live"
        model_cache["started_at"]   = today

        log.info("✓ Kroniq API ready — serving live regime")

    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        model_cache["ready"] = False
        # App starts in degraded mode — /health returns 503

    yield

    model_cache.clear()
    log.info("Kroniq API shutdown — cache cleared")

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Kroniq Regime Radar API",
    description=(
        "Cross-asset HMM regime detection. "
        "K=5 | 9 features. "
        "Model trained on fixed 2015-2020 window. "
        "Live inference on latest available data downloaded at startup. "
        "Walk-forward history: backtested 2021-2024 (1005 OOS days). "
        "Refresh requires app restart."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Readiness Guard ──────────────────────────────────────────
def require_ready():
    if not model_cache.get("ready", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not ready. "
                "Startup may still be in progress — retry in 60s."
            )
        )

# ── GET /regime ──────────────────────────────────────────────
@app.get("/regime", response_model=RegimeResponse)
def get_regime():
    """
    Returns the latest available market regime classification.

    Mode: LIVE.
    - Model trained on fixed 2015-2020 window (preserves OOS integrity).
    - Features include data through the latest available trading day
      as of startup time — not necessarily today's date.
    - as_of reflects the latest fully aligned feature date.
      FRED credit spreads update with 1-3 day lag, so as_of may
      trail today's date by up to 4 calendar days.
    - Does not auto-refresh intraday — restart for fresh data.
    - signal_driver_hint is a rule-based heuristic, not dynamic
      feature attribution. Dynamic attribution planned for Week 9.
    """
    require_ready()
    try:
        result = get_current_regime(
            model_cache["features"],
            model_cache["model_bundle"],
        )
        return RegimeResponse(
            regime             = result["regime"],
            confidence         = result["confidence"],
            allocation         = result["allocation"],
            signal_driver_hint = result["signal_driver_hint"],
            all_posteriors     = [
                PosteriorItem(**p) for p in result["all_posteriors"]
            ],
            as_of           = result["as_of"],
            as_of_note      = result["as_of_note"],
            model_train_end = model_cache["model_bundle"]["train_end"],
            mode            = model_cache.get("mode", "live"),
        )
    except Exception as e:
        log.error(f"/regime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Regime inference failed."
        )

# ── GET /regime/history ──────────────────────────────────────
@app.get("/regime/history", response_model=HistoryResponse)
def get_regime_history(
    days: int = Query(
        default=252,
        ge=5,
        le=1005,
        description=(
            "Number of trading days to return (5–1005). "
            "Full OOS period = 1005 days (2021-2024)."
        )
    )
):
    """
    Returns backtested walk-forward regime history.

    Source: pre-computed at startup via quarterly expanding-window
    retraining over 2021-2024 (1005 OOS trading days total).
    Clearly labeled as backtested — not a live per-request replay.
    """
    require_ready()
    try:
        wf = model_cache["wf_labels"].tail(days)
        history = [
            HistoryItem(
                date       = d.strftime("%Y-%m-%d"),
                regime     = lbl,
                allocation = ROUTE.get(lbl, 0.5),
            )
            for d, lbl in wf.items()
        ]
        return HistoryResponse(
            days_returned = len(history),
            source        = "backtested walk-forward (quarterly retraining)",
            period        = "2021-2024",
            history       = history,
        )
    except Exception as e:
        log.error(f"/regime/history error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="History retrieval failed."
        )

# ── POST /regime/explain ─────────────────────────────────────
@app.post("/regime/explain", response_model=ExplainResponse)
def explain_regime(req: ExplainRequest):
    """
    Returns a rule-based explanation template for a given regime.

    Note: explanations are static templates derived from the
    model's training-period state profiles. They are NOT dynamic
    feature attribution — the same explanation is returned for a
    given regime label regardless of current feature values.
    Dynamic feature attribution (Engine 3) is planned for Week 9.
    """
    require_ready()
    ex = EXPLANATIONS.get(req.regime)
    if not ex:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown regime '{req.regime}'. "
                f"Valid values: {list(EXPLANATIONS.keys())}"
            )
        )
    return ExplainResponse(
        regime               = req.regime,
        summary              = ex["summary"],
        drivers              = ex["drivers"],
        allocation           = ex["allocation"],
        historical_analogue  = ex["historical_analogue"],
        explanation_type     = "rule-based template (static, not dynamic feature attribution)",
        model                = (
            "Kroniq Regime Radar v5 | K=5 | "
            "9-feature cross-asset HMM"
        ),
    )

# ── GET /health ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Readiness check.
    Returns 200 with ready=True if model is loaded.
    Returns 503 with ready=False if startup failed or in progress.
    Response shape is consistent in both branches.
    started_at shows when live data was downloaded at startup.
    """
    ready  = model_cache.get("ready", False)
    bundle = model_cache.get("model_bundle")
    feats  = model_cache.get("features")

    response = HealthResponse(
        status     = "ok" if ready else "unavailable",
        ready      = ready,
        mode       = model_cache.get("mode")         if ready else None,
        model      = "kroniq_regime_radar_v5"         if ready else None,
        trained_on = bundle.get("trained_on")         if bundle else None,
        train_end  = bundle.get("train_end")          if bundle else None,
        as_of      = (
            feats.index[-1].strftime("%Y-%m-%d")
            if feats is not None else None
        ),
        started_at = model_cache.get("started_at")   if ready else None,
    )

    if not ready:
        return JSONResponse(
            status_code=503,
            content=response.model_dump()
        )

    return response