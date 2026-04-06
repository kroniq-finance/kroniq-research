# ============================================================
# api/main.py
# Kroniq Regime Radar — FastAPI
#
# Mode: LIVE
#   - Data downloaded through latest available trading day at startup
#   - Model trained on fixed 2015-2020 window (OOS integrity)
#   - Data does NOT auto-refresh after startup
#   - Refresh requires app restart
#
# Auth: API key via X-API-Key header
# Rate limiting: 100 requests/day per key (UTC day, in-memory)
#
# Endpoints:
#   GET  /regime          — latest regime from data downloaded at startup
#   GET  /regime/history  — backtested walk-forward 2021-2024
#   POST /regime/explain  — rule-based explanation template
#   GET  /health          — readiness check (no auth required)
#   GET  /usage           — quota status for your key
#
# Week 6 — April 15 2026
# ============================================================

import os
import hashlib
import logging
from datetime import date, datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
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
CORS_ORIGINS   = os.getenv("CORS_ORIGINS", "*").split(",")
RATE_LIMIT_DAY = int(os.getenv("RATE_LIMIT_DAY", "100"))

_raw_keys = os.getenv("KRONIQ_API_KEYS", "")
VALID_API_KEYS: set = {
    k.strip() for k in _raw_keys.split(",") if k.strip()
}
if not VALID_API_KEYS:
    log.warning(
        "No API keys configured. Set KRONIQ_API_KEYS in .env. "
        "All authenticated endpoints will reject requests."
    )

# ── API Key Header ────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Rate Limit Store (in-memory) ──────────────────────────────
# Structure: { "key_hash": { "2026-04-15": count } }
# UTC date used throughout — quota resets at midnight UTC.
# In-memory only — resets on restart.
# Replace with Redis in production (Week 10+).
rate_limit_store: dict = defaultdict(dict)

def _utc_today() -> str:
    """Returns today's UTC date as ISO string. Used for rate limit buckets."""
    return datetime.now(timezone.utc).date().isoformat()

def _key_hash(api_key: str) -> str:
    """Returns first 8 chars of SHA-256 hash of key for safe logging."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]

def _cleanup_old_buckets(key_hash: str) -> None:
    """
    Remove stale date buckets for a key.
    Keeps only today's UTC bucket. Prevents unbounded memory growth
    for long-running instances.
    """
    today = _utc_today()
    stale = [d for d in rate_limit_store[key_hash] if d != today]
    for d in stale:
        del rate_limit_store[key_hash][d]

def check_rate_limit(key_hash: str) -> None:
    """
    Increment request count for key_hash and raise 429 if over limit.
    Uses UTC date for bucket. Cleans up stale date buckets on each call.
    Called AFTER readiness check and key validation — degraded/invalid
    requests do not consume quota.
    """
    today = _utc_today()
    _cleanup_old_buckets(key_hash)

    current = rate_limit_store[key_hash].get(today, 0)
    if current >= RATE_LIMIT_DAY:
        log.warning(
            f"Rate limit exceeded: hash={key_hash} count={current + 1}"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. "
                f"Maximum {RATE_LIMIT_DAY} requests per UTC day. "
                f"Today's count: {current}. Resets at midnight UTC."
            )
        )
    rate_limit_store[key_hash][today] = current + 1

def validate_api_key(api_key: Optional[str]) -> str:
    """
    Validate API key from X-API-Key header.
    Returns the validated key on success.
    Raises 401 if key is missing.
    Raises 403 if key is present but not in VALID_API_KEYS.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing API key. "
                "Include X-API-Key header with your request. "
                "Contact support@kroniq.finance to request access."
            )
        )
    if api_key not in VALID_API_KEYS:
        log.warning(
            f"Invalid API key attempt: hash={_key_hash(api_key)}"
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "Invalid API key. "
                "Contact support@kroniq.finance to request access."
            )
        )
    return api_key

# ── Model Cache ──────────────────────────────────────────────
model_cache: dict = {}

# ── Pydantic Response Models ─────────────────────────────────
class PosteriorItem(BaseModel):
    state_id:    int
    regime:      str
    probability: float

class RegimeResponse(BaseModel):
    regime:             str
    confidence:         float
    allocation:         float
    signal_driver_hint: str
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
    explanation_type:    str
    model:               str

class HealthResponse(BaseModel):
    status:     str
    ready:      bool
    mode:       Optional[str]      = None
    model:      Optional[str]      = None
    trained_on: Optional[int]      = None
    train_end:  Optional[str]      = None
    as_of:      Optional[str]      = None
    started_at: Optional[str]      = None   # UTC ISO timestamp

class RateLimitStatus(BaseModel):
    api_key_hash:    str
    requests_today:  int
    limit_per_day:   int
    remaining_today: int
    utc_date:        str
    resets:          str

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
    model_cache["ready"] = False
    started_at_utc = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date().isoformat()



    try:
        log.info("=" * 55)
        log.info("Kroniq API startup — LIVE mode")
        log.info(f"Downloading latest available data (started {started_at_utc})")
        log.info(f"API keys loaded: {len(VALID_API_KEYS)}")
        log.info(f"Rate limit: {RATE_LIMIT_DAY} req/UTC day/key")
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
        model_cache["started_at"]   = started_at_utc  # full UTC ISO timestamp

        log.info("✓ Kroniq API ready — serving live regime")

    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        model_cache["ready"] = False

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
        "Inference on latest available data downloaded at startup. "
        "Walk-forward history: backtested 2021-2024 (1005 OOS days). "
        "Auth: X-API-Key header required on all endpoints except /health. "
        "Rate limit: 100 requests/UTC day/key."
    ),
    version="0.6.0",
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
    """Raise 503 if model is not loaded. Called before quota increment."""
    if not model_cache.get("ready", False):
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Startup may still be in progress — retry in 60s."
        )

# ── Auth + Rate Limit helper ─────────────────────────────────
def authenticate_and_limit(api_key: Optional[str]) -> str:
    """
    Single entry point for all protected endpoints.
    Order: require_ready → validate_api_key → check_rate_limit.

    This order ensures:
    - Degraded/startup requests never consume quota (503 before increment)
    - Invalid keys never consume quota (403 before increment)
    - Quota is incremented for every valid authenticated request,
      including requests that subsequently result in a 500 error.
      All authenticated requests count against the daily limit
      regardless of endpoint success or failure.
    """
    require_ready()
    key = validate_api_key(api_key)
    key_hash = _key_hash(key)
    check_rate_limit(key_hash)
    return key

# ── GET /regime ──────────────────────────────────────────────
@app.get("/regime", response_model=RegimeResponse)
def get_regime(
    api_key: Optional[str] = Security(API_KEY_HEADER)
):
    """
    Returns the latest available market regime classification
    based on data downloaded at startup.

    Requires: X-API-Key header.
    Rate limit: 100 requests/UTC day/key.

    Mode: LIVE.
    - Model trained on fixed 2015-2020 window (preserves OOS integrity).
    - Features include latest available trading day as of startup.
    - as_of may trail today by up to 4 days (FRED update schedule + market calendar).
    - Data does not auto-refresh — restart required for fresh data.
    - signal_driver_hint is rule-based heuristic, not dynamic feature attribution.
    """
    key = authenticate_and_limit(api_key)
    try:
        result = get_current_regime(
            model_cache["features"],
            model_cache["model_bundle"],
        )
        log.info(
            f"GET /regime — hash={_key_hash(key)} "
            f"regime={result['regime']}"
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
        raise HTTPException(status_code=500, detail="Regime inference failed.")

# ── GET /regime/history ──────────────────────────────────────
@app.get("/regime/history", response_model=HistoryResponse)
def get_regime_history(
    days: int = Query(
        default=252,
        ge=5,
        le=1005,
        description="Trading days to return (5–1005). Full OOS = 1005."
    ),
    api_key: Optional[str] = Security(API_KEY_HEADER)
):
    """
    Returns backtested walk-forward regime history (2021-2024).
    Pre-computed at startup. Not a live per-request replay.

    Requires: X-API-Key header.
    Rate limit: 100 requests/UTC day/key.
    """
    key = authenticate_and_limit(api_key)
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
        log.info(
            f"GET /regime/history — hash={_key_hash(key)} days={days}"
        )
        return HistoryResponse(
            days_returned = len(history),
            source        = "backtested walk-forward (quarterly retraining)",
            period        = "2021-2024",
            history       = history,
        )
    except Exception as e:
        log.error(f"/regime/history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="History retrieval failed.")

# ── POST /regime/explain ─────────────────────────────────────
@app.post("/regime/explain", response_model=ExplainResponse)
def explain_regime(
    req: ExplainRequest,
    api_key: Optional[str] = Security(API_KEY_HEADER)
):
    """
    Returns a rule-based explanation template for a given regime.

    Note: static templates derived from training-period state profiles.
    Not dynamic feature attribution — same explanation returned for a
    given regime label regardless of current feature values.
    Dynamic attribution (Engine 3) planned for Week 9.

    Requires: X-API-Key header.
    Rate limit: 100 requests/UTC day/key.
    """
    key = authenticate_and_limit(api_key)
    ex = EXPLANATIONS.get(req.regime)
    if not ex:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown regime '{req.regime}'. "
                f"Valid: {list(EXPLANATIONS.keys())}"
            )
        )
    log.info(
        f"POST /regime/explain — hash={_key_hash(key)} "
        f"regime={req.regime}"
    )
    return ExplainResponse(
        regime               = req.regime,
        summary              = ex["summary"],
        drivers              = ex["drivers"],
        allocation           = ex["allocation"],
        historical_analogue  = ex["historical_analogue"],
        explanation_type     = "rule-based template (static, not dynamic feature attribution)",
        model                = "Kroniq Regime Radar v5 | K=5 | 9-feature cross-asset HMM",
    )

# ── GET /health ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Readiness check. No auth required.
    Returns 200 if model loaded, 503 if not.
    Response shape is consistent in both branches.
    started_at is a full UTC ISO timestamp of when data was loaded.
    """
    ready  = model_cache.get("ready", False)
    bundle = model_cache.get("model_bundle")
    feats  = model_cache.get("features")

    response = HealthResponse(
        status     = "ok" if ready else "unavailable",
        ready      = ready,
        mode       = model_cache.get("mode")       if ready else None,
        model      = "kroniq_regime_radar_v5"       if ready else None,
        trained_on = bundle.get("trained_on")       if bundle else None,
        train_end  = bundle.get("train_end")        if bundle else None,
        as_of      = (
            feats.index[-1].strftime("%Y-%m-%d")
            if feats is not None else None
        ),
        started_at = model_cache.get("started_at") if ready else None,
    )

    if not ready:
        return JSONResponse(
            status_code=503,
            content=response.model_dump()
        )
    return response

# ── GET /usage ───────────────────────────────────────────────
@app.get("/usage", response_model=RateLimitStatus)
def get_usage(
    api_key: Optional[str] = Security(API_KEY_HEADER)
):
    """
    Returns today's UTC request count and remaining quota for your key.
    Does not consume quota — usage checks are free.

    Requires: X-API-Key header.
    """
    require_ready()
    key      = validate_api_key(api_key)
    key_hash = _key_hash(key)
    today    = _utc_today()
    _cleanup_old_buckets(key_hash)
    count    = rate_limit_store[key_hash].get(today, 0)

    return RateLimitStatus(
        api_key_hash    = key_hash,
        requests_today  = count,
        limit_per_day   = RATE_LIMIT_DAY,
        remaining_today = max(0, RATE_LIMIT_DAY - count),
        utc_date        = today,
        resets          = "midnight UTC",
    )