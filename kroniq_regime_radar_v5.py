# ============================================================
# kroniq_regime_radar_v5.py
# Kroniq — Unified Production Regime Detection Script
# K=5 | 9 features | Absolute anchoring | 3-day persistence
# Week 6 — April 13 2026
# ============================================================
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import os
import time
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
TRAIN_START      = "2015-01-01"
TRAIN_END        = "2020-12-31"   # fixed OOS boundary — never changes
TEST_START       = "2021-01-01"
K                = 5
RANDOM_STATE     = 42
RETRAIN_FREQ     = 63             # trading days (quarterly)
PERSIST_DAYS     = 3              # days before label switch accepted
DOWNLOAD_RETRIES = 3
DOWNLOAD_DELAY   = 2              # seconds between retries

ROUTE = {
    'Bull':    1.00,
    'Low-Vol': 1.00,
    'Neutral': 0.70,
    'Macro':   0.30,
    'Crisis':  0.00,
}

FEATURE_COLS = [
    "spy_ret", "spy_vol", "vix", "vix_chg",
    "tlt_ret", "gld_ret", "spy_tlt_corr",
    "credit_spread", "cs_chg"
]

VALID_REGIMES = {"Bull", "Low-Vol", "Neutral", "Macro", "Crisis"}

# ── FRED key ─────────────────────────────────────────────────
def _get_fred_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key

# ── 1. Data Pipeline ─────────────────────────────────────────
def download_prices(start: str, end: str) -> pd.DataFrame:
    """
    Download SPY, VIX, TLT, GLD close prices.
    Uses curl_cffi browser impersonation to bypass Yahoo rate limiting.
    Columns extracted and validated by name — never positional.
    Retries on failure with individual ticker fallback.
    """
    from curl_cffi import requests as cffi_requests

    tickers  = ["SPY", "^VIX", "TLT", "GLD"]
    rename   = {"^VIX": "VIX"}
    required = ["SPY", "VIX", "TLT", "GLD"]

    session = cffi_requests.Session(impersonate="chrome")

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            log.info(f"Downloading prices {start} → {end} (attempt {attempt})")

            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                session=session,
            )

            if raw is None or raw.empty:
                raise ValueError("yfinance returned empty DataFrame")

            # Extract Close — handle both flat and MultiIndex
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:
                close = raw.copy()

            if not isinstance(close, pd.DataFrame):
                raise TypeError(
                    f"Expected DataFrame after Close extraction, "
                    f"got {type(close).__name__}"
                )

            close = close.rename(columns=rename)

            missing = [c for c in required if c not in close.columns]
            if missing:
                raise ValueError(
                    f"Missing columns after download: {missing}. "
                    f"Available: {list(close.columns)}"
                )

            prices = close[required].copy()

            all_nan = [c for c in required if prices[c].isnull().all()]
            if all_nan:
                raise ValueError(f"All-NaN columns detected: {all_nan}")

            if len(prices) < 200:
                raise ValueError(
                    f"Too few rows downloaded: {len(prices)}. "
                    "Check start/end dates."
                )

            log.info(
                f"Prices downloaded: {prices.shape[0]} rows "
                f"× {prices.shape[1]} cols"
            )
            return prices

        except Exception as e:
            log.warning(f"Batch download attempt {attempt} failed: {e}")

            # Fallback: download tickers individually
            if attempt == DOWNLOAD_RETRIES:
                log.info("Attempting individual ticker downloads as fallback...")
                frames = {}
                for ticker in tickers:
                    try:
                        t = yf.Ticker(ticker, session=session)
                        df = t.history(start=start, end=end, auto_adjust=True)
                        if not df.empty:
                            col_name = rename.get(ticker, ticker)
                            frames[col_name] = df["Close"]
                            log.info(f"  {ticker}: {len(df)} rows")
                        else:
                            log.warning(f"  {ticker}: empty")
                    except Exception as te:
                        log.warning(f"  {ticker} individual download failed: {te}")

                if len(frames) == len(required):
                    prices = pd.DataFrame(frames)
                    prices.index = pd.to_datetime(prices.index).tz_localize(None)
                    log.info(
                        f"Individual fallback succeeded: "
                        f"{prices.shape[0]} rows"
                    )
                    return prices
                else:
                    raise RuntimeError(
                        f"Price download failed after {DOWNLOAD_RETRIES} attempts "
                        f"including individual fallback. "
                        f"Got {list(frames.keys())}, need {required}. "
                        "Check internet connection and Yahoo Finance availability."
                    )

            time.sleep(DOWNLOAD_DELAY)


def download_credit_spreads(start: str, end: str) -> pd.Series:
    """
    Download ICE BofA HY OAS from FRED (BAMLH0A0HYM2).
    Retries on failure.
    """
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            log.info(
                f"Downloading credit spreads from FRED "
                f"(attempt {attempt})"
            )
            fred = Fred(api_key=_get_fred_key())
            cs   = fred.get_series(
                "BAMLH0A0HYM2",
                observation_start=start,
                observation_end=end
            )
            if cs is None or cs.empty:
                raise ValueError(
                    "FRED returned empty series. "
                    "Check API key and date range."
                )
            cs.name = "credit_spread"
            log.info(f"Credit spreads downloaded: {len(cs)} observations")
            return cs

        except Exception as e:
            log.warning(f"FRED download attempt {attempt} failed: {e}")
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_DELAY)
            else:
                raise RuntimeError(
                    f"FRED download failed after {DOWNLOAD_RETRIES} "
                    f"attempts. Last error: {e}"
                )


def build_features(prices: pd.DataFrame,
                   credit_spreads: pd.Series) -> pd.DataFrame:
    """Build all 9 features with full validation."""
    df = prices.copy()
    df.index = pd.to_datetime(df.index)

    cs_aligned          = credit_spreads.reindex(df.index, method="ffill")
    df["credit_spread"] = cs_aligned

    f = pd.DataFrame(index=df.index)
    f["spy_ret"]       = df["SPY"].pct_change()
    f["spy_vol"]       = f["spy_ret"].rolling(20).std() * np.sqrt(252)
    f["vix"]           = df["VIX"]
    f["vix_chg"]       = df["VIX"].pct_change(5)
    f["tlt_ret"]       = df["TLT"].pct_change()
    f["gld_ret"]       = df["GLD"].pct_change()
    f["spy_tlt_corr"]  = f["spy_ret"].rolling(20).corr(f["tlt_ret"])
    f["credit_spread"] = df["credit_spread"]
    f["cs_chg"]        = df["credit_spread"].diff()

    f = f[FEATURE_COLS].dropna()

    # Explicit exceptions — never silent asserts in production
    if not isinstance(f, pd.DataFrame):
        raise TypeError("Features must be a DataFrame")
    if f.empty:
        raise ValueError("Feature DataFrame is empty after dropna")
    if f.shape[0] <= 500:
        raise ValueError(
            f"Too few rows after feature engineering: {f.shape[0]}. "
            "Need > 500. Check date range and data availability."
        )
    if f.isnull().any().any():
        bad_cols = f.columns[f.isnull().any()].tolist()
        raise ValueError(
            f"NaNs remain after dropna in columns: {bad_cols}. "
            "Check feature pipeline."
        )
    if f.index.duplicated().any():
        dupes = f.index[f.index.duplicated()].tolist()
        raise ValueError(
            f"Duplicate dates in feature index: {dupes[:5]}. "
            "Check data source."
        )
    if not f.index.is_monotonic_increasing:
        raise ValueError(
            "Feature index is not sorted ascending. "
            "Check data source ordering."
        )
    if list(f.columns) != FEATURE_COLS:
        raise ValueError(
            f"Feature column mismatch.\n"
            f"  Expected: {FEATURE_COLS}\n"
            f"  Got:      {list(f.columns)}"
        )

    log.info(
        f"Features built: {f.shape[0]} rows × {f.shape[1]} cols "
        f"({f.index[0].date()} → {f.index[-1].date()})"
    )
    return f

# ── 2. Label Anchoring ───────────────────────────────────────
def assign_labels_absolute(profiles: dict) -> dict:
    """
    Absolute threshold label assignment.
    Same rules at every retrain — prevents label drift.
    """
    labels   = {}
    assigned = set()

    # Crisis: VIX > 30 AND credit spread > 6.5%
    crisis_cands = {
        s: p for s, p in profiles.items()
        if p["mean_vix"] > 30 and p["mean_cs"] > 6.5
    }
    crisis = (
        max(crisis_cands, key=lambda s: profiles[s]["mean_vix"])
        if crisis_cands
        else max(profiles, key=lambda s: profiles[s]["mean_vix"])
    )
    labels[crisis] = "Crisis"
    assigned.add(crisis)

    # Macro: negative return AND VIX > 18
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    macro_cands = {
        s: p for s, p in rem.items()
        if p["mean_ret"] < 0 and p["mean_vix"] > 18
    }
    macro = (
        min(macro_cands, key=lambda s: profiles[s]["mean_ret"])
        if macro_cands
        else min(rem, key=lambda s: profiles[s]["mean_ret"])
    )
    labels[macro] = "Macro"
    assigned.add(macro)

    # Low-Vol: VIX < 15 AND positive return
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    lv_cands = {
        s: p for s, p in rem.items()
        if p["mean_vix"] < 15 and p["mean_ret"] > 0
    }
    lv = (
        min(lv_cands, key=lambda s: profiles[s]["mean_vix"])
        if lv_cands
        else min(rem, key=lambda s: profiles[s]["mean_vix"])
    )
    labels[lv] = "Low-Vol"
    assigned.add(lv)

    # Bull: highest return among remaining
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    bull = max(rem, key=lambda s: profiles[s]["mean_ret"])
    labels[bull] = "Bull"
    assigned.add(bull)

    # Neutral: leftover
    for s in profiles:
        if s not in assigned:
            labels[s] = "Neutral"

    # Explicit label validation — report missing and unexpected labels
    assigned_labels = set(labels.values())
    missing_labels  = VALID_REGIMES - assigned_labels
    extra_labels    = assigned_labels - VALID_REGIMES

    if missing_labels or extra_labels:
        raise ValueError(
            f"Label assignment error.\n"
            f"  Expected:    {sorted(VALID_REGIMES)}\n"
            f"  Got:         {sorted(assigned_labels)}\n"
            f"  Missing:     {sorted(missing_labels)}\n"
            f"  Unexpected:  {sorted(extra_labels)}\n"
            f"  Full map:    {labels}"
        )

    return labels


def build_profiles(features: pd.DataFrame,
                   state_seq: np.ndarray) -> dict:
    profiles = {}
    for s in range(K):
        mask = (state_seq == s)
        if mask.sum() == 0:
            profiles[s] = {
                "mean_ret": 0.0,
                "mean_vix": 15.0,
                "mean_cs":  4.0
            }
        else:
            profiles[s] = {
                "mean_ret": float(features.loc[mask, "spy_ret"].mean() * 252),
                "mean_vix": float(features.loc[mask, "vix"].mean()),
                "mean_cs":  float(features.loc[mask, "credit_spread"].mean()),
            }
    return profiles

# ── 3. Model Training ────────────────────────────────────────
def fit_hmm(X_scaled: np.ndarray) -> GaussianHMM:
    """Fit K=5 Gaussian HMM with convergence check."""
    model = GaussianHMM(
        n_components=K,
        covariance_type="full",
        n_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_scaled)

    if not model.monitor_.converged:
        log.warning(
            f"HMM did not converge after {model.monitor_.iter} iterations. "
            "Results may be unstable."
        )
    else:
        log.info(f"HMM converged in {model.monitor_.iter} iterations")

    return model

# ── 4. Walk-Forward Validation (offline only) ─────────────────
def walk_forward(features: pd.DataFrame,
                 n_train: int) -> pd.Series:
    """
    Expanding-window walk-forward validation.
    Offline use and pre-computed at API startup for /regime/history.
    Not called per-request.
    Returns Series of regime labels indexed by date.
    """
    n_test        = len(features) - n_train
    wf_labels     = []
    pending_label = None
    pending_count = 0
    current_label = None
    scaler        = None
    model         = None
    label_map     = {}

    log.info(
        f"Walk-forward: {n_test} OOS days, "
        f"retraining every {RETRAIN_FREQ} days"
    )

    for i in range(n_test):
        abs_i = n_train + i

        if i == 0 or i % RETRAIN_FREQ == 0:
            log.info(
                f"  Retrain step {i} "
                f"({features.index[abs_i].date()})"
            )
            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(features.iloc[:abs_i])
            model     = fit_hmm(X_scaled)
            states    = model.predict(X_scaled)
            profiles  = build_profiles(features.iloc[:abs_i], states)
            label_map = assign_labels_absolute(profiles)
            pending_label = None
            pending_count = 0

        X_hist    = scaler.transform(features.iloc[:abs_i + 1])
        raw_state = model.predict(X_hist)[-1]
        raw_label = label_map.get(raw_state, "Neutral")

        # Bootstrap first prediction
        if current_label is None:
            current_label = raw_label
            wf_labels.append(current_label)
            continue

        # 3-day persistence filter
        if raw_label == current_label:
            pending_label = None
            pending_count = 0
        else:
            if raw_label == pending_label:
                pending_count += 1
            else:
                pending_label = raw_label
                pending_count = 1
            if pending_count >= PERSIST_DAYS:
                current_label = pending_label
                pending_label = None
                pending_count = 0

        wf_labels.append(current_label)

    return pd.Series(wf_labels, index=features.index[n_train:])

# ── 5. Performance Metrics ───────────────────────────────────
def sharpe(returns: pd.Series) -> float:
    mu  = returns.mean() * 252
    sig = returns.std()  * np.sqrt(252)
    return round(float(mu / sig), 4) if sig > 0 else 0.0

def max_dd(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return round(float(((cum / cum.cummax()) - 1).min()), 4)

def cagr(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod().iloc[-1]
    yrs = len(returns) / 252
    return round(float(cum ** (1 / yrs) - 1), 4)

# ── 6. Static Model Preparation ──────────────────────────────
def prepare_static_model(features: pd.DataFrame,
                         train_end: str = TRAIN_END) -> dict:
    """
    Fits the production HMM on the fixed 2015-2020 training window.

    Named 'static' because the training boundary is fixed at TRAIN_END
    to preserve OOS integrity. Features passed in may extend through
    today — only the training slice is used for fitting.

    Called once at API startup. Returns model bundle for inference.
    """
    train = features[features.index <= train_end]

    if len(train) < 200:
        raise ValueError(
            f"Insufficient training data: {len(train)} rows. "
            "Check TRAIN_START / TRAIN_END config."
        )

    log.info(
        f"Fitting static model: {len(train)} training days "
        f"({train.index[0].date()} → {train.index[-1].date()})"
    )

    scaler    = StandardScaler()
    X_train   = scaler.fit_transform(train[FEATURE_COLS])
    model     = fit_hmm(X_train)
    states    = model.predict(X_train)
    profiles  = build_profiles(train, states)
    label_map = assign_labels_absolute(profiles)

    log.info(f"Static model ready. Label map: {label_map}")

    return {
        "model":        model,
        "scaler":       scaler,
        "label_map":    label_map,
        "train_end":    train_end,
        "trained_on":   len(train),
        "features_end": features.index[-1].strftime("%Y-%m-%d"),
    }

# ── 7. Live Inference ────────────────────────────────────────
def get_current_regime(features: pd.DataFrame,
                       model_bundle: dict) -> dict:
    """
    Infers the latest available regime from current features.
    Uses static model trained on 2015-2020.
    Features extend through the latest available trading day.

    Notes:
    - signal_driver_hint is a rule-based heuristic, not
      feature attribution. Macro/Crisis → credit spread primary.
      All other regimes → equity/vol primary.
    - as_of is the latest date in the aligned feature index,
      which may lag today by 1-4 days due to FRED update schedule
      and market calendar alignment.
    - Confidence = 1.0 with all other posteriors = 0.0 is a known
      numerical behaviour of Gaussian HMM in well-separated regimes.
      Raw float precision is preserved — rounding is display only.
    """
    model     = model_bundle["model"]
    scaler    = model_bundle["scaler"]
    label_map = model_bundle["label_map"]

    window = features.tail(60)
    if len(window) < 20:
        raise ValueError(
            f"Insufficient rows for inference: {len(window)}. Need >= 20."
        )

    X_scaled   = scaler.transform(window[FEATURE_COLS])
    posteriors = model.predict_proba(X_scaled)[-1]
    raw_state  = int(np.argmax(posteriors))
    regime     = label_map.get(raw_state, "Neutral")
    allocation = ROUTE.get(regime, 0.5)

    # Raw float confidence — not rounded to preserve precision signal
    confidence_raw = float(posteriors[raw_state])

    # Log raw precision for debugging
    log.debug(
        f"Raw posteriors: "
        + ", ".join(
            f"state {s}={float(p):.6e}"
            for s, p in enumerate(posteriors)
        )
    )

    # Posteriors sorted by probability — state_id included for debugging
    all_posteriors = sorted(
        [
            {
                "state_id":    s,
                "regime":      label_map.get(s, "Neutral"),
                "probability": float(p),   # full float, no rounding
            }
            for s, p in enumerate(posteriors)
        ],
        key=lambda x: x["probability"],
        reverse=True,
    )

    # Determine latest dates for each data source
    price_as_of  = features.index[-1].strftime("%Y-%m-%d")
    feature_as_of = features.index[-1].strftime("%Y-%m-%d")

    return {
        "regime":             regime,
        "confidence":         confidence_raw,
        "allocation":         allocation,
        # Renamed to _hint — rule-based heuristic, not feature attribution
        "signal_driver_hint": (
            "credit_spread OAS — primary (Macro/Crisis rule)"
            if regime in ["Crisis", "Macro"]
            else "spy_vol + vix — primary (calm regime rule)"
        ),
        "all_posteriors":     all_posteriors,
        # as_of = latest date in aligned feature index
        # May lag today by 1-4 days (FRED update schedule + market calendar)
        "as_of":              feature_as_of,
        "as_of_note":         (
            "Latest fully aligned feature date. "
            "FRED credit spreads update with 1-3 day lag. "
            "Run date may differ from as_of."
        ),
    }

# ── 8. Main — Offline Validation ─────────────────────────────
if __name__ == "__main__":
    from datetime import date
    TODAY = date.today().isoformat()

    log.info("=" * 60)
    log.info("Kroniq Regime Radar v5 — Offline Validation")
    log.info("=" * 60)

    prices   = download_prices(TRAIN_START, TODAY)
    cs       = download_credit_spreads(TRAIN_START, TODAY)
    features = build_features(prices, cs)

    train = features[features.index <= TRAIN_END]
    test  = features[
        (features.index >= TEST_START) &
        (features.index <= "2024-12-31")
    ]
    n_tr = len(train)

    log.info(f"Train: {n_tr} days | OOS test: {len(test)} days")
    log.info(f"Features extend through: {features.index[-1].date()}")

    bundle = prepare_static_model(features, TRAIN_END)

    log.info("\nRunning walk-forward validation (2021-2024)...")
    wf_labels  = walk_forward(
        features[features.index <= "2024-12-31"], n_tr
    )
    oos_ret    = test["spy_ret"]
    wf_weights = wf_labels.reindex(oos_ret.index).map(ROUTE)
    wf_strat   = wf_weights.shift(1) * oos_ret
    bh_strat   = oos_ret

    log.info("=" * 60)
    log.info("WALK-FORWARD OOS RESULTS (2021-2024)")
    log.info("=" * 60)
    log.info(f"  Sharpe  : {sharpe(wf_strat)}")
    log.info(f"  MaxDD   : {max_dd(wf_strat):.2%}")
    log.info(f"  CAGR    : {cagr(wf_strat):.2%}")
    log.info(f"  BH Sharpe: {sharpe(bh_strat)}")
    log.info("=" * 60)

    log.info("\nLive regime (based on data through today):")
    result = get_current_regime(features, bundle)
    for k, v in result.items():
        log.info(f"  {k}: {v}")

    log.info("\n✓ Validation complete")
    log.info("  Next: uvicorn api.main:app --reload --port 8000")