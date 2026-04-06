# ============================================================
# kroniq_regime_radar_v5.py
# Kroniq — Unified Production Regime Detection Script
# K=5 | 9 features | Absolute anchoring | 3-day persistence
# Walk-forward Sharpe 0.881 | MaxDD -17.46% | CAGR 9.97%
# Week 6 — April 13 2026
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ── Config ──────────────────────────────────────────────────
TRAIN_START   = "2015-01-01"
TRAIN_END     = "2020-12-31"
TEST_START    = "2021-01-01"
TEST_END      = "2024-12-31"
K             = 5
RANDOM_STATE  = 42
RETRAIN_FREQ  = 63      # trading days (quarterly)
PERSIST_DAYS  = 3       # min consecutive days before label switch
MIN_HOLD_DAYS = 5       # min days to hold allocation after switch (FastAPI layer)
FRED_API_KEY  = "your_fred_api_key_here"

ROUTE = {
    'Bull':    1.00,
    'Low-Vol': 1.00,
    'Neutral': 0.70,
    'Macro':   0.30,
    'Crisis':  0.00,
}

# ── 1. Data Pipeline ─────────────────────────────────────────
def download_prices(start: str, end: str) -> pd.DataFrame:
    tickers = ["SPY", "^VIX", "TLT", "GLD"]
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = ["GLD", "SPY", "TLT", "VIX"]
    return raw

def download_credit_spreads(start: str, end: str) -> pd.Series:
    fred = Fred(api_key=FRED_API_KEY)
    cs = fred.get_series("BAMLH0A0HYM2", observation_start=start,
                          observation_end=end)
    cs.name = "credit_spread"
    return cs

def build_features(prices: pd.DataFrame,
                   credit_spreads: pd.Series) -> pd.DataFrame:
    df = prices.copy()
    df["credit_spread"] = credit_spreads.reindex(df.index, method="ffill")
    df.dropna(subset=["credit_spread"], inplace=True)

    f = pd.DataFrame(index=df.index)
    f["spy_ret"]      = df["SPY"].pct_change()
    f["spy_vol"]      = f["spy_ret"].rolling(20).std() * np.sqrt(252)
    f["vix"]          = df["VIX"]
    f["vix_chg"]      = df["VIX"].pct_change(5)
    f["tlt_ret"]      = df["TLT"].pct_change()
    f["gld_ret"]      = df["GLD"].pct_change()
    f["spy_tlt_corr"] = f["spy_ret"].rolling(20).corr(f["tlt_ret"])
    f["credit_spread"]= df["credit_spread"]
    f["cs_chg"]       = df["credit_spread"].diff()

    f.dropna(inplace=True)
    return f

# ── 2. Label Anchoring ───────────────────────────────────────
def assign_labels_absolute(profiles: dict) -> dict:
    labels  = {}
    assigned = set()

    # Crisis: VIX > 30 AND credit spread > 6.5%
    crisis_cands = {s: p for s, p in profiles.items()
                    if p["mean_vix"] > 30 and p["mean_cs"] > 6.5}
    crisis = (max(crisis_cands, key=lambda s: profiles[s]["mean_vix"])
              if crisis_cands
              else max(profiles, key=lambda s: profiles[s]["mean_vix"]))
    labels[crisis] = "Crisis"
    assigned.add(crisis)

    # Macro: negative return AND VIX > 18
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    macro_cands = {s: p for s, p in rem.items()
                   if p["mean_ret"] < 0 and p["mean_vix"] > 18}
    macro = (min(macro_cands, key=lambda s: profiles[s]["mean_ret"])
             if macro_cands
             else min(rem, key=lambda s: profiles[s]["mean_ret"]))
    labels[macro] = "Macro"
    assigned.add(macro)

    # Low-Vol: VIX < 15 AND positive return
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    lv_cands = {s: p for s, p in rem.items()
                if p["mean_vix"] < 15 and p["mean_ret"] > 0}
    lv = (min(lv_cands, key=lambda s: profiles[s]["mean_vix"])
          if lv_cands
          else min(rem, key=lambda s: profiles[s]["mean_vix"]))
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

    return labels

def build_profiles(features: pd.DataFrame,
                   state_seq: np.ndarray) -> dict:
    profiles = {}
    for s in range(K):
        mask = (state_seq == s)
        if mask.sum() == 0:
            profiles[s] = {"mean_ret": 0, "mean_vix": 15, "mean_cs": 4.0}
        else:
            profiles[s] = {
                "mean_ret": features.loc[mask, "spy_ret"].mean() * 252,
                "mean_vix": features.loc[mask, "vix"].mean(),
                "mean_cs":  features.loc[mask, "credit_spread"].mean(),
            }
    return profiles

# ── 3. Model Training ────────────────────────────────────────
def fit_hmm(X_scaled: np.ndarray) -> GaussianHMM:
    model = GaussianHMM(
        n_components=K,
        covariance_type="full",
        n_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_scaled)
    return model

# ── 4. Walk-Forward Validation ───────────────────────────────
def walk_forward(features: pd.DataFrame,
                 n_train: int) -> pd.Series:
    n_test        = len(features) - n_train
    wf_labels     = []
    pending_label = None
    pending_count = 0
    current_label = None
    scaler        = None
    model         = None
    label_map     = {}

    for i in range(n_test):
        abs_i = n_train + i

        if i == 0 or i % RETRAIN_FREQ == 0:
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(features.iloc[:abs_i])
            model   = fit_hmm(X_scaled)
            states  = model.predict(X_scaled)
            profiles = build_profiles(features.iloc[:abs_i], states)
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
    return mu / sig if sig > 0 else 0.0

def max_dd(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return ((cum / cum.cummax()) - 1).min()

def cagr(returns: pd.Series) -> float:
    cum  = (1 + returns).cumprod().iloc[-1]
    yrs  = len(returns) / 252
    return cum ** (1 / yrs) - 1

# ── 6. Live Regime API Function ──────────────────────────────
def get_current_regime(features: pd.DataFrame,
                       model: GaussianHMM,
                       scaler: StandardScaler,
                       label_map: dict) -> dict:
    """
    API-ready function. Returns current regime with confidence.
    Called by FastAPI GET /regime endpoint.

    MIN_HOLD_DAYS enforced at FastAPI routing layer — not here.
    """
    X_scaled  = scaler.transform(features.values)
    posteriors = model.predict_proba(X_scaled)[-1]
    raw_state  = int(np.argmax(posteriors))
    regime     = label_map.get(raw_state, "Neutral")
    confidence = float(posteriors[raw_state])
    allocation = ROUTE.get(regime, 0.5)

    return {
        "regime":     regime,
        "confidence": round(confidence, 4),
        "allocation": allocation,
        "all_posteriors": {
            label_map.get(s, "Neutral"): round(float(p), 4)
            for s, p in enumerate(posteriors)
        },
        "signal_driver": (
            "credit_spread OAS — primary"
            if regime in ["Crisis", "Macro"]
            else "spy_vol + vix — primary"
        )
    }

# ── 7. Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Kroniq Regime Radar v5 — Production Script")
    print("=" * 60)

    # Load data
    print("\n[1/5] Downloading price data...")
    prices = download_prices(TRAIN_START, TEST_END)

    print("[2/5] Downloading credit spreads...")
    cs = download_credit_spreads(TRAIN_START, TEST_END)

    print("[3/5] Building features...")
    features = build_features(prices, cs)
    print(f"      Features shape: {features.shape}")
    print(f"      Date range: {features.index[0].date()} → {features.index[-1].date()}")

    # Train/test split
    train = features[features.index <= TRAIN_END]
    test  = features[features.index >= TEST_START]
    n_tr  = len(train)
    print(f"      Train: {n_tr} days | Test: {len(test)} days")

    # Fit static model on train set
    print("\n[4/5] Fitting static K=5 HMM on training data...")
    scaler_static = StandardScaler()
    X_train = scaler_static.fit_transform(train)
    model_static = fit_hmm(X_train)
    train_states  = model_static.predict(X_train)
    profiles      = build_profiles(train, train_states)
    label_map     = assign_labels_absolute(profiles)

    print("\n      State profiles:")
    for s, lbl in sorted(label_map.items(), key=lambda x: x[1]):
        p = profiles[s]
        print(f"        State {s}: {lbl:8s} | "
              f"VIX={p['mean_vix']:.1f} | "
              f"ret={p['mean_ret']*100:+.1f}% | "
              f"CS={p['mean_cs']:.2f}%")

    # Walk-forward validation
    print("\n[5/5] Running walk-forward validation (16 quarterly retrains)...")
    wf_labels = walk_forward(features, n_tr)

    # Performance
    oos_ret      = features.loc[test.index, "spy_ret"]
    wf_weights   = wf_labels.map(ROUTE)
    wf_strat     = wf_weights.shift(1) * oos_ret
    bh_strat     = oos_ret

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Walk-Forward Sharpe : {sharpe(wf_strat):.3f}")
    print(f"  Walk-Forward MaxDD  : {max_dd(wf_strat):.2%}")
    print(f"  Walk-Forward CAGR   : {cagr(wf_strat):.2%}")
    print(f"  Buy-Hold Sharpe     : {sharpe(bh_strat):.3f}")
    print("=" * 60)

    # Live regime snapshot
    print("\n── Current Regime (live snapshot) ──")
    result = get_current_regime(
        features.tail(60),
        model_static,
        scaler_static,
        label_map
    )
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n✓ kroniq_regime_radar_v5.py complete")
    print("  Next: wrap get_current_regime() in FastAPI → GET /regime")