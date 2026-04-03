"""
Kroniq Regime Radar v4.1 — Production Prototype
================================================
Clean, fully corrected version incorporating:
- VIX fix: individual ticker downloads (eliminates column misalignment)
- BIC: -2*score(X) + k*log(n), score() = total LL (runtime verified)
- K=2..4 constrained BIC selection (interpretable states)
- MAP decoding: full-sequence predict_proba, slice after
- Hungarian state alignment across walk-forward retrains
- Confidence gating: threshold 0.65, min 3-day persistence
- Posterior fix: state_N_label keys (no overwriting)
- Correct multi-asset turnover: sum(|Δweight|) per asset per day
- Realistic routing: SPY/TLT/GLD/Cash by regime + 2bps transaction costs
- Honest labelling: post-hoc interpretations of unsupervised clusters
- Period sanity checks: 2017, Feb-Apr 2020, 2022
- Regime history CSV export
- 60/40 and 60/30/10 benchmarks
- Error handling throughout
"""

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CFG = {
    "start":          "2015-01-01",
    "end":            "2024-12-31",
    "split":          "2020-01-01",
    "k_min":          2,
    "k_max":          4,           # constrained: forces interpretable states
    "bic_seeds":      10,
    "bic_iters":      1000,
    "wf_seeds":       5,
    "wf_iters":       500,
    "retrain_freq":   63,          # ~quarterly
    "conf_threshold": 0.65,
    "min_persist":    3,           # days before regime switch confirmed
    "tc_bps":         0.0002,      # 2bps per asset per rebalance
    "output_csv":     "kroniq_regime_history.csv",
    "output_chart":   "kroniq_hmm_regimes_final.png",
}

# ══════════════════════════════════════════════════════════════
# IMPORTS & DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════
import sys, warnings
warnings.filterwarnings('ignore')

REQUIRED = ["numpy","pandas","matplotlib","hmmlearn",
            "sklearn","scipy","yfinance"]
missing = [p for p in REQUIRED if __import__("importlib").util.find_spec(p) is None]
if missing:
    print(f"Missing packages: {missing}\nRun: pip install {' '.join(missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import hmmlearn
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import yfinance as yf

print(f"hmmlearn {hmmlearn.__version__}")

# Runtime sanity check for score() behavior
# Assumption: score(X) = total sequence log-likelihood (hmmlearn >=0.3)
# This is a heuristic check, not definitive proof from source.
def _verify_score():
    m = GaussianHMM(n_components=2, covariance_type="diag",
                    n_iter=50, random_state=0)
    X = np.random.RandomState(0).randn(100, 2)
    m.fit(X)
    r = m.score(X[:20]) / m.score(X[:10])
    status = "OK" if 1.5 < r < 2.5 else "WARNING — check version"
    print(f"score() sanity: ratio={r:.2f} → {status}")
    print("BIC = -2*score(X) + k*log(n)  [total LL, no *n needed]")
_verify_score()
print("="*60)

# ══════════════════════════════════════════════════════════════
# MODULE 1 — DATA
# VIX FIX: download each ticker individually to avoid
# multi-index column ordering bugs in batch yf.download()
# ══════════════════════════════════════════════════════════════
def load_data(cfg):
    tickers = {
        "SPY":  "SP500",
        "QQQ":  "NASDAQ",
        "^VIX": "VIX",
        "TLT":  "Bonds",
        "GLD":  "Gold",
    }
    series = {}
    print("\nDownloading data (individual tickers to avoid column misalignment)...")
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, start=cfg["start"], end=cfg["end"],
                             auto_adjust=True, progress=False)
            s = df["Close"].squeeze()
            s.name = name
            series[name] = s
            print(f"  {name:8s} ({ticker}): "
                  f"min={s.min():.2f}  max={s.max():.2f}  n={len(s)}")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
            sys.exit(1)

    raw = pd.concat(series.values(), axis=1).dropna()

    # Hard validation
    assert raw["VIX"].max() < 100, \
        f"VIX max={raw['VIX'].max():.2f} — still unrealistic, check download"
    assert len(raw) > 1000, f"Insufficient data: {len(raw)} rows"

    print(f"\nData loaded: {len(raw)} days, {raw.shape[1]} assets")
    print(f"VIX range: {raw['VIX'].min():.2f} — {raw['VIX'].max():.2f}  ✓")
    return raw

# ══════════════════════════════════════════════════════════════
# MODULE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def build_features(raw):
    """
    7 cross-asset features. All economically motivated.
    spy_ret/spy_vol: primary equity signal
    vix/vix_chg:    fear gauge and acceleration
    tlt_ret/gld_ret: safe-haven signals
    spy_tlt:         flight-to-quality correlation (key crisis signal)
    """
    rets    = raw.pct_change(fill_method=None).dropna()
    spy     = rets["SP500"]
    feat    = pd.DataFrame({
        "spy_ret":  spy,
        "spy_vol":  spy.rolling(20).std() * np.sqrt(252),
        "vix":      raw["VIX"],
        "vix_chg":  raw["VIX"].pct_change(5).fillna(0),
        "tlt_ret":  rets["Bonds"],
        "gld_ret":  rets["Gold"],
        "spy_tlt":  spy.rolling(20).corr(rets["Bonds"]),
    }).dropna()

    # Feature integrity check
    print(f"\nFeature summary ({len(feat)} days):")
    print(f"  {'Feature':12s} {'Mean':>9s} {'Std':>9s} {'Min':>9s} {'Max':>9s}")
    for col in feat.columns:
        s = feat[col]
        print(f"  {col:12s} {s.mean():9.4f} {s.std():9.4f} "
              f"{s.min():9.4f} {s.max():9.4f}")

    assert feat["vix"].max() < 100
    assert feat["spy_vol"].max() < 5
    assert -1.01 <= feat["spy_tlt"].min() and feat["spy_tlt"].max() <= 1.01
    print("  Feature integrity: PASSED")

    return feat, rets

# ══════════════════════════════════════════════════════════════
# MODULE 3 — BIC MODEL SELECTION
# BIC = -2*LL + k*log(n)
# score(X) = total sequence LL (verified above)
# Param count (diagonal): (K-1) + K(K-1) + Kd + Kd
# ══════════════════════════════════════════════════════════════
def compute_bic(model, X):
    n, d = X.shape
    K    = model.n_components
    k    = (K-1) + K*(K-1) + K*d + K*d
    return -2 * model.score(X) + k * np.log(n)

def select_model(X_train, cfg):
    print(f"\nBIC selection (K={cfg['k_min']}..{cfg['k_max']}, "
          f"{cfg['bic_seeds']} seeds)...")
    best_bic, best_model, best_K = np.inf, None, None
    for K in range(cfg["k_min"], cfg["k_max"] + 1):
        bk_bic, bk_model = np.inf, None
        for seed in range(cfg["bic_seeds"]):
            try:
                m = GaussianHMM(n_components=K, covariance_type="diag",
                                n_iter=cfg["bic_iters"],
                                random_state=seed, tol=1e-4)
                m.fit(X_train)
                b = compute_bic(m, X_train)
                if b < bk_bic: bk_bic, bk_model = b, m
            except Exception: pass
        if bk_model:
            print(f"  K={K}: BIC={bk_bic:.2f}")
            if bk_bic < best_bic:
                best_bic, best_model, best_K = bk_bic, bk_model, K
    if best_model is None:
        print("ERROR: No model converged"); sys.exit(1)
    print(f"Optimal K={best_K} (BIC={best_bic:.2f})")
    return best_model, best_K

# ══════════════════════════════════════════════════════════════
# MODULE 4 — MAP DECODING
#
# Design choice: MAP (posterior-argmax via predict_proba)
# NOT Viterbi, because:
#   - MAP gives per-step confidence scores (the posterior itself)
#   - Viterbi optimises globally but gives no uncertainty estimate
#   - API needs (regime, confidence) — MAP provides both naturally
#
# FIX: always decode full available sequence, then slice.
# Avoids cold-start at train/test boundary (Jan 2020).
# ══════════════════════════════════════════════════════════════
def map_decode(model, X_full, s=None, e=None):
    try:
        post   = model.predict_proba(X_full)
        states = np.argmax(post, axis=1)
        return states[s:e], post[s:e]
    except Exception as ex:
        print(f"  Decode warning: {ex} — using fallback")
        n = len(X_full[s:e]) if (s is not None or e is not None) else len(X_full)
        K = model.n_components
        return np.zeros(n, dtype=int), np.ones((n, K)) / K

# ══════════════════════════════════════════════════════════════
# MODULE 5 — STATE PROFILING & LABELLING
#
# NOTE: Labels are POST-HOC interpretations of unsupervised clusters.
# The HMM finds statistical structure; we assign economic meaning
# using theory-grounded thresholds. These are directional
# interpretations, not ground truth regime definitions.
# Thresholds: VIX>25+loss>5% inspired by Ang-Bekaert (2004);
# positive SPY-TLT corr as liquidity stress indicator.
# ══════════════════════════════════════════════════════════════
COLORS = {
    "Bull":      "#1D9E75",
    "Neutral":   "#5DCAA5",
    "Low-Vol":   "#888780",
    "Bear":      "#BA7517",
    "Macro":     "#534AB7",
    "Crisis":    "#993C1D",
    "Liquidity": "#185FA5",
}

def build_profiles(states, feat_df):
    profiles = {}
    for s in np.unique(states):
        mask = states == s
        d    = feat_df.iloc[mask]
        r252 = d["spy_ret"].mean() * 252
        v252 = d["spy_ret"].std()  * np.sqrt(252)
        profiles[s] = {
            "ret":    r252,
            "vol":    v252,
            "vix":    d["vix"].mean(),
            "corr":   d["spy_tlt"].mean(),
            "sharpe": r252 / (v252 + 1e-9),
            "pct":    mask.sum() / len(states) * 100,
        }
    return profiles

def assign_label(p):
    r, v, vix, c = p["ret"], p["vol"], p["vix"], p["corr"]
    # Extreme low-vol bull (VIX<13, vol<8%) — checked before general Bull
    if vix < 13 and v < 0.08 and r > 0:     return "Low-Vol"
    # Crisis: VIX>25 + annual loss>5% (Ang-Bekaert 2004 inspired)
    if vix > 25 and r < -0.05:              return "Crisis"
    # Liquidity stress: positive SPY-TLT corr = flight-to-quality breaks
    if c > 0.05 and r < 0:                  return "Liquidity"
    # Macro: high vol + negative returns (rate-shock type)
    if v > 0.18 and r < 0:                  return "Macro"
    # Bear: moderate-high vol with negative returns
    if r < -0.02 and v > 0.12:              return "Bear"
    # Bull: positive returns, moderate-low vol
    if r > 0.05 and v < 0.22:               return "Bull"
    return "Neutral"

def label_states(profiles):
    labels = {}
    used   = {}
    for s, p in sorted(profiles.items(),
                        key=lambda x: x[1]["ret"], reverse=True):
        base = assign_label(p)
        # Deduplicate: if label already used, append _2, _3 etc.
        if base in used:
            used[base] += 1
            labels[s]   = f"{base}-{used[base]}"
        else:
            used[base] = 1
            labels[s]  = base
    return labels

def print_profiles(profiles, labels):
    print(f"\n{'State':7s} {'Label':14s} {'Ret%':>8s} {'Vol%':>8s} "
          f"{'VIX':>7s} {'Corr':>8s} {'Days%':>7s}")
    for s, p in sorted(profiles.items()):
        print(f"  {s:<5d} {labels[s]:14s} {p['ret']*100:>+8.1f} "
              f"{p['vol']*100:>8.1f} {p['vix']:>7.1f} "
              f"{p['corr']:>8.3f} {p['pct']:>7.1f}%")

# ══════════════════════════════════════════════════════════════
# MODULE 6 — HUNGARIAN STATE ALIGNMENT
# Globally optimal mapping via scipy.optimize.linear_sum_assignment
# ══════════════════════════════════════════════════════════════
def profile_vec(p):
    return np.array([p["ret"], p["vol"], p["vix"] / 50.0, p["corr"]])

def hungarian_align(ref_prof, new_prof):
    rk   = sorted(ref_prof.keys())
    nk   = sorted(new_prof.keys())
    cost = np.array([[np.linalg.norm(
        profile_vec(new_prof[n]) - profile_vec(ref_prof[r]))
        for r in rk] for n in nk])
    rows, cols = linear_sum_assignment(cost)
    mapping = {nk[i]: rk[j] for i, j in zip(rows, cols)}
    for n in nk:
        if n not in mapping:
            mapping[n] = min(rk, key=lambda r: np.linalg.norm(
                profile_vec(new_prof[n]) - profile_vec(ref_prof[r])))
    return mapping

# ══════════════════════════════════════════════════════════════
# MODULE 7 — CONFIDENCE GATING
# Switch regime only when confident AND persistent
# ══════════════════════════════════════════════════════════════
def gate(states, posteriors, threshold, min_persist):
    T       = len(states)
    gated   = states.copy()
    current = states[0]
    cand    = states[0]
    streak  = 1
    for t in range(1, T):
        K    = posteriors.shape[1]
        s    = states[t]
        conf = posteriors[t, s] if s < K else 0.5
        if s == cand:
            streak += 1
        else:
            cand, streak = s, 1
        if streak >= min_persist and conf >= threshold:
            current = cand
        gated[t] = current
    return gated

# ══════════════════════════════════════════════════════════════
# MODULE 8 — ROUTING & STRATEGY
# ══════════════════════════════════════════════════════════════
ROUTING = {
    "Bull":      {"SP500":1.00,"Bonds":0.00,"Gold":0.00,"Cash":0.00},
    "Bull-2":    {"SP500":1.00,"Bonds":0.00,"Gold":0.00,"Cash":0.00},
    "Neutral":   {"SP500":0.60,"Bonds":0.30,"Gold":0.10,"Cash":0.00},
    "Low-Vol":   {"SP500":0.70,"Bonds":0.20,"Gold":0.10,"Cash":0.00},
    "Low-Vol-2": {"SP500":0.70,"Bonds":0.20,"Gold":0.10,"Cash":0.00},
    "Bear":      {"SP500":0.20,"Bonds":0.50,"Gold":0.10,"Cash":0.20},
    "Macro":     {"SP500":0.10,"Bonds":0.20,"Gold":0.30,"Cash":0.40},
    "Macro-2":   {"SP500":0.10,"Bonds":0.20,"Gold":0.30,"Cash":0.40},
    "Crisis":    {"SP500":0.00,"Bonds":0.40,"Gold":0.30,"Cash":0.30},
    "Liquidity": {"SP500":0.00,"Bonds":0.20,"Gold":0.40,"Cash":0.40},
}
ASSETS = ["SP500", "Bonds", "Gold"]

def run_strategy(day_labels, asset_rets, tc_bps):
    """
    day_labels: list of label strings, one per day.
    Turnover = sum(|Δweight|) per asset per day (correct multi-asset).
    """
    n        = len(day_labels)
    port     = np.zeros(n)
    turnover = np.zeros(n)
    prev_w   = {a: 0.0 for a in ASSETS}
    wt_hist  = []

    for t, lbl in enumerate(day_labels):
        w  = ROUTING.get(lbl, ROUTING["Neutral"])
        tv = sum(abs(w.get(a,0) - prev_w.get(a,0)) for a in ASSETS)
        tc = tv * tc_bps
        r  = sum(w.get(a,0) * asset_rets.iloc[t].get(a,0)
                 for a in ASSETS if a in asset_rets.columns)
        port[t]     = r - tc
        turnover[t] = tv
        prev_w      = {a: w.get(a,0) for a in ASSETS}
        wt_hist.append({**w, "turnover":tv})

    return port, turnover, pd.DataFrame(wt_hist)

def perf(rets_arr, label, tv=None):
    r    = np.asarray(rets_arr, dtype=float)
    ann  = r.mean() * 252
    vol  = r.std()  * np.sqrt(252) + 1e-9
    sr   = ann / vol
    cum  = np.cumprod(1 + r)
    dd   = (cum / np.maximum.accumulate(cum)) - 1
    mdd  = dd.min()
    cagr = cum[-1]**(252 / len(r)) - 1
    tv_s = f"  AnnTurnover={tv.mean()*252:.2f}" if tv is not None else ""
    print(f"  {label:35s} CAGR={cagr*100:+.2f}%  "
          f"Sharpe={sr:.3f}  MaxDD={mdd*100:.2f}%{tv_s}")
    return {"label":label,"cagr":cagr,"sharpe":sr,"mdd":mdd,"cum":cum}

# ══════════════════════════════════════════════════════════════
# MODULE 9 — WALK-FORWARD RETRAINING
# Expanding window, quarterly retraining.
# Each window: fit → build own profiles → Hungarian align →
# full-sequence decode → real posteriors (Fix 6).
# ══════════════════════════════════════════════════════════════
def run_walk_forward(X_full, f_tr, f_te, ref_prof, ref_labels,
                     best_K, cfg, init_model):
    n_tr, n_te = len(f_tr), len(f_te)
    FREQ       = cfg["retrain_freq"]

    wf_labels  = ["Neutral"] * n_te
    wf_conf    = np.zeros(n_te)

    cur_model    = init_model
    cur_profiles = ref_prof
    cur_labels   = ref_labels

    print(f"\nWalk-forward: {n_te} days, retrain every {FREQ} days...")

    for i in range(n_te):
        abs_i = n_tr + i

        if i > 0 and i % FREQ == 0:
            X_wf = X_full[:abs_i]
            best_wf_bic, best_wf_m = np.inf, None
            for seed in range(cfg["wf_seeds"]):
                try:
                    m = GaussianHMM(n_components=best_K,
                                    covariance_type="diag",
                                    n_iter=cfg["wf_iters"],
                                    random_state=seed, tol=1e-4)
                    m.fit(X_wf)
                    b = compute_bic(m, X_wf)
                    if b < best_wf_bic:
                        best_wf_bic, best_wf_m = b, m
                except Exception: pass

            if best_wf_m:
                cur_model = best_wf_m
                wf_states_full, _ = map_decode(cur_model, X_wf)
                raw_wf = pd.concat([f_tr, f_te.iloc[:i]])
                cur_profiles = build_profiles(wf_states_full, raw_wf)
                mapping      = hungarian_align(ref_prof, cur_profiles)
                cur_labels   = {
                    new_s: ref_labels.get(ref_s, "Neutral")
                    for new_s, ref_s in mapping.items()
                }
            if i % 252 == 0:
                print(f"  Retrained at day {i}/{n_te}")

        # Full sequence decode — real posteriors, no synthetic reconstruction
        X_sofar   = X_full[:abs_i + 1]
        states_sf, post_sf = map_decode(cur_model, X_sofar)
        last_s    = states_sf[-1]
        last_post = post_sf[-1]
        last_conf = float(last_post[last_s])

        # Align last state to reference labels
        mapping = hungarian_align(ref_prof, cur_profiles)
        ref_s   = mapping.get(last_s, last_s)
        lbl     = ref_labels.get(ref_s, "Neutral")

        wf_labels[i] = lbl
        wf_conf[i]   = last_conf

    return wf_labels, wf_conf

# ══════════════════════════════════════════════════════════════
# MODULE 10 — PERIOD SANITY CHECKS
# ══════════════════════════════════════════════════════════════
def sanity_checks(regime_series):
    periods = {
        "2017 (low-vol bull)":  ("2017-01-01","2017-12-31",
                                  ["Bull","Low-Vol","Neutral","Bull-2"]),
        "Feb-Apr 2020 (COVID)": ("2020-02-01","2020-04-30",
                                  ["Crisis","Bear","Liquidity","Macro"]),
        "Full year 2022":       ("2022-01-01","2022-12-31",
                                  ["Macro","Bear","Crisis","Macro-2"]),
    }
    print("\n═══ PERIOD SANITY CHECKS ═══")
    for name, (s, e, expected) in periods.items():
        sub = regime_series[(regime_series.index>=s) &
                             (regime_series.index<=e)]
        if len(sub) == 0:
            print(f"  {name}: NO DATA (in training window)"); continue
        counts   = sub.value_counts(normalize=True) * 100
        dominant = counts.idxmax()
        flag     = "✓ PASS" if dominant in expected else "✗ FAIL"
        print(f"\n  {name}: {flag}")
        print(f"    Dominant: {dominant} ({counts.max():.0f}%)")
        for lbl, pct in counts.items():
            print(f"    {lbl:14s}: {pct:.1f}%")

# ══════════════════════════════════════════════════════════════
# MODULE 11 — API-STYLE OUTPUT
# Posterior keys use state_N_label format (no overwriting)
# ══════════════════════════════════════════════════════════════
def get_current_regime(model, X_full, labels, routing, conf_threshold):
    states, post = map_decode(model, X_full)
    last_s    = states[-1]
    last_post = post[-1]
    lbl       = labels.get(last_s, "Neutral")
    conf      = float(last_post[last_s])
    alloc     = routing.get(lbl, routing.get("Neutral",
                {"SP500":0.6,"Bonds":0.3,"Gold":0.1,"Cash":0.0}))

    # Print raw posteriors for validation
    print("\nRaw posterior vector (last day):")
    for s in range(len(last_post)):
        print(f"  state_{s} ({labels.get(s,'?')}): {last_post[s]:.6f}")

    return {
        "regime":     lbl,
        "confidence": round(conf, 4),
        "signal":     "ACTIVE" if conf >= conf_threshold else "LOW_CONFIDENCE",
        "allocation": {
            "SP500_weight": alloc.get("SP500", 0),
            "Bonds_weight": alloc.get("Bonds", 0),
            "Gold_weight":  alloc.get("Gold",  0),
            "Cash_weight":  alloc.get("Cash",  0),
        },
        # FIX: unique keys prevent overwriting when multiple states share label
        "all_posteriors": {
            f"state_{s}_{labels.get(s,'?')}": round(float(last_post[s]), 4)
            for s in range(len(last_post))
        }
    }

# ══════════════════════════════════════════════════════════════
# MODULE 12 — REGIME TRANSITION STATISTICS
# ══════════════════════════════════════════════════════════════
def transition_stats(regime_series):
    seq  = list(regime_series)
    runs = []
    cur, start = seq[0], 0
    for i in range(1, len(seq)):
        if seq[i] != cur:
            runs.append((cur, i - start))
            cur, start = seq[i], i
    runs.append((cur, len(seq) - start))

    dur_by = {}
    for lbl, dur in runs:
        dur_by.setdefault(lbl, []).append(dur)

    print("\n═══ REGIME TRANSITION STATISTICS ═══")
    print(f"  {'Regime':14s} {'Count':>8s} {'AvgDays':>9s} "
          f"{'Min':>7s} {'Max':>7s}")
    for lbl in sorted(dur_by):
        d = dur_by[lbl]
        print(f"  {lbl:14s} {len(d):>8d} {np.mean(d):>9.1f} "
              f"{min(d):>7d} {max(d):>7d}")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main(cfg):
    # 1. Data
    raw         = load_data(cfg)
    feat, rets  = build_features(raw)

    SPLIT = cfg["split"]
    f_tr  = feat[feat.index <  SPLIT]
    f_te  = feat[feat.index >= SPLIT]
    n_tr, n_te = len(f_tr), len(f_te)

    scaler  = StandardScaler().fit(f_tr)
    X_train = scaler.transform(f_tr)
    X_test  = scaler.transform(f_te)
    X_full  = np.vstack([X_train, X_test])
    print(f"\nTrain: {n_tr} days | OOS Test: {n_te} days")

    # 2. Model selection
    model, best_K = select_model(X_train, cfg)

    # 3. Full-sequence MAP decode (no cold-start at boundary)
    tr_states, tr_post = map_decode(model, X_full, e=n_tr)
    te_states, te_post = map_decode(model, X_full, s=n_tr)

    # 4. State profiles + labels
    tr_prof   = build_profiles(tr_states, f_tr)
    tr_labels = label_states(tr_prof)
    tr_colors = {s: COLORS.get(tr_labels[s].split("-")[0], "#888780")
                 for s in tr_prof}

    print("\n═══ STATE PROFILES ═══")
    print_profiles(tr_prof, tr_labels)

    # 5. Confidence gating (static)
    te_conf  = te_post.max(axis=1)
    te_gated = gate(te_states, te_post,
                    cfg["conf_threshold"], cfg["min_persist"])
    static_labels = [tr_labels.get(s, "Neutral") for s in te_gated]

    # 6. Walk-forward
    wf_labels, wf_conf = run_walk_forward(
        X_full, f_tr, f_te, tr_prof, tr_labels, best_K, cfg, model)

    # Gate walk-forward using real confidence
    # NOTE: gating here uses approximated label-space posteriors.
    # Each day's confidence from the true model posterior is placed
    # on the aligned label index — a known approximation.
    rl = list(ROUTING.keys())
    n_r = len(rl)
    wf_states_int = np.array([
        rl.index(l) if l in rl else 0 for l in wf_labels])
    wf_post_g = np.ones((n_te, n_r)) * (0.05 / max(n_r-1, 1))
    for i in range(n_te):
        s = wf_states_int[i]
        if s < n_r: wf_post_g[i, s] = wf_conf[i]
    wf_gated_int    = gate(wf_states_int, wf_post_g,
                           cfg["conf_threshold"], cfg["min_persist"])
    wf_gated_labels = [rl[min(s, n_r-1)] for s in wf_gated_int]

    # 7. Asset returns + benchmarks
    asset_rets = rets[ASSETS].reindex(f_te.index).fillna(0)
    spy_rets   = rets["SP500"].reindex(f_te.index).fillna(0).values
    _6040      = (0.60 * rets["SP500"].reindex(f_te.index).fillna(0).values +
                  0.40 * rets["Bonds"].reindex(f_te.index).fillna(0).values)
    _63010     = (0.60 * rets["SP500"].reindex(f_te.index).fillna(0).values +
                  0.30 * rets["Bonds"].reindex(f_te.index).fillna(0).values +
                  0.10 * rets["Gold"].reindex(f_te.index).fillna(0).values)

    # 8. Run strategies
    kr_rets, kr_tv, kr_wt = run_strategy(static_labels, asset_rets,
                                          cfg["tc_bps"])
    wf_rets, wf_tv, wf_wt = run_strategy(wf_gated_labels, asset_rets,
                                          cfg["tc_bps"])

    # 9. Performance table
    print("\n═══ PERFORMANCE SUMMARY (OOS 2020-2024) ═══")
    results = []
    results.append(perf(spy_rets, "Buy-and-Hold SPY"))
    results.append(perf(_6040,    "60/40 SPY/TLT"))
    results.append(perf(_63010,   "60/30/10 SPY/TLT/GLD"))
    results.append(perf(kr_rets,  "Kroniq Static Router", kr_tv))
    results.append(perf(wf_rets,  "Kroniq Walk-Forward",  wf_tv))

    # 10. Regime history CSV
    all_states = np.concatenate([tr_states, te_states])
    all_labels = [tr_labels.get(s, "?") for s in all_states]
    regime_ser = pd.Series(all_labels, index=feat.index)

    rh = pd.DataFrame({
        "regime":        [tr_labels.get(s,"?") for s in te_states],
        "confidence":    te_conf.round(4),
        "gated_regime":  static_labels,
        "wf_regime":     wf_gated_labels,
        "alloc_SP500":   kr_wt["SP500"].values,
        "alloc_Bonds":   kr_wt["Bonds"].values,
        "alloc_Gold":    kr_wt["Gold"].values,
        "alloc_Cash":    kr_wt["Cash"].values,
        "strategy_ret":  kr_rets.round(6),
        "spy_ret":       spy_rets.round(6),
    }, index=f_te.index)
    rh.index.name = "date"
    rh.to_csv(cfg["output_csv"])
    print(f"\nRegime history saved: {cfg['output_csv']}")

    # 11. API output
    print("\n═══ API RESPONSE (Dec 31 2024) ═══")
    api = get_current_regime(model, X_full, tr_labels, ROUTING,
                              cfg["conf_threshold"])
    for k, v in api.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items(): print(f"    {kk}: {vv}")
        else: print(f"  {k}: {v}")

    # 12. Sanity checks
    sanity_checks(regime_ser)

    # 13. Transition stats
    transition_stats(regime_ser)

    # 14. Lead time analysis
    print("\n═══ LEAD TIME — MARCH 2020 ═══")
    peak   = pd.Timestamp("2020-02-19")
    rs_te  = pd.Series([tr_labels.get(s,"?") for s in te_states],
                        index=f_te.index)
    # Find the most crisis-like state (lowest return)
    worst_s   = min(tr_prof, key=lambda s: tr_prof[s]["ret"])
    worst_lbl = tr_labels[worst_s]
    pre       = rs_te[(rs_te.index >= "2020-01-01") &
                       (rs_te.index <= peak)]
    flags     = pre[pre == worst_lbl]
    if len(flags):
        ff   = flags.index[0]
        lead = (peak - ff).days
        bot  = pd.Timestamp("2020-03-23")
        pres = ((raw["SP500"].loc[ff] - raw["SP500"].loc[bot]) /
                 raw["SP500"].loc[ff] * 100)
        print(f"  Worst regime: {worst_lbl}")
        print(f"  First flag:   {ff.date()}")
        print(f"  SPY peak:     {peak.date()}")
        print(f"  Lead time:    {lead} calendar days")
        print(f"  Drawdown avoidable: {pres:.1f}%  "
              f"(retrospective — model was not live in 2020)")
    else:
        print(f"  No {worst_lbl} flag before Feb 19 2020 peak.")
        print("  Credit spreads (Week 5) will add the missing signal.")

    # 15. Feature Z-scores
    print("\n═══ FEATURE Z-SCORES BY STATE ═══")
    print("  (>|1.5| = strongly distinctive)")
    fnames = list(feat.columns)
    print(f"  {'Feature':12s}", end="")
    for s in sorted(tr_prof.keys()):
        print(f"  {tr_labels[s]:12s}", end="")
    print()
    for fn in fnames:
        gm, gs = f_tr[fn].mean(), f_tr[fn].std() + 1e-9
        print(f"  {fn:12s}", end="")
        for s in sorted(tr_prof.keys()):
            mask = tr_states == s
            sm   = f_tr.iloc[mask][fn].mean()
            print(f"  {(sm-gm)/gs:>+10.2f}  ", end="")
        print()

    # 16. 4-panel chart
    all_d    = feat.index
    te_d     = f_te.index
    all_s    = np.concatenate([tr_states, te_gated])
    spy_full = raw["SP500"].reindex(all_d)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"Kroniq Regime Radar — Final Prototype\n"
        f"K={best_K} | VIX fixed | MAP decode | "
        f"Conf≥{cfg['conf_threshold']} | Persist≥{cfg['min_persist']}d\n"
        f"Train 2015-2019 | OOS 2020-2024",
        fontsize=12, fontweight="bold", color="#042C53")

    # Panel 1: SPY + regimes
    for i in range(len(all_d)-1):
        s   = all_s[i]
        col = tr_colors.get(s, "#888780")
        axes[0].axvspan(all_d[i], all_d[i+1], alpha=0.3, color=col)
    axes[0].plot(all_d, spy_full, color="#042C53", lw=1.0, zorder=5)
    axes[0].axvline(pd.Timestamp(SPLIT), color="black",
                    ls="--", lw=1.5, zorder=6)
    patches = [mpatches.Patch(
        color=tr_colors[s], alpha=0.7,
        label=f"{tr_labels[s]} ({tr_prof[s]['pct']:.0f}%)")
        for s in sorted(tr_prof.keys())]
    patches.append(mpatches.Patch(color="black", label="Train/Test split"))
    axes[0].legend(handles=patches, loc="upper left", fontsize=8, ncol=4)
    axes[0].set_ylabel("SPY ($)")
    axes[0].set_title("SPY Price — Gated Regime Classification")

    # Panel 2: Confidence (OOS)
    axes[1].fill_between(te_d, te_conf, cfg["conf_threshold"],
                         where=te_conf >= cfg["conf_threshold"],
                         color="#1D9E75", alpha=0.6, label="Active")
    axes[1].fill_between(te_d, te_conf, cfg["conf_threshold"],
                         where=te_conf < cfg["conf_threshold"],
                         color="#993C1D", alpha=0.5, label="Hold")
    axes[1].axhline(cfg["conf_threshold"], color="#042C53", ls="--", lw=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("MAP Posterior Confidence (OOS only)")
    axes[1].legend(fontsize=8, loc="lower right")

    # Panel 3: VIX (corrected)
    axes[2].plot(all_d, raw["VIX"].reindex(all_d),
                 color="#993C1D", lw=0.9)
    for i in range(len(all_d)-1):
        axes[2].axvspan(all_d[i], all_d[i+1], alpha=0.12,
                        color=tr_colors.get(all_s[i], "#888780"))
    axes[2].axhline(20, color="#5DCAA5", ls="--", lw=1, label="VIX=20")
    axes[2].axhline(30, color="#BA7517", ls="--", lw=1, label="VIX=30")
    axes[2].axhline(50, color="#993C1D", ls="--", lw=1, label="VIX=50")
    axes[2].set_ylabel("VIX (correct)")
    axes[2].set_title("VIX Level 2015-2024 (9-83 range)")
    axes[2].legend(fontsize=7, loc="upper left")

    # Panel 4: Cumulative performance
    axes[3].plot(te_d, np.cumprod(1+spy_rets),
                 color="#185FA5", lw=1.5, label="Buy-and-Hold SPY")
    axes[3].plot(te_d, np.cumprod(1+_6040),
                 color="#888780", lw=1.2, ls="--", label="60/40 SPY/TLT")
    axes[3].plot(te_d, np.cumprod(1+_63010),
                 color="#BA7517", lw=1.2, ls="--",
                 label="60/30/10 SPY/TLT/GLD")
    axes[3].plot(te_d, results[3]["cum"],
                 color="#1D9E75", lw=1.5,
                 label=f"Kroniq Static (CAGR {results[3]['cagr']*100:.1f}%)")
    axes[3].plot(te_d, results[4]["cum"],
                 color="#534AB7", lw=1.5,
                 label=f"Kroniq Walk-Fwd (CAGR {results[4]['cagr']*100:.1f}%)")
    axes[3].axhline(1.0, color="black", lw=0.5)
    axes[3].set_ylabel("Cumulative Return")
    axes[3].set_title("OOS Performance — 5-Way Comparison")
    axes[3].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(cfg["output_chart"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {cfg['output_chart']}")
    print("\n✓ Kroniq Regime Radar — Final Prototype complete.")
    print("Come back with this output to build the journal and SSRN v0.3.")

    return model, tr_labels, tr_prof, regime_ser

if __name__ == "__main__":
    main(CFG)
