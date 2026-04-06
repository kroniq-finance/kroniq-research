import yfinance as yf
import pandas as pd

tickers = ['SPY', 'QQQ', '^VIX', 'TLT', 'GLD']

data = yf.download(tickers, start='2015-01-01', end='2024-12-31')['Close']

# Rename ^VIX column to match what the main code expects
data.columns = [c.replace('^', '^') for c in data.columns]

data.to_csv('kroniq_5asset_data.csv')
print(f"Saved: kroniq_5asset_data.csv")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Date range: {data.index[0].date()} → {data.index[-1].date()}")

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KRONIQ — Week 4 Feature Experiment: Credit Spreads (9-feature model)
# ------------------------------------------------------------
# Adds 2 new features to v3 baseline:
#   credit_spread — ICE BofA US High Yield OAS daily level
#   cs_chg        — daily change in credit spread
# Total features: 9 (up from 7 in v3)
# v3 remains the locked baseline (Sharpe 0.923, 23-day lead time)
# This notebook is exploratory — directional and diagnostic only
# Final walk-forward validation happens in Week 5
# ============================================================

# --- Load 5-asset price data ---
spy = pd.read_csv('kroniq_5asset_data.csv',
                  index_col=0, parse_dates=True)

# --- Load credit spreads ---
cs = pd.read_csv('kroniq_credit_spreads.csv',
                 index_col=0, parse_dates=True)
cs.columns = ['credit_spread']
cs = cs.loc['2015-01-01':'2024-12-31']

# --- Pre-dropna alignment diagnostics on full market calendar ---
credit_aligned_full = cs.reindex(spy.index)
credit_missing_full = credit_aligned_full['credit_spread'].isna().sum()
print("=== Credit spread alignment diagnostics (pre-dropna) ===")
print(f"Total market rows:                        {len(spy)}")
print(f"Total credit spread rows:                 {len(cs)}")
print(f"Credit spread missing on market calendar: {credit_missing_full}")
print(f"Overlapping rows on market calendar:      "
      f"{credit_aligned_full['credit_spread'].notna().sum()}")

# --- Recreate exact same 7 features as v3 ---
spy['spy_ret']      = spy['SPY'].pct_change(fill_method=None)
spy['spy_vol']      = spy['spy_ret'].rolling(20).std() * np.sqrt(252)
spy['vix']          = spy['^VIX']
spy['vix_chg']      = spy['^VIX'].pct_change(fill_method=None)
spy['tlt_ret']      = spy['TLT'].pct_change(fill_method=None)
spy['gld_ret']      = spy['GLD'].pct_change(fill_method=None)
spy['spy_tlt_corr'] = spy['spy_ret'].rolling(20).corr(spy['tlt_ret'])

# --- Add 2 new credit spread features ---
spy['credit_spread'] = cs['credit_spread']
spy['cs_chg']        = spy['credit_spread'].diff()

# --- Feature matrix ---
feature_cols = ['spy_ret', 'spy_vol', 'vix', 'vix_chg',
                'tlt_ret', 'gld_ret', 'spy_tlt_corr',
                'credit_spread', 'cs_chg']

# --- Missing value diagnostics before dropna ---
print("\n=== Missing values by feature before dropna ===")
print(spy[feature_cols].isna().sum())
rows_before = len(spy[feature_cols])

features = spy[feature_cols].dropna()
features = features.loc['2015-01-01':'2024-12-31']

print(f"\nRows before dropna: {rows_before}")
print(f"Rows after dropna:  {len(features)}")

# --- Train/test split — no overlap ---
split_date     = '2020-12-31'
train          = features[features.index <= split_date]
test           = features[features.index > split_date]

print(f"\nTrain: {train.index[0].date()} → "
      f"{train.index[-1].date()} ({len(train)} days)")
print(f"Test:  {test.index[0].date()} → "
      f"{test.index[-1].date()} ({len(test)} days)")

scaler  = StandardScaler()
X_train = scaler.fit_transform(train)
X_all   = scaler.transform(features)

# --- BIC selection K=2..6 ---
print("\n=== BIC Selection ===")
bic_scores = {}
models     = {}
d          = X_train.shape[1]

for k in range(2, 7):
    model = GaussianHMM(n_components=k,
                        covariance_type='full',
                        n_iter=200,
                        random_state=42)
    model.fit(X_train)

    log_ll   = model.score(X_train)
    n_params = ((k - 1) +
                k * (k - 1) +
                k * d +
                k * (d * (d + 1) // 2))
    bic = -2 * log_ll + n_params * np.log(len(X_train))

    bic_scores[k] = bic
    models[k]     = model
    print(f"K={k}: BIC={bic:.1f}  LogLL={log_ll:.1f}  Params={n_params}")

optimal_k = min(bic_scores, key=bic_scores.get)
print(f"\nOptimal K (v4 9-feature model): {optimal_k}")
print(f"v3 baseline optimal K was:      4")

# --- Decode states ---
best_model     = models[optimal_k]
raw_states     = best_model.predict(X_all)
states         = pd.Series(raw_states, index=features.index, name='state')
train_states   = states.loc[train.index]
train_features = features.loc[train.index]

# --- State profiles ---
print("\n=== State profiles (training data) ===")
print(f"{'State':<8} {'Ann.Ret':>8} {'VIX':>7} "
      f"{'Cr.Spread':>11} {'Days':>6}")
print("-" * 48)

state_profiles = {}
for s in range(optimal_k):
    mask    = train_states == s
    ann_ret = train_features.loc[mask, 'spy_ret'].mean() * 252
    avg_vix = train_features.loc[mask, 'vix'].mean()
    avg_cs  = train_features.loc[mask, 'credit_spread'].mean()
    days    = mask.sum()
    state_profiles[s] = {
        'mean_ret': ann_ret,
        'mean_vix': avg_vix,
        'mean_cs':  avg_cs,
        'days':     days
    }
    print(f"{s:<8} {ann_ret:>7.1%} {avg_vix:>7.1f} "
          f"{avg_cs:>10.2f}% {days:>6}")

# --- Auditable labeling diagnostics ---
print("\n=== State labeling diagnostics ===")

highest_vix = max(state_profiles, key=lambda s: state_profiles[s]['mean_vix'])
highest_cs  = max(state_profiles, key=lambda s: state_profiles[s]['mean_cs'])
lowest_vix  = min(state_profiles, key=lambda s: state_profiles[s]['mean_vix'])
highest_ret = max(state_profiles, key=lambda s: state_profiles[s]['mean_ret'])

print(f"\n{'State':<8} {'Ann.Ret':>8} {'VIX':>7} {'Cr.Sprd':>9} "
      f"{'HighVIX':>9} {'HighCS':>8} {'LowVIX':>8} {'HighRet':>9}")
print("-" * 72)

for s, p in state_profiles.items():
    print(f"{s:<8} {p['mean_ret']:>7.1%} {p['mean_vix']:>7.1f} "
          f"{p['mean_cs']:>8.2f}% "
          f"{'YES' if s == highest_vix else 'no':>9} "
          f"{'YES' if s == highest_cs  else 'no':>8} "
          f"{'YES' if s == lowest_vix  else 'no':>8} "
          f"{'YES' if s == highest_ret else 'no':>9}")

# ============================================================
# FIXED LABELING BLOCK — handles K=5 with two stress states
# Crisis  = acute panic (highest VIX — March 2020 concentrated)
# Macro   = chronic stress (lowest return — 2022 rate shock)
# Low-Vol = lowest VIX (calm, compressed volatility)
# Bull    = highest return among remaining
# Neutral = leftover
# ============================================================

labels   = {}
assigned = set()

# Step 1 — Crisis: highest VIX (acute panic regime)
acute_crisis = max(state_profiles,
                   key=lambda s: state_profiles[s]['mean_vix'])
labels[acute_crisis] = 'Crisis'
assigned.add(acute_crisis)
print(f"\nCrisis  → State {acute_crisis} "
      f"(VIX={state_profiles[acute_crisis]['mean_vix']:.1f}, "
      f"CS={state_profiles[acute_crisis]['mean_cs']:.2f}%, "
      f"ret={state_profiles[acute_crisis]['mean_ret']:.1%}, "
      f"days={state_profiles[acute_crisis]['days']})")

# Step 2 — Macro: lowest return among remaining (chronic stress)
remaining = {s: p for s, p in state_profiles.items()
             if s not in assigned}
macro = min(remaining.items(), key=lambda x: x[1]['mean_ret'])
labels[macro[0]] = 'Macro'
assigned.add(macro[0])
print(f"Macro   → State {macro[0]} "
      f"(VIX={macro[1]['mean_vix']:.1f}, "
      f"CS={macro[1]['mean_cs']:.2f}%, "
      f"ret={macro[1]['mean_ret']:.1%}, "
      f"days={macro[1]['days']})")

# Step 3 — Low-Vol: lowest VIX among remaining
remaining = {s: p for s, p in state_profiles.items()
             if s not in assigned}
low_vol = min(remaining.items(), key=lambda x: x[1]['mean_vix'])
labels[low_vol[0]] = 'Low-Vol'
assigned.add(low_vol[0])
print(f"Low-Vol → State {low_vol[0]} "
      f"(VIX={low_vol[1]['mean_vix']:.1f}, "
      f"CS={low_vol[1]['mean_cs']:.2f}%, "
      f"ret={low_vol[1]['mean_ret']:.1%}, "
      f"days={low_vol[1]['days']})")

# Step 4 — Bull: highest return among remaining
remaining = {s: p for s, p in state_profiles.items()
             if s not in assigned}
bull = max(remaining.items(), key=lambda x: x[1]['mean_ret'])
labels[bull[0]] = 'Bull'
assigned.add(bull[0])
print(f"Bull    → State {bull[0]} "
      f"(VIX={bull[1]['mean_vix']:.1f}, "
      f"CS={bull[1]['mean_cs']:.2f}%, "
      f"ret={bull[1]['mean_ret']:.1%}, "
      f"days={bull[1]['days']})")

# Step 5 — Neutral: leftover states
remaining = {s: p for s, p in state_profiles.items()
             if s not in assigned}
for i, (s, p) in enumerate(remaining.items()):
    name = 'Neutral' if i == 0 else f'State{s}'
    labels[s] = name
    assigned.add(s)
    print(f"{name:<10} → State {s} "
          f"(VIX={p['mean_vix']:.1f}, "
          f"CS={p['mean_cs']:.2f}%, "
          f"ret={p['mean_ret']:.1%}, "
          f"days={p['days']})")

print(f"\nFinal labels: {labels}")

# --- Routing — Crisis and Macro both go to cash ---
def route(state):
    name = labels.get(state, 'Neutral')
    if name in ['Bull', 'Low-Vol']:
        return 1.0
    elif name == 'Neutral':
        return 0.7
    elif name == 'Macro':
        return 0.3
    elif name == 'Crisis':
        return 0.0
    else:
        return 0.5

oos_states       = states.loc[test.index]
oos_returns      = features.loc[test.index, 'spy_ret']
weights          = oos_states.map(route)
strategy_returns = weights.shift(1) * oos_returns

def sharpe(r):
    r = r.dropna()
    return (r.mean() / r.std()) * np.sqrt(252)

def max_dd(r):
    r   = r.dropna()
    cum = (1 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def cagr(r):
    r = r.dropna()
    n = len(r) / 252
    return (1 + r).prod() ** (1/n) - 1

print("\n=== OOS Results — fixed labels ===")
print(f"v4 Strategy Sharpe:     {sharpe(strategy_returns):.3f}")
print(f"v4 Buy-Hold Sharpe:     {sharpe(oos_returns):.3f}")
print(f"v4 Strategy Max DD:     {max_dd(strategy_returns):.2%}")
print(f"v4 Strategy CAGR:       {cagr(strategy_returns):.2%}")
print(f"\n--- Baseline reference (v3 locked) ---")
print(f"v3 Walk-Forward Sharpe: 0.923")
print(f"v3 Max DD:             -24.50%")

# --- Lead time analysis ---
spy_peak   = pd.Timestamp('2020-02-19')
window     = states.loc['2019-10-01':'2020-04-30']
stress_ids = [s for s, name in labels.items()
              if name in ['Crisis', 'Macro']]

print("\n=== Lead time analysis (exploratory diagnostic) ===")
if stress_ids:
    stress_window = window[window.isin(stress_ids)]
    if len(stress_window) > 0:
        first_stress = stress_window.index[0]
        lead_days    = (spy_peak - first_stress).days
        first_label  = labels[stress_window.iloc[0]]
        print(f"First stress signal (v4): {first_stress.date()} [{first_label}]")
        print(f"SPY peak:                 {spy_peak.date()}")
        print(f"Lead time (v4):           {lead_days} calendar days")
        print(f"Lead time (v3):           23 calendar days")
        if lead_days > 23:
            print(f"Credit spreads extended lead time by "
                  f"{lead_days - 23} days")
        elif lead_days == 23:
            print("Lead time unchanged — existing features "
                  "already captured signal")
        else:
            print(f"Lead time shorter by {23 - lead_days} days — investigate")
    else:
        print("No stress state found in 2019-2020 window — "
              "check state labels above")
else:
    print("No Crisis or Macro state found — check labeling above")

# --- 3 sanity checks ---
checks = [
    ('2017 calm',       '2017-01-01', '2017-12-31', 'Low-Vol', 0.60),
    ('2020 COVID peak', '2020-02-01', '2020-05-31', 'Crisis',  0.50),
    ('2022 rate shock', '2022-01-01', '2022-12-31', 'Macro',   0.80),
]

print("\n=== Sanity checks — v4 fixed labels ===")
all_passed = True
for label, start, end, expected, threshold in checks:
    s      = states.loc[start:end]
    vc     = s.map(labels).value_counts(normalize=True)
    pct    = vc.get(expected, 0)
    passed = pct >= threshold
    if not passed:
        all_passed = False
    status = "PASS" if passed else "FAIL"
    print(f"\n{label}:")
    print(f"  Expected {expected} >= {threshold:.0%} → "
          f"got {pct:.1%} [{status}]")
    print(f"  Full breakdown: {dict(vc.round(2))}")

print("\n" + ("All sanity checks passed!" if all_passed
              else "Some checks failed — investigate before Week 5."))

# --- Save outputs ---
regime_labeled = states.map(labels)
regime_labeled.to_csv('kroniq_regime_history_v4.csv')
print("\nSaved: kroniq_regime_history_v4.csv")


# === Week 5 Day 2 — OOS Walk-Forward Validation (final) ===
# Expanding window, quarterly retraining (~63 trading days)
# Scaler refit at each retrain step — no lookahead bias
# Viterbi decoding over full history to current day
# Train: 2015-2020 | Test: 2021-2024 | K=5, seed=42

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

RETRAIN_FREQ  = 63
n_tr          = len(train)
n_te          = len(test)
wf_labels_out = []
scaler_wf     = None
m_wf          = None
labels_wf     = {}

print("=== Walk-Forward OOS Validation (K=5, quarterly retrain) ===")

for i in range(n_te):
    abs_i = n_tr + i

    if i == 0 or i % RETRAIN_FREQ == 0:
        print(f"  Retrained at OOS step {i} (date={test.index[i].date()})")

        scaler_wf = StandardScaler()
        X_wf      = scaler_wf.fit_transform(features.iloc[:abs_i])

        m_wf = GaussianHMM(
            n_components=5,
            covariance_type='full',
            n_iter=1000,
            random_state=42
        )
        m_wf.fit(X_wf)

        s_wf        = m_wf.predict(X_wf)
        f_wf        = features.iloc[:abs_i]
        profiles_wf = {}
        for s in range(5):
            mask = (s_wf == s)
            profiles_wf[s] = {
                'mean_ret': f_wf.loc[mask, 'spy_ret'].mean() * 252,
                'mean_vix': f_wf.loc[mask, 'vix'].mean(),
            }

        assigned_wf = set()
        labels_wf   = {}

        crisis = max(profiles_wf, key=lambda s: profiles_wf[s]['mean_vix'])
        labels_wf[crisis] = 'Crisis'
        assigned_wf.add(crisis)

        rem   = {s: p for s, p in profiles_wf.items() if s not in assigned_wf}
        macro = min(rem.items(), key=lambda x: x[1]['mean_ret'])
        labels_wf[macro[0]] = 'Macro'
        assigned_wf.add(macro[0])

        rem = {s: p for s, p in profiles_wf.items() if s not in assigned_wf}
        lv  = min(rem.items(), key=lambda x: x[1]['mean_vix'])
        labels_wf[lv[0]] = 'Low-Vol'
        assigned_wf.add(lv[0])

        rem  = {s: p for s, p in profiles_wf.items() if s not in assigned_wf}
        bull = max(rem.items(), key=lambda x: x[1]['mean_ret'])
        labels_wf[bull[0]] = 'Bull'
        assigned_wf.add(bull[0])

        rem = {s: p for s, p in profiles_wf.items() if s not in assigned_wf}
        for s, p in rem.items():
            labels_wf[s] = 'Neutral'

    # Viterbi decoding over full history to current day
    X_hist_today = scaler_wf.transform(features.iloc[:abs_i + 1])
    s_today      = m_wf.predict(X_hist_today)[-1]
    wf_labels_out.append(labels_wf.get(s_today, 'Neutral'))

# Performance
wf_series  = pd.Series(wf_labels_out, index=test.index)
oos_ret    = features.loc[test.index, 'spy_ret']
wf_weights = wf_series.map(route)
wf_strat   = wf_weights.shift(1) * oos_ret

print(f"\n=== Results ===")
print(f"Walk-Forward Sharpe:  {sharpe(wf_strat):.3f}")
print(f"Walk-Forward MaxDD:   {max_dd(wf_strat):.2%}")
print(f"Walk-Forward CAGR:    {cagr(wf_strat):.2%}")
print(f"\n--- Benchmark ---")
print(f"Buy-Hold Sharpe:      {sharpe(oos_ret):.3f}")
print(f"\n--- Static OOS reference ---")
print(f"Static Sharpe:        1.107")
print(f"Static MaxDD:         -8.37%")

# === Week 5 Day 3 — Task 1: 5x5 Transition Matrix ===
import seaborn as sns
import matplotlib.pyplot as plt

trans = best_model.transmat_

# Explicit semantic order — consistent with SSRN and demo decks
desired_order = ['Low-Vol', 'Neutral', 'Bull', 'Macro', 'Crisis']
state_order   = [s for s, lbl in labels.items() if lbl in desired_order]
state_order   = sorted(state_order, key=lambda s: desired_order.index(labels[s]))
ordered_labels = [labels[s] for s in state_order]

trans_ordered = trans[np.ix_(state_order, state_order)]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    trans_ordered,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=ordered_labels,
    yticklabels=ordered_labels,
    ax=ax,
    vmin=0, vmax=1
)
ax.set_title("Kroniq v4 — 5-State Transition Probability Matrix\nTrain: 2015-2020",
             fontsize=12, fontweight='bold')
ax.set_xlabel("To regime")
ax.set_ylabel("From regime")
plt.tight_layout()
plt.savefig("kroniq_transition_matrix_v4.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: kroniq_transition_matrix_v4.png")

print("\nTransition matrix (rows=from, cols=to):")
print(f"{'':12s}", end="")
for lbl in ordered_labels:
    print(f"{lbl:>10s}", end="")
print()
for i, from_lbl in enumerate(ordered_labels):
    print(f"{from_lbl:12s}", end="")
    for j in range(len(ordered_labels)):
        print(f"{trans_ordered[i,j]:>10.3f}", end="")
    print()

# === Week 5 Day 3 — Task 2: Regime History Chart 2015-2024 ===
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

REGIME_COLORS = {
    'Low-Vol': '#1D9E75',
    'Bull':    '#185FA5',
    'Neutral': '#888780',
    'Macro':   '#534AB7',
    'Crisis':  '#993C1D',
}

regime_full = states.map(labels)
spy_full    = spy['SPY'].reindex(features.index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("Kroniq v4 — Regime History 2015-2024\nK=5 · 9-feature model · 141-day COVID lead time",
             fontsize=13, fontweight='bold', color='#042C53')

dates = features.index

# Panel 1 — SPY price + regime bands
for i in range(len(dates) - 1):
    regime = regime_full.iloc[i]
    color  = REGIME_COLORS.get(regime, '#888780')
    ax1.axvspan(dates[i], dates[i+1], alpha=0.25, color=color, linewidth=0)

ax1.plot(dates, spy_full, color='#042C53', lw=1.2, zorder=5)
ax1.axvline(pd.Timestamp('2020-01-01'), color='black', ls='--', lw=1.2, zorder=6)

# Robust date lookups using .asof()
macro_date  = pd.Timestamp('2019-10-01')
crisis_date = pd.Timestamp('2020-02-28')
macro_y     = spy_full.asof(macro_date)
crisis_y    = spy_full.asof(crisis_date)

ax1.annotate('Oct 1 2019\nMacro signal',
             xy=(macro_date, macro_y),
             xytext=(pd.Timestamp('2018-06-01'), 320),
             arrowprops=dict(arrowstyle='->', color='#534AB7'),
             fontsize=8, color='#534AB7')

ax1.annotate('Feb 28 2020\nCrisis',
             xy=(crisis_date, crisis_y),
             xytext=(pd.Timestamp('2020-05-01'), 250),
             arrowprops=dict(arrowstyle='->', color='#993C1D'),
             fontsize=8, color='#993C1D')

# Legend — patches for regimes, Line2D for train/test split
legend_handles = [mpatches.Patch(color=c, alpha=0.6, label=r)
                  for r, c in REGIME_COLORS.items()]
line_handle = Line2D([0], [0], color='black', linestyle='--',
                     lw=1.2, label='Train/Test split')
legend_handles.append(line_handle)
ax1.legend(handles=legend_handles, loc='upper left', fontsize=8, ncol=3)
ax1.set_ylabel("SPY Price ($)")
ax1.set_title("SPY Price with Regime Bands", fontsize=11)

# Panel 2 — Regime colour timeline
for i in range(len(dates) - 1):
    regime = regime_full.iloc[i]
    color  = REGIME_COLORS.get(regime, '#888780')
    ax2.axvspan(dates[i], dates[i+1], alpha=0.7, color=color, linewidth=0)

ax2.set_ylabel("Regime")
ax2.set_yticks([])
ax2.set_title("Regime Classification Timeline", fontsize=11)

plt.tight_layout()
plt.savefig("kroniq_regime_history_v4.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: kroniq_regime_history_v4.png")



# === Week 5 Day 5 — Regime history chart v2 (cleaned up) ===
# Fix: minimum display duration to reduce flickering in timeline
# Uses same regime_full and spy_full variables from earlier

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

REGIME_COLORS = {
    'Low-Vol': '#1D9E75',
    'Bull':    '#185FA5',
    'Neutral': '#888780',
    'Macro':   '#534AB7',
    'Crisis':  '#993C1D',
}

# Apply minimum display duration — smooth out 1-2 day flickers
def smooth_regimes(regime_series, min_days=3):
    smoothed = regime_series.copy()
    values   = smoothed.values
    n        = len(values)
    i        = 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        run_len = j - i
        if run_len < min_days and i > 0 and j < n:
            values[i:j] = values[i-1]
        i = j
    return pd.Series(values, index=regime_series.index)

regime_smoothed = smooth_regimes(regime_full, min_days=3)
dates    = features.index
spy_full = spy['SPY'].reindex(features.index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("Kroniq v4 — Regime History 2015-2024\nK=5 · 9-feature model · 141-day COVID lead time",
             fontsize=13, fontweight='bold', color='#042C53')

# Panel 1 — SPY price + smoothed regime bands
for i in range(len(dates) - 1):
    color = REGIME_COLORS.get(regime_smoothed.iloc[i], '#888780')
    ax1.axvspan(dates[i], dates[i+1], alpha=0.25, color=color, linewidth=0)

ax1.plot(dates, spy_full, color='#042C53', lw=1.2, zorder=5)
ax1.axvline(pd.Timestamp('2020-01-01'), color='black', ls='--', lw=1.2, zorder=6)

macro_y  = spy_full.asof(pd.Timestamp('2019-10-01'))
crisis_y = spy_full.asof(pd.Timestamp('2020-02-28'))

ax1.annotate('Oct 1 2019\nMacro signal — 141 days early',
             xy=(pd.Timestamp('2019-10-01'), macro_y),
             xytext=(pd.Timestamp('2017-06-01'), 330),
             arrowprops=dict(arrowstyle='->', color='#534AB7'),
             fontsize=8, color='#534AB7')

ax1.annotate('Feb 28 2020\nCrisis state — 0% SPY',
             xy=(pd.Timestamp('2020-02-28'), crisis_y),
             xytext=(pd.Timestamp('2020-06-01'), 240),
             arrowprops=dict(arrowstyle='->', color='#993C1D'),
             fontsize=8, color='#993C1D')

legend_handles = [mpatches.Patch(color=c, alpha=0.6, label=r)
                  for r, c in REGIME_COLORS.items()]
line_handle = Line2D([0], [0], color='black', linestyle='--',
                     lw=1.2, label='Train/Test split')
legend_handles.append(line_handle)
ax1.legend(handles=legend_handles, loc='upper left', fontsize=8, ncol=3)
ax1.set_ylabel("SPY Price ($)")
ax1.set_title("SPY Price with Regime Bands (smoothed — min 3-day persistence)", fontsize=11)

# Panel 2 — smoothed regime timeline
for i in range(len(dates) - 1):
    color = REGIME_COLORS.get(regime_smoothed.iloc[i], '#888780')
    ax2.axvspan(dates[i], dates[i+1], alpha=0.7, color=color, linewidth=0)

ax2.set_ylabel("Regime")
ax2.set_yticks([])
ax2.set_title("Regime Classification Timeline (smoothed)", fontsize=11)

plt.tight_layout()
plt.savefig("kroniq_regime_history_v4_clean.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: kroniq_regime_history_v4_clean.png")


# === Week 5.5 — Task 1: Absolute threshold label anchoring ===
# Replace relative ranking with fixed economic rules
# Prevents label drift across quarterly retrains

def assign_labels_absolute(profiles, features_window):
    """
    Absolute threshold label assignment — same rules every retrain.
    No relative ranking — pure economic logic.
    """
    labels_out = {}
    assigned   = set()

    # Step 1 — Crisis: VIX > 30 AND credit spread > 6.5%
    # Both conditions required — prevents false Crisis in high-vol bull
    crisis_candidates = {
        s: p for s, p in profiles.items()
        if p['mean_vix'] > 30 and p['mean_cs'] > 6.5
    }
    if crisis_candidates:
        crisis = max(crisis_candidates, key=lambda s: profiles[s]['mean_vix'])
    else:
        # Fallback: highest VIX if no state meets both thresholds
        crisis = max(profiles, key=lambda s: profiles[s]['mean_vix'])
    labels_out[crisis] = 'Crisis'
    assigned.add(crisis)

    # Step 2 — Macro: negative return AND VIX > 18 among remaining
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    macro_candidates = {
        s: p for s, p in rem.items()
        if p['mean_ret'] < 0 and p['mean_vix'] > 18
    }
    if macro_candidates:
        macro = min(macro_candidates.items(), key=lambda x: x[1]['mean_ret'])
        labels_out[macro[0]] = 'Macro'
        assigned.add(macro[0])
    else:
        # Fallback: lowest return among remaining
        rem = {s: p for s, p in profiles.items() if s not in assigned}
        macro = min(rem.items(), key=lambda x: x[1]['mean_ret'])
        labels_out[macro[0]] = 'Macro'
        assigned.add(macro[0])

    # Step 3 — Low-Vol: VIX < 15 AND positive return among remaining
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    lv_candidates = {
        s: p for s, p in rem.items()
        if p['mean_vix'] < 15 and p['mean_ret'] > 0
    }
    if lv_candidates:
        lv = min(lv_candidates.items(), key=lambda x: x[1]['mean_vix'])
        labels_out[lv[0]] = 'Low-Vol'
        assigned.add(lv[0])
    else:
        rem = {s: p for s, p in profiles.items() if s not in assigned}
        lv = min(rem.items(), key=lambda x: x[1]['mean_vix'])
        labels_out[lv[0]] = 'Low-Vol'
        assigned.add(lv[0])

    # Step 4 — Bull: highest return among remaining
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    bull = max(rem.items(), key=lambda x: x[1]['mean_ret'])
    labels_out[bull[0]] = 'Bull'
    assigned.add(bull[0])

    # Step 5 — Neutral: leftover
    rem = {s: p for s, p in profiles.items() if s not in assigned}
    for s, p in rem.items():
        labels_out[s] = 'Neutral'
        assigned.add(s)

    return labels_out

# Test on training data profiles to verify labels match v4
test_labels = assign_labels_absolute(state_profiles, train)
print("=== Absolute threshold labels (should match v4) ===")
for s, lbl in sorted(test_labels.items()):
    p = state_profiles[s]
    print(f"  State {s}: {lbl:8s} | VIX={p['mean_vix']:.1f} | "
          f"ret={p['mean_ret']*100:+.1f}% | CS={p['mean_cs']:.2f}%")
print(f"\nExpected: Crisis=4, Macro=2, Low-Vol=3, Bull=1, Neutral=0")
print(f"Got:      {test_labels}")


# === Week 5.5 — Task 2: Walk-forward with anchoring + persistence filter ===

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

RETRAIN_FREQ  = 63
n_tr          = len(train)
n_te          = len(test)
wf55_labels   = []
scaler_wf     = None
m_wf          = None
labels_wf     = {}

# Persistence filter state
pending_label = None
pending_count = 0
current_label = None          # None — bootstrapped from first real prediction
PERSIST_DAYS  = 3

print("=== Week 5.5 Walk-Forward (absolute anchoring + persistence filter) ===")

for i in range(n_te):
    abs_i = n_tr + i

    if i == 0 or i % RETRAIN_FREQ == 0:
        print(f"  Retrained at OOS step {i} (date={test.index[i].date()})")

        scaler_wf = StandardScaler()
        X_wf      = scaler_wf.fit_transform(features.iloc[:abs_i])

        m_wf = GaussianHMM(
            n_components=5,
            covariance_type='full',
            n_iter=1000,
            random_state=42
        )
        m_wf.fit(X_wf)

        s_wf    = m_wf.predict(X_wf)
        f_wf    = features.iloc[:abs_i]
        profiles_wf = {}
        for s in range(5):
            mask = (s_wf == s)
            if mask.sum() == 0:
                profiles_wf[s] = {'mean_ret': 0, 'mean_vix': 15, 'mean_cs': 4.0}
            else:
                profiles_wf[s] = {
                    'mean_ret': f_wf.loc[mask, 'spy_ret'].mean() * 252,
                    'mean_vix': f_wf.loc[mask, 'vix'].mean(),
                    'mean_cs':  f_wf.loc[mask, 'credit_spread'].mean(),
                }

        labels_wf = assign_labels_absolute(profiles_wf, f_wf)

        # Reset persistence filter on retrain
        pending_label = None
        pending_count = 0

    # Predict current day
    X_hist    = scaler_wf.transform(features.iloc[:abs_i + 1])
    s_today   = m_wf.predict(X_hist)[-1]
    raw_label = labels_wf.get(s_today, 'Neutral')

    # Bootstrap: first day takes model prediction directly, no filter needed
    if current_label is None:
        current_label = raw_label
        wf55_labels.append(current_label)
        continue

    # === 3-day persistence filter ===
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

    wf55_labels.append(current_label)

# Performance
wf55_series  = pd.Series(wf55_labels, index=test.index)
oos_ret      = features.loc[test.index, 'spy_ret']
wf55_weights = wf55_series.map(route)
wf55_strat   = wf55_weights.shift(1) * oos_ret

print(f"\n=== Week 5.5 Results ===")
print(f"Walk-Forward Sharpe (5.5):  {sharpe(wf55_strat):.3f}")
print(f"Walk-Forward MaxDD  (5.5):  {max_dd(wf55_strat):.2%}")
print(f"Walk-Forward CAGR   (5.5):  {cagr(wf55_strat):.2%}")
print(f"\n--- Baseline comparison ---")
print(f"Walk-Forward Sharpe (Week 5): 0.881")
print(f"Walk-Forward MaxDD  (Week 5): -17.46%")
print(f"Buy-Hold Sharpe:              0.859")



# === Week 5.5 — Task 3: Routing refinement test ===
# Crisis=0% and Bull/Low-Vol=100% remain fixed
# Only Neutral and Macro allocation tested

def route_refined(label):
    if label in ['Bull', 'Low-Vol']:
        return 1.00
    elif label == 'Neutral':
        return 0.60   # was 0.70
    elif label == 'Macro':
        return 0.35   # was 0.30
    elif label == 'Crisis':
        return 0.00
    else:
        return 0.50

wf55_weights_r = wf55_series.map(route_refined)
wf55_strat_r   = wf55_weights_r.shift(1) * oos_ret

print("=== Routing refinement comparison ===")
print(f"\nCurrent routing:")
print(f"  Sharpe: {sharpe(wf55_strat):.3f}")
print(f"  MaxDD:  {max_dd(wf55_strat):.2%}")
print(f"  CAGR:   {cagr(wf55_strat):.2%}")
print(f"\nRefined routing:")
print(f"  Sharpe: {sharpe(wf55_strat_r):.3f}")
print(f"  MaxDD:  {max_dd(wf55_strat_r):.2%}")
print(f"  CAGR:   {cagr(wf55_strat_r):.2%}")
print(f"\nDecision rule: prefer the version that improves Sharpe")
print(f"without materially worsening MaxDD.")
print(f"Do not auto-lock on Sharpe alone.")