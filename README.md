# Kroniq Research
### The Intelligence of Market Timing

Real-time market regime detection infrastructure for quantitative finance.
Building the API that tells every quant fund what the market is actually doing.

---

## Week 1 — Python for Finance (March 26–30, 2026)

### What has been built so far

| File | Description |
|------|-------------|
| kroniq_week1_spy.ipynb | SPY 5-year price chart — first data pull |
| kroniq_week1_5asset_data.ipynb | 5-asset data pipeline (SPY, QQQ, VIX, TLT, GLD) |
| kroniq_5asset_prices.csv | 10 years daily closing prices — 2,765 rows |
| kroniq_5asset_returns.csv | Daily percentage returns — all 5 assets |
| kroniq_5asset_volatility.png | Rolling 20-day volatility chart with regime events |

### Research thesis
Quant algorithms are blind to market regime shifts.
When the regime changes, strategies built for one regime
destroy capital in another. Kroniq detects regime shifts
in real time using cross-asset Hidden Markov Models.

### Stack
Python | pandas | numpy | matplotlib | yfinance | hmmlearn

## Week 2 — Quant Risk Analysis (Mar 30 – Apr 3) | File | Key Result | |------|-----------| | kroniq_week2_montecarlo.ipynb | Sharpe 0.5308 · Drawdown -33.72% | | kroniq_week2_var_cvar.ipynb | VaR -1.93% · CVaR -3.20% · 11.58% gap | | kroniq_week2_regime_routing.ipynb | 7×6 routing table — Engine 2 |

### Founder
Sumanth Polavarapu — Data Engineer @ Capital One.
Building Kroniq: the AWS of quantitative finance.

## Week 3 Results (April 6-12, 2026)

### Model selection — BIC sweep K=3..9
| K | BIC | Min state days | Decision |
|---|-----|---------------|----------|
| 3 | 24,724.7 | — | Rejected |
| 4 | 23,852.4 | — | Rejected |
| **5** | **22,867.2** | **233 days** | **★ Production model** |
| 6 | 23,035.0 | 24 days | Rejected (unstable) |
| 7 | 23,048.8 | 9 days | Rejected (unstable) |
| 8 | 22,521.7 | 13 days | Rejected (unstable) |
| 9 | 22,720.8 | — | Rejected |

### v4 Production model (K=5, 9 features)
- Static OOS Sharpe: **1.107** | MaxDD: **-8.37%** | CAGR: 7.94%
- Walk-forward OOS Sharpe: **0.881** | MaxDD: -17.46% | CAGR: 9.97%
- Buy-and-hold Sharpe: 0.859
- 141-day COVID early warning (credit spreads)
- All 3 sanity checks pass

### Transition matrix persistence
Low-Vol: 95.5% | Bull: 92.7% | Neutral: 92.5% | Macro: 81.7% | Crisis: 73.0%
