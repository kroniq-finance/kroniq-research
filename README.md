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

### Founder
Sumanth Polavarapu — Data Engineer @ Capital One
Building Kroniq: the AWS of quantitative finance.
