Project structure:

regime_momentum_pairs_vol/
├── README.md
├── requirements.txt
├── run_backtest.py
├── data/
│   └── cache/                      # CSVs auto-saved here
├── signals/
│   ├── __init__.py
│   ├── momentum.py
│   ├── pairs.py
│   └── volatility.py
├── strategy/
│   ├── __init__.py
│   ├── position_sizing.py
│   └── regime_strategy.py
├── backtest/
│   ├── __init__.py
│   ├── backtester.py
│   └── metrics.py
└── tests/
    ├── test_data.py
    ├── test_signals.py
    ├── test_strategy.py
    └── test_backtest.py

# Regime-Aware Momentum & Pairs Trading with Volatility Intelligence

**Universe** (daily, 2014-01-01 → today, free data via yfinance):

["AAPL", "MSFT", "AMZN", "NVDA", "SPY", "QQQ", "IWM",
"XLE", "XLK", "USO", "^VIX"] # ^VIX only for regime detection


| Regime | Definition | Active Signals |
|--------|------------|----------------|
| **LOW** | VIX < 15 | 6-month cross-sectional momentum |
| **MID** | 15 ≤ VIX ≤ 25 | momentum (half size) |
| **HIGH** | VIX > 25 | market-neutral pairs trades (SPY–QQQ, XLE–USO) |

* **Volatility spread** = VIX – realised 20-day σ<sub>SPY</sub>.  
  Extremes auto-scale risk (Kelly-fraction × vol-target).
* **Execution**: positions re-balanced daily at close, 10 bp one-way cost.
* **Back-test**: vectorised, single-pass, ~1 s runtime on a laptop.

Run:

```bash
pip install -r requirements.txt
python run_backtest.py        # prints CAGR / Sharpe / max-DD, plots equity
pytest                        # all unit + integration tests
