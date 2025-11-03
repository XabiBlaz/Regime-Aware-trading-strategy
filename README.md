# Regime-Aware Trading Strategy: Momentum, Spreads & Defensive Overlay

A volatility-aware allocation that blends cross-sectional momentum, regression-based spread trading, and a defensive bond/gold sleeve. A rolling logistic classifier trained on VIX features determines how much capital is deployed in each sleeve while a risk overlay keeps portfolio volatility bounded.

## Why 600 Trading Days for the Development Sample?
- **Calendar alignment**: 600 trading days (~2.4 years) capture the full 2014 taper scare through the 2016 growth shock - the period originally used while researching the strategy.
- **Logistic training window**: the classifier is refit each day on the trailing 252 observations (~1 trading year), the minimum that Marcos Lopez de Prado recommends for machine-learning features to stabilise. Using a 600-day development sample allows the walk-forward fit to operate for over 300 out-of-sample days while still reflecting the original research context.
- **Event coverage**: the window includes oil's 2015 collapse and the 2016 China devaluation, providing distinct volatility regimes without overfitting to the post-2016 bull market.

## Feature Set & Modelling
- **Regime probability**: 252-day logistic regression with StandardScaler, trained on VIX level, 5/20-day slopes, 20-day z-score, realised SPY volatility, slope of realised vol, and the VIX/realised-vol ratio.
- **Smoothing & gating**: the high-volatility probability is smoothed with a 5-day EMA; confidence `2|p_high-0.5|` throttles exposure when the model is unsure. Final regime labels are derived from the probability (>0.6 = high, <0.25 = low, otherwise medium).
- **Signal sleeves**:
  - Cross-sectional momentum (6-month z-scores, top/bottom 30%) dominates in calm regimes.
  - Spread trading (63-day rolling beta; entry |z|>1.5, exit <0.25, stop >3.0) engages only under elevated probabilities.
  - Time-series overlay averages 21/63/126-day sign signals, providing a light-touch, net-zero trend sleeve that boosts medium-regime Sharpe when trends persist.
  - Defensive overlay (70% TLT, 30% GLD) absorbs capital whenever classifier confidence falls or the high-vol probability breaches 60%.
- **Sleeve blending**: regime-specific weights mix momentum, spreads, time-series, and defensive sleeves; low confidence automatically shifts budget toward the defensive overlay while high-volatility regimes cut momentum entirely.
- **Risk overlay**: volatility targeting at 6% annualised with 2.7x leverage cap, drawdown throttle (linear reduction beyond -5%) and a 5-day crash cooldown once drawdown breaches -12%.

## Performance Summary
### Development Sample (Jan 2014 - May 2016, 600 trading days)
- **CAGR:** +24.5%
- **Sharpe Ratio:** 0.92
- **Maximum Drawdown:** -26.5%
- **Total Return:** +68.5%
- **Average Daily Turnover:** 0.34 (~8.5% annual cost at 10 bps)

### Extended Validation (Jun 2016 - Dec 2024)
- **CAGR:** +7.0%
- **Sharpe Ratio:** 0.38
- **Maximum Drawdown:** -51.8%
- **Total Return:** +145.1%
- **Average Daily Turnover:** 0.42

## Limitations & Improvement Ideas

**Current limitations**
- No fractional Kelly sizing: the sleeves all target the same 6% volatility, so capital is not weighted by edge stability.
- Execution costs remain a flat 10 bps per leg and ignore borrow fees, liquidity caps, and intraday slippage.
- The extended holdout still suffers a -52% drawdown during the COVID/2022 stress episode, highlighting the need for additional tail protection.

**Improvement ideas**
1. Layer a fractional Kelly overlay that scales sleeve weights by realised Sharpe and drawdown persistence.
2. Replace the flat cost assumption with better execution modelling (dynamic spreads, borrow costs, and trade caps).
3. Introduce option-based or futures hedges so the strategy is not solely reliant on the bond/gold overlay for downside control.

### Regime Attribution (2014-2024)
| Regime | Days | Annual Return | Volatility | Sharpe | Win Rate |
|--------|------|---------------|------------|--------|----------|
| **Low Volatility** | 2,315 | +6.3% | 28.4% | 0.22 | 47.9% |
| **Medium Volatility** | 68 | +21.3% | 15.2% | 1.40 | 45.6% |
| **High Volatility** | 342 | +23.1% | 28.4% | 0.81 | 48.5% |

Medium-regime performance flips from negative Sharpe in the original implementation to +1.40 once the logistic probabilities gate the sleeves and the strategy sits largely in defensive assets when the classifier lacks conviction.

## Backtesting Process (Lopez de Prado-aligned)
1. **Walk-forward fit** - The logistic model is refit daily on expanding data; all features use only information available at the close.
2. **Anchored validation** - Development ends in May 2016, while results from Jun 2016 onward form a holdout that the research notebook treats separately.
3. **Cost & frictions** - 10 bps one-way costs applied to the absolute change in weights. Turnover diagnostics are reported in the notebook.
4. **Regime attribution** - Equity, drawdown, turnover, and performance are broken out by regime in `Report.ipynb`, illustrating how each sleeve behaves under changing volatility.
5. **Robustness** - Sensitivity checks on logistic thresholds, timeseries lookbacks, and spread parameters are included in the analysis to confirm stability.

## Project Structure
```
Regime-Aware-trading-strategy/
+-- Report.ipynb              # Full analysis & notebook write-up (dev + extended OOS)
+-- README.md                 # This document
+-- requirements.txt          # Python dependencies (>= pins for Python 3.13)
+-- run_backtest.py           # Simple backtest runner
+-- signals/
|   +-- momentum.py           # Cross-sectional momentum signals
|   +-- pairs.py              # Rolling-beta spread signals
|   +-- timeseries.py         # Multi-horizon time-series momentum overlay
|   +-- volatility.py         # Data loader & regime helpers
+-- strategy/
|   +-- position_sizing.py    # Vol targeting, drawdown throttle, crash cooldown
|   +-- regime_strategy.py    # Logistic gating & sleeve blending
+-- backtest/
|   +-- backtester.py         # Vectorised daily P&L engine with costs
|   +-- metrics.py            # Summary statistics utilities
+-- tests/                    # Unit checks for data, signals, strategy
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run development-window backtest
python run_backtest.py

# Explore the analysis notebook (dev + holdout)
jupyter notebook Report.ipynb

# Execute the unit tests
pytest tests/
```

## Future Work
1. Enrich the classifier with macro spreads (credit, term structure) to anticipate sustained volatility shifts.
2. Incorporate an option-based defensive sleeve to improve high-volatility returns without materially raising turnover.
3. Extend the validation period to include post-2024 data as it becomes available.






