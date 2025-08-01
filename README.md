# Regime-Aware Trading Strategy: Momentum & Pairs with VIX-Based Regime Switching

A comprehensive quantitative trading strategy that dynamically allocates between momentum and pairs trading based on market volatility regimes. This project demonstrates sophisticated regime detection, strategy implementation, and rigorous backtesting methodologies commonly used in quantitative finance.

## Strategy Overview

**Core Concept**: Switch between momentum and pairs trading strategies based on VIX-derived volatility regimes to adapt to changing market conditions.

### Regime Classification & Strategy Allocation

| VIX Regime | Definition | Strategy | Rationale |
|------------|------------|----------|-----------|
| **Low Volatility** | VIX < 15 | 100% Momentum | Trending markets favor momentum strategies |
| **Medium Volatility** | 15 ≤ VIX < 25 | 50% Momentum | Mixed allocation during transition periods |
| **High Volatility** | VIX ≥ 25 | 100% Pairs Trading | Market-neutral approach during stress periods |

### Asset Universe (10 Assets + VIX)
- **Tech Stocks**: AAPL, MSFT, AMZN, NVDA (40%)
- **Broad Market ETFs**: SPY, QQQ, IWM (30%)  
- **Sector ETFs**: XLE (Energy), XLK (Technology) (20%)
- **Commodities**: USO (Oil) (10%)
- **Regime Indicator**: ^VIX (for classification only)

## Key Findings & Performance

**Backtest Period**: January 2014 - May 2016 (600 trading days)

### Performance Metrics
- **CAGR**: -55.65% (severe underperformance)
- **Sharpe Ratio**: -1.291 (poor risk-adjusted returns)
- **Maximum Drawdown**: -88.75% (catastrophic loss)

### Critical Insights by Regime
| Regime | Duration | Ann. Return | Sharpe | Win Rate | Key Finding |
|--------|----------|-------------|---------|----------|-------------|
| High Volatility | 4.5% (27 days) | +195.44% | 2.119 | 51.85% | **Pairs trading highly effective** |
| Low Volatility | 54.3% (326 days) | -72.33% | -2.137 | 36.20% | **Momentum strategy fundamentally flawed** |
| Medium Volatility | 41.2% (247 days) | -89.81% | -1.383 | 31.98% | **Mixed allocation performs worst** |

## Research Contributions

Despite poor performance, this analysis provides valuable quantitative insights:

1. **Regime Detection Limitations**: Static VIX thresholds inadequate for real-time classification
2. **Strategy Interaction Effects**: Momentum and pairs trading signals interfere destructively during regime transitions
3. **Risk Management Gaps**: Critical importance of position sizing and drawdown controls in multi-strategy frameworks
4. **Market Microstructure Evolution**: Strategy performance degradation reflects changing factor exposures (2014-2016)

## Project Structure

```
Project - QUANT/
├── Report.ipynb                    # Comprehensive analysis & results
├── adaptive_kelly_simulation.ipynb # Kelly criterion position sizing research
├── README.md                       # This file
├── requirements.txt               # Python dependencies
├── run_backtest.py               # Main execution script
├── signals/                      # Signal generation modules
│   ├── momentum.py               # Cross-sectional momentum signals
│   ├── pairs.py                  # Market-neutral pairs trading
│   └── volatility.py             # VIX-based regime classification
├── strategy/                     # Strategy implementation
│   ├── regime_strategy.py        # Main regime-switching logic
│   └── position_sizing.py        # Position sizing & risk management
├── backtest/                     # Backtesting framework
│   ├── backtester.py            # Vectorized backtesting engine
│   └── metrics.py               # Performance & risk metrics
└── tests/                        # Unit & integration tests
    ├── test_signals.py
    ├── test_strategy.py
    └── test_backtest.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete backtest
python run_backtest.py

# View comprehensive analysis
jupyter notebook Report.ipynb

# Explore Kelly criterion research
jupyter notebook adaptive_kelly_simulation.ipynb

# Run tests
pytest tests/
```

## Technical Implementation

- **Data Source**: Yahoo Finance (yfinance) - 600 daily observations
- **Execution**: Daily rebalancing with 10 bps transaction costs
- **Performance**: Vectorized backtesting (~1 second runtime)
- **Regime Detection**: Historical VIX-based classification
- **Risk Management**: Basic position sizing (improvement opportunities identified)

## Key Improvements Identified

1. **Dynamic Regime Detection**: Replace static thresholds with probabilistic models
2. **Enhanced Position Sizing**: Implement Kelly criterion and volatility targeting
3. **Risk Management**: Add stop-losses and drawdown controls
4. **Strategy Diversification**: Include additional uncorrelated strategies
5. **Transition Management**: Smooth regime transitions to reduce whipsaw effects

## Educational Value

This project demonstrates:
- **Regime-switching strategies** in quantitative finance
- **Multi-strategy portfolio construction** and allocation
- **Rigorous backtesting methodology** with proper attribution analysis
- **Professional risk assessment** and performance evaluation
- **Research-oriented approach** to strategy development

## Important Notes

- **Research Purpose**: This is an educational/research project demonstrating quantitative methods
- **Performance Disclaimer**: Strategy shows significant losses; not suitable for live trading
- **Learning Focus**: Emphasizes methodology, analysis, and improvement identification over profitability
- **Academic Context**: Suitable for quantitative finance education and interview preparation

## License

This project is for educational purposes. All data sourced from publicly available Yahoo Finance feeds.

---

*This project showcases systematic quantitative research methodologies and demonstrates how sophisticated analysis can extract valuable insights even from unsuccessful strategies.*