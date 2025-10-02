"""Performance metric helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summary_stats(returns: pd.Series) -> dict[str, float]:
    """CAGR, Sharpe, max draw-down (log equity) with guard rails."""
    if returns is None or len(returns) == 0:
        nan = float('nan')
        return {"CAGR": nan, "Sharpe": nan, "MaxDD": nan}

    clean = returns.fillna(0.0)
    n = len(clean)

    cumulative = (1.0 + clean).prod()
    if cumulative > 0:
        cagr = cumulative ** (TRADING_DAYS / n) - 1
    else:
        cagr = float('nan')

    volatility = clean.std(ddof=0)
    if volatility and not np.isclose(volatility, 0.0):
        sharpe = np.sqrt(TRADING_DAYS) * clean.mean() / volatility
    else:
        sharpe = float('nan')

    equity = (1.0 + clean).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}

