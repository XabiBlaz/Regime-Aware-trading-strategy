"""Performance metric helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd


def summary_stats(returns: pd.Series) -> dict[str, float]:
    """CAGR, Sharpe, max draw-down (log equity)."""
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": ann_ret, "Sharpe": sharpe, "MaxDD": max_dd}
