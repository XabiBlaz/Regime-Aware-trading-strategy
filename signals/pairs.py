"""
Mean-reversion pairs signals.

Pairs:
    1. (SPY, QQQ)
    2. (XLE, USO)

* Spread : price_A − beta · price_B  (beta = 1, daily close)
* Z-score: 60-day rolling mean / std
* Entry  : |Z| > 2
* Exit   : |Z| < 0.5
* Positions: ±0.5 long / short per leg (dollar-neutral)

Returns a weight matrix (rows = date, cols = tickers).
"""
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

PAIRS: List[Tuple[str, str]] = [("SPY", "QQQ"), ("XLE", "USO")]
WINDOW = 60
ENTRY_Z = 2.0
EXIT_Z = 0.5
LEG_WEIGHT = 0.5      # gross 1.0, net 0.0


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def pairs_positions(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily pair weights (dollar-neutral)."""
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, b in PAIRS:
        spread = prices[a] - prices[b]      # beta=1
        z = _zscore(spread, WINDOW)

        state = 0      # 0 = flat,  1 = long spread (long a, short b),  -1 = short spread
        for t in z.index:
            if state == 0:
                if z[t] > ENTRY_Z:
                    state = -1
                elif z[t] < -ENTRY_Z:
                    state = 1
            elif state == 1 and abs(z[t]) < EXIT_Z:
                state = 0
            elif state == -1 and abs(z[t]) < EXIT_Z:
                state = 0

            if state == 1:
                pos.at[t, a] = +LEG_WEIGHT
                pos.at[t, b] = -LEG_WEIGHT
            elif state == -1:
                pos.at[t, a] = -LEG_WEIGHT
                pos.at[t, b] = +LEG_WEIGHT

    return pos
