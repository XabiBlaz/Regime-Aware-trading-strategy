"""
Mean-reversion pairs signals.

Pairs:
    1. (SPY, QQQ)
    2. (XLE, USO)
    3. (XLK, QQQ)

* Spread : price_A − beta · price_B  (beta estimated via rolling OLS)
* Z-score: 63-day rolling mean / std
* Entry  : |Z| > 1.5
* Exit   : |Z| < 0.25
* Stop   : |Z| > 3.0
* Positions: ±LEG_WEIGHT per leg (dollar-neutral)

Returns a weight matrix (rows = date, cols = tickers).
"""
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

PAIRS: List[Tuple[str, str]] = [("SPY", "QQQ"), ("XLE", "USO"), ("XLK", "QQQ")]
WINDOW = 63
ENTRY_Z = 1.5
EXIT_Z = 0.25
STOP_Z = 3.0
LEG_WEIGHT = 0.35      # gross ~0.7, net 0.0


def _hedge_ratio(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling hedge ratio via OLS beta."""
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta = cov / var.replace(0, np.nan)
    return beta.fillna(1.0)


def pairs_positions(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily pair weights (dollar-neutral)."""
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, b in PAIRS:
        if a not in prices or b not in prices:
            continue

        beta = _hedge_ratio(prices[a], prices[b], WINDOW)
        spread = prices[a] - beta * prices[b]
        mean = spread.rolling(WINDOW).mean()
        std = spread.rolling(WINDOW).std().replace(0, np.nan)
        z = (spread - mean) / std

        pair_pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        state = 0      # 0 = flat, 1 = long spread, -1 = short spread
        for t in z.index:
            if state == 0:
                if z[t] > ENTRY_Z:
                    state = -1
                elif z[t] < -ENTRY_Z:
                    state = 1
            else:
                if abs(z[t]) < EXIT_Z or abs(z[t]) > STOP_Z:
                    state = 0

            if state == 1:
                pair_pos.at[t, a] = +LEG_WEIGHT
                pair_pos.at[t, b] = -LEG_WEIGHT
            elif state == -1:
                pair_pos.at[t, a] = -LEG_WEIGHT
                pair_pos.at[t, b] = +LEG_WEIGHT

        pos = pos.add(pair_pos, fill_value=0.0)

    return pos
