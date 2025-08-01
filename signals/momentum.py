"""
Cross-sectional 6-month momentum signal.

* Return horizon: 126 trading days (~6 months)
* Signal: z-score across universe each day
* Position construction:
    long  top-30 %  (equal weight)
    short bottom-30 %
"""
from __future__ import annotations
import pandas as pd
import numpy as np


LOOKBACK = 126
PCT = 0.30


def momentum_zscores(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily cross-sectional z-scores of 6-month returns."""
    six_m_ret = prices.pct_change(LOOKBACK)
    mean = six_m_ret.mean(axis=1)
    std = six_m_ret.std(axis=1).replace(0, np.nan)
    z = (six_m_ret.sub(mean, axis=0)).div(std, axis=0)
    return z.fillna(0.0).clip(-5, 5)


def momentum_positions(zscores: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight long top PCT, short bottom PCT.
    Weights sum to +1 (long) and −1 (short) but net 0 by design.
    """
    n_assets = zscores.shape[1]
    k = max(1, int(PCT * n_assets))

    ranks = zscores.rank(axis=1, ascending=False, method="first")
    long_mask = ranks <= k
    short_mask = ranks >= n_assets - k + 1

    w_long = 0.5 / k          # +0.5 gross long
    w_short = -0.5 / k        # −0.5 gross short

    pos = (long_mask * w_long) + (short_mask * w_short)
    return pos.astype(float)
