"""
Volatility-target position scaling.
Target annualised vol = 10 %.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


TARGET_VOL = 0.10
ROLL = 20


def scale_weights(raw_weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Scale weights each day so that predicted portfolio vol (20-day)
    equals TARGET_VOL. Uses ex-post realised vol of SPY as proxy.
    """
    spy_returns = prices["SPY"].pct_change().fillna(0)
    port_returns = (raw_weights.shift().fillna(0) * prices.pct_change()).sum(axis=1)

    daily_vol = port_returns.rolling(ROLL).std()
    scaler = TARGET_VOL / (daily_vol.replace(0, np.nan))
    scaler = scaler.clip(upper=5).fillna(1.0)

    return raw_weights.mul(scaler, axis=0)
