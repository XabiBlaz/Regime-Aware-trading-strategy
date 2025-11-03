"""
Volatility-target position scaling.
Target annualised vol = 6 %.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


TARGET_VOL = 0.06
ROLL = 20


def scale_weights(raw_weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Scale weights each day so that realised portfolio vol over ROLL days
    targets TARGET_VOL, falling back to SPY volatility when the portfolio
    history is too short.
    """
    returns = prices.pct_change().fillna(0.0)
    port_returns = (raw_weights.shift().fillna(0.0) * returns).sum(axis=1)

    daily_vol = port_returns.rolling(ROLL).std()
    spy_vol = returns["SPY"].rolling(ROLL).std()
    effective_vol = daily_vol.fillna(spy_vol)

    scaler = TARGET_VOL / (effective_vol.replace(0, np.nan))
    scaler = scaler.clip(upper=2.7).fillna(1.0)

    equity = (1 + port_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    dd_start = -0.05
    dd_cutoff = -0.25
    dd_scaler = drawdown.copy()
    dd_scaler = (dd_scaler - dd_cutoff) / (dd_start - dd_cutoff)
    dd_scaler = dd_scaler.clip(lower=0.0, upper=1.0)
    dd_scaler = dd_scaler.where(drawdown <= dd_start, 1.0).fillna(1.0)

    crash = (drawdown <= -0.12).astype(int)
    cooldown = crash.rolling(5, min_periods=1).max()
    dd_scaler = dd_scaler * (1 - cooldown)

    final_scaler = scaler.mul(dd_scaler, axis=0)

    return raw_weights.mul(final_scaler, axis=0)


