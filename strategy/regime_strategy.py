"""
Combine momentum & pairs based on volatility regime,
then apply volatility-target scaling.
"""
from __future__ import annotations
import pandas as pd
from signals.momentum import momentum_zscores, momentum_positions
from signals.pairs import pairs_positions
from signals.volatility import Regime, classify_regime, realised_vol
from strategy.position_sizing import scale_weights


class RegimeAwareStrategy:
    """
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices (columns = tickers, excl. ^VIX).
    vix : pd.Series
        Daily VIX levels (same index).
    """

    def __init__(self, prices: pd.DataFrame, vix: pd.Series) -> None:
        self.prices = prices
        self.vix = vix
        self.regimes = classify_regime(vix)
        self.realised = realised_vol(prices["SPY"])

        # pre-compute signals
        self._mom_pos = momentum_positions(momentum_zscores(prices))
        self._pair_pos = pairs_positions(prices)

    def positions(self) -> pd.DataFrame:
        """Final daily weights after regime switch & vol targeting."""
        weights = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)

        for t in weights.index:
            reg = self.regimes.loc[t]
            if reg in (Regime.LOW, Regime.MEDIUM):
                w = self._mom_pos.loc[t].copy()
                if reg is Regime.MEDIUM:
                    w *= 0.5                # half-size in MEDIUM regime
            else:                           # HIGH
                w = self._pair_pos.loc[t].copy()
            weights.loc[t] = w

        # volatility-target scaling
        return scale_weights(weights, self.prices)
