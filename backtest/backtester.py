"""
Vectorised daily back-tester with 10 bp one-way transaction cost.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest.metrics import summary_stats


class Backtester:
    def __init__(self, prices: pd.DataFrame, strategy) -> None:
        self.prices = prices
        self.strategy = strategy

    # --- internal helpers -------------------------------------------------

    def _pnl(self, weights: pd.DataFrame) -> pd.Series:
        r = self.prices.pct_change().fillna(0)
        gross = (weights.shift().fillna(0) * r).sum(axis=1)

        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * 0.001          # 10 bp
        return gross - costs

    # --- public -----------------------------------------------------------

    def run(self) -> dict[str, float]:
        w = self.strategy.positions()
        self.returns = self._pnl(w)
        return summary_stats(self.returns)

    def plot_equity_curve(self, save: Path | None = None) -> None:
        eq = (1 + self.returns).cumprod()
        eq.plot(figsize=(10, 4), title="Equity Curve")
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()
