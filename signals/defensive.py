"""
Defensive overlay for turbulent / uncertain regimes.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

DEFENSIVE_ALLOC: Mapping[str, float] = {
    "TLT": 0.7,
    "GLD": 0.3,
}


def _normalise_alloc(alloc: Mapping[str, float], available: set[str]) -> dict[str, float]:
    filtered = {
        ticker: float(weight)
        for ticker, weight in alloc.items()
        if ticker in available and float(weight) > 0.0
    }
    total = sum(filtered.values())
    if total <= 0.0:
        return {ticker: 0.0 for ticker in available}
    return {ticker: weight / total for ticker, weight in filtered.items()}


def defensive_overlay(prices: pd.DataFrame, alloc: Mapping[str, float] | None = None) -> pd.DataFrame:
    """
    Create a long-only defensive sleeve that can be blended with other signals.

    Parameters
    ----------
    prices
        Price history for the investable universe.
    alloc
        Optional custom allocation mapping. Falls back to 70 % TLT / 30 % GLD.
    """
    alloc = alloc or DEFENSIVE_ALLOC
    available = set(prices.columns)
    normalised = _normalise_alloc(alloc, available)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ticker, weight in normalised.items():
        weights[ticker] = weight
    return weights
