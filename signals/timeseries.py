"""
Multi-horizon time-series momentum overlay.

Creates a net-zero overlay by averaging the sign of medium and long-horizon
returns across the universe. Designed to complement the cross-sectional sleeve
without doubling gross exposure.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_LOOKBACKS: Sequence[int] = (21, 63, 126)


def _ts_weights(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Net-zero weights from the sign of lookback-period log returns."""
    lookback = int(lookback)
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    log_rets = np.log(prices).diff(lookback)
    signal = log_rets.apply(np.sign).replace(0.0, np.nan)

    demeaned = signal.sub(signal.mean(axis=1), axis=0)
    weights = demeaned.div(demeaned.abs().sum(axis=1), axis=0)
    return weights.fillna(0.0)


def _coerce_lookbacks(lookback: int | Iterable[int] | None) -> list[int]:
    if lookback is None:
        return [int(lb) for lb in DEFAULT_LOOKBACKS]
    if isinstance(lookback, Iterable) and not isinstance(lookback, (str, bytes)):
        return [int(lb) for lb in lookback]
    return [int(lookback)]


def timeseries_momentum(
    prices: pd.DataFrame,
    lookback: int | Iterable[int] | None = None,
    return_components: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create time-series momentum weights.

    Parameters
    ----------
    prices
        Price history aligned across the asset universe.
    lookback
        Single window or collection of windows to blend. Defaults to
        (21, 63, 126) if omitted.
    return_components
        When True, also return a MultiIndex DataFrame containing the per-horizon
        overlays (useful for diagnostics).
    """
    lookbacks = _coerce_lookbacks(lookback)

    components: dict[str, pd.DataFrame] = {}
    for lb in lookbacks:
        component = _ts_weights(prices, lb)
        components[f"L{lb}"] = component

    if not components:
        combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if return_components:
            return combined, pd.DataFrame(
                0.0,
                index=prices.index,
                columns=pd.MultiIndex.from_product([["L0"], prices.columns]),
            )
        return combined

    combined = sum(components.values()) / len(components)
    if not return_components:
        return combined.fillna(0.0)

    horizon_panel = pd.concat(components, axis=1).sort_index(axis=1)
    return combined.fillna(0.0), horizon_panel.fillna(0.0)
