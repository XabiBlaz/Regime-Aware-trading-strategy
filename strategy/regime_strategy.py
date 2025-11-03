"""
Blend cross-sectional, spread, time-series, and defensive sleeves based on the
volatility regime, then apply volatility targeting.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from signals.defensive import defensive_overlay
from signals.momentum import momentum_zscores, momentum_positions
from signals.pairs import pairs_positions
from signals.timeseries import timeseries_momentum
from signals.volatility import Regime, classify_regime, realised_vol
from strategy.position_sizing import scale_weights

try:  # pragma: no cover - optional dependency guard
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    LogisticRegression = None
    StandardScaler = None


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
        self._mom_z = momentum_zscores(prices)
        self._mom_pos = momentum_positions(self._mom_z)
        self._pair_pos = pairs_positions(prices)
        self._ts_pos, self._ts_components = timeseries_momentum(prices, return_components=True)
        self._def_pos = defensive_overlay(prices)

        self._mom_strength = self._mom_z.abs().mean(axis=1)
        self._pair_intensity = self._pair_pos.abs().sum(axis=1)
        self._ts_intensity = self._ts_pos.abs().sum(axis=1)
        self._def_intensity = self._def_pos.abs().sum(axis=1)
        self._prob_high = self._estimate_high_regime_prob()
        self.regimes = pd.Series(
            np.where(
                self._prob_high > 0.6,
                Regime.HIGH.value,
                np.where(self._prob_high < 0.25, Regime.LOW.value, Regime.MEDIUM.value),
            ),
            index=self.vix.index,
        )

    def _feature_matrix(self) -> pd.DataFrame:
        """Feature set for regime classifier."""
        features = pd.DataFrame(index=self.vix.index, dtype=float)
        vix = self.vix.astype(float)
        spy_realised = self.realised.reindex(self.vix.index).astype(float)

        features["vix_level"] = vix
        features["vix_pct_5"] = vix.pct_change(5)
        features["vix_pct_20"] = vix.pct_change(20)
        features["vix_z_20"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
        features["vix_percentile"] = vix.rank(pct=True)

        features["realised_vol_20"] = spy_realised
        features["realised_vol_slope"] = spy_realised.diff(5)
        vol_ratio = vix / spy_realised.replace(0, np.nan)
        features["vol_ratio"] = vol_ratio

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0.0)
        return features

    def _estimate_high_regime_prob(self) -> pd.Series:
        """Fit rolling logistic models to estimate high-vol regime probability."""
        high_flag = (self.regimes == Regime.HIGH.value).astype(int)
        if LogisticRegression is None or StandardScaler is None or high_flag.nunique() < 2:
            return high_flag.astype(float)

        features = self._feature_matrix()
        proba = pd.Series(np.nan, index=features.index, dtype=float)

        min_train = 252
        for pos, ts in enumerate(features.index):
            if pos < min_train:
                window = high_flag.iloc[:pos] if pos else high_flag.iloc[:1]
                baseline = window.mean() if len(window) else 0.0
                proba.iloc[pos] = float(baseline) if np.isfinite(baseline) else 0.0
                continue

            X_train = features.iloc[pos - min_train : pos]
            y_train = high_flag.iloc[pos - min_train : pos]

            if y_train.nunique() < 2:
                proba.iloc[pos] = float(y_train.iloc[-1])
                continue

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            clf = LogisticRegression(max_iter=200, solver="lbfgs")
            clf.fit(X_train_scaled, y_train)
            X_pred = scaler.transform(features.iloc[pos : pos + 1])
            proba.iloc[pos] = clf.predict_proba(X_pred)[0, 1]

        proba = proba.ffill().fillna(high_flag.mean())
        return proba.clip(0.0, 1.0).rename("p_high")

    def positions(self) -> pd.DataFrame:
        """Final daily weights after regime switch & vol targeting."""
        weights = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        sleeve_contrib = {
            "momentum": weights.copy(),
            "pairs": weights.copy(),
            "timeseries": weights.copy(),
            "defensive": weights.copy(),
        }

        prob_high_raw = self._prob_high.reindex(weights.index).fillna(0.0).clip(0.0, 1.0)
        prob_high = prob_high_raw.ewm(span=5, adjust=False).mean().clip(0.0, 1.0)
        confidence = (prob_high - 0.5).abs() * 2.0
        transition_scaler = confidence.clip(lower=0.2)

        mom_strength = self._mom_strength.reindex(weights.index)
        mom_scaler = (
            mom_strength / mom_strength.rolling(252, min_periods=21).median()
        ).clip(lower=0.3, upper=1.5).fillna(1.0)
        pair_intensity = self._pair_intensity.reindex(weights.index)
        pair_scaler = (
            pair_intensity / pair_intensity.rolling(252, min_periods=21).median()
        ).clip(lower=0.3, upper=1.6).fillna(1.0)
        ts_intensity = self._ts_intensity.reindex(weights.index)
        ts_scaler = (
            ts_intensity / ts_intensity.rolling(252, min_periods=21).median()
        ).clip(lower=0.4, upper=1.4).fillna(1.0)
        def_scaler = (0.85 + 0.6 * prob_high).clip(0.5, 1.6)

        for t in weights.index:
            regime = self.regimes.loc[t]
            mix = self._regime_mix(regime, prob_high.loc[t], confidence.loc[t])

            mom_component = mix["momentum"] * mom_scaler.loc[t] * self._mom_pos.loc[t]
            pair_component = mix["pairs"] * pair_scaler.loc[t] * self._pair_pos.loc[t]
            ts_component = mix["timeseries"] * ts_scaler.loc[t] * self._ts_pos.loc[t]
            defensive_component = mix["defensive"] * def_scaler.loc[t] * self._def_pos.loc[t]

            combined = mom_component + pair_component + ts_component + defensive_component
            weights.loc[t] = transition_scaler.loc[t] * combined

            sleeve_contrib["momentum"].loc[t] = mom_component
            sleeve_contrib["pairs"].loc[t] = pair_component
            sleeve_contrib["timeseries"].loc[t] = ts_component
            sleeve_contrib["defensive"].loc[t] = defensive_component

        scaled = scale_weights(weights, self.prices)

        # expose diagnostics for the notebook / downstream analysis
        self.analysis_payload: dict[str, object] = {
            "prob_high": prob_high,
            "confidence": confidence,
            "transition_scaler": transition_scaler,
            "raw_weights": weights,
            "scaled_weights": scaled,
            "sleeve_contributions": sleeve_contrib,
            "timeseries_components": self._ts_components,
        }

        return scaled

    @staticmethod
    def _regime_mix(regime: str, prob_high: float, confidence: float) -> Mapping[str, float]:
        """Regime-specific blend of sleeves."""
        base_weights: dict[str, dict[str, float]] = {
            Regime.LOW.value: {"momentum": 0.55, "pairs": 0.15, "timeseries": 0.20, "defensive": 0.10},
            Regime.MEDIUM.value: {"momentum": 0.30, "pairs": 0.20, "timeseries": 0.25, "defensive": 0.25},
            Regime.HIGH.value: {"momentum": 0.05, "pairs": 0.20, "timeseries": 0.15, "defensive": 0.60},
        }

        mix = base_weights.get(regime, base_weights[Regime.MEDIUM.value]).copy()

        uncertainty = 1.0 - confidence
        mix["defensive"] += 0.25 * uncertainty + 0.35 * prob_high
        mix["momentum"] *= max(0.0, 1.0 - 0.7 * prob_high)

        if regime == Regime.MEDIUM.value:
            mix["defensive"] += 0.1
            mix["momentum"] *= 0.5 + 0.5 * confidence
            mix["pairs"] *= 0.7 + 0.3 * prob_high
        elif regime == Regime.HIGH.value:
            mix["momentum"] = 0.0
            mix["pairs"] *= 0.8 + 0.4 * prob_high
            mix["timeseries"] *= 0.6 + 0.4 * confidence
        else:  # LOW
            mix["defensive"] += 0.05 * uncertainty
            mix["pairs"] *= 0.5 + 0.5 * prob_high
            mix["timeseries"] *= 0.8 + 0.2 * confidence

        total = sum(mix.values())
        if total <= 0.0:
            return {"momentum": 0.0, "pairs": 0.0, "timeseries": 0.0, "defensive": 1.0}

        return {k: max(0.0, v) / total for k, v in mix.items()}










