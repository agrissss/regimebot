"""Daily trend following: a target position per symbol, rebalanced at each daily close.

A different shape from the playbook strategies: no signal to enter, no stop, no target, no regime.
At every decision the strategy names a signed target weight (fraction of equity) per symbol and
the engine trades toward it (``Engine`` target-position step); the risk manager caps the result.

- Signal: the mean of sign(close - SMA_n(close)) over ``lookbacks``, the SMA including the current
  close, in [-1, 1]; ``long_flat`` sets negative signals to 0.
- Volatility: the standard deviation (ddof 1) of daily close-to-close returns over ``vol_window``
  days, times sqrt(365).
- Weight: signal * min(asset cap, target_vol / n_symbols / vol), n_symbols being the configured
  symbols (dividing by n, not sqrt(n): the coins are correlated); all weights scaled down alike
  when their gross exceeds the gross cap. The caps are the risk config's.

Prototype: ``research/daily_trend.py`` (vectorised, outside the engine, sqrt(n) sizing).
"""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from regimebot.config import DailyTrendConfig, RiskConfig

NAME = "daily_trend"
DAYS_PER_YEAR = 365


def trend_features(candles: pd.DataFrame, cfg: DailyTrendConfig) -> pd.DataFrame:
    """``close``, ``signal`` and ``vol`` on daily decision bars (UTC close-time index).

    Causal: row t uses closes up to and including t only (trailing rolling windows). NaN until
    every lookback is full (``max(lookbacks)`` closes) and the volatility window is full
    (``vol_window + 1`` closes); warm-up rows are never filled.
    """
    close = candles["close"].astype(float)
    gaps = pd.DataFrame({i: close - close.rolling(n).mean() for i, n in enumerate(cfg.lookbacks)})
    signal = pd.DataFrame(np.sign(gaps.to_numpy()), index=gaps.index).mean(axis=1, skipna=False)
    if cfg.mode == "long_flat":
        signal = signal.clip(lower=0.0)
    returns = close / close.shift(1) - 1.0
    vol = returns.rolling(cfg.vol_window).std() * math.sqrt(DAYS_PER_YEAR)
    return pd.DataFrame({"close": close, "signal": signal, "vol": vol}, index=candles.index)


class DailyTrend:
    """The daily trend rule as a ``TargetStrategy``."""

    name = NAME

    def __init__(self, cfg: DailyTrendConfig, risk: RiskConfig, symbols: Sequence[str]) -> None:
        self.cfg = cfg
        self.asset_cap = risk.max_asset_exposure
        self.gross_cap = risk.max_gross_leverage
        self.n_symbols = len(symbols)

    def weights(self, ts: datetime, rows: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Target weight of every symbol whose decision row has a finite signal and volatility; a
        symbol without one (window not full yet) gets none and is not traded."""
        out = {}
        for symbol, row in rows.items():
            signal, vol = row["signal"], row["vol"]
            if not (math.isfinite(signal) and math.isfinite(vol)):
                continue
            # vol 0 (constant closes) is an infinite size before the cap
            size = self.cfg.target_vol / self.n_symbols / vol if vol > 0 else math.inf
            out[symbol] = signal * min(self.asset_cap, size)
        gross = sum(abs(w) for w in out.values())
        if gross > self.gross_cap:
            out = {symbol: w / (gross / self.gross_cap) for symbol, w in out.items()}
        return out
