"""Daily trend following: a target position per symbol, rebalanced at each daily close.

A different shape from the playbook strategies: no signal to enter, no stop, no target, no regime.
At every decision the strategy names a signed target weight (fraction of equity) per symbol and
the engine trades toward it (``Engine`` target-position step); the risk manager caps the result.

- Signal (``signal``), in [-1, 1]: the mean over ``lookbacks`` of a sign per lookback n
  (``SIGNALS``): ``sma`` sign(close - SMA_n), ``donchian`` sign(close - the midpoint of the highest
  and lowest close of the last n days), both windows including the current close; ``momentum``
  the sign of the n-day return. ``hold`` is always 1 (volatility-targeted buy and hold).
  ``long_flat`` sets negative signals to 0.
- Volatility: the standard deviation (ddof 1) of daily close-to-close returns over ``vol_window``
  days, times sqrt(365).
- Weight: signal * min(asset cap, target_vol / n_symbols / vol), n_symbols being the configured
  symbols (dividing by n, not sqrt(n): the coins are correlated); all weights scaled down alike
  when their gross exceeds the gross cap. The caps are the risk config's.

Prototype: ``research/daily_trend.py`` (vectorised, outside the engine, sqrt(n) sizing).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from regimebot.config import DailyTrendConfig, RiskConfig

NAME = "daily_trend"
DAYS_PER_YEAR = 365


def _sign(x: pd.Series) -> pd.Series:
    """-1, 0 or 1 per row (NaN stays NaN)."""
    return pd.Series(np.sign(x.to_numpy()), index=x.index)


def sma_sign(close: pd.Series, n: int) -> pd.Series:
    """sign(close - SMA_n(close)), the SMA including the current close (n closes)."""
    return _sign(close - close.rolling(n).mean())


def donchian_sign(close: pd.Series, n: int) -> pd.Series:
    """sign(close - midpoint of the highest and lowest close of the last n days), the current
    close included (n closes)."""
    window = close.rolling(n)
    return _sign(close - (window.max() + window.min()) / 2)


def momentum_sign(close: pd.Series, n: int) -> pd.Series:
    """Sign of the n-day return close / close n days ago - 1 (n + 1 closes), taken as
    sign(close - close n days ago): the same for positive closes, and exact on a tie."""
    return _sign(close - close.shift(n))


SIGNALS: dict[str, Callable[[pd.Series, int], pd.Series]] = {
    "sma": sma_sign,
    "donchian": donchian_sign,
    "momentum": momentum_sign,
}


def trend_features(candles: pd.DataFrame, cfg: DailyTrendConfig) -> pd.DataFrame:
    """``close``, ``signal`` and ``vol`` on daily decision bars (UTC close-time index).

    Causal: row t uses closes up to and including t only (trailing windows). NaN until every
    lookback is full (n closes, n + 1 for momentum) and the volatility window is full
    (``vol_window + 1`` closes); warm-up rows are never filled. ``hold``: 1 on every row.
    """
    close = candles["close"].astype(float)
    if cfg.signal == "hold":
        signal = pd.Series(1.0, index=close.index)
    else:
        rule = SIGNALS[cfg.signal]
        signs = pd.concat([rule(close, n) for n in cfg.lookbacks], axis=1)
        signal = signs.mean(axis=1, skipna=False)
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
