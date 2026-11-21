"""Synthetic market data and stub strategies for engine and backtest tests (deterministic).

Bars follow the project convention: a bar's ``ts`` is its close, so bar ``i`` (0-based) of a
series starting at ``start`` closes at ``start + (i + 1) * 1m`` and opens at ``start + i * 1m``.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from regimebot.types import Regime, Side, Signal, side_sign

T0 = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
MINUTE = timedelta(minutes=1)
BTC, ETH, SOL = "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"


def candles_from_closes(
    closes: list[float] | np.ndarray,
    *,
    start: datetime = T0,
    volume: float = 1e6,
    wick: float = 0.05,
) -> pd.DataFrame:
    """1m bars with the given closes: open = previous close (the first bar opens at its own
    close), high/low = max/min(open, close) +/- ``wick``, constant ``volume``. Tests edit single
    bars afterwards with ``df.loc[ts, col] = ...``."""
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate([close[:1], close[:-1]])
    idx = pd.date_range(start + MINUTE, periods=len(close), freq="1min", name="ts")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def regime_walk(
    n: int, *, seed: int, start_price: float = 100.0, start: datetime = T0
) -> pd.DataFrame:
    """Random 1m bars that switch every few hours between quiet, trending and volatile
    stretches, with volume rising with volatility. Rich enough to make the real features,
    regimes and strategies trade."""
    rng = np.random.default_rng(seed)
    vols, drifts = np.empty(n), np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(120, 900))
        kind = rng.choice(["quiet", "trend", "volatile"], p=[0.4, 0.35, 0.25])
        vol = {"quiet": 0.0003, "trend": 0.0008, "volatile": 0.002}[kind]
        drift = rng.choice([-1.0, 1.0]) * vol * 0.35 if kind == "trend" else 0.0
        vols[i : i + length], drifts[i : i + length] = vol, drift
        i += length
    rets = drifts + vols * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.standard_normal(n)) * vols * close * 0.5
    idx = pd.date_range(start + MINUTE, periods=n, freq="1min", name="ts")
    volume = 50.0 * (vols / 0.0003) * rng.lognormal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def funding_every_8h(start: datetime, n: int, *, seed: int) -> pd.Series:
    """``n`` funding rates stamped every 8h from ``start`` (random, mostly positive)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="8h", name="ts")
    return pd.Series(rng.normal(1e-4, 2e-4, n), index=idx, name="rate")


@dataclass
class Scripted:
    """Stub ``Strategy``: a signal with fixed stop/target distances from the decision close,
    at the decision times in ``at`` (every decision when ``at`` is None)."""

    name: str
    side: Side = Side.LONG
    stop_dist: float = 2.0
    target_dist: float = 4.0
    at: set[datetime] | None = None
    max_hold_bars: int = 1000
    trail_atr_mult: float | None = None

    def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None:
        ts = pd.Timestamp(window.index[-1]).to_pydatetime()
        if self.at is not None and ts not in self.at:
            return None
        close, atr = float(window["close"].iloc[-1]), float(window["atr"].iloc[-1])
        sign = side_sign(self.side)
        return Signal(
            strategy=self.name,
            symbol=symbol,
            ts=ts,
            side=self.side,
            entry=close,
            stop=close - sign * self.stop_dist,
            target=close + sign * self.target_dist,
            atr=atr,
            regime=regime,
            max_hold_bars=self.max_hold_bars,
            trail_atr_mult=self.trail_atr_mult,
        )
