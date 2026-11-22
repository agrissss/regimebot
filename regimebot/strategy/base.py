"""Strategy protocols and the shared trade-geometry helpers.

Strategies are pure: they read the features frame (last row = decision bar) and
return a ``Signal`` or ``None``. Sizing and playbook permission live elsewhere.
A ``TargetStrategy`` (daily trend) instead names a target weight per symbol.
"""

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from regimebot.config import BreakoutConfig, MeanReversionConfig, TrendConfig
from regimebot.types import Regime, Side, Signal, side_sign

StrategyConfig = BreakoutConfig | TrendConfig | MeanReversionConfig


class Strategy(Protocol):
    name: str

    def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None: ...


@runtime_checkable
class TargetStrategy(Protocol):
    """Holds a target position per symbol instead of signalling entries: at each decision, the
    signed target weight (fraction of equity) of every symbol it has a view on, from the decision
    rows (feature values at ``ts``). The engine trades toward it; risk caps it."""

    name: str

    def weights(
        self, ts: datetime, rows: Mapping[str, Mapping[str, float]]
    ) -> dict[str, float]: ...


AnyStrategy = Strategy | TargetStrategy


def all_finite(*values: float) -> bool:
    """True when every value is finite (NaN and inf are not)."""
    return all(math.isfinite(v) for v in values)


def target_is_plausible(
    target_pct: float, rv_per_bar: float, max_hold_bars: int, mult: float
) -> bool:
    """A target further than ``mult`` hold-horizon standard deviations is unreachable."""
    return target_pct <= mult * rv_per_bar * math.sqrt(max_hold_bars)


def build_signal(
    *,
    name: str,
    symbol: str,
    window: pd.DataFrame,
    regime: Regime,
    side: Side,
    stop: float,
    cfg: StrategyConfig,
    target: float | None = None,
) -> Signal | None:
    """Turn a raw stop into a validated ``Signal`` (entry = last close).

    Default target is ``max(min_target_pct, min_rr * stop_pct)`` from entry, so
    reward-to-risk is always ``>= min_rr``, and it must pass the plausibility
    check against the last ``rv``. An explicit ``target`` (mean reversion) must
    lie on the profit side with reward-to-risk ``>= min_rr``; plausibility is
    skipped for it. Returns None if any input is non-finite or a check fails.
    """
    last = window.iloc[-1]
    entry, atr = float(last["close"]), float(last["atr"])
    if not all_finite(entry, atr, stop) or entry <= 0:
        return None
    sign = side_sign(side)
    risk = sign * (entry - stop)  # > 0 only when the stop is on the loss side
    if risk <= 0:
        return None
    stop_pct = risk / entry
    if stop_pct > cfg.max_stop_pct:
        return None
    if target is None:
        rv = float(last["rv"])
        target_pct = max(cfg.min_target_pct, cfg.min_rr * stop_pct)
        if not all_finite(rv) or not target_is_plausible(
            target_pct, rv, cfg.max_hold_bars, cfg.plausibility_mult
        ):
            return None
        target = entry * (1 + sign * target_pct)
    else:
        if not all_finite(target) or sign * (target - entry) / risk < cfg.min_rr:
            return None
    return Signal(
        strategy=name,
        symbol=symbol,
        ts=pd.Timestamp(window.index[-1]).to_pydatetime(),
        side=side,
        entry=entry,
        stop=stop,
        target=target,
        atr=atr,
        regime=regime,
        max_hold_bars=cfg.max_hold_bars,
        trail_atr_mult=cfg.trail_atr_mult,
    )
