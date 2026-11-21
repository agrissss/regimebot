"""Pure position rules used by the engine at decisions and fills."""

import math

from regimebot.engine.portfolio import Position
from regimebot.types import ExitReason, side_sign


def trailed_stop(position: Position, atr: float) -> float:
    """The position's stop after trailing at a decision, never looser than the current one.

    Trailing starts once ``best_price`` is at least 1R (``|entry - initial_stop|``) in favour;
    the candidate is ``best_price -/+ trail_atr_mult * atr`` (long/short).
    """
    p = position
    sign = side_sign(p.side)
    one_r = abs(p.entry_price - p.initial_stop)
    if (
        p.trail_atr_mult is None
        or not math.isfinite(atr)
        or sign * (p.best_price - p.entry_price) < one_r
    ):
        return p.stop
    candidate = p.best_price - sign * p.trail_atr_mult * atr
    return max(p.stop, candidate) if sign > 0 else min(p.stop, candidate)


def stop_reason(position: Position) -> ExitReason:
    """TRAIL when trailing moved the stop beyond the initial stop in the trade's favour."""
    moved = side_sign(position.side) * (position.stop - position.initial_stop) > 0
    return ExitReason.TRAIL if moved else ExitReason.STOP
