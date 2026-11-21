"""Portfolio accounting for linear USDT perpetuals: positions, cash, equity and drawdown.

``cash`` is realised equity. The entry fee leaves cash at open, funding leaves (or joins) cash
when applied, and at close cash moves by the gross pnl minus the exit fee. Hence, once flat,
the sum of ``Trade.net_pnl`` equals ``cash - initial_equity``.

Gross pnl is ``qty * (exit - entry) * side_sign``. There is no margin or liquidation model:
the risk module caps gross leverage at 1.5x.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from regimebot.execution.fill_model import funding_payment
from regimebot.types import ExitReason, Fill, OrderSide, Regime, Side, Signal, Trade, side_sign

QTY_REL_TOL = 1e-9  # an exit fill's qty must match the position's qty this closely

_ENTRY_ORDER = {Side.LONG: OrderSide.BUY, Side.SHORT: OrderSide.SELL}
_EXIT_ORDER = {Side.LONG: OrderSide.SELL, Side.SHORT: OrderSide.BUY}


@dataclass(kw_only=True, slots=True)
class Position:
    """One open position. The engine moves ``stop`` and ``best_price`` and counts ``bars_held``."""

    symbol: str
    strategy: str
    side: Side
    regime: Regime  # regime at entry
    qty: float  # base units, > 0
    entry_price: float  # executed entry price
    entry_ts: datetime
    stop: float  # current stop (a trailing stop moves it)
    initial_stop: float  # stop at entry: defines 1R
    target: float
    atr_at_entry: float
    max_hold_bars: int
    trail_atr_mult: float | None
    bars_held: int = 0
    best_price: float  # most favourable price seen, for the trailing stop
    fees: float = 0.0  # fees paid so far (the entry fee), quote currency
    funding: float = 0.0  # funding paid so far, negative = received

    def unrealised_pnl(self, price: float) -> float:
        """Gross pnl if the whole position were closed at ``price`` (no fees)."""
        return self.qty * (price - self.entry_price) * side_sign(self.side)

    def to_state(self) -> dict[str, Any]:
        """JSON-safe dict for ``open_positions``: enums as values, ``entry_ts`` as ISO 8601."""
        state = asdict(self)
        state.update(
            side=self.side.value, regime=self.regime.value, entry_ts=self.entry_ts.isoformat()
        )
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "Position":
        """Inverse of ``to_state``; raises ValueError on an unknown enum or a naive timestamp."""
        entry_ts = datetime.fromisoformat(state["entry_ts"])
        if entry_ts.utcoffset() is None:
            raise ValueError(f"entry_ts {state['entry_ts']!r} has no timezone")
        return cls(
            **{
                **state,
                "side": Side(state["side"]),
                "regime": Regime(state["regime"]),
                "entry_ts": entry_ts,
            }
        )


class Portfolio:
    """Cash, open positions (at most one per symbol), last marks and the equity peak.

    Every position is marked at its entry fill on open, so ``equity`` is always defined.
    The peak is updated on every change of equity (open, close, funding, mark), so
    ``drawdown`` is always >= 0.
    """

    def __init__(self, initial_equity: float) -> None:
        if not (math.isfinite(initial_equity) and initial_equity > 0):
            raise ValueError(f"initial_equity must be positive, got {initial_equity!r}")
        self.initial_equity = initial_equity
        self.cash = initial_equity  # realised equity
        self.positions: dict[str, Position] = {}
        self.last_prices: dict[str, float] = {}
        self.peak_equity = initial_equity

    @property
    def equity(self) -> float:
        """Cash plus the unrealised pnl of every position at its last mark."""
        unrealised = sum(p.unrealised_pnl(self.last_prices[s]) for s, p in self.positions.items())
        return self.cash + unrealised

    @property
    def drawdown(self) -> float:
        """Fraction of equity below its peak, >= 0."""
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def gross_exposure(self) -> float:
        """Sum of the absolute notionals of all positions at their last marks."""
        return sum(self.exposure(symbol) for symbol in self.positions)

    def exposure(self, symbol: str) -> float:
        """Absolute notional of the symbol's position at its last mark, 0.0 when flat."""
        position = self.positions.get(symbol)
        return 0.0 if position is None else abs(position.qty * self.last_prices[symbol])

    def open(self, signal: Signal, fill: Fill) -> Position:
        """Open ``signal`` at ``fill`` (whose qty may be below the requested qty); pays its fee."""
        if signal.symbol in self.positions:
            raise ValueError(f"{signal.symbol} already has an open position")
        if fill.symbol != signal.symbol or fill.side != _ENTRY_ORDER[signal.side]:
            raise ValueError(
                f"{fill.side} fill for {fill.symbol} cannot open {signal.side} {signal.symbol}"
            )
        if fill.price == signal.stop:
            raise ValueError("entry price equals the stop: 1R would be zero")
        position = Position(
            symbol=signal.symbol,
            strategy=signal.strategy,
            side=signal.side,
            regime=signal.regime,
            qty=fill.qty,
            entry_price=fill.price,
            entry_ts=fill.ts,
            stop=signal.stop,
            initial_stop=signal.stop,
            target=signal.target,
            atr_at_entry=signal.atr,
            max_hold_bars=signal.max_hold_bars,
            trail_atr_mult=signal.trail_atr_mult,
            best_price=fill.price,
            fees=fill.fee,
        )
        self.positions[signal.symbol] = position
        self.last_prices[signal.symbol] = fill.price
        self.cash -= fill.fee
        self._update_peak()
        return position

    def close(self, symbol: str, fill: Fill, reason: ExitReason) -> Trade:
        """Close the whole position at ``fill``: cash moves by gross pnl minus the exit fee."""
        position = self.positions.get(symbol)
        if position is None:
            raise ValueError(f"no open position in {symbol}")
        if fill.symbol != symbol or fill.side != _EXIT_ORDER[position.side]:
            raise ValueError(
                f"{fill.side} fill for {fill.symbol} cannot close {position.side} {symbol}"
            )
        if not math.isclose(fill.qty, position.qty, rel_tol=QTY_REL_TOL):
            raise ValueError(
                f"exit qty {fill.qty} != position qty {position.qty} (no partial exits)"
            )
        if fill.ts < position.entry_ts:
            raise ValueError(f"exit at {fill.ts} is before entry at {position.entry_ts}")
        gross = position.unrealised_pnl(fill.price)
        fees = position.fees + fill.fee
        net = gross - fees - position.funding
        del self.positions[symbol]
        self.cash += gross - fill.fee
        self._update_peak()
        return Trade(
            symbol=symbol,
            strategy=position.strategy,
            side=position.side,
            regime=position.regime,
            qty=position.qty,
            entry_ts=position.entry_ts,
            exit_ts=fill.ts,
            entry_price=position.entry_price,
            exit_price=fill.price,
            initial_stop=position.initial_stop,
            target=position.target,
            gross_pnl=gross,
            fees=fees,
            funding=position.funding,
            net_pnl=net,
            exit_reason=reason,
            r_multiple=net / (position.qty * abs(position.entry_price - position.initial_stop)),
            bars_held=position.bars_held,
        )

    def apply_funding(self, symbol: str, mark_price: float, rate: float) -> None:
        """Settle one funding event on the symbol's position; a no-op when flat."""
        position = self.positions.get(symbol)
        if position is None:
            return
        payment = funding_payment(position.side, position.qty, mark_price, rate)
        self.cash -= payment
        position.funding += payment
        self._update_peak()

    def mark(self, prices: Mapping[str, float]) -> None:
        """Store last prices (any symbols) and update the equity peak."""
        for symbol, price in prices.items():
            if not (math.isfinite(price) and price > 0):
                raise ValueError(f"mark price for {symbol} must be positive, got {price!r}")
        self.last_prices.update(prices)
        self._update_peak()

    def _update_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity)
