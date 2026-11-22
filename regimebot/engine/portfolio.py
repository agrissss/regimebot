"""Portfolio accounting for linear USDT perpetuals: positions, cash, equity and drawdown.

``cash`` is realised equity. The entry fee leaves cash at open, funding leaves (or joins) cash
when applied, and at close cash moves by the gross pnl minus the exit fee. Hence, once flat,
the sum of ``Trade.net_pnl`` equals ``cash - initial_equity``.

Gross pnl is ``qty * (exit - entry) * side_sign``. There is no margin or liquidation model:
the risk module caps gross leverage at 1.5x.

A target position (daily_trend, ``rebalance``) also grows and shrinks before it closes: it is
held at its average cost, a partial exit moves cash by its gross pnl at that cost minus its fee,
and the trade it ends as aggregates every fill (``Trade``).
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from typing import Any

from regimebot.execution.fill_model import funding_payment
from regimebot.types import (
    ExitReason,
    Fill,
    OrderSide,
    Regime,
    Side,
    Signal,
    Trade,
    entry_order_side,
    exit_order_side,
    side_sign,
)

QTY_REL_TOL = 1e-9  # an exit fill's qty must match the position's qty this closely
# Added with target positions: a state saved before them lacks these keys and is a position
# opened from a signal, for which the defaults are exact.
_LATER_STATE_KEYS = frozenset({"entry_qty", "entry_notional", "realised"})


@dataclass(kw_only=True, slots=True)
class Position:
    """One open position. The engine moves ``stop`` and ``best_price`` and counts ``bars_held``.

    A target position (``Portfolio.rebalance``) has no stop, target, ATR or regime (NaN, NaN,
    NaN, None; ``max_hold_bars`` 0 and unused) and records its trade's legs in ``entry_qty``,
    ``entry_notional`` and ``realised``; a position opened from a signal keeps them at their
    defaults (one entry fill, no partial exit).
    """

    symbol: str
    strategy: str
    side: Side
    regime: Regime | None  # regime at entry; None for a target position
    qty: float  # base units, > 0
    entry_price: float  # executed entry price (a target position: average cost of ``qty``)
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
    # A target position's trade so far (None: just ``qty`` at ``entry_price``, one entry fill):
    entry_qty: float | None = None  # total qty of its entry-side fills
    entry_notional: float | None = None  # their total notional
    realised: float = 0.0  # gross pnl of its partial exits, already in cash

    def unrealised_pnl(self, price: float) -> float:
        """Gross pnl if the whole position were closed at ``price`` (no fees)."""
        return self.qty * (price - self.entry_price) * side_sign(self.side)

    def to_state(self) -> dict[str, Any]:
        """JSON-safe dict for ``open_positions``: enums as values, ``entry_ts`` as ISO 8601 (a
        target position's NaN stop and target are JSON ``NaN``, which ``json`` reads back)."""
        state = asdict(self)
        state.update(
            side=self.side.value,
            regime=None if self.regime is None else self.regime.value,
            entry_ts=self.entry_ts.isoformat(),
        )
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "Position":
        """Inverse of ``to_state``; ``entry_ts`` comes back in UTC.

        Raises ValueError on missing or unknown keys (a missing ``fees`` must not silently
        default to 0; only the target-position keys, which a state saved before them lacks, take
        their defaults), an unknown enum value or a naive timestamp.
        """
        expected = {f.name for f in fields(cls)}
        missing = sorted(expected - set(state) - _LATER_STATE_KEYS)
        unknown = sorted(set(state) - expected)
        if missing or unknown:
            raise ValueError(f"position state: missing keys {missing}, unknown keys {unknown}")
        entry_ts = datetime.fromisoformat(state["entry_ts"])
        if entry_ts.utcoffset() is None:
            raise ValueError(f"entry_ts {state['entry_ts']!r} has no timezone")
        return cls(
            **{
                **state,
                "side": Side(state["side"]),
                "regime": None if state["regime"] is None else Regime(state["regime"]),
                "entry_ts": entry_ts.astimezone(UTC),
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

    def signed_qty(self, symbol: str) -> float:
        """The symbol's position in base units, positive long and negative short; 0.0 when flat."""
        position = self.positions.get(symbol)
        return 0.0 if position is None else position.qty * side_sign(position.side)

    def open(self, signal: Signal, fill: Fill) -> Position:
        """Open ``signal`` at ``fill`` (whose qty may be below the requested qty); pays its fee.

        Raises ValueError for a fill that does not match the signal, or one at or beyond the stop
        (1R would be zero or negative: the engine cancels such an entry before it gets here)."""
        if signal.symbol in self.positions:
            raise ValueError(f"{signal.symbol} already has an open position")
        if fill.symbol != signal.symbol or fill.side != entry_order_side(signal.side):
            raise ValueError(
                f"{fill.side} fill for {fill.symbol} cannot open {signal.side} {signal.symbol}"
            )
        if fill.ts < signal.ts:
            raise ValueError(f"entry fill at {fill.ts} is before its signal at {signal.ts}")
        if side_sign(signal.side) * (fill.price - signal.stop) <= 0:
            raise ValueError(
                f"{signal.side} entry at {fill.price} is at or beyond the stop {signal.stop}: "
                "1R would not be positive"
            )
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
        """Close the whole position at ``fill``: cash moves by gross pnl minus the exit fee.

        A target position that was partly reduced before closes as the trade of all its fills:
        gross pnl adds the partial exits' realised pnl, entry and exit prices are its legs'
        quantity-weighted averages (the exit price follows from the gross pnl)."""
        position = self.positions.get(symbol)
        if position is None:
            raise ValueError(f"no open position in {symbol}")
        if fill.symbol != symbol or fill.side != exit_order_side(position.side):
            raise ValueError(
                f"{fill.side} fill for {fill.symbol} cannot close {position.side} {symbol}"
            )
        if not math.isclose(fill.qty, position.qty, rel_tol=QTY_REL_TOL):
            raise ValueError(
                f"exit qty {fill.qty} != position qty {position.qty} (no partial exits)"
            )
        if fill.ts < position.entry_ts:
            raise ValueError(f"exit at {fill.ts} is before entry at {position.entry_ts}")
        last = position.unrealised_pnl(fill.price)  # what is left, at its (average) cost
        gross = position.realised + last
        qty, entry, exit_ = position.qty, position.entry_price, fill.price
        if position.entry_qty is not None and position.entry_qty != position.qty:  # partial exits
            assert position.entry_notional is not None  # set together with entry_qty
            qty = position.entry_qty
            entry = position.entry_notional / qty
            exit_ = entry + side_sign(position.side) * gross / qty
        fees = position.fees + fill.fee
        net = gross - fees - position.funding
        del self.positions[symbol]
        self.cash += last - fill.fee
        self._update_peak()
        return Trade(
            symbol=symbol,
            strategy=position.strategy,
            side=position.side,
            regime=position.regime,
            qty=qty,
            entry_ts=position.entry_ts,
            exit_ts=fill.ts,
            entry_price=entry,
            exit_price=exit_,
            initial_stop=position.initial_stop,
            target=position.target,
            gross_pnl=gross,
            fees=fees,
            funding=position.funding,
            net_pnl=net,
            exit_reason=reason,
            r_multiple=net / (qty * abs(entry - position.initial_stop)),  # NaN without a stop
            bars_held=position.bars_held,
        )

    def rebalance(self, fill: Fill, strategy: str, reason: ExitReason) -> Trade | None:
        """Apply one fill of a target position (no stop, target or regime) of ``strategy``:
        open, add to, reduce, close, or flip it with one fill through zero. Returns the trade the
        fill closed (a flip closes it and opens the other side with the rest of the fill, the fee
        split pro rata by qty), else None.

        An add moves the average cost (``entry_price``) and pays its fee; a partial exit moves
        cash by its gross pnl at the average cost minus its fee. The symbol is marked at the
        fill, as ``open`` does."""
        symbol = fill.symbol
        position = self.positions.get(symbol)
        if position is not None and fill.ts < position.entry_ts:
            raise ValueError(f"fill at {fill.ts} is before entry at {position.entry_ts}")
        if position is None:
            self._open_target(fill, strategy)
            return None
        if position.entry_qty is None or position.entry_notional is None:
            raise ValueError(f"{symbol} was opened from a signal: it is not a target position")
        if fill.side == entry_order_side(position.side):  # add
            qty = position.qty + fill.qty
            position.entry_price = (
                position.qty * position.entry_price + fill.qty * fill.price
            ) / qty
            position.qty = qty
            position.entry_qty += fill.qty
            position.entry_notional += fill.qty * fill.price
            position.fees += fill.fee
            self.cash -= fill.fee
        elif fill.qty < position.qty * (1 - QTY_REL_TOL):  # partial exit
            gross = fill.qty * (fill.price - position.entry_price) * side_sign(position.side)
            position.qty -= fill.qty
            position.realised += gross
            position.fees += fill.fee
            self.cash += gross - fill.fee
        elif fill.qty <= position.qty * (1 + QTY_REL_TOL):  # the whole position
            return self.close(symbol, fill, reason)
        else:  # through zero: close, then open the other side with the rest
            closing = replace(fill, qty=position.qty, fee=fill.fee * position.qty / fill.qty)
            trade = self.close(symbol, closing, reason)
            self._open_target(
                replace(fill, qty=fill.qty - closing.qty, fee=fill.fee - closing.fee), strategy
            )
            return trade
        self.last_prices[symbol] = fill.price
        self._update_peak()
        return None

    def _open_target(self, fill: Fill, strategy: str) -> None:
        if not (math.isfinite(fill.qty) and fill.qty > 0):
            raise ValueError(f"fill qty must be positive, got {fill.qty!r}")
        self.positions[fill.symbol] = Position(
            symbol=fill.symbol,
            strategy=strategy,
            side=Side.LONG if fill.side == OrderSide.BUY else Side.SHORT,
            regime=None,
            qty=fill.qty,
            entry_price=fill.price,
            entry_ts=fill.ts,
            stop=math.nan,
            initial_stop=math.nan,
            target=math.nan,
            atr_at_entry=math.nan,
            max_hold_bars=0,
            trail_atr_mult=None,
            best_price=fill.price,
            fees=fill.fee,
            entry_qty=fill.qty,
            entry_notional=fill.qty * fill.price,
        )
        self.last_prices[fill.symbol] = fill.price
        self.cash -= fill.fee
        self._update_peak()

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
