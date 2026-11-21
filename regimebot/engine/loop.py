"""The event loop shared by backtest and paper trading.

The ``Engine`` turns decision-bar features into regimes, signals, risk decisions, fills and
trades. A feed drives it with four events: ``on_decision`` at each decision-bar close,
``fill_queued`` at the first price after a decision (entries and exits decided at a close fill
there, never at the close itself), ``on_minute`` for every base bar while a position is open
(stops and targets), and ``on_funding``. Everything it does is appended to a ``RunLog``.
A long-running feed (paper trading) also uses ``to_state``/``load_state`` across restarts,
``request_flatten`` for an emergency stop between decisions and ``cancel_queued_entries`` at
shutdown.

Causality: a decision at ``ts`` reads only ``windows`` (features up to and including ``ts``)
and state built from earlier events. Orders it queues fill no earlier than the next event.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import structlog

from regimebot.config import AppConfig, timeframe_minutes
from regimebot.engine.portfolio import Portfolio, Position
from regimebot.engine.rules import stop_reason, trailed_stop
from regimebot.execution.fill_model import market_fill, quote_from_bar_open, stop_fill, target_fill
from regimebot.market_state.regime import REQUIRED_FEATURES, classify, playbook_for
from regimebot.risk.manager import MIN_NOTIONAL, RiskManager
from regimebot.strategy.base import Strategy
from regimebot.types import (
    Bar,
    ExitReason,
    Fill,
    Regime,
    RiskDecision,
    RiskEvent,
    Signal,
    Trade,
    entry_order_side,
    exit_order_side,
    side_sign,
)

log = structlog.get_logger(__name__)

REGIME_INPUTS = ("close", *REQUIRED_FEATURES)  # every value ``classify`` reads


@dataclass
class RunLog:
    """Everything an engine run produced, in event order (the rows ``persist_result`` saves).

    ``decisions`` are ``(index into signals, ts, RiskDecision)``; an entry cancelled after
    approval adds a second decision for the same signal. ``regimes`` carry the inputs the
    classifier read (NaN stored as None). ``equity`` rows are ``(ts, equity, gross_exposure,
    drawdown)``, one per decision (and per ``flatten_all``).
    """

    signals: list[Signal] = field(default_factory=list)
    decisions: list[tuple[int, datetime, RiskDecision]] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    regimes: list[tuple[str, datetime, Regime, dict[str, float | None]]] = field(
        default_factory=list
    )
    risk_events: list[RiskEvent] = field(default_factory=list)
    equity: list[tuple[datetime, float, float, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _QueuedEntry:
    signal_index: int
    signal: Signal
    qty: float  # approved qty; the fill may cut it


def _last_row(window: pd.DataFrame | None, ts: datetime) -> tuple[dict[str, float] | None, str]:
    """The window's last row if it is the decision bar at ``ts`` with a usable close, else
    (None, why). Reads one numpy row: ``iloc`` per decision is several times slower."""
    if window is None or window.empty:
        return None, "no features"
    last = window.index[-1]
    if last != ts:
        return None, f"last features row at {last}"
    row = dict(zip(window.columns.tolist(), window.to_numpy(dtype=float)[-1].tolist(), strict=True))
    close = row.get("close", math.nan)
    if not (math.isfinite(close) and close > 0):
        return None, f"invalid close {close!r}"
    return row, ""


class Engine:
    """Regimes, signals, risk, fills and trades for ``cfg.exchange.symbols``.

    At most one position or queued order per symbol. A queued exit takes the whole position;
    a queued entry fills once, possibly cut by the participation cap, and only before the next
    decision. ``strategies`` are looked up by name in each regime's playbook order.
    """

    def __init__(
        self,
        cfg: AppConfig,
        strategies: Sequence[Strategy],
        risk: RiskManager,
        portfolio: Portfolio,
        log: RunLog,
    ) -> None:
        self.cfg = cfg
        self.strategies = {s.name: s for s in strategies}
        if len(self.strategies) != len(strategies):
            raise ValueError(f"duplicate strategy names in {[s.name for s in strategies]}")
        self._decision_width = timedelta(minutes=timeframe_minutes(cfg.exchange.decision_timeframe))
        self.risk = risk
        self.portfolio = portfolio
        self.log = log
        self._entries: dict[str, _QueuedEntry] = {}
        self._exits: dict[str, ExitReason] = {}
        self._last_close: dict[str, float] = {}  # last decision close: exit ref_price

    def is_active(self, symbol: str) -> bool:
        """True when ``symbol`` has an open position or a queued order (it needs bars)."""
        return (
            symbol in self.portfolio.positions or symbol in self._entries or symbol in self._exits
        )

    @property
    def queued_exits(self) -> dict[str, ExitReason]:
        """A copy of the queued exits by symbol."""
        return dict(self._exits)

    def to_state(self) -> dict[str, Any]:
        """JSON-safe snapshot of what a restart needs besides the portfolio and risk state: the
        queued exits and the last decision closes (an exit's ``ref_price``). Queued entries are
        not included: an entry only fills right after its own decision."""
        return {
            "queued_exits": {s: r.value for s, r in self._exits.items()},
            "last_close": dict(self._last_close),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore a ``to_state`` snapshot. Raises ValueError, changing nothing, on a missing
        key or a wrong type, an unknown exit reason or symbol, a close that is not finite and
        positive, or a queued exit without a last close (it could not be priced)."""
        if not isinstance(state, Mapping):
            raise ValueError(f"engine state must be a mapping, got {type(state).__name__}")
        if missing := sorted({"queued_exits", "last_close"} - set(state)):
            raise ValueError(f"engine state: missing keys {missing}")
        for key in ("queued_exits", "last_close"):
            if not isinstance(state[key], Mapping):
                raise ValueError(f"engine state: {key} must be a mapping")
        try:
            exits = {str(s): ExitReason(r) for s, r in state["queued_exits"].items()}
            closes = {str(s): float(v) for s, v in state["last_close"].items()}
        except (TypeError, ValueError) as exc:  # e.g. an unknown reason, a None close
            raise ValueError(f"engine state: {exc}") from None
        if unknown := sorted((set(exits) | set(closes)) - set(self.cfg.exchange.symbols)):
            raise ValueError(f"engine state: unknown symbols {unknown}")
        if bad := sorted(s for s, c in closes.items() if not (math.isfinite(c) and c > 0)):
            raise ValueError(f"engine state: last close of {bad} must be finite and positive")
        if unpriced := sorted(set(exits) - set(closes)):
            raise ValueError(f"engine state: queued exits {unpriced} have no last close")
        self._exits, self._last_close = exits, closes

    # --- events --------------------------------------------------------------------------

    def on_decision(
        self,
        ts: datetime,
        windows: Mapping[str, pd.DataFrame],
        spreads: Mapping[str, float] | None = None,
    ) -> None:
        """Process the decision bar closing at ``ts``: mark, classify, manage open positions,
        check loss limits, then look for entries. A symbol without a window ending at ``ts``
        is ``NO_TRADE`` (one ``stale_data`` event); its open position keeps only its stop and
        target. Ends with an equity snapshot."""
        rows: dict[str, dict[str, float]] = {}
        for symbol in self.cfg.exchange.symbols:
            row, why = _last_row(windows.get(symbol), ts)
            if row is None:
                detail = f"{symbol}: {why}"
                self.log.risk_events.append(RiskEvent(ts, "stale_data", detail, "skip_symbol"))
                log.warning("stale_data", symbol=symbol, ts=ts.isoformat(), detail=why)
            else:
                rows[symbol] = row
        closes = {symbol: row["close"] for symbol, row in rows.items()}
        self.portfolio.mark(closes)
        self._last_close.update(closes)

        regimes = {
            symbol: self._classify(symbol, ts, rows.get(symbol), spreads)
            for symbol in self.cfg.exchange.symbols
        }
        self._manage_positions(rows, regimes)

        self.log.risk_events.extend(self.risk.check_limits(self.portfolio, ts))
        if self.risk.must_flatten:
            self.request_flatten(ts)
        else:
            for symbol, entry in list(self._entries.items()):
                if entry.signal.ts < ts:  # never offered a fill (no bars since its decision)
                    self._cancel_entry(symbol, ts, "expired", f"not filled since {entry.signal.ts}")
            self._look_for_entries(ts, windows, rows, regimes)
        self._snapshot(ts)

    def fill_queued(
        self, symbol: str, ts: datetime, bid: float, ask: float, bar_volume: float
    ) -> None:
        """Execute ``symbol``'s queued order at the quote (taker, ``market_fill``).

        An exit closes the whole position (``ref_price`` = last decision close). An entry is
        capped at ``max_participation`` of ``bar_volume`` (``ref_price`` = signal entry). An entry
        is cancelled, with a second decision for its signal, when it is first offered at or after
        ``signal.ts`` + one decision width (``expired``: the next decision has passed), when there
        is no fill or one under ``MIN_NOTIONAL`` (``no_liquidity``), or when the fill price is at
        or beyond the stop (``gapped_through_stop``: the price gapped past it, 1R would not be
        positive).

        The order is priced before it is removed: when pricing raises (a crossed quote, a
        non-finite volume), the order stays queued and the exception propagates."""
        reason = self._exits.get(symbol)
        position = self.portfolio.positions.get(symbol)
        if reason is not None and position is not None:
            fill = self._exit_fill(position, ts, bid, ask, self._last_close[symbol], bar_volume)
            self._close(fill, reason)  # removes the queued exit
        self._exits.pop(symbol, None)  # an exit whose position is already closed
        entry = self._entries.get(symbol)
        if entry is not None:
            self._fill_entry(entry, ts, bid, ask, bar_volume)

    def on_minute(self, bar: Bar) -> None:
        """Check ``bar`` against the open position's stop, then its target (stop first when one
        bar touches both: the order inside a bar is unknown, so assume the worse)."""
        position = self.portfolio.positions.get(bar.symbol)
        if position is None:
            return
        exit_side = exit_order_side(position.side)
        fill = stop_fill(bar.symbol, bar, exit_side, position.qty, position.stop, self.cfg.costs)
        if fill is not None:
            self._close(fill, stop_reason(position))
            return
        fill = target_fill(
            bar.symbol, bar, exit_side, position.qty, position.target, self.cfg.costs
        )
        if fill is not None:
            self._close(fill, ExitReason.TARGET)

    def on_funding(self, symbol: str, ts: datetime, rate: float, mark_price: float) -> None:
        """Settle a funding event stamped ``ts`` on a position held at that time. A position
        entered at or after ``ts`` was not held then and pays nothing."""
        position = self.portfolio.positions.get(symbol)
        if position is None or position.entry_ts >= ts:
            return
        self.portfolio.apply_funding(symbol, mark_price, rate)

    def flatten_all(self, ts: datetime, prices: Mapping[str, float], reason: ExitReason) -> None:
        """Close every position at ``prices`` and cancel queued entries (end of data, shutdown).

        Each exit is a taker fill on ``quote_from_bar_open(price)`` at full participation
        (``bar_volume=0``: no volume information, so the worst bounded impact), stamped ``ts``.
        A position with an exit already queued keeps that exit's reason. The equity snapshot at
        ``ts`` is replaced by the flat one."""
        self.cancel_queued_entries(ts, "cancelled", f"flatten_all: {reason}")
        for symbol, position in list(self.portfolio.positions.items()):
            bid, ask = quote_from_bar_open(prices[symbol], self.cfg.costs)
            fill = self._exit_fill(position, ts, bid, ask, prices[symbol], bar_volume=0.0)
            self._close(fill, self._exits.get(symbol, reason))
        self._exits.clear()
        self._snapshot(ts)

    def request_flatten(self, ts: datetime, reason: ExitReason = ExitReason.KILL) -> None:
        """Queue a ``reason`` exit for every open position (an exit already queued keeps its
        reason) and cancel queued entries as ``kill_switch``; the exits fill at the next
        ``fill_queued``. Used at a halted decision and for an emergency stop between them."""
        for symbol in self.portfolio.positions:
            self._exits.setdefault(symbol, reason)
        self.cancel_queued_entries(ts, "kill_switch", f"halted: {self.risk.halted_reason}")

    def cancel_queued_entries(
        self, ts: datetime, rule: str = "cancelled", detail: str = "session stopped"
    ) -> None:
        """Drop every queued entry, recording a closing decision (``rule``) for each."""
        for symbol in list(self._entries):
            self._cancel_entry(symbol, ts, rule, detail)

    # --- decision steps ------------------------------------------------------------------

    def _classify(
        self,
        symbol: str,
        ts: datetime,
        row: dict[str, float] | None,
        spreads: Mapping[str, float] | None,
    ) -> Regime:
        if row is None:
            regime, inputs = Regime.NO_TRADE, {}
        else:
            spread = None if spreads is None else spreads.get(symbol)
            regime = classify(row, self.cfg.regime, ts, spread)
            inputs = {k: None if math.isnan(v := row[k]) else v for k in REGIME_INPUTS}
        self.log.regimes.append((symbol, ts, regime, inputs))
        return regime

    def _manage_positions(
        self, rows: Mapping[str, dict[str, float]], regimes: Mapping[str, Regime]
    ) -> None:
        """Step 3: count the bar, trail the stop, queue time and regime exits. A stale symbol
        only counts the bar."""
        for symbol, position in self.portfolio.positions.items():
            position.bars_held += 1
            row = rows.get(symbol)
            if row is None:
                continue
            close, sign = row["close"], side_sign(position.side)
            if sign * (close - position.best_price) > 0:
                position.best_price = close
            position.stop = trailed_stop(position, row["atr"])
            if symbol in self._exits:
                continue
            if position.bars_held >= position.max_hold_bars:
                self._exits[symbol] = ExitReason.TIME
            elif playbook_for(regimes[symbol], self.cfg).force_close:
                self._exits[symbol] = ExitReason.REGIME

    def _look_for_entries(
        self,
        ts: datetime,
        windows: Mapping[str, pd.DataFrame],
        rows: Mapping[str, dict[str, float]],
        regimes: Mapping[str, Regime],
    ) -> None:
        """Step 5, in configured symbol order: the first signal from the strategies the regime's
        playbook allows (in playbook order) goes to risk, with queued entries as ``pending``."""
        for symbol in self.cfg.exchange.symbols:
            if symbol not in rows or self.is_active(symbol):
                continue
            playbook = playbook_for(regimes[symbol], self.cfg)
            signal = self._first_signal(
                symbol, windows[symbol], regimes[symbol], playbook.strategies
            )
            if signal is None:
                continue
            index = len(self.log.signals)
            self.log.signals.append(signal)
            pending = {s: e.qty * e.signal.entry for s, e in self._entries.items()}
            decision = self.risk.evaluate(signal, self.portfolio, playbook, ts, pending)
            self.log.decisions.append((index, ts, decision))
            if decision.approved:
                self._entries[symbol] = _QueuedEntry(index, signal, decision.qty)

    def _first_signal(
        self, symbol: str, window: pd.DataFrame, regime: Regime, allowed: Sequence[str]
    ) -> Signal | None:
        """First signal of the playbook's strategies, in the playbook's order (its priority);
        a listed strategy that is disabled in the config is skipped."""
        for name in allowed:
            strategy = self.strategies.get(name)
            if strategy is None:
                continue
            signal = strategy.generate(symbol, window, regime)
            if signal is None:
                continue
            if signal.symbol != symbol or signal.ts != window.index[-1]:
                raise ValueError(
                    f"{strategy.name} returned a signal for {signal.symbol} at {signal.ts}, "
                    f"expected {symbol} at {window.index[-1]}"
                )
            return signal
        return None

    # --- orders --------------------------------------------------------------------------

    def _fill_entry(
        self, entry: _QueuedEntry, ts: datetime, bid: float, ask: float, bar_volume: float
    ) -> None:
        """Fill ``entry`` (still queued), or cancel it; it leaves the queue only once decided."""
        signal = entry.signal
        if ts >= signal.ts + self._decision_width:  # the next decision has passed: stale signal
            detail = f"first fill offered at {ts}, signal at {signal.ts}"
            self._cancel_entry(signal.symbol, ts, "expired", detail)
            return
        fill = market_fill(
            signal.symbol,
            ts,
            entry_order_side(signal.side),
            entry.qty,
            bid,
            ask,
            signal.entry,
            bar_volume,
            self.cfg.costs,
            cap_participation=True,
        )
        if fill is None or fill.qty * fill.price < MIN_NOTIONAL:
            why = "no volume" if fill is None else f"fill notional {fill.qty * fill.price:.2f}"
            detail = f"{why} at bar volume {bar_volume}, min notional {MIN_NOTIONAL}"
            self._cancel_entry(signal.symbol, ts, "no_liquidity", detail)
            return
        if side_sign(signal.side) * (fill.price - signal.stop) <= 0:  # 1R zero or negative
            detail = f"{signal.side} fill at {fill.price} is at or beyond the stop {signal.stop}"
            self._cancel_entry(signal.symbol, ts, "gapped_through_stop", detail)
            return
        self.portfolio.open(signal, fill)
        del self._entries[signal.symbol]
        self.log.fills.append(fill)

    def _cancel_entry(self, symbol: str, ts: datetime, rule: str, detail: str) -> None:
        self._reject(self._entries.pop(symbol), ts, rule, detail)

    def _reject(self, entry: _QueuedEntry, ts: datetime, rule: str, detail: str) -> None:
        """Record that an approved entry will not fill: a second decision for its signal."""
        self.log.decisions.append((entry.signal_index, ts, RiskDecision(False, 0.0, rule, detail)))

    def _exit_fill(
        self,
        position: Position,
        ts: datetime,
        bid: float,
        ask: float,
        ref_price: float,
        bar_volume: float,
    ) -> Fill:
        fill = market_fill(
            position.symbol,
            ts,
            exit_order_side(position.side),
            position.qty,
            bid,
            ask,
            ref_price,
            bar_volume,
            self.cfg.costs,
            cap_participation=False,
        )
        if fill is None:  # market_fill always fills an uncapped order
            raise RuntimeError(f"exit for {position.symbol} got no fill")
        return fill

    def _close(self, fill: Fill, reason: ExitReason) -> None:
        trade = self.portfolio.close(fill.symbol, fill, reason)
        self._exits.pop(fill.symbol, None)
        self.risk.on_trade_closed(trade)
        self.log.fills.append(fill)
        self.log.trades.append(trade)

    def _snapshot(self, ts: datetime) -> None:
        """Append (ts, equity, gross exposure, drawdown); replaces a snapshot at the same ts."""
        p = self.portfolio
        snapshot = (ts, p.equity, p.gross_exposure, p.drawdown)
        if self.log.equity and self.log.equity[-1][0] == ts:
            self.log.equity[-1] = snapshot
        else:
            self.log.equity.append(snapshot)
