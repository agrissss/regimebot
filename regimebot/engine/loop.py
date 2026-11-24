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

With a ``TargetStrategy`` (daily_trend) the decision step is different and everything around it
is the same: no regime, no stop, target or time exit; at each decision every symbol with a target
weight is traded toward it (``_rebalance``), and the order fills like an entry at the next
``fill_queued`` (with ``strategies.daily_trend.fill_minutes`` above one bar, it keeps working: the
unfilled rest fills at each later bar's ``fill_queued`` until its window ends). Halts, queued
exits, funding, flattening and the equity snapshot work as for the playbook strategies; the
loss-streak rules do not apply (no discrete stopped trades). Because a day has one decision, the
loss limits are also checked at every hour boundary: ``on_minute`` marks each open position at
the bar closing on the hour and, once all are marked, checks them on that equity; a halt queues
KILL exits, which fill at the next ``fill_queued`` (the next bar's open in a backtest, the next
poll's quote in paper).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import structlog

from regimebot.config import AppConfig, timeframe_minutes
from regimebot.engine.portfolio import Portfolio, Position
from regimebot.engine.rules import stop_reason, trailed_stop
from regimebot.execution.fill_model import market_fill, quote_from_bar_open, stop_fill, target_fill
from regimebot.market_state.regime import REQUIRED_FEATURES, classify, playbook_for
from regimebot.risk.manager import RiskManager
from regimebot.strategy.base import AnyStrategy, Strategy, TargetStrategy
from regimebot.types import (
    Bar,
    ExitReason,
    Fill,
    OrderSide,
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
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
LIMIT_CHECK = timedelta(hours=1)  # target positions: loss limits checked on each hour boundary


@dataclass
class RunLog:
    """Everything an engine run produced, in event order (the rows ``persist_result`` saves).

    ``decisions`` are ``(index into signals, ts, RiskDecision)``; an entry cancelled after
    approval adds a second decision for the same signal. A target position's orders have no
    signal (index None); their decision details start with the symbol. ``regimes`` carry the
    inputs the classifier read (NaN stored as None; ``spread_bps`` only when a live spread was
    given). ``equity`` rows are ``(ts, equity, gross_exposure, drawdown)``, one per decision (and
    per ``flatten_all``).
    """

    signals: list[Signal] = field(default_factory=list)
    decisions: list[tuple[int | None, datetime, RiskDecision]] = field(default_factory=list)
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


@dataclass(frozen=True, slots=True)
class _QueuedTarget:
    ts: datetime  # the decision that queued it
    target: float  # approved target position, signed base units
    ref_price: float  # decision close
    offered: bool = False  # offered a fill already: it is working (fill_minutes)


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

    At most one position or queued order per symbol (a target position may also have one
    queued order, toward its target). A queued exit takes the whole position; a queued entry
    fills once, possibly cut by the participation cap, and only before the next decision (a
    target order too, unless it keeps working: ``working_until``).
    ``strategies`` are looked up by name in each regime's playbook order. A daily_trend config
    takes exactly one ``TargetStrategy`` instead (and nothing else).
    """

    def __init__(
        self,
        cfg: AppConfig,
        strategies: Sequence[AnyStrategy],
        risk: RiskManager,
        portfolio: Portfolio,
        log: RunLog,
    ) -> None:
        self.cfg = cfg
        targets = [s for s in strategies if isinstance(s, TargetStrategy)]
        signals = [s for s in strategies if not isinstance(s, TargetStrategy)]
        if (
            cfg.strategies.daily_trend.enabled != bool(targets)
            or len(targets) > 1
            or (targets and signals)
        ):
            names = [s.name for s in strategies]
            raise ValueError(
                f"strategies {names}: a daily_trend config runs exactly one target-weight "
                "strategy and nothing else, any other config none"
            )
        self._target: TargetStrategy | None = targets[0] if targets else None
        self.strategies: dict[str, Strategy] = {s.name: s for s in signals}
        if len(self.strategies) != len(signals):
            raise ValueError(f"duplicate strategy names in {[s.name for s in signals]}")
        self._decision_width = timedelta(minutes=timeframe_minutes(cfg.exchange.decision_timeframe))
        self._base_width = timedelta(minutes=timeframe_minutes(cfg.exchange.base_timeframe))
        self._fill_window = timedelta(minutes=cfg.strategies.daily_trend.fill_minutes)
        self._works = self._fill_window > self._base_width  # target orders fill over later bars
        self.risk = risk
        self.portfolio = portfolio
        self.log = log
        self._entries: dict[str, _QueuedEntry] = {}
        self._targets: dict[str, _QueuedTarget] = {}  # a target position's queued order
        self._exits: dict[str, ExitReason] = {}
        self._last_close: dict[str, float] = {}  # last decision close: exit ref_price
        # target positions: the hour each was last marked at (loss limits once all are)
        # ponytail: not saved, so a restart between two symbols' bars of one hour skips that
        # hour's check; save it with the engine state if feeds ever split an hour across polls.
        self._marked_hour: dict[str, datetime] = {}

    def is_active(self, symbol: str) -> bool:
        """True when ``symbol`` has an open position or a queued order (it needs bars)."""
        return (
            symbol in self.portfolio.positions
            or self.has_queued_order(symbol)
            or symbol in self._exits
        )

    def has_queued_order(self, symbol: str) -> bool:
        """True when ``symbol`` has a queued entry or target-position order: it fills right after
        its decision (a working target order also at later bars, ``working_until``; an exit is
        retried until it fills)."""
        return symbol in self._entries or symbol in self._targets

    def working_until(self, symbol: str) -> datetime | None:
        """End of the fill window of ``symbol``'s queued target-position order when it may keep
        filling at later base bars (``strategies.daily_trend.fill_minutes`` longer than one bar),
        else None. A feed offers it one ``fill_queued`` per base bar opening before then."""
        order = self._targets.get(symbol)
        if order is None or not self._works:
            return None
        return order.ts + self._fill_window

    @property
    def needs_minute_bars(self) -> bool:
        """False for a target strategy: its positions have no stop or target to check, so a feed
        may send ``on_minute`` only the bars closing on an hour boundary (the loss-limit checks)
        and a ``fill_queued`` after each (a halt's exits), plus one at each bar a working order
        may fill at (``working_until``)."""
        return self._target is None

    @property
    def queued_exits(self) -> dict[str, ExitReason]:
        """A copy of the queued exits by symbol."""
        return dict(self._exits)

    def to_state(self) -> dict[str, Any]:
        """JSON-safe snapshot of what a restart needs besides the portfolio and risk state: the
        queued exits and the last decision closes (an exit's ``ref_price``). Queued entries and
        target-position orders are not included: they only fill right after their decision. A
        working target order is therefore dropped by a restart; what it filled is in the
        portfolio, so nothing fills twice, and the next decision trades from there."""
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
        check loss limits, then look for entries (a target strategy: rebalance). A symbol without
        a window ending at ``ts`` is ``NO_TRADE`` (one ``stale_data`` event); its open position
        keeps only its stop and target (a target position is held as it is). Ends with an equity
        snapshot."""
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

        regimes: dict[str, Regime] = {}
        if self._target is None:
            regimes = {
                symbol: self._classify(symbol, ts, rows.get(symbol), spreads)
                for symbol in self.cfg.exchange.symbols
            }
            self._manage_positions(rows, regimes)
        else:  # no regime, and no trailing, time or regime exit for a target position
            for position in self.portfolio.positions.values():
                position.bars_held += 1

        self.log.risk_events.extend(self.risk.check_limits(self.portfolio, ts))
        if self.risk.must_flatten:
            self.request_flatten(ts)
        else:
            for symbol, entry in list(self._entries.items()):
                if entry.signal.ts < ts:  # never offered a fill (no bars since its decision)
                    self._cancel_entry(symbol, ts, "expired", f"not filled since {entry.signal.ts}")
            for symbol, order in list(self._targets.items()):
                if order.ts < ts:  # this decision replaces it
                    why = "still working, replaced" if order.offered else "not filled"
                    self._cancel_target(symbol, ts, "expired", f"{why} since {order.ts}")
            if self._target is None:
                self._look_for_entries(ts, windows, rows, regimes)
            else:
                self._rebalance(ts, rows, self._target)
        self._snapshot(ts)

    def fill_queued(
        self, symbol: str, ts: datetime, bid: float, ask: float, bar_volume: float
    ) -> None:
        """Execute ``symbol``'s queued order at the quote (taker, ``market_fill``).

        An exit closes the whole position (``ref_price`` = last decision close). An entry is
        capped at ``max_participation`` of ``bar_volume`` (``ref_price`` = signal entry). An entry
        is cancelled, with a second decision for its signal, when it is first offered at or after
        ``signal.ts`` + one decision width (``expired``: the next decision has passed), when there
        is no fill or one under ``risk.min_notional`` (``no_liquidity``), or when the fill price is
        at or beyond the stop (``gapped_through_stop``: the price gapped past it, 1R would not be
        positive). A target-position order is handled like an entry (``_fill_target``).

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
        order = self._targets.get(symbol)
        if order is not None:
            self._fill_target(symbol, order, ts, bid, ask, bar_volume)

    def on_minute(self, bar: Bar) -> None:
        """Check ``bar`` against the open position's stop, then its target (stop first when one
        bar touches both: the order inside a bar is unknown, so assume the worse). A target
        position has neither: its bar closing on an hour boundary marks it, and checks the loss
        limits once every open position is marked at that hour (``_check_hour``)."""
        if self._target is not None:
            self._check_hour(bar)
            return
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

    def _check_hour(self, bar: Bar) -> None:
        """Target positions: mark ``bar.symbol`` at the close of its bar ending on an hour
        boundary (also its exit ``ref_price``); once every open position is marked at that hour,
        check the loss limits on that equity and, if halted, flatten as at a decision. A day and
        week start at the check at their 00:00 roll, the previous daily close."""
        if bar.symbol not in self.portfolio.positions or (bar.ts - _EPOCH) % LIMIT_CHECK:
            return
        self.portfolio.mark({bar.symbol: bar.close})
        self._last_close[bar.symbol] = bar.close
        self._marked_hour[bar.symbol] = bar.ts
        if all(self._marked_hour.get(s) == bar.ts for s in self.portfolio.positions):
            self.log.risk_events.extend(self.risk.check_limits(self.portfolio, bar.ts))
            if self.risk.must_flatten:
                self.request_flatten(bar.ts)

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
        """Drop every queued entry and target-position order, recording a closing decision
        (``rule``) for each."""
        for symbol in list(self._entries):
            self._cancel_entry(symbol, ts, rule, detail)
        for symbol in list(self._targets):
            self._cancel_target(symbol, ts, rule, detail)

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
            if spread is not None:  # a live quote (paper): the classifier read it too
                inputs["spread_bps"] = spread
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

    def _rebalance(
        self, ts: datetime, rows: Mapping[str, dict[str, float]], strategy: TargetStrategy
    ) -> None:
        """Target-position step, in configured symbol order. Each symbol the strategy gives a
        weight is traded toward ``weight * equity / close`` when the change is at least
        ``rebalance_band`` x equity in notional, or when the target is zero or changes sign while
        a position is open (a flip is one order through zero). The risk manager may cut the
        target to the cap room or reject the order (one decision each, without a signal);
        orders approved earlier in this decision count toward the caps, so a cap that binds
        serves the symbols in configured order (not pro rata), and a queued reduction frees
        room only once it fills (the rest of a cut target follows at the next decision). A
        symbol without a weight (stale, not warm) or with a queued exit is left as it is. A
        target under ``risk.min_notional`` counts as zero, so a position is always closed in
        full."""
        weights = strategy.weights(ts, rows)
        equity = self.portfolio.equity
        band = self.cfg.strategies.daily_trend.rebalance_band * equity
        pending: dict[str, float] = {}
        for symbol in self.cfg.exchange.symbols:
            if symbol not in weights or symbol not in rows or symbol in self._exits:
                continue
            price = rows[symbol]["close"]
            current = self.portfolio.signed_qty(symbol)
            target = weights[symbol] * equity / price
            if abs(target) * price < self.cfg.risk.min_notional:
                target = 0.0
            closing = current != 0 and (target == 0 or (target > 0) != (current > 0))
            if target == current or not (closing or abs(target - current) * price >= band):
                continue
            decision, approved = self.risk.evaluate_target(
                symbol, current, target, price, self.portfolio, ts, pending
            )
            self.log.decisions.append((None, ts, decision))
            if decision.approved:
                self._targets[symbol] = _QueuedTarget(ts, approved, price)
                added = abs(approved) * price - self.portfolio.exposure(symbol)
                pending[symbol] = max(added, 0.0)  # a reduction frees room only once filled

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
        min_notional = self.cfg.risk.min_notional
        if fill is None or fill.qty * fill.price < min_notional:
            why = "no volume" if fill is None else f"fill notional {fill.qty * fill.price:.2f}"
            detail = f"{why} at bar volume {bar_volume}, min notional {min_notional}"
            self._cancel_entry(signal.symbol, ts, "no_liquidity", detail)
            return
        if side_sign(signal.side) * (fill.price - signal.stop) <= 0:  # 1R zero or negative
            detail = f"{signal.side} fill at {fill.price} is at or beyond the stop {signal.stop}"
            self._cancel_entry(signal.symbol, ts, "gapped_through_stop", detail)
            return
        self.portfolio.open(signal, fill)
        del self._entries[signal.symbol]
        self.log.fills.append(fill)

    def _fill_target(
        self,
        symbol: str,
        order: _QueuedTarget,
        ts: datetime,
        bid: float,
        ask: float,
        bar_volume: float,
    ) -> None:
        """Trade ``symbol`` toward its queued target (still queued), or cancel the order; it
        leaves the queue only once decided. An order that only adds exposure is cut to
        ``max_participation`` of ``bar_volume`` and cancelled (``no_liquidity``) with no fill or
        one under ``risk.min_notional``, as an entry is; one that reduces or closes the position
        always fills in full, as an exit does. A flip closes in full and opens only as much as an
        entry could (``max_participation`` of ``bar_volume``; nothing when that is worth less
        than ``risk.min_notional`` at the decision close): still one order. ``expired`` as for
        an entry. The order is cancelled (``kill_switch``) while halted or while the kill file
        exists, the risk manager's first rules.

        ``fill_minutes`` 1: one fill, and the rest of the target is traded at the next decision.
        Longer, the order keeps working: it is offered again at each later base bar opening
        within ``fill_minutes`` of the decision, each time for what is left of the approved
        target (never more), priced and capped as above; an offer that cannot fill (no volume,
        under ``risk.min_notional``) waits for the next bar. It ends when the rest is worth less
        than ``risk.min_notional`` at the decision close (done, no row), or with the last bar of
        the window, or at an offer after it (the rest dropped, ``no_liquidity``)."""
        if ts >= order.ts + self._decision_width:
            detail = f"first fill offered at {ts}, decision at {order.ts}"
            self._cancel_target(symbol, ts, "expired", detail)
            return
        if (halt := self.risk.halted_reason) or self.risk.kill_file.exists():
            why = f"halted: {halt}" if halt else f"kill file {self.risk.kill_file} exists"
            self._cancel_target(symbol, ts, "kill_switch", why)
            return
        window_end = order.ts + self._fill_window
        if order.offered and ts >= window_end:
            self._drop_rest(symbol, order, ts)
            return
        last = not self._works or ts + self._base_width >= window_end  # no later bar in it
        current = self.portfolio.signed_qty(symbol)
        adds = current * order.target >= 0 and abs(order.target) > abs(current)
        qty = abs(order.target - current)
        if current * order.target < 0:  # a flip: cap its opening leg like an entry
            room = self.cfg.costs.max_participation * max(bar_volume, 0.0)
            opening = min(abs(order.target), room)
            if opening * order.ref_price < self.cfg.risk.min_notional:
                opening = 0.0
            qty = abs(current) + opening
        fill = market_fill(
            symbol,
            ts,
            OrderSide.BUY if order.target > current else OrderSide.SELL,
            qty,
            bid,
            ask,
            order.ref_price,
            bar_volume,
            self.cfg.costs,
            cap_participation=adds,
        )
        min_notional = self.cfg.risk.min_notional
        if fill is None or (adds and fill.qty * fill.price < min_notional):
            if not last:  # a working order waits for the next bar
                self._targets[symbol] = replace(order, offered=True)
                return
            why = "no volume" if fill is None else f"fill notional {fill.qty * fill.price:.2f}"
            detail = f"{why} at bar volume {bar_volume}, min notional {min_notional}"
            self._cancel_target(symbol, ts, "no_liquidity", detail)
            return
        assert self._target is not None  # only a target strategy queues target orders
        trade = self.portfolio.rebalance(fill, self._target.name, ExitReason.SIGNAL)
        self.log.fills.append(fill)
        if trade is not None:
            self.log.trades.append(trade)
        rest = abs(order.target - self.portfolio.signed_qty(symbol))
        if rest * order.ref_price < min_notional:  # done: the target reached, or only dust left
            del self._targets[symbol]
        elif not last:
            self._targets[symbol] = replace(order, offered=True)
        elif self._works:
            self._drop_rest(symbol, order, ts)
        else:  # fill_minutes 1: the rest is traded at the next decision, as before (no row)
            del self._targets[symbol]

    def _drop_rest(self, symbol: str, order: _QueuedTarget, ts: datetime) -> None:
        """End a working target order whose fill window is over, recording the rest it drops."""
        rest = abs(order.target - self.portfolio.signed_qty(symbol))
        minutes = self.cfg.strategies.daily_trend.fill_minutes
        detail = (
            f"{rest:.8g} of target {order.target:+.8g} left after the {minutes}-minute fill window"
        )
        self._cancel_target(symbol, ts, "no_liquidity", detail)

    def _cancel_target(self, symbol: str, ts: datetime, rule: str, detail: str) -> None:
        """Drop ``symbol``'s queued target order, recording that it will not fill."""
        del self._targets[symbol]
        decision = RiskDecision(False, 0.0, rule, f"{symbol}: {detail}")
        self.log.decisions.append((None, ts, decision))

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
        if self._target is None:  # the loss-streak rules do not apply to target positions
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
