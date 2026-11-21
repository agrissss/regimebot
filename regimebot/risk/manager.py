"""Risk manager: position sizing, exposure caps, loss limits and the kill switch.

Risk overrides strategy: nothing trades without an approved ``RiskDecision``. ``evaluate``
sizes a signal so that a stop hit loses at most ``risk_per_trade`` of equity (before costs),
then caps its notional by the remaining per-asset, per-group and gross exposure room.
``check_limits`` halts trading on a daily, weekly or drawdown loss limit or the kill file.
Daily and weekly halts clear at the next UTC day / ISO week (Monday 00:00 UTC); drawdown and
manual-kill halts stay until ``clear_halt``. All exposures are valued at the portfolio's marks.
"""

import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from regimebot.config import AppConfig, PlaybookEntry
from regimebot.engine.portfolio import Portfolio
from regimebot.types import RiskDecision, RiskEvent, Signal, Trade, side_sign

log = structlog.get_logger(__name__)

DEFAULT_KILL_FILE = Path(__file__).resolve().parents[2] / "data" / "KILL"  # <repo>/data/KILL
STREAK_SIZE_MULT = 0.5  # size multiplier once the loss streak reaches streak_half
# Halts that stay until cleared by hand (``clear_halt``): a run that hits one has ended trading.
PERMANENT_HALTS = ("manual_kill", "max_drawdown")
HALT_SEVERITY = (*PERMANENT_HALTS, "weekly_loss", "daily_loss")  # most severe first


def _utc(ts: datetime, name: str) -> datetime:
    """``ts`` in UTC. A naive datetime is a bug upstream (its UTC day is unknown): ValueError."""
    if ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC, got naive {ts!r}")
    return ts.astimezone(UTC)


def _week_start(day: date) -> date:
    """Monday of the ISO week containing ``day``; it identifies the week."""
    return day - timedelta(days=day.weekday())


def _iso(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def _from_iso(text: str | None) -> date | None:
    return None if text is None else date.fromisoformat(text)


def _reject(rule: str, detail: str) -> RiskDecision:
    return RiskDecision(approved=False, qty=0.0, rule=rule, detail=detail)


def _stop_problem(signal: Signal) -> str | None:
    """Why the signal cannot be sized, or None: prices must be finite and positive and the
    stop must be on the losing side of entry (below it for LONG, above it for SHORT)."""
    prices = (signal.entry, signal.stop, signal.target)
    if not all(math.isfinite(x) and x > 0 for x in prices):
        return f"entry/stop/target must be finite and positive, got {prices}"
    if side_sign(signal.side) * (signal.entry - signal.stop) <= 0:
        return f"{signal.side} stop {signal.stop} is not on the losing side of entry {signal.entry}"
    return None


class RiskManager:
    """Approves and sizes signals, enforces loss limits and owns the kill switch.

    State (saved by ``to_state``): ``loss_streak`` (consecutive losing trades), ``day_stopped``
    (UTC date on which the streak stop was hit: no entries that day), the active halts (see
    ``halted_reason``) and the equity at the start of the current day/week.
    """

    def __init__(self, cfg: AppConfig, kill_file: Path) -> None:
        self.risk = cfg.risk
        self.kill_file = kill_file
        self.loss_streak = 0
        self.day_stopped: date | None = None
        self._halts: set[str] = set()
        self._day: date | None = None  # current UTC day and ISO-week Monday
        self._week: date | None = None
        self._day_start_equity: float | None = None  # set by the period's first check_limits
        self._week_start_equity: float | None = None

    @property
    def halted_reason(self) -> str | None:
        """Most severe active halt (manual_kill > max_drawdown > weekly_loss > daily_loss)."""
        return next((kind for kind in HALT_SEVERITY if kind in self._halts), None)

    @property
    def must_flatten(self) -> bool:
        """True while halted: the engine must close every position and open none."""
        return self.halted_reason is not None

    def evaluate(
        self,
        signal: Signal,
        portfolio: Portfolio,
        playbook: PlaybookEntry,
        now: datetime,
        pending: Mapping[str, float] | None = None,
    ) -> RiskDecision:
        """Approve and size ``signal``, or reject it with the first failing rule.

        Rules in order: kill_switch, no_equity, day_stopped, regime_blocks_entries,
        strategy_not_allowed, position_exists, invalid_stop; then sizing, where a cap with no
        room left rejects with its rule (cap_asset, cap_group, cap_gross) and a notional under
        ``risk.min_notional`` rejects as too_small. Day/week rollovers for ``now`` apply first.

        ``pending`` maps symbol -> notional (quote units) of entries approved but not yet
        filled. It counts as exposure for every cap, and a pending symbol is position_exists,
        so several approvals in one decision bar can never jointly breach a cap. Raises
        ValueError on a pending notional that is not finite and >= 0.
        """
        now = _utc(now, "now")
        pending = pending or {}
        for symbol, notional in pending.items():
            if not (math.isfinite(notional) and notional >= 0):
                raise ValueError(f"pending notional for {symbol} must be >= 0, got {notional!r}")
        self._roll(now)
        equity = portfolio.equity
        if self.halted_reason is not None:
            return _reject("kill_switch", f"halted: {self.halted_reason}")
        if self.kill_file.exists():
            return _reject("kill_switch", f"kill file {self.kill_file} exists")
        if not (math.isfinite(equity) and equity > 0):
            return _reject("no_equity", f"equity {equity!r} is not positive")
        if self.day_stopped is not None:
            return _reject("day_stopped", f"loss-streak stop hit on {self.day_stopped}")
        if not playbook.allow_entries:
            return _reject("regime_blocks_entries", f"no entries allowed in regime {signal.regime}")
        if signal.strategy not in playbook.strategies:
            detail = f"{signal.strategy} not in {playbook.strategies}"
            return _reject("strategy_not_allowed", detail)
        if signal.symbol in portfolio.positions or signal.symbol in pending:
            return _reject("position_exists", f"{signal.symbol} is already open or pending")
        if problem := _stop_problem(signal):
            return _reject("invalid_stop", problem)
        return self._size(signal, portfolio, pending, equity, playbook.size_mult)

    def _size(
        self,
        signal: Signal,
        portfolio: Portfolio,
        pending: Mapping[str, float],
        equity: float,
        size_mult: float,
    ) -> RiskDecision:
        """Risk-based qty, reduced to the smallest cap room; ties go to the earlier cap."""
        streak_mult = STREAK_SIZE_MULT if self.loss_streak >= self.risk.streak_half else 1.0
        risk_amount = equity * self.risk.risk_per_trade * size_mult * streak_mult
        qty = risk_amount / abs(signal.entry - signal.stop)
        notional, rule, detail = qty * signal.entry, "ok", f"risk {risk_amount:.2f} at the stop"
        for cap_rule, label, room in self._rooms(signal.symbol, portfolio, pending, equity):
            if room < notional:
                notional, rule = room, cap_rule
                detail = f"{label} room {room:.2f} caps risk {risk_amount:.2f}"
        if notional <= 0 and rule != "ok":
            return _reject(rule, f"no {detail}")
        if notional < self.risk.min_notional:
            return _reject("too_small", f"notional {notional:.2f} < {self.risk.min_notional}")
        if rule != "ok":
            qty = notional / signal.entry
        return RiskDecision(approved=True, qty=qty, rule=rule, detail=detail)

    def _rooms(
        self, symbol: str, portfolio: Portfolio, pending: Mapping[str, float], equity: float
    ) -> list[tuple[str, str, float]]:
        """(rule, label, remaining notional) for each cap on ``symbol``: asset, its groups, gross.

        Used exposure is open positions at their marks plus pending notional. The asset room
        needs neither: ``position_exists`` rejected open and pending symbols.
        """
        r = self.risk
        rooms = [("cap_asset", "asset", r.max_asset_exposure * equity)]
        for group, members in r.groups.items():
            if symbol in members:
                used = sum(portfolio.exposure(s) + pending.get(s, 0.0) for s in set(members))
                rooms.append(("cap_group", f"group {group}", r.max_group_exposure * equity - used))
        gross_used = portfolio.gross_exposure + sum(pending.values())
        rooms.append(("cap_gross", "gross", r.max_gross_leverage * equity - gross_used))
        return rooms

    def on_trade_closed(self, trade: Trade) -> None:
        """Update the loss streak with the trade's net pnl.

        ``net_pnl < 0`` extends the streak, anything else resets it. Reaching ``streak_stop``
        stops entries for the rest of the exit's UTC day. A trade exiting after the stopped day
        first applies that day's reset, so it starts a new streak.
        """
        if not math.isfinite(trade.net_pnl):
            raise ValueError(f"trade net_pnl must be finite, got {trade.net_pnl!r}")
        day = _utc(trade.exit_ts, "trade.exit_ts").date()
        self._clear_day_stop(day)
        self.loss_streak = self.loss_streak + 1 if trade.net_pnl < 0 else 0
        if self.loss_streak >= self.risk.streak_stop and self.day_stopped != day:
            self.day_stopped = day
            log.warning("risk_day_stopped", day=day.isoformat(), loss_streak=self.loss_streak)

    def check_limits(self, portfolio: Portfolio, now: datetime) -> list[RiskEvent]:
        """Halt on a breached limit; return one "halt" event per new incident, most severe first.

        Breaches: equity <= day/week start equity * (1 - daily_loss/weekly_loss), drawdown >=
        max_drawdown, kill file present. The first call of each UTC day / ISO week records that
        period's start equity. A breach of an already active halt returns nothing, so an event
        recurs only once its halt has cleared and the condition holds again (``clear_halt``
        while drawdown is still at the limit halts again on the next call). Raises ValueError,
        before changing any state, on non-finite equity or drawdown: NaN compares False and
        would silently disable every limit.
        """
        now = _utc(now, "now")
        equity, drawdown = portfolio.equity, portfolio.drawdown
        if not (math.isfinite(equity) and math.isfinite(drawdown)):
            raise ValueError(f"equity {equity!r} and drawdown {drawdown!r} must be finite")
        self._roll(now)
        r = self.risk
        if self._day_start_equity is None:
            self._day_start_equity = equity
        if self._week_start_equity is None:
            self._week_start_equity = equity
        day_floor = self._day_start_equity * (1 - r.daily_loss)
        week_floor = self._week_start_equity * (1 - r.weekly_loss)
        checks = (  # in HALT_SEVERITY order
            ("manual_kill", self.kill_file.exists(), f"kill file {self.kill_file} exists"),
            ("max_drawdown", drawdown >= r.max_drawdown, f"drawdown {drawdown:.2%}"),
            ("weekly_loss", equity <= week_floor, f"equity {equity:.2f} <= {week_floor:.2f}"),
            ("daily_loss", equity <= day_floor, f"equity {equity:.2f} <= {day_floor:.2f}"),
        )
        events = []
        for kind, breached, detail in checks:
            if breached and kind not in self._halts:
                self._halts.add(kind)
                events.append(RiskEvent(ts=now, kind=kind, detail=detail, action="halt"))
                log.warning("risk_halt", kind=kind, detail=detail, ts=now.isoformat())
        return events

    def clear_halt(self) -> None:
        """Clear every active halt by hand (the only way out of max_drawdown and manual_kill).

        Raises RuntimeError while the kill file exists: run ``regimebot unkill`` first.
        """
        if self.kill_file.exists():
            raise RuntimeError(f"kill file {self.kill_file} exists: run `regimebot unkill` first")
        log.info("risk_halt_cleared", halts=sorted(self._halts))
        self._halts.clear()

    def _roll(self, now: datetime) -> None:
        """Enter ``now``'s UTC day / ISO week: clear that period's halt, forget its start equity
        and lift a streak stop from an earlier day. Time moving backwards changes nothing."""
        today = now.date()
        self._clear_day_stop(today)
        if self._day is None or today > self._day:
            self._day, self._day_start_equity = today, None
            self._halts.discard("daily_loss")
        week = _week_start(today)
        if self._week is None or week > self._week:
            self._week, self._week_start_equity = week, None
            self._halts.discard("weekly_loss")

    def _clear_day_stop(self, today: date) -> None:
        if self.day_stopped is not None and today > self.day_stopped:
            self.day_stopped, self.loss_streak = None, 0

    def to_state(self) -> dict[str, Any]:
        """JSON-safe snapshot (dates as ISO strings) for paper-session restarts."""
        return {
            "loss_streak": self.loss_streak,
            "day_stopped": _iso(self.day_stopped),
            "halts": sorted(self._halts),
            "day": _iso(self._day),
            "week": _iso(self._week),
            "day_start_equity": self._day_start_equity,
            "week_start_equity": self._week_start_equity,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore a ``to_state`` snapshot. Raises ValueError on an unknown halt kind (dropping
        it would silently resume trading); nothing changes when it raises. Keys it does not read
        are ignored, so states saved with the old ``pnl_by_day`` / ``pnl_by_week`` still load."""
        unknown = set(state["halts"]) - set(HALT_SEVERITY)
        if unknown:
            raise ValueError(f"unknown halt kinds in state: {sorted(unknown)}")
        day_start, week_start = state["day_start_equity"], state["week_start_equity"]
        parsed = (  # parse everything before assigning anything
            int(state["loss_streak"]),
            _from_iso(state["day_stopped"]),
            set(state["halts"]),
            _from_iso(state["day"]),
            _from_iso(state["week"]),
            None if day_start is None else float(day_start),
            None if week_start is None else float(week_start),
        )
        (
            self.loss_streak,
            self.day_stopped,
            self._halts,
            self._day,
            self._week,
            self._day_start_equity,
            self._week_start_equity,
        ) = parsed
