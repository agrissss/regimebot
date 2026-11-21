"""Paper-session persistence: append new log rows, save and restore the resumable state, status.

A paper run keeps its resumable state in ``open_positions``: one row per open position (its
``Position.to_state``), the risk manager's ``to_state`` under ``"__risk__"``, and under
``"__portfolio__"`` the portfolio's cash, peak equity and marks, the engine's ``to_state`` (queued
exits, last decision closes) and the session's cursors (``Cursors``). The ``"__portfolio__"`` row
is rewritten on every poll, so its ``updated_at`` is the session's heartbeat. A running session
holds an exclusive lock on ``paper_run_<id>.lock`` next to the kill file (``lock_run``).
"""

import fcntl
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from regimebot.config import AppConfig, config_hash
from regimebot.db import repo
from regimebot.db.models import EquitySnapshot, OpenPosition, RegimeRow, RiskEventRow, Run
from regimebot.engine.loop import Engine, RunLog
from regimebot.engine.portfolio import Position
from regimebot.risk.manager import HALT_SEVERITY, RiskManager
from regimebot.types import Mode, RiskEvent

PORTFOLIO_KEY = "__portfolio__"
RISK_KEY = "__risk__"
STATUS_EVENTS = 10  # risk events shown by ``paper status``


@dataclass(frozen=True)
class FlushMarks:
    """How much of each ``RunLog`` list is stored; ``signal_ids[i]`` is signal i's row id."""

    signal_ids: list[int] = field(default_factory=list)
    decisions: int = 0
    fills: int = 0
    trades: int = 0
    regimes: int = 0
    risk_events: int = 0
    equity: int = 0


@dataclass(frozen=True)
class Cursors:
    """Where a session stands: the last decision processed and, per symbol, the newest base bar
    and funding event the engine has seen (anything later is replayed after a restart)."""

    last_decision: datetime
    last_bar: dict[str, datetime]
    last_funding: dict[str, datetime]


def append_log(session: Session, run_id: int, log: RunLog, marks: FlushMarks) -> FlushMarks:
    """Insert the rows ``log`` gained since ``marks``; return the new marks.

    The caller adopts the new marks only once the transaction has committed, so a failed flush is
    retried in full by the next one. Equity rows are append-only here: the engine replaces its last
    snapshot only in ``flatten_all``, which a paper session never calls.
    """
    new_signals = log.signals[len(marks.signal_ids) :]
    ids = marks.signal_ids + repo.save_signals(session, run_id, new_signals)
    decisions = [(ids[i], ts, d) for i, ts, d in log.decisions[marks.decisions :]]
    repo.save_risk_decisions(session, run_id, decisions)
    repo.save_fills(session, run_id, log.fills[marks.fills :])
    repo.save_trades(session, run_id, log.trades[marks.trades :])
    repo.save_regimes(session, run_id, log.regimes[marks.regimes :])
    repo.save_risk_events(session, run_id, log.risk_events[marks.risk_events :])
    repo.save_equity(session, run_id, log.equity[marks.equity :])
    return FlushMarks(
        ids,
        len(log.decisions),
        len(log.fills),
        len(log.trades),
        len(log.regimes),
        len(log.risk_events),
        len(log.equity),
    )


# --- run lock -------------------------------------------------------------------------------


def lock_run(lock_dir: Path, run_id: int) -> int:
    """Take run ``run_id``'s exclusive lock and return its file descriptor; the lock holds until
    the descriptor is closed or the process exits. Raises ValueError when it is already held
    (another session runs this run), so two processes never book the same run."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"paper_run_{run_id}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ValueError(f"paper run {run_id} is already running (lock {path} is held)") from None
    return fd


# --- save and restore -----------------------------------------------------------------------


def save_state(session: Session, run_id: int, engine: Engine, cursors: Cursors) -> None:
    """Write the resumable state rows (a closed position's row is deleted)."""
    portfolio = engine.portfolio
    for symbol in engine.cfg.exchange.symbols:
        position = portfolio.positions.get(symbol)
        state = None if position is None else position.to_state()
        repo.save_open_position(session, run_id, symbol, state)
    repo.save_open_position(session, run_id, RISK_KEY, engine.risk.to_state())
    saved = {
        "cash": portfolio.cash,
        "peak_equity": portfolio.peak_equity,
        "last_prices": dict(portfolio.last_prices),
        **engine.to_state(),  # queued_exits, last_close
        "last_decision": cursors.last_decision.isoformat(),
        "last_bar": {s: ts.isoformat() for s, ts in cursors.last_bar.items()},
        "last_funding": {s: ts.isoformat() for s, ts in cursors.last_funding.items()},
    }
    repo.save_open_position(session, run_id, PORTFOLIO_KEY, saved)


def check_resumable(session: Session, run_id: int, cfg: AppConfig) -> None:
    """Raise ValueError unless ``run_id`` is a paper run started with this config."""
    run = repo.load_run(session, run_id)  # ValueError for an unknown id
    if run.mode != Mode.PAPER:
        raise ValueError(f"run {run_id} is a {run.mode} run, not a paper run")
    if run.config_hash != config_hash(cfg):
        raise ValueError(
            f"run {run_id} ran config {run.config_hash}, the current config is "
            f"{config_hash(cfg)}: resume it with the config it started with"
        )


def restore(session: Session, run_id: int, engine: Engine) -> Cursors:
    """Load ``run_id``'s saved state into a fresh ``engine`` (its portfolio, risk manager and
    queued exits) and return the session cursors.

    Raises ValueError when the run cannot be resumed or its state is missing or malformed; the
    portfolio is only assigned once everything else has loaded.
    """
    check_resumable(session, run_id, engine.cfg)
    rows = repo.load_open_positions(session, run_id)
    if PORTFOLIO_KEY not in rows or RISK_KEY not in rows:
        raise ValueError(f"run {run_id} has no saved paper state")
    saved, risk_state = rows.pop(PORTFOLIO_KEY), rows.pop(RISK_KEY)
    positions = {symbol: Position.from_state(state) for symbol, state in rows.items()}
    last_prices = {s: float(v) for s, v in saved["last_prices"].items()}
    if missing := sorted(set(positions) - set(last_prices)):
        raise ValueError(f"run {run_id}: no mark for open positions {missing}")
    cursors = Cursors(
        _utc(saved["last_decision"]),
        {s: _utc(v) for s, v in saved["last_bar"].items()},
        {s: _utc(v) for s, v in saved["last_funding"].items()},
    )
    cash, peak = float(saved["cash"]), float(saved["peak_equity"])
    engine.load_state({key: saved[key] for key in ("queued_exits", "last_close")})
    engine.risk.load_state(risk_state)  # both validate before assigning anything
    portfolio = engine.portfolio
    portfolio.cash, portfolio.peak_equity = cash, peak
    portfolio.last_prices, portfolio.positions = last_prices, positions
    return cursors


def clear_halts(session: Session, run_id: int, cfg: AppConfig, kill_file: Path) -> list[str]:
    """Clear paper run ``run_id``'s recorded halts in its stored risk state; returns the halts
    that were active. The caller holds the run's lock (``lock_run``) until this commits.

    Raises ValueError for a run that is not a paper run or has no saved state, RuntimeError
    while the kill file exists (``RiskManager.clear_halt``).
    """
    if (mode := repo.load_run(session, run_id).mode) != Mode.PAPER:
        raise ValueError(f"run {run_id} is a {mode} run, not a paper run")
    saved = repo.load_open_positions(session, run_id).get(RISK_KEY)
    if saved is None:
        raise ValueError(f"run {run_id} has no saved paper state")
    risk = RiskManager(cfg, kill_file)
    risk.load_state(saved)
    risk.clear_halt()
    repo.save_open_position(session, run_id, RISK_KEY, risk.to_state())
    return list(saved["halts"])


def _utc(text: str) -> datetime:
    ts = datetime.fromisoformat(text)
    if ts.utcoffset() is None:
        raise ValueError(f"saved timestamp {text!r} has no timezone")
    return ts.astimezone(UTC)


# --- status ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperStatus:
    """What ``regimebot paper status`` shows for one paper run."""

    run_id: int
    started_at: datetime
    is_open: bool  # never ended: ``paper run --resume`` continues it
    last_poll: datetime | None  # the session rewrites its state row on every poll
    equity: tuple[datetime, float] | None  # last snapshot (one per decision)
    cash: float | None
    positions: list[Position]
    last_regimes: dict[str, tuple[datetime, str]]
    halted_reason: str | None
    risk_events: list[RiskEvent]  # the last STATUS_EVENTS, oldest first


def latest_status(session: Session) -> PaperStatus | None:
    """Status of the most recent paper run, or None when there is none."""
    run = session.scalars(
        select(Run).where(Run.mode == str(Mode.PAPER)).order_by(Run.id.desc()).limit(1)
    ).first()
    if run is None:
        return None
    rows: dict[str, Any] = repo.load_open_positions(session, run.id)
    saved, risk_state = rows.pop(PORTFOLIO_KEY, None), rows.pop(RISK_KEY, None)
    halts = set() if risk_state is None else set(risk_state["halts"])
    last_poll = session.scalar(
        select(OpenPosition.updated_at).where(
            OpenPosition.run_id == run.id, OpenPosition.symbol == PORTFOLIO_KEY
        )
    )
    equity = session.execute(
        select(EquitySnapshot.ts, EquitySnapshot.equity)
        .where(EquitySnapshot.run_id == run.id)
        .order_by(EquitySnapshot.ts.desc())
        .limit(1)
    ).first()
    newest = (
        select(RegimeRow.symbol, func.max(RegimeRow.ts).label("ts"))
        .where(RegimeRow.run_id == run.id)
        .group_by(RegimeRow.symbol)
        .subquery()
    )
    regimes = session.execute(
        select(RegimeRow.symbol, RegimeRow.ts, RegimeRow.regime)
        .join(newest, (RegimeRow.symbol == newest.c.symbol) & (RegimeRow.ts == newest.c.ts))
        .where(RegimeRow.run_id == run.id)
        .order_by(RegimeRow.symbol)
    ).all()
    events = session.scalars(
        select(RiskEventRow)
        .where(RiskEventRow.run_id == run.id)
        .order_by(RiskEventRow.ts.desc(), RiskEventRow.id.desc())
        .limit(STATUS_EVENTS)
    ).all()
    return PaperStatus(
        run_id=run.id,
        started_at=run.started_at,
        is_open=run.ended_at is None,
        last_poll=last_poll,
        equity=None if equity is None else (equity[0], equity[1]),
        cash=None if saved is None else float(saved["cash"]),
        positions=[Position.from_state(state) for state in rows.values()],
        last_regimes={symbol: (ts, regime) for symbol, ts, regime in regimes},
        halted_reason=next((kind for kind in HALT_SEVERITY if kind in halts), None),
        risk_events=[RiskEvent(e.ts, e.kind, e.detail, e.action) for e in reversed(events)],
    )
