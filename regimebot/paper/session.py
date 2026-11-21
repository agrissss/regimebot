"""Paper trading: the backtest's ``Engine`` fed by live, polled market data, with simulated fills.

Each poll (``run_once``) downloads every symbol's newly closed base bars and funding events into
the database and replays them through the engine in time order (``on_minute``, ``on_funding``; a
bar closing at t before funding stamped t), as the backtest replays an interval. When a decision
bar has closed for every symbol, features are rebuilt from the stored trailing history (the
backtest's resample and ``compute_features``) and ``on_decision`` runs with live spreads between
the replayed events up to its close and those after it. Queued orders then fill at a live quote
(``fill_queued``) instead of at the next bar's open. All new rows and the resumable state are
flushed at the end of every poll in one transaction.

There is no order path: the session only calls the ``Exchange`` protocol's market-data methods
(``fetch_candles``, ``fetch_funding``, ``fetch_quote``); every fill is simulated.
"""

import math
import os
import signal
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType

import pandas as pd
import structlog
from sqlalchemy import Engine as Database
from sqlalchemy.orm import Session

from regimebot.backtest.runner import WINDOW_ROWS, decision_features
from regimebot.config import AppConfig, timeframe_minutes
from regimebot.data.exchange import Exchange
from regimebot.data.history import download_candles, download_funding
from regimebot.db import repo
from regimebot.db.session import session_scope
from regimebot.engine.loop import Engine, RunLog
from regimebot.engine.portfolio import Portfolio
from regimebot.paper import state
from regimebot.reporting.alerts import alert
from regimebot.risk.manager import RiskManager
from regimebot.strategy import Strategy, build_strategies
from regimebot.types import Bar, Mode, Quote

log = structlog.get_logger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_TICK = timedelta(microseconds=1)  # stored timestamps have microsecond resolution
_BAR, _FUNDING = 0, 1  # event order at equal timestamps, as in the backtest
Event = tuple[datetime, int, int, Bar | float]  # (ts, kind, symbol order, bar or funding rate)


def _utc_now() -> datetime:
    """The default clock; the session reads time only through its injected ``clock``."""
    return datetime.now(UTC)


def _floor(ts: datetime, width: timedelta) -> datetime:
    """The latest epoch-aligned multiple of ``width`` at or before ``ts``."""
    return ts - (ts - _EPOCH) % width


class PaperSession:
    """A resumable paper-trading run over ``cfg.exchange.symbols``.

    Decisions: the decision bar closing at T is processed once, at the first poll where every
    symbol has its base bar T stored or is stale. A symbol is stale when its last candle download
    failed, it has no bars, or its newest bar is older than ``stale_after_seconds``; a symbol
    missing bar T therefore turns stale before T + ``stale_after_seconds``. A decision still
    unprocessed at T + ``stale_after_seconds`` (by a fresh clock read, taken again after the
    spread quotes, just before ``on_decision``) is missed: skipped for good, never processed late,
    and alerted as ``decision_missed`` unless it closed before this process started (a routine
    start or resume, logged at info). Only symbols with bar T stored
    get a window; the others are recorded as ``stale_data`` by the engine and not traded.
    ``alert("stale_data")`` fires once per incident, ``stale_data_recovered`` when it ends.

    Fills: spreads come from quotes fetched just before ``on_decision``; after it, each symbol
    with a queued order fills at a fresh quote, stamped with the session clock, with the last
    closed bar's volume for participation. A fill can lag the decision by those quote fetches
    (each up to about 91 s while the exchange retries); the engine's entry expiry at T + one
    decision width bounds the lag of an entry. A failed quote means no spread and no fill for that
    symbol at this decision: a queued entry then expires at the next decision (the engine's
    rule), a queued exit is retried on every poll until it fills. The kill file is checked on
    every poll: it halts (open or flat) and flattens at once, not only at the next decision.
    Every halt is alerted, and so is a halt that clears.
    """

    def __init__(
        self,
        cfg: AppConfig,
        exchange: Exchange,
        engine_db: Database,
        kill_file: Path,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
        strategies: Sequence[Strategy] | None = None,
    ) -> None:
        self.cfg = cfg
        self.exchange = exchange
        self._db = engine_db
        self._clock = clock
        self._sleep = sleep
        self._symbols = list(cfg.exchange.symbols)
        self._tf = cfg.exchange.base_timeframe
        self._base = timedelta(minutes=timeframe_minutes(self._tf))
        self._width = timedelta(minutes=timeframe_minutes(cfg.exchange.decision_timeframe))
        self._stale_after = timedelta(seconds=cfg.paper.stale_after_seconds)
        self._history = cfg.paper.warmup_bars * self._width
        self.log = RunLog()
        self.portfolio = Portfolio(cfg.backtest.initial_equity)
        self.risk = RiskManager(cfg, kill_file)
        strategies = build_strategies(cfg) if strategies is None else strategies
        self.engine = Engine(cfg, strategies, self.risk, self.portfolio, self.log)
        self.run_id: int | None = None
        self._lock_fd: int | None = None
        self._marks = state.FlushMarks()
        self._newest: dict[str, Bar] = {}  # newest bar fed to the engine, per symbol
        self._last_funding: dict[str, datetime] = {}  # newest funding event fed, per symbol
        self._last_decision = _EPOCH
        self._incidents: set[str] = set()  # open alert incidents, "kind:subject"
        self._halts: set[str] = set()  # active halts last seen (alerted on change)
        self._events_alerted = 0  # risk events already scanned for halts
        self._stop_signal: int | None = None
        self._started_at = _EPOCH  # this process's start: earlier missed decisions are routine
        self._stopped_at: datetime | None = None  # set by a clean stop, cleared by a resume

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self, resume_run_id: int | None = None) -> int:
        """Lock the run, warm up, then create a paper run or resume ``resume_run_id``; returns
        the run id. Raises ValueError when the run cannot be resumed or is already running.

        Warm-up stores the last ``warmup_bars`` decision bars' worth of base candles (and the
        funding) per symbol with ``download_candles``; a symbol whose download fails stays stale
        until polling brings its data.

        Resume: the run must be a paper run started with this config. Positions, cash, peak
        equity, marks, risk state and queued exits come back from its saved rows, and new rows
        are appended to the same run. Base bars that closed while the process was down are
        replayed in order through ``on_minute`` at the first poll, so a stop or target hit during
        the downtime is honoured at the bar where it happened. Decisions missed during the
        downtime are NOT replayed: the first poll skips a decision whose wait has ended (logged
        at info, not alerted: it closed before this process started), and the session continues
        at the next live decision. A ``stopped_at`` marker of the previous clean stop is cleared
        as soon as the run is locked, before the warm-up. Queued entries are not
        saved: ``run_forever`` cancels them on a clean stop, while after a hard kill (SIGKILL)
        an approved entry is left without a closing decision row.
        """
        try:
            run_id = self._start(resume_run_id)
        except BaseException:
            self.close()  # a failed start keeps no lock
            raise
        log.info(
            "paper_started",
            run_id=run_id,
            resumed=resume_run_id is not None,
            positions=sorted(self.portfolio.positions),
            last_decision=self._last_decision.isoformat(),
        )
        return run_id

    def _start(self, resume_run_id: int | None) -> int:
        now = self._started_at = self._clock()
        if resume_run_id is not None:
            with Session(self._db) as session:
                state.check_resumable(session, resume_run_id, self.cfg)  # fail before warm-up
            self._lock(resume_run_id)
            with session_scope(self._db) as session:  # a resume dying in warm-up is not "stopped"
                state.clear_stopped(session, resume_run_id)
        for symbol in self._symbols:
            self._download(symbol, now - self._history, now)
        with session_scope(self._db) as session:
            if resume_run_id is None:
                self.run_id = repo.create_run(session, Mode.PAPER, self.cfg)
                self._lock(self.run_id)
                for symbol in self._symbols:
                    self._adopt(session, symbol, now)
            else:
                self.run_id = resume_run_id
                cursors = state.restore(session, resume_run_id, self.engine)
                for symbol, ts in cursors.last_bar.items():
                    self._newest[symbol] = self._stored_bar(session, symbol, ts)
                self._last_funding = dict(cursors.last_funding)
                self._last_decision = cursors.last_decision
            self._halts = set(self.risk.to_state()["halts"])
            self._marks = self._save(session)
        return self.run_id

    def close(self) -> None:
        """Release the run's lock (process exit does the same)."""
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _lock(self, run_id: int) -> None:
        self._lock_fd = state.lock_run(self.risk.kill_file.parent, run_id)

    def run_forever(self) -> None:
        """Poll every ``poll_seconds`` until SIGINT or SIGTERM (or a KeyboardInterrupt in sleep).

        A signal only records itself, so the current poll always finishes and is flushed. The
        session then cancels queued entries (each gets a closing decision), stores ``stopped_at``
        (so ``paper status`` tells a clean stop from a hung session), flushes and returns without
        flattening; the run stays open for a resume. An exception in a poll is logged and alerted
        once per incident, and the next poll runs as usual.
        """
        self._stop_signal = None
        self._stopped_at = None
        previous = {
            sig: signal.signal(sig, self._request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            while self._stop_signal is None:
                self._guarded_poll()
                if self._stop_signal is not None:
                    break
                try:
                    self._sleep(self.cfg.paper.poll_seconds)
                except KeyboardInterrupt:
                    break
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        if self._stop_signal is not None:
            log.info("paper_stop_requested", signal=signal.Signals(self._stop_signal).name)
        self._stopped_at = self._clock()
        self.engine.cancel_queued_entries(self._stopped_at)
        try:
            self.flush()
        except Exception:
            log.exception("paper_flush_failed", run_id=self.run_id)
        log.info("paper_stopped", run_id=self.run_id, positions=sorted(self.portfolio.positions))

    def _request_stop(self, signum: int, frame: FrameType | None) -> None:
        # Only record it: logging here could deadlock on the stdout lock the poll may hold.
        self._stop_signal = signum

    def _guarded_poll(self) -> None:
        try:
            self.run_once()
        except Exception as exc:  # the loop must survive anything; the incident is alerted
            log.exception("paper_poll_failed", run_id=self.run_id)
            self._track("poll_error", "session", repr(exc))
        else:
            self._track("poll_error", "session", None)

    # --- one poll ----------------------------------------------------------------------------

    def run_once(self) -> None:
        """One poll: download; replay new bars and funding up to the open decision bar, decide
        it once ready, then replay the rest; act on the kill file; fill queued orders; alert
        halts; flush. While a decision waits for a late symbol, later events are held back so
        that every bar after T is checked against the stops as the decision at T left them.
        Residual: during that wait a retried exit (or a KILL exit) fills at the live quote before
        its symbol's held-back bars are replayed, so a stop in them can be missed; the wait bounds
        this to under ``stale_after_seconds``. Exchange errors are contained per symbol and call."""
        now = self._clock()
        downloaded = {}
        for symbol in self._symbols:
            newest = self._newest.get(symbol)
            since = now - self._history if newest is None else newest.ts + self._base
            downloaded[symbol] = self._download(symbol, since, now)
        events, latest = self._new_events(now)
        stale = self._stale_symbols(now, downloaded, latest)
        quotes: dict[str, Quote] | None = None  # pre-decision quotes, when deciding
        ts, ready = self._open_decision(now, stale, latest)
        if ts is None:
            self._feed(events)
        else:  # events after an open decision wait for it: they stay stored and come back
            self._feed(e for e in events if e[0] <= ts)
            if ready:
                quotes = self._decide(ts, latest)  # None when it turned out late: missed
                self._feed(e for e in events if e[0] > ts)
        self._check_kill_file(now)
        self._fill_queued(quotes)
        self._alert_halts()
        self.flush()

    def flush(self) -> None:
        """Append the new log rows and rewrite the resumable state, in one transaction."""
        with session_scope(self._db) as session:
            marks = self._save(session)
        self._marks = marks  # only once committed: a failed flush is retried in full

    def _save(self, session: Session) -> state.FlushMarks:
        if self.run_id is None:
            raise RuntimeError("start() the session first")
        marks = state.append_log(session, self.run_id, self.log, self._marks)
        cursors = state.Cursors(
            self._last_decision,
            {s: bar.ts for s, bar in self._newest.items()},
            dict(self._last_funding),
        )
        state.save_state(session, self.run_id, self.engine, cursors, self._stopped_at)
        return marks

    def _download(self, symbol: str, since: datetime, now: datetime) -> bool:
        """Store ``symbol``'s closed base bars in ``[since, now]`` and its new funding events;
        False when the candle download failed."""
        ok = True
        try:
            with session_scope(self._db) as session:
                download_candles(self.exchange, session, symbol, self._tf, since, now)
        except Exception as exc:  # any exchange or data error: the symbol is stale this poll
            log.warning("paper_download_failed", symbol=symbol, data="candles", error=repr(exc))
            ok = False
        try:
            with session_scope(self._db) as session:
                download_funding(self.exchange, session, symbol, now - self._history, now)
        except Exception as exc:  # retried next poll
            log.warning("paper_download_failed", symbol=symbol, data="funding", error=repr(exc))
        return ok

    def _new_events(self, now: datetime) -> tuple[list[Event], dict[str, datetime]]:
        """Stored bars and funding events newer than the engine has seen, all symbols in time
        order, and each symbol's newest stored bar ts. Funding waits until the symbol's bars reach
        its stamp; its mark is the close of the last bar at or before it (a record arriving over
        a bar late gets a later close)."""
        events: list[Event] = []
        latest: dict[str, datetime] = {}
        with Session(self._db) as session:
            for order, symbol in enumerate(self._symbols):
                newest = self._newest.get(symbol)
                if newest is None:  # first data for this symbol: no position can need it
                    self._adopt(session, symbol, now)
                    if (adopted := self._newest.get(symbol)) is not None:
                        latest[symbol] = adopted.ts
                    continue
                bars = repo.load_candles(session, symbol, self._tf, newest.ts + self._base)
                stamps = list(pd.DatetimeIndex(bars.index).to_pydatetime())
                for ts, row in zip(stamps, bars.to_numpy(dtype=float).tolist(), strict=True):
                    events.append((ts, _BAR, order, Bar(symbol, ts, *row)))
                latest[symbol] = stamps[-1] if stamps else newest.ts
                since = self._last_funding.get(symbol)
                after = None if since is None else since + _TICK
                rates = repo.load_funding(session, symbol, after, latest[symbol] + _TICK)
                for ts, rate in zip(
                    pd.DatetimeIndex(rates.index).to_pydatetime(), rates.tolist(), strict=True
                ):
                    events.append((ts, _FUNDING, order, rate))
        events.sort(key=lambda event: event[:3])
        return events, latest

    def _feed(self, events: Iterable[Event]) -> None:
        for ts, _, order, payload in events:
            symbol = self._symbols[order]
            if isinstance(payload, Bar):
                self._feed_bar(payload)
            else:
                self._last_funding[symbol] = ts
                self.engine.on_funding(symbol, ts, payload, self._newest[symbol].close)

    def _feed_bar(self, bar: Bar) -> None:
        """Check ``bar`` against the symbol's open position (``on_minute``)."""
        self._newest[bar.symbol] = bar  # first: a bar the engine rejects is not fed again
        position = self.portfolio.positions.get(bar.symbol)
        if position is not None and bar.ts <= position.entry_ts:
            return  # it closed before the live fill that opened the position
        self.engine.on_minute(bar)

    def _adopt(self, session: Session, symbol: str, now: datetime) -> None:
        """Follow ``symbol`` from its newest stored bar and funding event, replaying nothing."""
        stamps = repo.candle_timestamps(session, symbol, self._tf, now - self._history, now)
        if len(stamps):
            self._newest[symbol] = self._stored_bar(session, symbol, stamps[-1].to_pydatetime())
        if bounds := repo.funding_ts_bounds(session, symbol):
            self._last_funding[symbol] = bounds[1]

    def _stored_bar(self, session: Session, symbol: str, ts: datetime) -> Bar:
        frame = repo.load_candles(session, symbol, self._tf, ts, ts + self._base)
        if frame.empty:
            raise ValueError(f"no stored {self._tf} bar for {symbol} at {ts.isoformat()}")
        open_, high, low, close, volume = frame.to_numpy(dtype=float)[0].tolist()
        return Bar(symbol, ts, open_, high, low, close, volume)

    def _stale_symbols(
        self, now: datetime, downloaded: dict[str, bool], latest: dict[str, datetime]
    ) -> set[str]:
        """Symbols whose candle download failed, with no bars, or whose newest bar is too old
        (alerted once per incident)."""
        stale = set()
        for symbol in self._symbols:
            problem = None
            if not downloaded[symbol]:
                problem = "candle download failed"
            elif symbol not in latest:
                problem = "no bars stored"
            elif now - latest[symbol] > self._stale_after:
                problem = f"newest bar closed at {latest[symbol].isoformat()}"
            if problem is not None:
                stale.add(symbol)
            self._track("stale_data", symbol, problem)
        return stale

    def _track(self, kind: str, subject: str, problem: str | None) -> None:
        """Alert ``kind`` once when ``subject`` gets a problem and ``kind_recovered`` once when
        it is gone."""
        key = f"{kind}:{subject}"
        if problem is not None and key not in self._incidents:
            self._incidents.add(key)
            alert(kind, f"{subject}: {problem}")
        elif problem is None and key in self._incidents:
            self._incidents.remove(key)
            alert(f"{kind}_recovered", f"{subject}: recovered")

    # --- decisions and fills -----------------------------------------------------------------

    def _open_decision(
        self, now: datetime, stale: set[str], latest: dict[str, datetime]
    ) -> tuple[datetime | None, bool]:
        """The unprocessed decision bar closing at or before ``now`` (None when there is none)
        and whether it is ready: every symbol has its bar or is stale. One whose wait has ended
        is missed (``_missed``), so it is never processed late."""
        ts = _floor(now, self._width)
        if ts <= self._last_decision or self._missed(ts):
            return None, False
        waiting = [s for s in self._symbols if s not in stale and latest[s] < ts]
        return ts, not waiting

    def _missed(self, ts: datetime) -> bool:
        """True, with ``ts`` marked processed so this happens once, when its wait has ended by a
        fresh clock read (downloads, a warm-up or quotes may have taken a while). Alerted, unless
        ``ts`` closed before this process started: that is a routine start or resume."""
        late = (self._clock() - ts).total_seconds()
        if late < self._stale_after.total_seconds():
            return False
        self._last_decision = ts
        if ts < self._started_at:
            reason = "before session start"
            log.info("decision_missed", ts=ts.isoformat(), late_s=late, reason=reason)
        else:
            log.warning("decision_missed", ts=ts.isoformat(), late_s=late)
            alert(
                "decision_missed", f"decision at {ts.isoformat()} reached {late:.0f}s late: skipped"
            )
        return True

    def _decide(self, ts: datetime, latest: dict[str, datetime]) -> dict[str, Quote] | None:
        """Run the decision at ``ts`` with a window for every symbol whose bar ``ts`` is stored;
        returns the pre-decision quotes (a symbol without one gets no fill this poll), or None
        when building windows and quotes took it past its wait (missed, not run)."""
        with_bar = [s for s in self._symbols if s in latest and latest[s] >= ts]
        windows = self._windows(ts, with_bar)
        quotes = {s: q for s in self._symbols if (q := self._quote(s)) is not None}
        if self._missed(ts):  # a quote can take about 91 s: 5 retries (31 s backoff) + timeouts
            return None
        spreads = {s: (q.ask - q.bid) / (q.ask + q.bid) * 2e4 for s, q in quotes.items()}  # bps
        self._last_decision = ts  # set before the engine runs: never processed twice
        self.engine.on_decision(ts, windows, spreads)
        log.info(
            "paper_decision",
            ts=ts.isoformat(),
            windows=sorted(windows),
            equity=self.portfolio.equity,
            positions=sorted(self.portfolio.positions),
        )
        return quotes

    def _windows(self, ts: datetime, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Trailing ``WINDOW_ROWS`` feature rows of ``symbols`` from the stored history, exactly
        as the backtest builds them; the engine checks that each ends at ``ts``."""
        # ponytail: recomputes warmup_bars of features at every decision (about a second at
        # 3000 bars); keep the frame and append each new decision bar if that ever matters.
        windows = {}
        end = ts + self._base  # half-open: bars up to and including ts
        with Session(self._db) as session:
            for symbol in symbols:
                start = ts - self._history + self._base  # whole decision bars only
                candles = repo.load_candles(session, symbol, self._tf, start, end)
                funding = repo.load_funding(session, symbol, None, end)
                if not candles.empty:
                    features = decision_features(self.cfg, candles, funding)
                    windows[symbol] = features.iloc[-WINDOW_ROWS:]
        return windows

    def _check_kill_file(self, now: datetime) -> None:
        """With the kill file present, halt now, open or flat (the risk manager records
        ``manual_kill`` once), and queue KILL exits, which fill on this poll."""
        if not self.risk.kill_file.exists():
            return
        self.log.risk_events.extend(self.risk.check_limits(self.portfolio, now))
        if self.risk.must_flatten:
            self.engine.request_flatten(now)

    def _fill_queued(self, quotes: dict[str, Quote] | None) -> None:
        """Fill queued orders at fresh quotes: exits on every poll, entries only on the poll of
        their decision (``quotes`` given). A symbol whose pre-decision quote failed is not filled
        on that poll: its entry expires at the next decision, its exit waits for the next poll."""
        exits = self.engine.queued_exits
        for symbol in self._symbols:
            if quotes is not None and symbol not in quotes:
                continue
            flat = symbol not in self.portfolio.positions
            entry = quotes is not None and flat and self.engine.is_active(symbol)
            if not (symbol in exits or entry):
                continue
            if (quote := self._quote(symbol)) is not None:
                self._fill(symbol, quote)

    def _alert_halts(self) -> None:
        """Alert every new halt event once, and every halt that has cleared."""
        for event in self.log.risk_events[self._events_alerted :]:
            if event.action == "halt":
                alert(event.kind, event.detail)
        self._events_alerted = len(self.log.risk_events)
        halts = set(self.risk.to_state()["halts"])
        for kind in sorted(self._halts - halts):
            alert(f"{kind}_recovered", f"{kind} halt cleared")
        self._halts = halts

    def _quote(self, symbol: str) -> Quote | None:
        """A live quote, or None when the fetch fails or the quote is unusable."""
        try:
            quote = self.exchange.fetch_quote(symbol)
        except Exception as exc:  # no spread and no fill this time
            log.warning("paper_quote_failed", symbol=symbol, error=repr(exc))
            return None
        bid, ask = quote.bid, quote.ask
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            log.warning("paper_quote_invalid", symbol=symbol, bid=bid, ask=ask)
            return None
        return quote

    def _fill(self, symbol: str, quote: Quote) -> None:
        newest = self._newest.get(symbol)
        volume = 0.0 if newest is None else newest.volume
        if not (math.isfinite(volume) and volume >= 0):
            log.warning("paper_bar_volume_invalid", symbol=symbol, volume=volume)
            volume = 0.0  # no usable volume: worst-case exit impact, no entry fill
        self.engine.fill_queued(symbol, self._clock(), quote.bid, quote.ask, volume)
