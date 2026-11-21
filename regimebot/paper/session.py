"""Paper trading: the backtest's ``Engine`` fed by live, polled market data, with simulated fills.

Each poll (``run_once``) downloads every symbol's newly closed base bars and funding events into
the database and replays them through the engine in time order (``on_minute``, ``on_funding``; a
bar closing at t before funding stamped t), as the backtest replays an interval. When a decision
bar has closed for every symbol, features are rebuilt from the stored trailing history (the
backtest's resample and ``compute_features``), ``on_decision`` runs with live spreads, and queued
orders fill at a live quote (``fill_queued``) instead of at the next bar's open. All new rows and
the resumable state are flushed at the end of every poll in one transaction.

There is no order path: the session only calls the ``Exchange`` protocol's market-data methods
(``fetch_candles``, ``fetch_funding``, ``fetch_quote``); every fill is simulated.
"""

import math
import signal
import time
from collections.abc import Callable, Sequence
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
    failed or its newest bar is older than ``stale_after_seconds``; a symbol missing bar T is
    therefore stale by T + ``stale_after_seconds`` at the latest, which bounds the wait. A stale
    symbol gets no window (the engine records ``stale_data`` and does not trade it), and
    ``alert("stale_data")`` fires once per incident, ``stale_data_recovered`` when it ends.

    Fills: spreads come from quotes fetched just before ``on_decision``; right after it, each
    symbol with a queued order fills at a fresh quote, stamped with the session clock, with the
    last closed bar's volume for participation. A failed quote means no spread and no fill for
    that symbol at this decision: a queued entry then expires at the next decision (the engine's
    rule), a queued exit is retried on every poll until it fills.
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
        self._marks = state.FlushMarks()
        self._newest: dict[str, Bar] = {}  # newest bar fed to the engine, per symbol
        self._last_funding: dict[str, datetime] = {}  # newest funding event fed, per symbol
        self._last_decision = _EPOCH
        self._incidents: set[str] = set()  # open alert incidents, "kind:subject"
        self._stop_requested = False

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self, resume_run_id: int | None = None) -> int:
        """Warm up, then create a paper run or resume ``resume_run_id``; returns the run id.

        Warm-up stores the last ``warmup_bars`` decision bars' worth of base candles (and the
        funding) per symbol with ``download_candles``; a symbol whose download fails stays stale
        until polling brings its data.

        Resume: the run must be a paper run started with this config. Positions, cash, peak
        equity, marks, risk state and queued exits come back from its saved rows, and new rows
        are appended to the same run. Base bars that closed while the process was down are
        replayed in order through ``on_minute`` at the first poll, so a stop or target hit during
        the downtime is honoured at the bar where it happened. Decisions missed during the
        downtime are NOT replayed: a decision bar whose wait (``stale_after_seconds``) ended
        before the restart is skipped and the session resumes at the next live decision. An
        entry still queued at shutdown is dropped (it could only have filled at its decision).
        """
        now = self._clock()
        if resume_run_id is not None:
            with Session(self._db) as session:
                state.check_resumable(session, resume_run_id, self.cfg)  # fail before warm-up
        for symbol in self._symbols:
            self._download(symbol, now - self._history, now)
        with session_scope(self._db) as session:
            if resume_run_id is None:
                self.run_id = repo.create_run(session, Mode.PAPER, self.cfg)
                for symbol in self._symbols:
                    self._adopt(session, symbol, now)
                last_decision = _EPOCH
            else:
                self.run_id = resume_run_id
                cursors = state.restore(session, resume_run_id, self.engine)
                for symbol, ts in cursors.last_bar.items():
                    self._newest[symbol] = self._stored_bar(session, symbol, ts)
                self._last_funding = dict(cursors.last_funding)
                last_decision = cursors.last_decision
            # a decision whose wait ended before now was missed: never process it late
            self._last_decision = max(last_decision, _floor(now - self._stale_after, self._width))
            self._marks = self._save(session)
        log.info(
            "paper_started",
            run_id=self.run_id,
            resumed=resume_run_id is not None,
            positions=sorted(self.portfolio.positions),
            last_decision=self._last_decision.isoformat(),
        )
        return self.run_id

    def run_forever(self) -> None:
        """Poll every ``poll_seconds`` until SIGINT or SIGTERM (or a KeyboardInterrupt in sleep).

        A signal only sets a flag, so the current poll always finishes and is flushed; the
        session then exits without flattening and the run stays open for a resume. An exception
        in a poll is logged and alerted once per incident, and the next poll runs as usual.
        """
        self._stop_requested = False
        previous = {
            sig: signal.signal(sig, self._request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            while not self._stop_requested:
                self._guarded_poll()
                if self._stop_requested:
                    break
                try:
                    self._sleep(self.cfg.paper.poll_seconds)
                except KeyboardInterrupt:
                    break
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        try:
            self.flush()  # normally a no-op rewrite: each poll flushed already
        except Exception:
            log.exception("paper_flush_failed", run_id=self.run_id)
        log.info("paper_stopped", run_id=self.run_id, positions=sorted(self.portfolio.positions))

    def _request_stop(self, signum: int, frame: FrameType | None) -> None:
        self._stop_requested = True
        log.info("paper_stop_requested", signal=signal.Signals(signum).name)

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
        """One poll: download, replay new bars and funding, retry queued exits, decide if a
        decision bar is due, flush. Exchange errors are contained per symbol and call."""
        now = self._clock()
        downloaded = {}
        for symbol in self._symbols:
            newest = self._newest.get(symbol)
            since = now - self._history if newest is None else newest.ts + self._base
            downloaded[symbol] = self._download(symbol, since, now)
        self._replay(now)
        stale = self._stale_symbols(now, downloaded)
        for symbol in [s for s in state.queued_exits(self.engine) if s in self.portfolio.positions]:
            if (quote := self._quote(symbol)) is not None:
                self._fill(symbol, quote)
        if (ts := self._due_decision(now, stale)) is not None:
            self._decide(ts, stale)
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
        state.save_state(session, self.run_id, self.engine, cursors)
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

    def _replay(self, now: datetime) -> None:
        """Feed every stored bar and funding event newer than the engine has seen, all symbols
        in time order. Funding waits until the symbol's bars reach its stamp; its mark is the
        close of the last bar at or before it (a record arriving over a bar late gets a later
        close)."""
        events: list[tuple[datetime, int, int, Bar | float]] = []
        with Session(self._db) as session:
            for order, symbol in enumerate(self._symbols):
                newest = self._newest.get(symbol)
                if newest is None:  # first data for this symbol: no position can need it
                    self._adopt(session, symbol, now)
                    continue
                bars = repo.load_candles(session, symbol, self._tf, newest.ts + self._base)
                stamps = list(pd.DatetimeIndex(bars.index).to_pydatetime())
                for ts, row in zip(stamps, bars.to_numpy(dtype=float).tolist(), strict=True):
                    events.append((ts, _BAR, order, Bar(symbol, ts, *row)))
                since = self._last_funding.get(symbol)
                upto = (stamps[-1] if stamps else newest.ts) + _TICK
                after = None if since is None else since + _TICK
                rates = repo.load_funding(session, symbol, after, upto)
                for ts, rate in zip(
                    pd.DatetimeIndex(rates.index).to_pydatetime(), rates.tolist(), strict=True
                ):
                    events.append((ts, _FUNDING, order, rate))
        events.sort(key=lambda event: event[:3])
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

    def _stale_symbols(self, now: datetime, downloaded: dict[str, bool]) -> set[str]:
        """Symbols whose candle download failed or whose newest bar is too old (alerted)."""
        stale = set()
        for symbol in self._symbols:
            newest = self._newest.get(symbol)
            problem = None
            if not downloaded[symbol]:
                problem = "candle download failed"
            elif newest is None:
                problem = "no bars stored"
            elif now - newest.ts > self._stale_after:
                problem = f"newest bar closed at {newest.ts.isoformat()}"
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

    def _due_decision(self, now: datetime, stale: set[str]) -> datetime | None:
        """The decision bar closing at or before ``now`` when it is unprocessed and every symbol
        has its bar or is stale; None otherwise."""
        ts = _floor(now, self._width)
        if ts <= self._last_decision:
            return None
        waiting = [s for s in self._symbols if s not in stale and self._newest[s].ts < ts]
        return None if waiting else ts

    def _decide(self, ts: datetime, stale: set[str]) -> None:
        windows = self._windows(ts, stale)
        quotes = {s: q for s in self._symbols if (q := self._quote(s)) is not None}
        spreads = {s: (q.ask - q.bid) / (q.ask + q.bid) * 2e4 for s, q in quotes.items()}  # bps
        self._last_decision = ts  # set before the engine runs: never processed twice
        self.engine.on_decision(ts, windows, spreads)
        for symbol in self._symbols:
            if not (symbol in quotes and self._has_queued_order(symbol)):
                continue
            if (quote := self._quote(symbol)) is not None:
                self._fill(symbol, quote)
        log.info(
            "paper_decision",
            ts=ts.isoformat(),
            windows=sorted(windows),
            equity=self.portfolio.equity,
            positions=sorted(self.portfolio.positions),
        )

    def _windows(self, ts: datetime, stale: set[str]) -> dict[str, pd.DataFrame]:
        """Trailing ``WINDOW_ROWS`` feature rows per fresh symbol from the stored history,
        exactly as the backtest builds them; the engine checks that each ends at ``ts``."""
        # ponytail: recomputes warmup_bars of features at every decision (about a second at
        # 3000 bars); keep the frame and append each new decision bar if that ever matters.
        windows = {}
        end = ts + self._base  # half-open: bars up to and including ts
        with Session(self._db) as session:
            for symbol in self._symbols:
                if symbol in stale:
                    continue
                start = ts - self._history + self._base  # whole decision bars only
                candles = repo.load_candles(session, symbol, self._tf, start, end)
                funding = repo.load_funding(session, symbol, None, end)
                if not candles.empty:
                    features = decision_features(self.cfg, candles, funding)
                    windows[symbol] = features.iloc[-WINDOW_ROWS:]
        return windows

    def _has_queued_order(self, symbol: str) -> bool:
        if symbol in state.queued_exits(self.engine):
            return True
        return self.engine.is_active(symbol) and symbol not in self.portfolio.positions

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
            return None  # checked here: the engine pops a queued order before pricing it
        return quote

    def _fill(self, symbol: str, quote: Quote) -> None:
        newest = self._newest.get(symbol)
        volume = 0.0 if newest is None else newest.volume  # no bar: worst-case exit impact
        self.engine.fill_queued(symbol, self._clock(), quote.bid, quote.ask, volume)
