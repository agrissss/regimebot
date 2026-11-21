"""Paper session: backtest parity, decision timing, stale data, exchange errors, exact resume,
the run lock, the kill switch, halts, queued exits that never get stuck, the poll loop and the CLI.

Every scenario replays scripted 1m candles through ``ReplayExchange`` on a test clock. Polls run
``LAG`` = 3 s after a minute (a bar is visible 2 s after its close), so a live fill at a decision
is stamped LAG after the backtest's, which stamps it at the next bar's open, and priced from that
same open: the fake quote is the forming bar's open -/+ the half spread. Polls between decisions
may be sparse: the bars in between are replayed in order. ``Scripted`` stubs (with a trailing
stop) trade whatever regime the features give: ``open_cfg`` lets every regime run "breakout".
"""

import os
import signal
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import structlog
import structlog.testing
from fakes import Clock, ReplayExchange
from sqlalchemy import Engine as Database
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from synth import BTC, ETH, MINUTE, T0, Scripted, candles_from_closes
from typer.testing import CliRunner

from regimebot.backtest.runner import run_backtest
from regimebot.cli import app
from regimebot.config import AppConfig, load_config
from regimebot.db import repo
from regimebot.db.models import Base, OpenPosition, RegimeRow, RiskDecisionRow, Run, SignalRow
from regimebot.db.session import make_engine, session_scope
from regimebot.engine.loop import Engine
from regimebot.paper import session as paper_session
from regimebot.paper import state
from regimebot.paper.session import PaperSession
from regimebot.types import ExitReason, Fill, Mode, Regime, Trade

HOUR = timedelta(hours=1)
D = timedelta(minutes=15)  # decision bar
LAG = timedelta(seconds=3)
START = T0 + 4 * HOUR  # first decision; the 4 h before it warm the features up
T1, T2 = START + D, START + 4 * D  # entry decisions (both symbols, long)
MAX_HOLD = 4  # decision bars: BTC's T2 entry leaves by TIME one hour later
END = T2 + (MAX_HOLD + 1) * D  # last bar and last decision
BARS = (END - T0) // MINUTE
EVERY_5M = 5 * MINUTE


def open_cfg(**overrides: object) -> AppConfig:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    return load_config(
        overrides={
            "exchange.symbols": [BTC, ETH],
            "paper.warmup_bars": 100,  # all the history: features equal the backtest's exactly
            **{f"playbook.{r.value}": entry for r in Regime},
            **overrides,
        }
    )


def funding_at(*stamps_and_rates: tuple[datetime, float]) -> pd.Series:
    stamps, rates = zip(*stamps_and_rates, strict=True)
    return pd.Series(rates, index=pd.DatetimeIndex(stamps, name="ts"), name="rate")


def market() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """1m bars from T0 to END, BTC at 100 and ETH at 50 (0.05 wicks, constant volume), scripted
    so that every exit kind happens: stops are entry - 2, targets entry + 4, trailing 2 ATR."""
    ramp = (T2 - T0) // MINUTE  # first bar after the T2 decision
    eth_closes = np.full(BARS, 50.0)
    eth_closes[ramp : ramp + 30] = np.linspace(50.1, 53.0, 30)  # +1.5R: the stop trails up
    eth_closes[ramp + 30 : ramp + 39] = 53.0
    eth_closes[ramp + 39 :] = 52.0  # the dip at T2 + 40m takes the trailed stop: TRAIL exit
    btc, eth = candles_from_closes([100.0] * BARS), candles_from_closes(eth_closes)
    btc.loc[T1 + 20 * MINUTE, "high"] = 104.5  # the T1 long hits its 104 target
    eth.loc[T1 + 10 * MINUTE, "low"] = 47.5  # the T1 long hits its 48 stop
    btc.loc[T2 + MAX_HOLD * D + MINUTE, ["open", "high"]] = [101.0, 101.05]  # time exit fill bar
    funding = {  # T2's is stamped at the entry decision: nobody holds a position then
        BTC: funding_at((T2, 0.5), (T2 + 2 * D, 0.0001)),
        ETH: funding_at((T2, 0.5), (T2 + 2 * D, -0.0002)),
    }
    return {BTC: btc, ETH: eth}, funding


def stub(at: set[datetime] | None = None) -> Scripted:
    return Scripted(
        "breakout", at={T1, T2} if at is None else at, max_hold_bars=MAX_HOLD, trail_atr_mult=2.0
    )


@dataclass
class Rig:
    cfg: AppConfig
    exchange: ReplayExchange
    clock: Clock
    db: Database
    kill_file: Path
    at: set[datetime] | None = None

    def session(self, **kwargs: Any) -> PaperSession:
        return PaperSession(
            self.cfg,
            kwargs.pop("exchange", self.exchange),
            self.db,
            self.kill_file,
            clock=self.clock,
            strategies=[stub(self.at)],
            **kwargs,
        )

    def poll(self, session: PaperSession, until: datetime, every: timedelta = MINUTE) -> None:
        """Poll ``every`` minute (or few) up to ``until`` + LAG."""
        while self.clock.now <= until + LAG:
            session.run_once()
            self.clock.now += every


def make_rig(
    tmp_path: Path,
    cfg: AppConfig | None = None,
    at: set[datetime] | None = None,
    shift: timedelta = timedelta(0),
) -> Rig:
    """A rig on ``market()`` moved ``shift`` later (``at`` must then be shifted too)."""
    cfg = cfg or open_cfg()
    candles, funding = market()
    candles = {s: df.shift(freq=shift) for s, df in candles.items()}
    funding = {s: rates.shift(freq=shift) for s, rates in funding.items()}
    clock = Clock(START + shift + LAG)
    db = make_engine(f"sqlite:///{tmp_path / 'paper.db'}")
    Base.metadata.create_all(db)
    exchange = ReplayExchange(candles, funding, cfg.costs, clock)
    return Rig(cfg, exchange, clock, db, tmp_path / "KILL", at)


@pytest.fixture(autouse=True)
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Alerts the session sent; never the real webhook."""
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(paper_session, "alert", lambda kind, detail: sent.append((kind, detail)))
    return sent


def unlag(ts: datetime) -> datetime:
    """Undo the one documented difference: a live fill is stamped at its poll, LAG after the
    minute, where the backtest stamps it at the minute (the next bar's open)."""
    return ts - LAG if (ts - T0) % MINUTE == LAG else ts


def unlag_trade(t: Trade) -> Trade:
    return replace(t, entry_ts=unlag(t.entry_ts), exit_ts=unlag(t.exit_ts))


def unlag_fill(f: Fill) -> Fill:
    return replace(f, ts=unlag(f.ts))


def stored(db: Database, run_id: int) -> dict[str, list[Any]]:
    """Everything a run stored, in a form comparable across databases."""
    with Session(db) as s:
        decisions = s.execute(
            select(SignalRow.symbol, SignalRow.ts, RiskDecisionRow.ts, RiskDecisionRow.rule)
            .join(SignalRow, RiskDecisionRow.signal_id == SignalRow.id)
            .where(RiskDecisionRow.run_id == run_id)
            .order_by(RiskDecisionRow.id)
        ).all()
        regimes = s.execute(
            select(RegimeRow.symbol, RegimeRow.ts, RegimeRow.regime, RegimeRow.inputs_json)
            .where(RegimeRow.run_id == run_id)
            .order_by(RegimeRow.ts, RegimeRow.symbol)
        ).all()
        return {
            "trades": repo.load_trades(s, run_id),
            "fills": repo.load_fills(s, run_id),
            "equity": list(repo.load_equity(s, run_id).items()),
            "risk_events": repo.load_risk_events(s, run_id),
            "decisions": [tuple(r) for r in decisions],
            "regimes": [tuple(r) for r in regimes],
            "open_positions": sorted(repo.load_open_positions(s, run_id)),
        }


def exits(rows: dict[str, list[Any]]) -> list[tuple[str, ExitReason, datetime]]:
    return [(t.symbol, t.exit_reason, t.exit_ts) for t in rows["trades"]]


class Spy:
    """Records every attribute the session reads from the exchange."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.names: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self.names.add(name)
        return getattr(self.inner, name)


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[Any]]:
    """What an uninterrupted run over the whole market stores (polls every 5 minutes)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paper_session, "alert", lambda kind, detail: None)
        rig = make_rig(tmp_path_factory.mktemp("uninterrupted"))
        session = rig.session()
        run_id = session.start()
        rig.poll(session, END, every=EVERY_5M)
        session.close()
        return stored(rig.db, run_id)


# --- parity -----------------------------------------------------------------------------------


def test_a_scripted_market_trades_exactly_like_the_backtest(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    spy = Spy(rig.exchange)
    session = rig.session(exchange=spy)
    run_id = session.start()
    rig.poll(session, END, every=EVERY_5M)
    candles, funding = market()
    ref = run_backtest(
        rig.cfg, candles, funding, START, END + MINUTE, tmp_path / "BT_KILL", strategies=[stub()]
    ).log
    paper = session.log

    assert [(t.symbol, t.exit_reason) for t in ref.trades] == [
        (ETH, ExitReason.STOP),
        (BTC, ExitReason.TARGET),
        (ETH, ExitReason.TRAIL),  # the trailing stop moved at T2 + 30m, the dip took it
        (BTC, ExitReason.TIME),
    ]
    assert len({t.funding for t in ref.trades}) > 2  # funding charged and received
    # Only live-fill timestamps differ (LAG later); prices, sizes, costs and pnl are equal.
    assert [unlag_trade(t) for t in paper.trades] == ref.trades
    assert [unlag_fill(f) for f in paper.fills] == ref.fills
    assert paper.fills != ref.fills  # the difference is real, and the only one
    assert paper.signals == ref.signals
    assert paper.decisions == ref.decisions
    assert paper.regimes == ref.regimes
    assert paper.risk_events == ref.risk_events == []
    assert paper.equity == ref.equity
    # one decision per decision bar, though three polls fall inside each
    assert [e[0] for e in paper.equity] == [START + k * D for k in range(10)]
    # the database holds the same records, and the exchange was only read
    rows = stored(rig.db, run_id)
    assert rows["trades"] == sorted(paper.trades, key=lambda t: t.entry_ts)
    assert rows["fills"] == paper.fills
    assert rows["equity"] == [(pd.Timestamp(ts), eq) for ts, eq, _, _ in paper.equity]
    assert rows["open_positions"] == [state.PORTFOLIO_KEY, state.RISK_KEY]
    assert spy.names == {"fetch_candles", "fetch_funding", "fetch_quote"}


def test_a_decision_fires_once_per_decision_bar_at_any_poll_rate_and_after_a_restart(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    run_id = session.start()
    ten_seconds = timedelta(seconds=10)

    def poll_every_10s(session: PaperSession, start: datetime, end: datetime) -> None:
        rig.clock.now = start
        while rig.clock.now < end:
            session.run_once()
            rig.clock.now += ten_seconds

    poll_every_10s(session, START + LAG, START + 3 * MINUTE)
    poll_every_10s(session, T1 - 2 * MINUTE, T1 + 3 * MINUTE)
    session.close()
    resumed = rig.session()  # restart inside the T1 bar, right after its decision
    resumed.start(resume_run_id=run_id)
    poll_every_10s(resumed, T1 + 3 * MINUTE, T1 + 5 * MINUTE)
    poll_every_10s(resumed, T1 + D - 2 * MINUTE, T1 + D + 3 * MINUTE)
    rows = stored(rig.db, run_id)
    assert [ts for ts, _ in rows["equity"]] == [START, T1, T1 + D]
    assert [(s, ts) for s, ts, _, _ in rows["regimes"]] == [
        (symbol, ts) for ts in (START, T1, T1 + D) for symbol in (BTC, ETH)
    ]


# --- decision timing --------------------------------------------------------------------------


def test_a_poll_that_comes_after_the_wait_skips_the_decision(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    run_id = session.start()
    rig.poll(session, T1 - 2 * MINUTE)
    rig.clock.now = T1 + 12 * MINUTE + LAG  # the process slept through T1 and its 3 min wait
    rig.poll(session, T1 + D)
    rows = stored(rig.db, run_id)
    assert [ts for ts, _ in rows["equity"]] == [START, T1 + D]  # T1 never ran late
    assert rows["fills"] == [] and session.log.signals == []
    assert alerts == [
        ("decision_missed", f"decision at {T1.isoformat()} reached 723s late: skipped")
    ]


class SlowWarmUp:
    """The first candle download takes 95 s of test time, like the live 94.5 s warm-up."""

    def __init__(self, inner: ReplayExchange) -> None:
        self.inner = inner
        self.delay = timedelta(seconds=95)

    def fetch_candles(self, *args: Any) -> Any:
        self.inner.clock.now += self.delay
        self.delay = timedelta(0)
        return self.inner.fetch_candles(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def test_a_start_whose_warm_up_outlasts_the_wait_skips_that_decision(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = make_rig(tmp_path)
    rig.clock.now = T1 + 2 * MINUTE  # within T1's wait, but the warm-up takes 95 s
    session = rig.session(exchange=SlowWarmUp(rig.exchange))
    run_id = session.start()
    assert rig.clock.now == T1 + 3 * MINUTE + 35 * timedelta(seconds=1)
    rig.poll(session, T1 + D + MINUTE)
    assert [ts for ts, _ in stored(rig.db, run_id)["equity"]] == [T1 + D]
    assert session.log.signals == []  # no T1 entry 3.5 minutes after its signal
    assert [kind for kind, _ in alerts] == ["decision_missed"]


def test_bars_after_a_late_decision_see_the_stop_it_trailed(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    rig.exchange.lag[BTC] = timedelta(seconds=100)  # every decision waits 2 minutes for BTC
    eth = rig.exchange.candles[ETH]
    eth.loc[T2 + 2 * D + MINUTE, ["close", "low"]] = [52.0, 51.95]  # right after the trail
    session = rig.session()
    session.start()
    rig.poll(session, T2 + 2 * D + 3 * MINUTE)
    (eth_t2,) = [t for t in session.log.trades if t.symbol == ETH and t.entry_ts > T2]
    # The T2 + 2D decision runs at T2 + 2D + 2m, after ETH's next two bars arrived; they are
    # held back and replayed after it, against the stop it trailed to (52.1, not 48).
    assert (eth_t2.exit_reason, eth_t2.exit_ts) == (ExitReason.TRAIL, T2 + 2 * D + MINUTE)
    assert eth_t2.exit_price > 52.0


# --- stale data, quotes and exchange errors ---------------------------------------------------


def test_a_failing_candle_feed_blocks_the_symbol_and_alerts_once_per_incident(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    session.start()
    rig.poll(session, T1 - 3 * MINUTE)
    rig.exchange.fail.add(("fetch_candles", ETH))
    rig.poll(session, T1 + 2 * MINUTE)  # six failing polls around the T1 decision
    rig.exchange.fail.clear()
    rig.poll(session, T1 + 12 * MINUTE)

    log = session.log
    assert [s.symbol for s in log.signals] == [BTC]  # ETH had no bar T1, so no window
    assert [(e.ts, e.kind, e.action) for e in log.risk_events] == [
        (T1, "stale_data", "skip_symbol")
    ]
    assert ETH in log.risk_events[0].detail
    assert alerts == [
        ("stale_data", f"{ETH}: candle download failed"),
        ("stale_data_recovered", f"{ETH}: recovered"),
    ]
    assert [e[0] for e in log.equity] == [START, T1]  # T1 was not held up by ETH
    with Session(rig.db) as s:  # the bars missed during the incident were caught up
        assert len(repo.load_candles(s, ETH, "1m", T1 - 3 * MINUTE, T1 + 3 * MINUTE)) == 6


def test_a_failed_download_keeps_the_window_when_bar_t_is_already_stored(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = make_rig(tmp_path)
    rig.exchange.lag[BTC] = timedelta(seconds=100)  # T1 is decided at T1 + 2m
    session = rig.session()
    session.start()
    rig.poll(session, T1)  # ETH's bar T1 is stored at T1 + 3s
    rig.exchange.fail.add(("fetch_candles", ETH))
    rig.poll(session, T1 + 2 * MINUTE)  # ETH downloads fail, then T1 is decided
    rig.exchange.fail.clear()
    rig.poll(session, T1 + 3 * MINUTE)
    log = session.log
    assert sorted(s.symbol for s in log.signals) == [BTC, ETH]  # ETH kept its window
    assert {f.symbol: f.ts for f in log.fills} == {
        BTC: T1 + 2 * MINUTE + LAG,
        ETH: T1 + 2 * MINUTE + LAG,
    }
    assert [e for e in log.risk_events if e.kind == "stale_data"] == []
    assert alerts == [  # the failure is still reported
        ("stale_data", f"{ETH}: candle download failed"),
        ("stale_data_recovered", f"{ETH}: recovered"),
    ]


@pytest.mark.parametrize(
    ("lag", "decided", "eth_trades"),
    [(timedelta(seconds=100), T1 + 2 * MINUTE, True), (timedelta(seconds=250), T1, False)],
    ids=["late_symbol_waited_for", "too_late_symbol_is_stale"],
)
def test_a_decision_waits_for_a_late_symbol_until_it_turns_stale(
    tmp_path: Path,
    alerts: list[tuple[str, str]],
    lag: timedelta,
    decided: datetime,
    eth_trades: bool,
) -> None:
    rig = make_rig(tmp_path)
    rig.exchange.lag[ETH] = lag  # ETH bars arrive `lag` after their close
    session = rig.session()
    session.start()
    rig.poll(session, T1 + 12 * MINUTE)
    entries = {f.symbol: f.ts for f in session.log.fills if f.side == "buy"}
    assert entries[BTC] == decided + LAG
    assert (ETH in entries) is eth_trades
    stale = [e for e in session.log.risk_events if e.kind == "stale_data"]
    assert bool(stale) is not eth_trades  # stale ETH is skipped at every decision
    assert [a[0] for a in alerts] == ([] if eth_trades else ["stale_data"])


def test_a_failed_quote_means_no_spread_and_no_fill_and_the_entry_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    spreads: list[dict[str, float] | None] = []
    decide = Engine.on_decision

    def recording(engine: Engine, ts: datetime, windows: Any, spread: Any = None) -> None:
        spreads.append(spread)
        decide(engine, ts, windows, spread)

    monkeypatch.setattr(Engine, "on_decision", recording)
    session.start()
    rig.poll(session, T1 - MINUTE)
    rig.exchange.fail.add(("fetch_quote", ETH))
    rig.poll(session, T1)
    rig.exchange.fail.clear()
    rig.poll(session, T1 + D)  # quotes work again: the entry is not retried, it expires

    log = session.log
    assert spreads[0] == {BTC: pytest.approx(2.0), ETH: pytest.approx(2.0)}  # 2 x 1 bp
    assert spreads[1] == {BTC: pytest.approx(2.0)}  # T1: no ETH quote, no ETH spread
    eth = [(ts, d.rule) for i, ts, d in log.decisions if log.signals[i].symbol == ETH]
    assert eth == [(T1, "ok"), (T1 + D, "expired")]
    assert [(f.symbol, f.ts) for f in log.fills] == [(BTC, T1 + LAG)]


def test_an_exchange_error_in_one_poll_does_not_stop_the_next(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    session.start()
    rig.poll(session, T1 + 9 * MINUTE)
    methods = ("fetch_candles", "fetch_funding", "fetch_quote")
    rig.exchange.fail |= {(m, s) for m in methods for s in (BTC, ETH)}
    rig.poll(session, T1 + 11 * MINUTE)  # nothing reachable while ETH's stop bar closes
    rig.exchange.fail.clear()
    rig.poll(session, T1 + 12 * MINUTE)
    (eth,) = [t for t in session.log.trades if t.symbol == ETH]
    assert (eth.exit_reason, eth.exit_ts) == (ExitReason.STOP, T1 + 10 * MINUTE)  # at its bar


# --- resume -----------------------------------------------------------------------------------


def resumable_state(session: PaperSession) -> dict[str, Any]:
    p = session.portfolio
    return {
        "positions": {s: pos.to_state() for s, pos in p.positions.items()},
        "cash": p.cash,
        "peak": p.peak_equity,
        "marks": p.last_prices,
        "risk": session.risk.to_state(),
        "engine": session.engine.to_state(),
    }


def test_resume_mid_position_three_times_matches_an_uninterrupted_run(
    tmp_path: Path, uninterrupted: dict[str, list[Any]]
) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    run_id = session.start()
    saved = []
    # Each stop lets a decisive bar close while the process is down: ETH's stop (T1 + 10m),
    # BTC's target (T1 + 20m), the dip through ETH's trailed stop (T2 + 40m).
    for stop_after in (T1 + 5 * MINUTE, T1 + D, T2 + 2 * D + 5 * MINUTE):
        rig.poll(session, stop_after, every=EVERY_5M)
        before = resumable_state(session)
        session.close()  # the process stops
        session = rig.session()
        assert session.start(resume_run_id=run_id) == run_id
        assert resumable_state(session) == before
        saved.append(before)
    rig.poll(session, END, every=EVERY_5M)

    assert saved[1]["risk"]["loss_streak"] == 1 and saved[1]["risk"]["pnl_by_day"]  # ETH lost
    assert sorted(saved[1]["positions"]) == [BTC]
    trailed = saved[2]["positions"][ETH]
    assert trailed["stop"] > trailed["initial_stop"]
    assert stored(rig.db, run_id) == uninterrupted
    assert exits(uninterrupted)[:3] == [
        (ETH, ExitReason.STOP, T1 + 10 * MINUTE),  # each at the bar where it happened
        (BTC, ExitReason.TARGET, T1 + 20 * MINUTE),
        (ETH, ExitReason.TRAIL, T2 + 40 * MINUTE),
    ]


def test_a_crash_between_the_decision_and_its_flush_converges_after_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uninterrupted: dict[str, list[Any]]
) -> None:
    rig = make_rig(tmp_path)
    session = rig.session()
    run_id = session.start()
    rig.poll(session, T2 - EVERY_5M, every=EVERY_5M)
    assert rig.clock.now == T2 + LAG

    def crash() -> None:
        raise RuntimeError("killed before the commit")

    monkeypatch.setattr(session, "flush", crash)
    with pytest.raises(RuntimeError, match="killed"):
        session.run_once()
    assert [s.ts for s in session.log.signals][-2:] == [T2, T2]  # T2 ran, nothing stored
    session.close()
    resumed = rig.session()  # restarted at once, still inside T2's wait
    resumed.start(resume_run_id=run_id)
    rig.poll(resumed, END, every=EVERY_5M)
    assert stored(rig.db, run_id) == uninterrupted  # no duplicate, same result


def test_decisions_missed_while_down_are_not_replayed(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = make_rig(tmp_path)
    first = rig.session()
    run_id = first.start()
    rig.poll(first, T1 + 2 * MINUTE)
    first.close()
    # Down over the T1 + D and T1 + 2D closes; back 2 min after T1 + 2D, while its wait (3 min)
    # is still open: T1 + D is missed, T1 + 2D is decided late.
    rig.clock.now = T1 + 2 * D + 2 * MINUTE + LAG
    resumed = rig.session()
    resumed.start(resume_run_id=run_id)
    rig.poll(resumed, T1 + 3 * D)
    rows = stored(rig.db, run_id)
    assert [ts for ts, _ in rows["equity"]] == [START, T1, T1 + 2 * D, T1 + 3 * D]
    assert exits(rows) == [
        (ETH, ExitReason.STOP, T1 + 10 * MINUTE),  # both exits happened while down
        (BTC, ExitReason.TARGET, T1 + 20 * MINUTE),
    ]
    assert alerts == []  # T1 + D was never observed by a running session


def test_resume_refuses_an_unknown_run_a_backtest_run_and_a_changed_config(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path)
    with session_scope(rig.db) as s:
        backtest_id = repo.create_run(s, Mode.BACKTEST, rig.cfg)
    paper = rig.session()
    paper_id = paper.start()
    paper.close()
    for run_id, cfg, error in [
        (99, rig.cfg, "unknown run_id 99"),
        (backtest_id, rig.cfg, "not a paper run"),
        (paper_id, open_cfg(**{"risk.risk_per_trade": 0.01}), "config"),
    ]:
        rig.cfg = cfg
        with pytest.raises(ValueError, match=error):
            rig.session().start(resume_run_id=run_id)


def test_a_run_cannot_be_run_by_two_sessions_at_once(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    first = rig.session()
    run_id = first.start()
    with pytest.raises(ValueError, match=f"paper run {run_id} is already running"):
        rig.session().start(resume_run_id=run_id)  # its own descriptor: the flock conflicts
    first.close()
    second = rig.session()
    assert second.start(resume_run_id=run_id) == run_id
    assert (rig.kill_file.parent / f"paper_run_{run_id}.lock").exists()
    second.close()


# --- exits, the kill switch and halts ---------------------------------------------------------


def test_a_queued_exit_is_retried_every_poll_until_it_fills_even_across_a_restart(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path)
    first = rig.session()
    run_id = first.start()
    exit_at = T2 + MAX_HOLD * D  # BTC's T2 entry gets its time exit queued here
    rig.poll(first, exit_at - EVERY_5M, every=EVERY_5M)
    rig.exchange.fail.add(("fetch_quote", BTC))
    rig.poll(first, exit_at + 2 * MINUTE)  # three polls without a BTC quote
    assert first.engine.queued_exits == {BTC: ExitReason.TIME}  # ETH left by its trail
    first.close()
    resumed = rig.session()  # restart while the exit waits
    resumed.start(resume_run_id=run_id)
    assert resumed.engine.queued_exits == {BTC: ExitReason.TIME}
    rig.poll(resumed, exit_at + 3 * MINUTE)  # still no quote
    rig.exchange.fail.clear()
    rig.poll(resumed, exit_at + 4 * MINUTE)

    btc = [t for t in stored(rig.db, run_id)["trades"] if t.symbol == BTC][-1]
    assert (btc.exit_reason, btc.exit_ts) == (ExitReason.TIME, exit_at + 4 * MINUTE + LAG)
    # sold at the live bid (forming bar open 100 - 1 bp) less 2 bps and impact on 1e6 volume
    slip_bps = 2.0 + 100 * btc.qty / 1e6
    assert btc.exit_price == pytest.approx(100 * (1 - 1e-4) * (1 - slip_bps / 1e4), rel=1e-12)
    assert resumed.engine.queued_exits == {}


def test_the_kill_file_flattens_within_one_poll_and_retries_a_failed_quote(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = make_rig(tmp_path, at={T1, T1 + D})
    session = rig.session()
    run_id = session.start()
    rig.poll(session, T1 + 5 * MINUTE)
    rig.kill_file.touch()  # mid-interval: the next decision is 9 minutes away
    rig.exchange.fail.add(("fetch_quote", BTC))
    rig.poll(session, T1 + 6 * MINUTE)  # one poll: halted, ETH flattened, BTC has no quote
    assert exits(stored(rig.db, run_id)) == [(ETH, ExitReason.KILL, T1 + 6 * MINUTE + LAG)]
    assert session.engine.queued_exits == {BTC: ExitReason.KILL}
    rig.exchange.fail.clear()
    rig.poll(session, T1 + D + 5 * MINUTE)

    rows = stored(rig.db, run_id)
    assert exits(rows) == [
        (ETH, ExitReason.KILL, T1 + 6 * MINUTE + LAG),
        (BTC, ExitReason.KILL, T1 + 7 * MINUTE + LAG),  # retried on the next poll
    ]
    assert [s.ts for s in session.log.signals] == [T1, T1]  # no entries once halted
    assert [(e.ts, e.kind, e.action) for e in rows["risk_events"]] == [
        (T1 + 6 * MINUTE + LAG, "manual_kill", "halt")  # once, not again at T1 + D
    ]
    assert [kind for kind, _ in alerts] == ["manual_kill"]
    assert rows["open_positions"] == [state.PORTFOLIO_KEY, state.RISK_KEY]


def test_halts_are_alerted_and_so_is_a_daily_halt_clearing(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    shift = 18 * HOUR  # T1 is 22:15 UTC: the next UTC day starts during the run
    rig = make_rig(tmp_path, open_cfg(**{"risk.daily_loss": 0.001}), at={T1 + shift}, shift=shift)
    session = rig.session()
    session.start()
    midnight = T0 + 24 * HOUR
    rig.poll(session, midnight, every=EVERY_5M)
    halts = [e for e in session.log.risk_events if e.action == "halt"]
    assert [(e.ts, e.kind) for e in halts] == [(T1 + D + shift, "daily_loss")]  # ETH's stop
    assert alerts == [
        ("daily_loss", halts[0].detail),
        ("daily_loss_recovered", "daily_loss halt cleared"),  # at the midnight decision
    ]
    assert [t.exit_reason for t in session.log.trades] == [ExitReason.STOP, ExitReason.KILL]


# --- the poll loop ----------------------------------------------------------------------------


@pytest.mark.parametrize("stop", ["sigterm", "keyboard_interrupt"])
def test_run_forever_survives_a_failing_poll_and_stops_without_flattening(
    tmp_path: Path, alerts: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    rig = make_rig(tmp_path)
    rig.clock.now = T1 + LAG
    rig.exchange.fail.add(("fetch_quote", ETH))  # ETH's T1 entry stays queued
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        rig.clock.now += MINUTE
        if len(sleeps) == 4:
            if stop == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                raise KeyboardInterrupt

    session = rig.session(sleep=sleep)
    run_id = session.start()
    flush = session.flush
    flushes = 0

    def flaky_flush() -> None:  # the database is away during the second poll
        nonlocal flushes
        flushes += 1
        if flushes == 2:
            raise RuntimeError("database is locked")
        flush()

    monkeypatch.setattr(session, "flush", flaky_flush)
    handler = signal.getsignal(signal.SIGTERM)
    with structlog.testing.capture_logs() as logs:
        session.run_forever()

    assert sleeps == [10, 10, 10, 10]  # four polls, one sleep after each
    assert signal.getsignal(signal.SIGTERM) == handler
    assert [kind for kind, _ in alerts] == ["poll_error", "poll_error_recovered"]
    events = [e["event"] for e in logs]
    assert events[-1] == "paper_stopped"
    assert ("paper_stop_requested" in events) is (stop == "sigterm")  # logged after the loop
    rows = stored(rig.db, run_id)
    assert rows["open_positions"] == sorted([state.PORTFOLIO_KEY, state.RISK_KEY, BTC])
    assert [ts for ts, _ in rows["equity"]] == [T1]
    eth = [(sig_ts, ts, rule) for s, sig_ts, ts, rule in rows["decisions"] if s == ETH]
    assert eth == [(T1, T1, "ok"), (T1, T1 + 4 * MINUTE + LAG, "cancelled")]  # at the stop
    with Session(rig.db) as s:
        run = s.get(Run, run_id)
        assert run is not None and run.ended_at is None  # left open for a resume
        saved = repo.load_open_positions(s, run_id)[state.PORTFOLIO_KEY]
    assert saved["last_bar"][BTC] == (T1 + 3 * MINUTE).isoformat()  # the last poll's bar
    session.close()


# --- CLI --------------------------------------------------------------------------------------


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    db = make_engine(url)
    Base.metadata.create_all(db)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr("regimebot.cli.DEFAULT_KILL_FILE", tmp_path / "KILL")
    yield db
    structlog.reset_defaults()  # `paper run` configures JSON logging
    db.dispose()


def cli_rig(tmp_path: Path, db: Database) -> Rig:
    rig = make_rig(tmp_path)
    rig.db = db
    return rig


def test_cli_paper_status_shows_the_latest_paper_run_and_its_liveness(
    tmp_path: Path, cli_db: Database
) -> None:
    assert "no paper runs yet" in CliRunner().invoke(app, ["paper", "status"]).output
    rig = cli_rig(tmp_path, cli_db)
    session = rig.session()
    run_id = session.start()
    rig.exchange.fail.add(("fetch_candles", ETH))
    rig.poll(session, T1 + 4 * MINUTE)

    def heartbeat() -> datetime:
        with Session(cli_db) as s:
            beat = s.scalar(
                select(OpenPosition.updated_at).where(OpenPosition.symbol == "__portfolio__")
            )
        assert beat is not None
        return beat

    before = heartbeat()
    rig.poll(session, T1 + 5 * MINUTE)  # a poll without a decision still touches the row
    assert heartbeat() > before

    result = CliRunner().invoke(app, ["paper", "status"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith(f"paper run {run_id}: open (resume with `regimebot paper run")
    assert lines[1] == "last poll: 0 s ago"
    snapshot, cash = session.log.equity[-1][1], session.portfolio.cash  # at the T1 decision
    assert lines[2] == f"equity: {snapshot:.2f} at {T1.isoformat()} (cash {cash:.2f})"
    assert "halted: no" in lines and "open positions: 1" in lines
    assert any(line.startswith(f"  {BTC} long qty=25 entry=") for line in lines)
    assert f"  {BTC} no_trade at {T1.isoformat()}" in lines
    assert f"  {ETH} no_trade at {T1.isoformat()}" in lines
    assert lines[-2:] == [
        "last 1 risk events:",
        f"  {T1.isoformat()} stale_data -> skip_symbol: {ETH}: no features",
    ]

    with session_scope(cli_db) as s:  # the session hangs: no poll for an hour
        s.execute(update(OpenPosition).values(updated_at=datetime.now(UTC) - HOUR))
    hung = CliRunner().invoke(app, ["paper", "status"])
    assert hung.exit_code == 1 and "last poll: 3600 s ago" in hung.output
    assert "NOT POLLING: no poll in the last 210 s" in hung.output  # 3 x 10 s + 180 s
    assert CliRunner().invoke(app, ["paper", "status", "--max-age", "7200"]).exit_code == 0
    session.close()


def test_cli_paper_run_prints_the_banner_and_leaves_the_run_open(
    cli_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    clock = Clock(datetime.now(UTC))  # the CLI session reads the real clock; no data is that new
    candles = {s: candles_from_closes([100.0] * 10) for s in cfg.exchange.symbols}
    monkeypatch.setattr(
        "regimebot.cli.CcxtExchange", lambda _id: ReplayExchange(candles, {}, cfg.costs, clock)
    )
    monkeypatch.setattr(PaperSession, "run_forever", lambda self: None)
    result = CliRunner().invoke(app, ["paper", "run"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "PAPER TRADING - simulated fills only, no orders are sent to any exchange" in out
    assert "run id:    1 (new)" in out and "KILL" in out and "Ctrl-C or SIGTERM" in out
    assert "the kill switch or a 10% drawdown halt flattens and stops this run" in out
    assert "`regimebot paper clear-halt 1`" in out and "a drawdown halt ends the run" in out
    assert "regimebot paper run --resume 1" in out
    with Session(cli_db) as s:
        run = s.get(Run, 1)
        assert run is not None and (run.mode, run.ended_at) == ("paper", None)

    bad = CliRunner().invoke(app, ["paper", "run", "--resume", "99"])
    assert bad.exit_code == 2 and "unknown run_id 99" in bad.output


def test_cli_paper_clear_halt_clears_a_manual_kill_only_when_safe(
    tmp_path: Path, cli_db: Database
) -> None:
    rig = cli_rig(tmp_path, cli_db)
    session = rig.session()
    run_id = session.start()
    rig.poll(session, T1 + 5 * MINUTE)
    rig.kill_file.touch()
    rig.poll(session, T1 + 6 * MINUTE)  # halted and flattened
    assert session.risk.halted_reason == "manual_kill"

    def clear(run: int) -> Any:
        return CliRunner().invoke(app, ["paper", "clear-halt", str(run)])

    running = clear(run_id)
    assert running.exit_code == 1 and f"paper run {run_id} is already running" in running.output
    session.close()
    killed = clear(run_id)
    assert killed.exit_code == 1 and "regimebot unkill" in killed.output
    rig.kill_file.unlink()
    result = clear(run_id)
    assert result.exit_code == 0, result.output
    assert f"run {run_id}: cleared manual_kill" in result.output
    with Session(cli_db) as s:
        assert repo.load_open_positions(s, run_id)[state.RISK_KEY]["halts"] == []
    assert "cleared nothing" in clear(run_id).output
    with session_scope(cli_db) as s:
        backtest_id = repo.create_run(s, Mode.BACKTEST, rig.cfg)
    assert "not a paper run" in clear(backtest_id).output
