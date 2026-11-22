"""Loss limits for target positions: the daily-loss, weekly-loss and drawdown halts are checked on
mark-to-market equity at every hour boundary, in the engine (``on_minute``), so backtest and paper
fire them at the same bar; a halt flattens at the next 1m open and blocks as any halt does.

Markets: 1m bars moving in straight lines between hourly prices (each minute opens at the previous
close, so an exit at the next open fills at the hour's close). 2024-01-01 (T0) is a Monday; the
first decision is Thursday 4 January, BTC is held at weight 0.5 (50 BTC at 100 of 10 000), costs
are zero unless a test says otherwise, so every number below is exact.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import Clock, ReplayExchange
from synth import BTC, ETH, MINUTE, T0, candles_from_closes, funding_every_8h
from test_daily_trend_engine import FREE, TREND_YAML, Fixed

from regimebot.backtest.runner import BacktestResult, run_backtest
from regimebot.config import AppConfig, load_config
from regimebot.db.models import Base
from regimebot.db.session import make_engine
from regimebot.engine.loop import RunLog
from regimebot.paper import session as paper_session
from regimebot.paper.session import PaperSession
from regimebot.types import ExitReason, OrderSide

DAY, HOUR = timedelta(days=1), timedelta(hours=1)
LAG = timedelta(seconds=3)
DAYS = 9
START = T0 + 3 * DAY  # Thursday: lookbacks [2] and vol_window 2 are warm after 3 closes
THU, FRI, SAT, SUN, MON = (T0 + k * DAY for k in range(3, 8))


def cfg(**overrides: object) -> AppConfig:
    return load_config(
        TREND_YAML,
        overrides={
            "exchange.symbols": [BTC],
            "strategies.daily_trend.lookbacks": [2],
            "strategies.daily_trend.vol_window": 2,
            "paper.warmup_bars": 12,
            **FREE,
            **overrides,
        },
    )


def prices(day: datetime, move: Callable[[int], float], start: float = 100.0) -> np.ndarray:
    """Hourly prices from T0: ``start`` until ``day`` 00:00, ``move(h)`` for its 24 hours, then
    flat at ``move(24)``."""
    out = np.full(DAYS * 24 + 1, start)
    first = (day - T0) // HOUR
    out[first : first + 25] = [move(h) for h in range(25)]
    out[first + 25 :] = move(24)
    return out


def candles(hourly: np.ndarray) -> pd.DataFrame:
    minutes = np.concatenate([np.linspace(a, b, 61)[1:] for a, b in pairwise(hourly)])
    return candles_from_closes(minutes, volume=1e9)


def weights(*symbols_and_weights: tuple[str, float]) -> Fixed:
    days = [START + k * DAY for k in range(DAYS)]
    return Fixed({ts: dict(symbols_and_weights) for ts in days})


def backtest(config: AppConfig, market: dict[str, pd.DataFrame], tmp_path: Path) -> BacktestResult:
    """BTC held at 0.5 from Thursday, no funding (only its absence is recorded)."""
    w = weights((BTC, 0.5))
    return run_backtest(config, market, {}, START, T0 + DAYS * DAY, tmp_path / "K", strategies=[w])


def halts(log: RunLog) -> list[tuple[datetime, str]]:
    return [(e.ts, e.kind) for e in log.risk_events if e.action == "halt"]


def fills(log: RunLog) -> list[tuple[datetime, OrderSide, float, float]]:
    return [(f.ts, f.side, f.qty, f.price) for f in log.fills]


def test_an_intraday_loss_past_the_daily_limit_halts_at_that_hour_and_flattens(
    tmp_path: Path,
) -> None:
    # Friday falls 0.45 an hour for 10 hours: 50 x 0.45 = 22.5 lost per hour; the day's floor is
    # 10 000 x (1 - 1.5%) = 9 850, first crossed at 07:00 (9 842.5); -2.25% by 10:00
    result = backtest(
        cfg(), {BTC: candles(prices(FRI, lambda h: 100 - 0.45 * min(h, 10)))}, tmp_path
    )
    assert halts(result.log) == [(FRI + 7 * HOUR, "daily_loss")]
    kill = result.trades[0]
    assert (kill.exit_reason, kill.exit_ts, kill.exit_price) == (
        ExitReason.KILL,
        FRI + 7 * HOUR,  # at the open of the next 1m bar
        pytest.approx(96.85),
    )
    assert kill.net_pnl == pytest.approx(-157.5)
    # flat for the rest of Friday; the halt clears at Saturday's close-of-day roll: re-entry
    assert fills(result.log)[:3] == [
        (THU, OrderSide.BUY, 50.0, 100.0),
        (FRI + 7 * HOUR, OrderSide.SELL, 50.0, pytest.approx(96.85)),
        (SAT, OrderSide.BUY, pytest.approx(0.5 * 9_842.5 / 95.5), pytest.approx(95.5)),
    ]


def test_the_weekly_window_includes_sunday(tmp_path: Path) -> None:
    # Only the weekly limit (daily off): the week started Thursday's first check at 10 000, its
    # floor is 9 600. Sunday's fall crosses it at 18:00 (9 595), before Monday rolls the week.
    market = {BTC: candles(prices(SUN, lambda h: 100 - 0.45 * min(h, 20)))}
    result = backtest(cfg(**{"risk.daily_loss": 1.0}), market, tmp_path)
    assert halts(result.log) == [(SUN + 18 * HOUR, "weekly_loss")]
    assert [(t.exit_reason, t.exit_ts) for t in result.trades][:1] == [
        (ExitReason.KILL, SUN + 18 * HOUR)
    ]
    assert [f[0] for f in fills(result.log)][:3] == [THU, SUN + 18 * HOUR, MON]  # Monday re-enters


def test_the_drawdown_halt_fires_intraday_from_an_intraday_peak_and_stays(
    tmp_path: Path,
) -> None:
    # Friday rises to 102 by 02:00 (peak equity 10 100), then falls 0.6 an hour: a 3% drawdown
    # (equity 9 797) is first passed at 13:00 (9 770)
    def move(h: int) -> float:
        return 100 + h if h <= 2 else 102 - 0.6 * (h - 2)

    limits = {"risk.max_drawdown": 0.03, "risk.daily_loss": 1.0, "risk.weekly_loss": 1.0}
    result = backtest(cfg(**limits), {BTC: candles(prices(FRI, move))}, tmp_path)
    assert halts(result.log) == [(FRI + 13 * HOUR, "max_drawdown")]
    assert fills(result.log) == [
        (THU, OrderSide.BUY, 50.0, 100.0),
        (FRI + 13 * HOUR, OrderSide.SELL, 50.0, pytest.approx(95.4)),
    ]  # permanent: no order at any later decision


def test_a_normal_day_marks_every_hour_and_does_not_halt(tmp_path: Path) -> None:
    # Friday rises to 101 at noon and is back at 100 at its close: +-1%, no limit near
    result = backtest(
        cfg(), {BTC: candles(prices(FRI, lambda h: 101 - abs(h - 12) / 12))}, tmp_path
    )
    assert halts(result.log) == [] and result.trades[0].exit_reason == ExitReason.END_OF_DATA
    # the noon high was marked: Saturday's snapshot is 50 below the 10 050 peak
    ((_, equity, _, drawdown),) = [row for row in result.log.equity if row[0] == SAT]
    assert (equity, drawdown) == (10_000.0, pytest.approx(50 / 10_050))


# --- paper: the same halt at the same bar -----------------------------------------------------


@pytest.fixture(autouse=True)
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(paper_session, "alert", lambda kind, detail: sent.append((kind, detail)))
    return sent


def two_symbol_market() -> tuple[AppConfig, dict[str, pd.DataFrame]]:
    """BTC long 0.5 falls through Friday as in the daily-loss test, ETH short 0.3 stays at 50;
    real costs and funding."""
    config = load_config(
        TREND_YAML,
        overrides={
            "exchange.symbols": [BTC, ETH],
            "strategies.daily_trend.lookbacks": [2],
            "strategies.daily_trend.vol_window": 2,
            "paper.warmup_bars": 12,
        },
    )
    btc = candles(prices(FRI, lambda h: 100 - 0.45 * min(h, 10)))
    return config, {BTC: btc, ETH: candles(np.full(DAYS * 24 + 1, 50.0))}


def paper_run(
    tmp_path: Path, config: AppConfig, market: dict[str, pd.DataFrame], kill_at: datetime | None
) -> PaperSession:
    """Poll LAG after every hour (a bar is visible 2 s after its close) to the last decision;
    touch the kill file just before the poll at ``kill_at``."""
    funding = {s: funding_every_8h(T0, 3 * DAYS, seed=i) for i, s in enumerate(market)}
    clock = Clock(START + LAG)
    exchange = ReplayExchange(market, funding, config.costs, clock)
    db = make_engine(f"sqlite:///{tmp_path / 'paper.db'}")
    Base.metadata.create_all(db)
    w = weights((BTC, 0.5), (ETH, -0.3))
    session = PaperSession(config, exchange, db, tmp_path / "KILL", clock=clock, strategies=[w])
    session.start()
    while clock.now <= T0 + (DAYS - 1) * DAY + LAG:
        if kill_at is not None and clock.now == kill_at + LAG:
            (tmp_path / "KILL").touch()
        session.run_once()
        clock.now += HOUR
    session.close()
    return session


def unlagged(log: RunLog, end: datetime) -> dict[str, object]:
    def unlag(ts: datetime) -> datetime:
        return ts - LAG if (ts - T0) % MINUTE == LAG else ts

    return {
        "decisions": [(i, ts, d) for i, ts, d in log.decisions if ts < end],
        "fills": [(unlag(f.ts), f.side, f.qty, f.price, f.fee) for f in log.fills if f.ts < end],
        "trades": [
            (t.symbol, unlag(t.exit_ts), t.exit_reason, t.net_pnl)
            for t in log.trades
            if t.exit_ts < end
        ],
        "equity": [e for e in log.equity if e[0] < end],
        "halts": halts(log),
    }


def test_paper_fires_the_intraday_halt_at_the_same_bar_and_fills_like_the_backtest(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    config, market = two_symbol_market()
    funding = {s: funding_every_8h(T0, 3 * DAYS, seed=i) for i, s in enumerate(market)}
    last = T0 + (DAYS - 1) * DAY
    w = weights((BTC, 0.5), (ETH, -0.3))
    ref = run_backtest(
        config, market, funding, START, last + MINUTE, tmp_path / "BT", strategies=[w]
    ).log
    paper = paper_run(tmp_path, config, market, kill_at=None).log
    assert halts(ref) and halts(ref)[0][0].date() == FRI.date()  # it does fire intraday
    assert {t.exit_reason for t in ref.trades if t.exit_ts.date() == FRI.date()} == {
        ExitReason.KILL
    }
    assert unlagged(paper, last) == unlagged(ref, last)
    assert halts(ref)[0][1] == "daily_loss" and ("daily_loss" in {k for k, _ in alerts})


def test_the_kill_file_flattens_target_positions_between_decisions_in_paper(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    config, market = two_symbol_market()
    market = {s: candles(np.full(DAYS * 24 + 1, p)) for s, p in ((BTC, 100.0), (ETH, 50.0))}
    session = paper_run(tmp_path, config, market, kill_at=FRI + 7 * HOUR)
    log = session.log
    # the poll's hourly check (07:00 bar) sees the kill file first; the exits fill on that poll
    assert halts(log) == [(FRI + 7 * HOUR, "manual_kill")]
    assert [(t.symbol, t.exit_reason, t.exit_ts) for t in log.trades] == [
        (BTC, ExitReason.KILL, FRI + 7 * HOUR + LAG),
        (ETH, ExitReason.KILL, FRI + 7 * HOUR + LAG),
    ]
    assert session.portfolio.positions == {}
    later = [d for _, ts, d in log.decisions if ts > FRI + 7 * HOUR and d.approved]
    assert later == [] and all(f.ts <= FRI + 7 * HOUR + LAG for f in log.fills)
    assert ("manual_kill", f"kill file {tmp_path / 'KILL'} exists") in alerts
