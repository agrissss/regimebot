"""Daily trend target orders that keep working (``strategies.daily_trend.fill_minutes``): the
unfilled rest fills at each later base bar's open, every slice capped by ``max_participation`` of
its bar and priced like the first fill, until the target is reached or the window ends; a halt,
the kill file, a new decision and ``risk.min_notional`` end it; backtest and paper run the same
engine (paper sizing each slice from the last closed bar, one bar behind the backtest);
``fill_minutes`` 1 (the default) is the previous single fill.

Engine tests: ``config/trend.yaml`` on BTC, zero costs unless a test says otherwise, 10 000
equity, weight 0.5 at 100 = a 50 BTC target, max participation 5%, min notional 10.
"""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import Clock, ReplayExchange
from pydantic import ValidationError
from synth import BTC, ETH, MINUTE, candles_from_closes
from test_daily_trend_engine import D1, D2, D3, FREE, Fixed, approx, decide, decisions, trend_cfg
from test_daily_trend_paper import LAG, START, market, unlag
from test_daily_trend_paper import cfg as paper_cfg

from regimebot.backtest.runner import run_backtest
from regimebot.config import (
    REPO_ROOT,
    AppConfig,
    config_hash,
    config_json,
    load_config,
    with_overrides,
)
from regimebot.db import repo
from regimebot.db.models import Base
from regimebot.db.session import make_engine, session_scope
from regimebot.engine.loop import Engine, RunLog
from regimebot.engine.portfolio import Portfolio
from regimebot.execution.fill_model import market_fill
from regimebot.paper import session as paper_session
from regimebot.paper.session import PaperSession
from regimebot.risk.manager import RiskManager
from regimebot.types import Bar, ExitReason, OrderSide

BUY = OrderSide.BUY


def engine(
    tmp_path: Path,
    fill_minutes: int,
    weights: dict[datetime, dict[str, float]] | None = None,
    costs: dict[str, float] = FREE,
) -> Engine:
    cfg = trend_cfg(
        **{"exchange.symbols": [BTC], "strategies.daily_trend.fill_minutes": fill_minutes, **costs}
    )
    return Engine(
        cfg,
        [Fixed(weights or {D1: {BTC: 0.5}})],
        RiskManager(cfg, tmp_path / "KILL"),
        Portfolio(cfg.backtest.initial_equity),
        RunLog(),
    )


def offer(engine: Engine, ts: datetime, volume: float, price: float = 100.0) -> None:
    engine.fill_queued(BTC, ts, price, price, volume)


def slices(engine: Engine) -> list[tuple[datetime, float]]:
    return [(f.ts, f.qty) for f in engine.log.fills]


# --- config ----------------------------------------------------------------------------------

# the hashes runs were stored under before fill_minutes existed (paper resumes check them)
STORED_HASHES = {
    "default.yaml": "128cd0ef746fcece",
    "trend.yaml": "a64d94bfba4fc61b",
    "kraken.yaml": "f5467cff7f15beda",
    "kraken_trend_longflat.yaml": "d54cd649595ccc54",
    "kraken_playbook_15m.yaml": "9595e9870b558a2f",
    "kraken_breakout_1h.yaml": "5910eb647997586a",
}


def test_fill_minutes_is_bounded_and_left_out_of_the_stored_config_at_its_default() -> None:
    for bad in (0, 1441):
        with pytest.raises(ValidationError):
            trend_cfg(**{"strategies.daily_trend.fill_minutes": bad})
    hashes = {name: config_hash(load_config(REPO_ROOT / "config" / name)) for name in STORED_HASHES}
    assert hashes == STORED_HASHES
    assert "fill_minutes" not in config_json(trend_cfg())
    working = trend_cfg(**{"strategies.daily_trend.fill_minutes": 30})
    assert '"fill_minutes":30' in config_json(working)
    assert config_hash(working) != STORED_HASHES["trend.yaml"]
    # a config stored without it validates back to the same config
    stored = json.loads(config_json(trend_cfg())) | {"database_url": "sqlite://"}
    assert AppConfig.model_validate(stored) == trend_cfg(**{"database_url": "sqlite://"})


# --- engine ----------------------------------------------------------------------------------


def test_a_working_order_fills_a_slice_at_each_bar_priced_like_the_first_fill(
    tmp_path: Path,
) -> None:
    e = engine(tmp_path, 30, costs={})  # trend.yaml's real costs
    decide(e, D1, 100.0)
    assert e.working_until(BTC) == D1 + 30 * MINUTE
    quotes = [(99.99, 100.01), (100.49, 100.51), (99.49, 99.51), (100.99, 101.01), (98.0, 98.02)]
    for k, (bid, ask) in enumerate([*quotes, (97.0, 97.02)]):
        e.fill_queued(BTC, D1 + k * MINUTE, bid, ask, 200.0)  # 5% of 200 = 10 BTC a slice
    # each slice is the target fill of what is left, at that bar: capped, ref the decision close
    rest = 50.0
    for k, (f, (bid, ask)) in enumerate(zip(e.log.fills, quotes, strict=True)):
        expected = market_fill(
            BTC, D1 + k * MINUTE, BUY, rest, bid, ask, 100.0, 200.0, e.cfg.costs, True
        )
        assert f == expected and f.qty == 10.0
        rest -= f.qty
    position = e.portfolio.positions[BTC]
    assert position.qty == approx(50.0) and not e.has_queued_order(BTC)  # the sixth: nothing
    assert position.entry_price == approx(sum(f.qty * f.price for f in e.log.fills) / 50)
    assert e.portfolio.cash == approx(10_000.0 - sum(f.fee for f in e.log.fills))
    assert decisions(e) == [(D1, "ok", 50.0)]  # a completed order leaves no closing row


def test_each_slice_is_capped_by_its_own_bar_and_never_overfills(tmp_path: Path) -> None:
    e = engine(tmp_path, 30)
    decide(e, D1, 100.0)
    for k, volume in enumerate([100.0, 0.0, 1.0, 300.0, 1e9]):
        offer(e, D1 + k * MINUTE, volume)
    # 5, none (no volume), 0.05 worth 5 < min notional 10 (skipped, still working), 15, the 30 left
    assert slices(e) == [(D1, 5.0), (D1 + 3 * MINUTE, 15.0), (D1 + 4 * MINUTE, 30.0)]
    assert e.portfolio.signed_qty(BTC) == 50.0 and not e.has_queued_order(BTC)
    assert decisions(e) == [(D1, "ok", 50.0)]


def test_the_rest_is_dropped_with_no_liquidity_when_the_window_ends(tmp_path: Path) -> None:
    e = engine(tmp_path, 3)
    decide(e, D1, 100.0)
    for k in range(5):
        offer(e, D1 + k * MINUTE, 100.0)  # 5 BTC a slice
    # the bars opening at D1, +1 and +2 minutes; the last one in the window drops the 35 left
    assert slices(e) == [(D1 + k * MINUTE, 5.0) for k in range(3)]
    assert decisions(e) == [(D1, "ok", 50.0), (D1 + 2 * MINUTE, "no_liquidity", 0.0)]
    assert "35 of target +50 left after the 3-minute fill window" in e.log.decisions[-1][2].detail
    assert e.portfolio.signed_qty(BTC) == 15.0 and not e.has_queued_order(BTC)
    # bars missing in the window: the next offer after it drops the rest without a fill
    gap = engine(tmp_path, 3)
    decide(gap, D1, 100.0)
    offer(gap, D1, 100.0)
    offer(gap, D1 + 5 * MINUTE, 100.0)
    assert slices(gap) == [(D1, 5.0)] and decisions(gap)[-1] == (D1 + 5 * MINUTE, "no_liquidity", 0)
    # a slice that gets nothing on the last bar of the window also ends it
    thin = engine(tmp_path, 2)
    decide(thin, D1, 100.0)
    offer(thin, D1, 0.0)
    assert thin.has_queued_order(BTC)
    offer(thin, D1 + MINUTE, 0.0)
    assert slices(thin) == [] and decisions(thin)[-1][1:] == ("no_liquidity", 0.0)


def test_the_kill_file_mid_fill_stops_further_slices(tmp_path: Path) -> None:
    e = engine(tmp_path, 30)
    decide(e, D1, 100.0)
    offer(e, D1, 100.0)
    offer(e, D1 + MINUTE, 100.0)
    (tmp_path / "KILL").touch()  # the risk manager's rule: no order while the kill file exists
    offer(e, D1 + 2 * MINUTE, 100.0)
    offer(e, D1 + 3 * MINUTE, 100.0)
    assert slices(e) == [(D1, 5.0), (D1 + MINUTE, 5.0)] and not e.has_queued_order(BTC)
    assert decisions(e)[-1] == (D1 + 2 * MINUTE, "kill_switch", 0.0)


def test_a_halt_mid_fill_cancels_the_rest_and_flattens(tmp_path: Path) -> None:
    e = engine(tmp_path, 120)
    decide(e, D1, 100.0)
    for k in range(60):
        offer(e, D1 + k * MINUTE, 10.0)  # 0.5 BTC a slice: 30 BTC by 00:59
    # the hour check marks 30 BTC at 60: equity 8 800, a 12% drawdown: halted, KILL exit queued
    e.on_minute(Bar(BTC, D1 + 60 * MINUTE, 60.0, 60.0, 60.0, 60.0, 10.0))
    assert e.risk.halted_reason == "max_drawdown" and not e.has_queued_order(BTC)
    assert decisions(e)[-1] == (D1 + 60 * MINUTE, "kill_switch", 0.0)
    offer(e, D1 + 60 * MINUTE, 10.0, price=60.0)  # the exit fills, no slice
    offer(e, D1 + 61 * MINUTE, 10.0, price=60.0)
    assert e.portfolio.positions == {} and len(e.log.fills) == 61
    assert (e.log.trades[-1].exit_reason, e.log.trades[-1].qty) == (ExitReason.KILL, approx(30))


def test_a_new_decision_replaces_a_working_order(tmp_path: Path) -> None:
    e = engine(tmp_path, 1440, {D1: {BTC: 0.5}, D2: {BTC: 0.3}})
    decide(e, D1, 100.0)
    for k in range(3):
        offer(e, D1 + k * MINUTE, 100.0)  # 15 of 50, then no more bars before D2
    decide(e, D2, 100.0)  # 0.3 x 10 000 / 100 = 30, traded from the 15 held
    assert decisions(e)[-2:] == [(D2, "expired", 0.0), (D2, "ok", 15.0)]
    assert "replaced" in e.log.decisions[-2][2].detail
    offer(e, D2, 1e9)
    assert e.portfolio.signed_qty(BTC) == 30.0 and not e.has_queued_order(BTC)


def test_a_flip_closes_in_full_then_its_opening_leg_works_as_capped_adds(tmp_path: Path) -> None:
    e = engine(tmp_path, 30, {D1: {BTC: 0.5}, D2: {BTC: -0.3}})
    decide(e, D1, 100.0)
    offer(e, D1, 1e9)  # long 50
    decide(e, D2, 100.0)  # target -30: one order of 80 through zero
    for k, volume in enumerate([200.0, 200.0, 400.0, 1e9]):
        offer(e, D2 + k * MINUTE, volume)
    # the closing 50 in full plus an opening 5% of 200 = 10; then adds of 10 and the 10 left (a
    # 20 cap); nothing past the target
    sells = [(f.ts, f.side, f.qty) for f in e.log.fills[1:]]
    assert sells == [
        (D2, OrderSide.SELL, 60.0),
        (D2 + MINUTE, OrderSide.SELL, 10.0),
        (D2 + 2 * MINUTE, OrderSide.SELL, 10.0),
    ]
    assert e.portfolio.signed_qty(BTC) == -30.0 and not e.has_queued_order(BTC)
    assert len(e.log.trades) == 1 and e.log.trades[0].qty == 50.0  # the long, closed by the flip
    assert decisions(e) == [(D1, "ok", 50.0), (D2, "ok", 80.0)]


def test_a_dust_rest_under_min_notional_ends_the_working_order(tmp_path: Path) -> None:
    e = engine(tmp_path, 30)
    decide(e, D1, 100.0)
    offer(e, D1, 999.0)  # 5% = 49.95 BTC: 0.05 left, worth 5 < 10
    assert not e.has_queued_order(BTC) and e.working_until(BTC) is None
    offer(e, D1 + MINUTE, 1e9)
    assert slices(e) == [(D1, approx(49.95))] and decisions(e) == [(D1, "ok", 50.0)]


def test_fill_minutes_1_is_one_fill_per_decision_as_before(tmp_path: Path) -> None:
    assert trend_cfg().strategies.daily_trend.fill_minutes == 1  # the default
    e = engine(tmp_path, 1)
    decide(e, D1, 100.0)
    assert e.working_until(BTC) is None
    offer(e, D1, 200.0)  # 10 of 50: the rest waits for the next decision, with no row
    offer(e, D1 + MINUTE, 1e9)
    assert slices(e) == [(D1, 10.0)] and decisions(e) == [(D1, "ok", 50.0)]
    none = engine(tmp_path, 1)
    decide(none, D1, 100.0)
    offer(none, D1, 0.0)  # no volume at the first bar: cancelled at once
    assert decisions(none)[-1] == (D1, "no_liquidity", 0.0) and not none.has_queued_order(BTC)


# --- backtest --------------------------------------------------------------------------------


def thin_backtest(tmp_path: Path, fill_minutes: int, volume: float) -> RunLog:
    """BTC flat at 100 for three days, target 0.5 at D1 and D2, decisions D1 to D3."""
    cfg = trend_cfg(
        **{"exchange.symbols": [BTC], **FREE, "strategies.daily_trend.fill_minutes": fill_minutes}
    )
    candles = {BTC: candles_from_closes(np.full(3 * 1440, 100.0), volume=volume)}
    w = Fixed({D1: {BTC: 0.5}, D2: {BTC: 0.5}})
    return run_backtest(cfg, candles, {}, D1, D3 + MINUTE, tmp_path / "K", strategies=[w]).log


def test_the_backtest_offers_a_working_order_every_bar_of_its_window(tmp_path: Path) -> None:
    buys = [(f.ts, f.qty) for f in thin_backtest(tmp_path, 30, 200.0).fills if f.side == BUY]
    assert buys == [(D1 + k * MINUTE, 10.0) for k in range(5)]  # 50 BTC by 00:04
    buys = [(f.ts, f.qty) for f in thin_backtest(tmp_path, 1, 200.0).fills if f.side == BUY]
    assert buys == [(D1, 10.0), (D2, 10.0)]  # the previous behaviour: one fill a decision
    # a window across the hour check: one slice a bar, not two at 01:00 (0.5 BTC each)
    log = thin_backtest(tmp_path, 90, 10.0)
    first_day = [f.ts for f in log.fills if f.side == BUY and f.ts < D2]
    assert first_day == [D1 + k * MINUTE for k in range(90)]
    assert [(ts, d.rule) for _, ts, d in log.decisions][:2] == [
        (D1, "ok"),
        (D1 + 89 * MINUTE, "no_liquidity"),
    ]


# --- paper -----------------------------------------------------------------------------------

FILL_MINUTES = 5
SECOND = timedelta(seconds=1)
LAST = START + timedelta(days=5)  # the last decision compared


@pytest.fixture(autouse=True)
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(paper_session, "alert", lambda kind, detail: sent.append((kind, detail)))
    return sent


def thin_cfg() -> AppConfig:
    """``test_daily_trend_paper``'s config with a 5-minute fill window and no loss limits."""
    limits_off = {"risk.daily_loss": 1.0, "risk.weekly_loss": 1.0, "risk.max_drawdown": 1.0}
    return with_overrides(
        paper_cfg(), {"strategies.daily_trend.fill_minutes": FILL_MINUTES, **limits_off}
    )


def thin_market(varying: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """``test_daily_trend_paper``'s market with 20 a bar: 5% is 1 BTC or ETH, so most orders
    need many slices and many end with their window. ``varying``: 10 and 30 a bar in turn."""
    candles, funding = market()
    volume = (lambda n: np.where(np.arange(n) % 2 == 0, 10.0, 30.0)) if varying else None
    return {
        s: df.assign(volume=20.0 if volume is None else volume(len(df)))
        for s, df in candles.items()
    }, funding


class Rig:
    def __init__(self, tmp_path: Path, now: datetime, varying: bool = False) -> None:
        self.cfg = thin_cfg()
        candles, funding = thin_market(varying)
        self.clock = Clock(now)
        self.exchange = ReplayExchange(candles, funding, self.cfg.costs, self.clock)
        self.db = make_engine(f"sqlite:///{tmp_path / 'paper.db'}")
        Base.metadata.create_all(self.db)
        self.kill_file = tmp_path / "KILL"

    def session(self) -> PaperSession:
        return PaperSession(self.cfg, self.exchange, self.db, self.kill_file, clock=self.clock)

    def poll(self, session: PaperSession, day: datetime, minutes: range) -> None:
        """Poll ``LAG`` and 30 s after each minute in ``minutes`` after the close ``day`` (the
        second poll sees no new bar: no slice)."""
        for k in minutes:
            for after in (LAG, 30 * SECOND):
                self.clock.now = day + k * MINUTE + after
                session.run_once()

    def stored_fills(self, run_id: int) -> list[str]:
        with session_scope(self.db) as s:
            return [repr(f) for f in repo.load_fills(s, run_id)]


def unlagged(log: RunLog, end: datetime) -> dict[str, object]:
    return {
        "decisions": [(i, unlag(ts), d) for i, ts, d in log.decisions if ts < end],
        "fills": [repr(replace(f, ts=unlag(f.ts))) for f in log.fills if f.ts < end],
        "trades": [
            repr(replace(t, entry_ts=unlag(t.entry_ts), exit_ts=unlag(t.exit_ts)))
            for t in log.trades
            if t.exit_ts < end
        ],
        "equity": [e for e in log.equity if e[0] < end],
        "risk_events": log.risk_events,
    }


def test_at_constant_volume_paper_works_an_order_like_the_backtest(tmp_path: Path) -> None:
    """Constant bar volume: paper reproduces the backtest row for row (live fills LAG later).
    With varying volume it does not, by one bar (next test)."""
    rig = Rig(tmp_path, START + LAG)
    session = rig.session()
    session.start()
    for k in range((LAST - START).days + 1):
        day = START + k * timedelta(days=1)
        rig.poll(session, day, range(FILL_MINUTES + 2))  # each minute of the window, and after
    candles, funding = thin_market()
    ref = run_backtest(thin_cfg(), candles, funding, START, LAST + MINUTE, tmp_path / "BT").log
    assert unlagged(session.log, LAST) == unlagged(ref, LAST)
    # the orders did work over several minutes, and some ended with their window
    first = [f for f in ref.fills if START <= f.ts < START + timedelta(days=1)]
    assert len(first) == 2 * FILL_MINUTES  # BTC and ETH, a slice at each minute of the window
    assert sum(d.rule == "no_liquidity" for _, _, d in ref.decisions) >= 5


def test_paper_sizes_each_slice_one_bar_behind_the_backtest(tmp_path: Path) -> None:
    """The known offset (ADR 0004, ADR 0006): slice k fills at the bar opening at START + k
    minutes in both, priced alike; the backtest caps it at 5% of that bar's volume, paper at 5% of
    the last closed bar's, the one that closed at START + k minutes."""
    rig = Rig(tmp_path, START + LAG, varying=True)
    session = rig.session()
    session.start()
    rig.poll(session, START, range(FILL_MINUTES + 2))
    candles, funding = thin_market(varying=True)
    day = START + timedelta(days=1)
    ref = run_backtest(thin_cfg(), candles, funding, START, day + MINUTE, tmp_path / "BT").log
    for symbol in (BTC, ETH):
        volume = candles[symbol]["volume"]
        paper = [(unlag(f.ts), f.qty, f.bid) for f in session.log.fills if f.symbol == symbol]
        back = [(f.ts, f.qty, f.bid) for f in ref.fills if f.symbol == symbol and f.ts < day]
        bars = [START + k * MINUTE for k in range(FILL_MINUTES)]
        assert [ts for ts, _, _ in paper] == [ts for ts, _, _ in back] == bars
        assert [bid for *_, bid in paper] == [bid for *_, bid in back]  # the same quotes
        assert [q for _, q, _ in paper] == [approx(0.05 * volume[ts]) for ts in bars]
        assert [q for _, q, _ in back] == [approx(0.05 * volume[ts + MINUTE]) for ts in bars]
        assert {round(q, 9) for _, q, _ in paper + back} == {0.5, 1.5}  # 5% of 10 and of 30
        assert all(p != b for (_, p, _), (_, b, _) in zip(paper, back, strict=True))


def test_a_restart_mid_fill_drops_the_working_order_without_filling_twice(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, START + LAG)
    first = rig.session()
    run_id = first.start()
    rig.poll(first, START, range(3))  # the decision and two more slices per symbol
    filled = rig.stored_fills(run_id)
    held = {s: first.portfolio.signed_qty(s) for s in first.portfolio.positions}
    assert len(filled) == 6 and first.engine.working_until(BTC) is not None
    first.close()  # a hard stop: the working orders are not saved
    rig.clock.now = START + 3 * MINUTE + LAG
    resumed = rig.session()
    resumed.start(resume_run_id=run_id)
    assert resumed.engine.working_until(BTC) is None
    rig.poll(resumed, START, range(3, FILL_MINUTES + 2))
    assert rig.stored_fills(run_id) == filled  # nothing more until the next decision
    assert {s: resumed.portfolio.signed_qty(s) for s in resumed.portfolio.positions} == held
    rig.poll(resumed, START + timedelta(days=1), range(1))  # trades from the restored position
    assert len(rig.stored_fills(run_id)) > len(filled)
