"""Backtest runner end to end: fill timing, a hand-computed trade, funding, accounting
identity, no look-ahead, persistence and the CLI.

Stub-strategy runs use ``open_cfg``: every regime's playbook allows "breakout" at full size,
so a ``Scripted("breakout")`` trades whatever regime the synthetic features produce.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from synth import (
    BTC,
    ETH,
    MINUTE,
    SOL,
    T0,
    Scripted,
    candles_from_closes,
    funding_every_8h,
    regime_walk,
)
from typer.testing import CliRunner

from regimebot.backtest.runner import BacktestResult, persist_result, run_backtest
from regimebot.cli import app
from regimebot.config import AppConfig, load_config
from regimebot.db import repo
from regimebot.db.models import Base, RiskDecisionRow, Run, SignalRow
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.types import ExitReason, OrderSide, Regime

HOUR = timedelta(hours=1)


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-12, abs=1e-9)


def open_cfg(symbols: tuple[str, ...] = (BTC,)) -> AppConfig:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    overrides: dict[str, object] = {f"playbook.{r.value}": entry for r in Regime}
    return load_config(overrides={"exchange.symbols": list(symbols), **overrides})


def flat(minutes: int, volume: float = 1e6) -> pd.DataFrame:
    """1m bars at 100 with 0.05 wicks: they touch neither a 98 stop nor a 104 target."""
    return candles_from_closes([100.0] * minutes, volume=volume)


def stub_run(
    tmp_path: Path,
    candles: pd.DataFrame,
    signal_at: datetime,
    funding: pd.Series | None = None,
    start: datetime = T0,
) -> BacktestResult:
    """BTC backtest over all of ``candles`` with one long signal (stop -2, target +4) at
    ``signal_at``."""
    end = candles.index[-1].to_pydatetime() + MINUTE
    return run_backtest(
        open_cfg(),
        {BTC: candles},
        {} if funding is None else {BTC: funding},
        start,
        end,
        tmp_path / "KILL",
        strategies=[Scripted("breakout", at={signal_at})],
    )


def test_entry_fills_at_the_open_of_the_first_base_bar_after_the_decision(tmp_path: Path) -> None:
    candles = flat(120)
    t = T0 + HOUR  # decision close, price 100
    candles.loc[t + MINUTE, ["open", "high"]] = [101.0, 101.05]  # the next bar opens at 101
    result = stub_run(tmp_path, candles, t)
    entry = result.log.fills[0]
    assert (entry.ts, entry.side, entry.ref_price) == (t, OrderSide.BUY, 100.0)
    # ask 101 * (1 + 1 bp); 25 units of 1e6 volume: slippage 2 + impact 0.0025 bps
    assert entry.price == approx(101.0 * 1.0001 * (1 + 2.0025e-4))
    assert result.log.signals[0].ts == t and result.trades[0].entry_ts == t


def test_a_position_is_checked_against_the_bar_it_was_filled_on(tmp_path: Path) -> None:
    candles = flat(120)
    t = T0 + HOUR
    candles.loc[t + MINUTE, "low"] = 97.5  # the fill bar itself trades through the 98 stop
    result = stub_run(tmp_path, candles, t)
    (trade,) = result.trades
    assert (trade.entry_ts, trade.exit_ts, trade.exit_reason) == (t, t + MINUTE, ExitReason.STOP)


def test_scripted_long_hits_its_target_with_hand_checked_accounting(tmp_path: Path) -> None:
    candles = flat(240)
    t = T0 + HOUR
    candles.loc[t + MINUTE, "volume"] = 2500.0  # entry bar: 25 units = 1% participation
    candles.loc[t + 21 * MINUTE, "high"] = 104.5  # trades through the 104 target
    result = stub_run(tmp_path, candles, t, start=T0 + 30 * MINUTE)

    # Sizing: risk 0.005 * 10 000 = 50; stop distance 100 - 98 = 2  ->  qty 25.
    # Entry at the next bar's open 100: ask = 100 * (1 + 1 bp) = 100.01; slippage 2 bps +
    # impact 1 bp (1% of the bar) = 3 bps -> 100.01 * 1.0003 = 100.040003.
    # Entry fee (taker 5 bps): 25 * 100.040003 * 0.0005 = 1.2505000375.
    # Exit at the 104 target (maker 2 bps, no slippage): fee 25 * 104 * 0.0002 = 0.52.
    # Gross: 25 * (104 - 100.040003) = 98.999925.
    # Net: 98.999925 - 1.2505000375 - 0.52 = 97.2294249625; final equity 10 097.2294249625.
    # R: 97.2294249625 / (25 * (100.040003 - 98)) = 97.2294249625 / 51.000075.
    entry, exit_ = result.log.fills
    assert (entry.qty, entry.price, entry.fee) == (
        approx(25),
        approx(100.040003),
        approx(1.2505000375),
    )
    assert (exit_.price, exit_.fee, exit_.liquidity) == (104.0, approx(0.52), "maker")
    (trade,) = result.trades
    assert (trade.exit_reason, trade.entry_ts, trade.exit_ts) == (
        ExitReason.TARGET,
        t,
        t + 21 * MINUTE,
    )
    assert trade.gross_pnl == approx(98.999925)
    assert trade.fees == approx(1.7705000375)
    assert trade.net_pnl == approx(97.2294249625)
    assert trade.r_multiple == approx(97.2294249625 / 51.000075)
    assert trade.bars_held == 1  # one decision (t + 15m) while open
    assert result.equity.iloc[-1] == approx(10_097.2294249625)
    assert round(result.equity.iloc[-1], 2) == 10_097.23
    # decisions start at ``start``; earlier bars were warm-up only
    assert min(r[1] for r in result.log.regimes) == T0 + 30 * MINUTE == result.equity.index[0]


def test_symbols_are_replayed_together_in_time_order(tmp_path: Path) -> None:
    # Both enter at t; ETH (second in symbol order) is stopped first. Risk must see the trades
    # close in time order (loss streaks, day boundaries), not symbol by symbol.
    t = T0 + HOUR
    candles = {BTC: flat(120), ETH: flat(120)}
    candles[ETH].loc[t + 3 * MINUTE, "low"] = 97.5
    candles[BTC].loc[t + 9 * MINUTE, "low"] = 97.5
    result = run_backtest(
        open_cfg((BTC, ETH)),
        candles,
        {},
        T0,
        T0 + 2 * HOUR,
        tmp_path / "KILL",
        strategies=[Scripted("breakout", at={t})],
    )
    assert [(tr.symbol, tr.exit_ts) for tr in result.trades] == [
        (ETH, t + 3 * MINUTE),
        (BTC, t + 9 * MINUTE),
    ]


def funding_at(*stamps_and_rates: tuple[datetime, float]) -> pd.Series:
    stamps, rates = zip(*stamps_and_rates, strict=True)
    return pd.Series(rates, index=pd.DatetimeIndex(stamps, name="ts"), name="rate")


def test_funding_is_charged_for_a_position_held_across_a_funding_time(tmp_path: Path) -> None:
    candles = flat(10 * 60)
    t = T0 + 7 * HOUR  # entry decision; the fill is stamped 07:00 too
    candles.loc[T0 + 8 * HOUR, "close"] = 100.5  # mark for the 08:00 funding
    candles.loc[T0 + 9 * HOUR, "high"] = 104.5  # target at 09:00
    # 00:00 and 07:00 come before the position was held, 16:00 after it closed
    rates = funding_at((T0, 0.5), (t, 0.5), (T0 + 8 * HOUR, 0.0001), (T0 + 16 * HOUR, 0.5))
    result = stub_run(tmp_path, candles, t, funding=rates)
    (trade,) = result.trades
    assert trade.exit_ts == T0 + 9 * HOUR
    assert trade.funding == approx(25 * 100.5 * 0.0001)  # qty * 08:00 close * rate
    assert trade.net_pnl == approx(trade.gross_pnl - trade.fees - trade.funding)
    assert result.equity.iloc[-1] == approx(10_000 + trade.net_pnl)


def test_a_position_stopped_in_the_bar_closing_at_a_funding_time_pays_none(
    tmp_path: Path,
) -> None:
    candles = flat(10 * 60)
    t = T0 + 7 * HOUR
    candles.loc[T0 + 8 * HOUR, "low"] = 97.5  # stopped in the bar closing at 08:00
    result = stub_run(tmp_path, candles, t, funding=funding_at((T0 + 8 * HOUR, 0.5)))
    (trade,) = result.trades
    assert (trade.exit_ts, trade.exit_reason, trade.funding) == (
        T0 + 8 * HOUR,
        ExitReason.STOP,
        0.0,
    )


# --- real strategies on random data -----------------------------------------------------------

DAYS = 20
# 7 minutes after the 07:30 decision that enters ETH (open until 10:37): see test_no_look_ahead
CUT = datetime(2024, 1, 13, 7, 37, tzinfo=UTC)


def walk_inputs(seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    candles = {
        BTC: regime_walk(DAYS * 1440, seed=seed),
        ETH: regime_walk(DAYS * 1440, seed=seed + 1, start_price=50.0),
    }
    funding = {s: funding_every_8h(T0, DAYS * 3, seed=seed + i) for i, s in enumerate(candles)}
    return candles, funding


def walk_run(
    tmp_path: Path, candles: dict[str, pd.DataFrame], funding: dict[str, pd.Series]
) -> BacktestResult:
    cfg = load_config(overrides={"exchange.symbols": [BTC, ETH]})
    return run_backtest(cfg, candles, funding, T0, T0 + timedelta(days=DAYS), tmp_path / "KILL")


@pytest.fixture(scope="module")
def walk(tmp_path_factory: pytest.TempPathFactory) -> BacktestResult:
    return walk_run(tmp_path_factory.mktemp("walk"), *walk_inputs(seed=1))


def test_random_walk_run_trades_both_symbols_with_every_strategy(walk: BacktestResult) -> None:
    assert {t.symbol for t in walk.trades} == {BTC, ETH}
    assert {t.strategy for t in walk.trades} == {"breakout", "trend", "mean_reversion"}
    assert len(walk.trades) >= 10


def test_final_equity_is_initial_equity_plus_the_sum_of_net_pnl(walk: BacktestResult) -> None:
    assert walk.log.equity[-1][2] == 0.0  # flat at the end: equity is all cash
    for t in walk.trades:
        assert t.net_pnl == approx(t.gross_pnl - t.fees - t.funding)
    assert any(t.funding != 0 for t in walk.trades)
    total = sum(t.net_pnl for t in walk.trades)
    assert walk.equity.iloc[-1] == pytest.approx(10_000 + total, rel=1e-12)
    assert len(walk.equity) == len({e[0] for e in walk.log.equity})  # one snapshot per ts


def test_events_of_all_symbols_are_processed_in_time_order(walk: BacktestResult) -> None:
    # Replaying a bar early (before the decision it follows) or one symbol at a time would
    # log a later fill before an earlier one.
    fills = [f.ts for f in walk.log.fills]
    assert fills == sorted(fills)
    assert [t.exit_ts for t in walk.trades] == sorted(t.exit_ts for t in walk.trades)
    assert all(t.entry_ts <= t.exit_ts for t in walk.trades)


def replace_after(df: pd.DataFrame, cut: datetime, seed: int) -> pd.DataFrame:
    """``df`` with every bar closing after ``cut`` replaced by an unrelated random walk."""
    out = df.copy()
    later = out.index > cut
    fresh = regime_walk(
        int(later.sum()), seed=seed, start_price=float(out["close"][~later].iloc[-1]) * 1.1
    )
    out.loc[later, ["open", "high", "low", "close", "volume"]] = fresh.to_numpy()
    return out


def test_no_look_ahead_end_to_end(tmp_path: Path, walk: BacktestResult) -> None:
    """Replacing everything after CUT (bars and funding) leaves every record up to CUT intact.

    CUT sits where a leak would show first: just after a decision that queued an entry, while
    that position is open. It is 7 minutes into the decision interval because an order queued
    at a decision close D fills at the open of the bar closing at D + 1m and is stamped D: with
    CUT on a decision close, that one fill would legitimately read a bar closing after CUT.
    """
    assert any(f.ts == CUT - 7 * MINUTE for f in walk.log.fills)
    assert any(t.entry_ts < CUT < t.exit_ts for t in walk.trades)
    candles, funding = walk_inputs(seed=1)
    candles = {s: replace_after(df, CUT, seed=99 + i) for i, (s, df) in enumerate(candles.items())}
    for i, (s, rates) in enumerate(funding.items()):
        rates = rates.copy()
        rates[rates.index > CUT] = np.random.default_rng(7 + i).normal(
            0, 1e-3, int((rates.index > CUT).sum())
        )
        funding[s] = rates
    other = walk_run(tmp_path, candles, funding)

    def upto(result: BacktestResult) -> dict[str, list[object]]:
        log = result.log
        return {
            "signals": [s for s in log.signals if s.ts <= CUT],
            "decisions": [(log.signals[i], ts, d) for i, ts, d in log.decisions if ts <= CUT],
            "fills": [f for f in log.fills if f.ts <= CUT],
            "trades": [t for t in log.trades if t.exit_ts <= CUT],
            "regimes": [r for r in log.regimes if r[1] <= CUT],
            "risk_events": [e for e in log.risk_events if e.ts <= CUT],
            "equity": [e for e in log.equity if e[0] <= CUT],
        }

    before, after = upto(walk), upto(other)
    for key in before:
        assert before[key] == after[key], key
    closed = [t for t in walk.trades if t.exit_ts <= CUT]
    assert {t.symbol for t in closed} == {BTC, ETH} and len(closed) >= 5  # data is rich
    assert walk.log.signals != other.log.signals  # the replaced data did change the run
    assert walk.trades != other.trades


# --- persistence and CLI ----------------------------------------------------------------------


def test_persist_result_round_trips_trades_equity_and_links(
    tmp_path: Path, walk: BacktestResult
) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'bt.db'}")
    Base.metadata.create_all(engine)
    with session_scope(engine) as session:
        run_id = persist_result(session, walk, note="unit")
    with session_scope(engine) as session:
        run = session.get(Run, run_id)
        assert run is not None and (run.mode, run.note) == ("backtest", "unit")
        assert run.ended_at is not None
        assert repo.load_trades(session, run_id) == sorted(walk.trades, key=lambda t: t.entry_ts)
        pd.testing.assert_series_equal(
            repo.load_equity(session, run_id), walk.equity, check_index_type=False, check_freq=False
        )
        links = session.execute(
            select(SignalRow.symbol, SignalRow.ts, RiskDecisionRow.rule)
            .join(RiskDecisionRow, RiskDecisionRow.signal_id == SignalRow.id)
            .order_by(RiskDecisionRow.id)
        ).all()
        expected = [
            (walk.log.signals[i].symbol, walk.log.signals[i].ts, d.rule)
            for i, _, d in walk.log.decisions
        ]
        assert [tuple(r) for r in links] == expected
    engine.dispose()


def test_cli_backtest_run_persists_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    upgrade_to_head(url)
    engine = make_engine(url)
    with session_scope(engine) as session:
        for i, symbol in enumerate((BTC, ETH, SOL)):
            repo.upsert_candles(session, symbol, "1m", regime_walk(3 * 1440, seed=i))
            repo.upsert_funding(session, symbol, funding_every_8h(T0, 9, seed=i))
    monkeypatch.setenv("DATABASE_URL", url)
    args = ["backtest", "run", "--start", "2024-01-02", "--end", "2024-01-04", "--note", "cli"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    with session_scope(engine) as session:
        (run,) = session.scalars(select(Run)).all()
        assert run.note == "cli" and f"run {run.id}" in result.output
        assert '"start":"2024-01-02T00:00:00Z"' in run.config_json  # the range actually run
        equity = repo.load_equity(session, run.id)
    assert equity.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert equity.index[-1] < pd.Timestamp("2024-01-04", tz="UTC")
    engine.dispose()
