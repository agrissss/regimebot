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

from regimebot.backtest.runner import BacktestResult, load_result, persist_result, run_backtest
from regimebot.cli import app
from regimebot.config import AppConfig, load_config
from regimebot.db import repo
from regimebot.db.models import Base, RiskDecisionRow, RiskEventRow, Run, SignalRow
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.reporting.metrics import first_permanent_halt
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


@pytest.mark.parametrize(
    ("with_eth", "expired_at"),
    [(False, 30 * MINUTE), (True, 15 * MINUTE)],
    ids=["fill_after_gap", "decision_in_gap"],
)
def test_an_entry_signalled_before_a_gap_in_base_bars_expires(
    tmp_path: Path, with_eth: bool, expired_at: timedelta
) -> None:
    # BTC has no bars for 30 minutes after the signal at t. Alone, the first fill it is offered
    # opens at t + 30m, two decision bars late; next to ETH, the ETH decision at t + 15m comes
    # first. Either way the stale signal must not fill.
    t = T0 + HOUR
    btc = flat(240)
    btc = btc[(btc.index <= t) | (btc.index > t + 30 * MINUTE)]
    candles = {BTC: btc, ETH: flat(240)} if with_eth else {BTC: btc}
    result = run_backtest(
        open_cfg(tuple(candles)),
        candles,
        {},
        T0,
        T0 + 4 * HOUR,
        tmp_path / "KILL",
        strategies=[Scripted("breakout", at={t})],
    )
    btc_decisions = [
        (ts, d.rule) for i, ts, d in result.log.decisions if result.log.signals[i].symbol == BTC
    ]
    assert btc_decisions == [(t, "ok"), (t + expired_at, "expired")]
    assert [f.symbol for f in result.log.fills if f.symbol == BTC] == []


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


@pytest.mark.parametrize(
    ("hours", "gaps"),
    [
        ((0, 8, 32, 40), [(8, 32)]),  # 16:00 and 24:00 missing: 24 h without a stamp
        ((0, 8, 24, 40), []),  # one missed event (16 h) is within the tolerance
        ((), [(0, 48)]),  # no funding at all: the whole range
    ],
    ids=["interior_gap", "one_missed_event", "no_funding"],
)
def test_a_funding_gap_in_the_range_is_recorded_once_and_is_not_a_halt(
    tmp_path: Path, hours: tuple[int, ...], gaps: list[tuple[int, int]]
) -> None:
    candles = flat(48 * 60 - 1)  # decisions in [T0, T0 + 48 h)
    rates = funding_at(*((T0 + h * HOUR, 1e-4) for h in hours)) if hours else None
    result = stub_run(tmp_path, candles, T0 + HOUR, funding=rates)
    events = [e for e in result.log.risk_events if e.kind == "funding_missing"]
    assert [(e.ts, e.action, e.detail) for e in events] == [
        (
            T0,
            "funding_not_charged",
            f"{BTC}: no funding stamp from {(T0 + a * HOUR).isoformat()} to "
            f"{(T0 + b * HOUR).isoformat()}",
        )
        for a, b in gaps
    ]
    assert first_permanent_halt(result.log.risk_events) is None
    assert len(result.trades) == 1  # recorded, nothing else changes


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


# --- feature warm-up --------------------------------------------------------------------------

# Every window short: features are fully valid from the 24th decision bar (6 h of history).
SHORT_WINDOWS = {
    "features.atr_window": 5,
    "features.rv_window": 8,
    "features.pctile_window": 16,
    "features.adx_window": 5,
    "features.ema_fast": 4,
    "features.ema_slow": 12,
    "features.volume_window": 8,
    "features.donchian_window": 8,
    "features.zscore_window": 8,
}


@pytest.mark.parametrize(
    ("start", "cold"),
    [
        (T0, {BTC: T0 + 6 * HOUR, ETH: T0 + 7 * HOUR}),  # no history before start
        (T0 + 6 * HOUR, {ETH: T0 + 7 * HOUR}),  # BTC has its 6 h, ETH (listed 1 h later) not
        (T0 + 7 * HOUR, {}),  # both warm: no event
    ],
    ids=["no_history", "eth_short", "both_warm"],
)
def test_an_under_warmed_start_is_recorded_once_per_cold_symbol(
    tmp_path: Path, start: datetime, cold: dict[str, datetime]
) -> None:
    cfg = load_config(overrides={"exchange.symbols": [BTC, ETH], **SHORT_WINDOWS})
    candles = {
        BTC: regime_walk(12 * 60, seed=4),
        ETH: regime_walk(11 * 60, seed=5, start=T0 + HOUR),
    }
    result = run_backtest(cfg, candles, {}, start, T0 + 12 * HOUR, tmp_path / "KILL", [])
    events = [e for e in result.log.risk_events if e.kind == "warmup_incomplete"]
    assert [(e.ts, e.action) for e in events] == [(start, "no_trade_until_warm")] * len(cold)
    for event, (symbol, warm_at) in zip(events, cold.items(), strict=True):
        assert event.detail.startswith(f"{symbol}: ")
        assert event.detail.endswith(f"first fully valid row: {warm_at.isoformat()}")


def test_a_symbol_that_never_warms_up_is_reported_without_a_warm_row(tmp_path: Path) -> None:
    result = stub_run(tmp_path, flat(120), T0 + HOUR)  # two hours, default 31-day windows
    (event,) = result.log.risk_events
    assert event.kind == "warmup_incomplete" and event.ts == T0
    end = (T0 + 121 * MINUTE).isoformat()
    assert event.detail == (
        f"{BTC}: features not fully valid at start; first fully valid row: none before {end}"
    )
    assert len(result.trades) == 1  # NaN features are NO_TRADE, which open_cfg still trades


# --- real strategies on random data -----------------------------------------------------------

DAYS = 20
# A 4-day percentile window (default 30 days): with no history before T0 the features warm up
# by day 5 (96 + 384 decision bars) and leave 15 days of trading.
WALK_OVERRIDES = {"features.pctile_window": 384}
# close of the fill bar of the 06:45 decision that enters ETH (open until 09:30)
CUT = datetime(2024, 1, 13, 6, 46, tzinfo=UTC)


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
    cfg = load_config(overrides={"exchange.symbols": [BTC, ETH], **WALK_OVERRIDES})
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

    CUT sits where a leak would show first: just after a decision D that queued an entry, while
    that position is open. It is D + 1m, the close of the bar the entry fills on: an order
    queued at D fills at that bar's open and is stamped D, so with CUT = D that one fill would
    legitimately read a bar closing after CUT. Every later bar is replaced.
    """
    assert any(f.ts == CUT - MINUTE for f in walk.log.fills)
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


@pytest.mark.db
def test_persist_result_round_trips_trades_equity_and_links(
    database_url: str, walk: BacktestResult
) -> None:
    engine = make_engine(database_url)
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
        assert repo.load_equity_snapshots(session, run_id) == walk.log.equity  # with gross
        loaded = load_result(session, run_id, walk.cfg)  # what `report` rebuilds
        assert loaded.trades == sorted(walk.trades, key=lambda t: t.entry_ts)
        assert (loaded.log.fills, loaded.log.risk_events) == (walk.log.fills, walk.log.risk_events)
        assert loaded.log.equity == walk.log.equity
        assert (loaded.start, loaded.end) == (walk.cfg.backtest.start, walk.cfg.backtest.end)
        pd.testing.assert_series_equal(loaded.equity, walk.equity, check_index_type=False)
        pd.testing.assert_series_equal(
            loaded.gross_exposure, walk.gross_exposure, check_index_type=False
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
    assert "halted at" not in result.output
    # one day of history cannot warm the default 31-day windows up
    (warning,) = [line for line in result.output.splitlines() if line.startswith("warning:")]
    assert "under-warmed start" in warning and all(s in warning for s in (BTC, ETH, SOL))

    # Short feature windows (warm after 120 bars, 30 h) so the two days before 2024-01-03 warm
    # them up and one day trades; any entry fee then breaches the drawdown.
    short = ["features.ema_slow=48", "features.ema_fast=12", "features.pctile_window=96"]
    short += ["features.rv_window=24", "risk.max_drawdown=0.000001"]
    args = ["backtest", "run", "--start", "2024-01-03", "--end", "2024-01-04"]
    halted = CliRunner().invoke(app, [*args, *(a for kv in short for a in ("--set", kv))])
    assert halted.exit_code == 0, halted.output
    assert "warning:" not in halted.output
    with session_scope(engine) as session:
        halt_ts, detail = session.execute(
            select(RiskEventRow.ts, RiskEventRow.detail).where(RiskEventRow.kind == "max_drawdown")
        ).one()
    assert f"halted at {halt_ts.isoformat()}: max_drawdown ({detail})" in halted.output
    engine.dispose()
