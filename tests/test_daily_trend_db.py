"""Daily trend persistence: a target trade's missing stop, target, R and regime are NULL in the
database (never a number that could pass for a real one), migration 0003 relaxes the columns on
a database at 0002, and a stored trend backtest comes back as it ran. Marked ``db``: the CI
Postgres job runs them against ``TEST_DATABASE_URL``."""

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from synth import BTC, ETH, MINUTE, candles_from_closes, funding_every_8h
from test_daily_trend_engine import FREE, TREND_YAML

from regimebot.backtest.runner import load_result, persist_result, run_backtest
from regimebot.config import load_config
from regimebot.db import repo
from regimebot.db.models import RiskDecisionRow, TradeRow
from regimebot.db.session import ALEMBIC_INI, make_engine, session_scope, upgrade_to_head
from regimebot.types import ExitReason, Mode, Regime, Side, Trade

pytestmark = pytest.mark.db

T0 = datetime(2024, 1, 1, tzinfo=UTC)
NAN_TRADE = Trade(
    symbol=BTC,
    strategy="daily_trend",
    side=Side.SHORT,
    regime=None,
    qty=15.0,
    entry_ts=T0,
    exit_ts=T0 + timedelta(days=9),
    entry_price=110.0,
    exit_price=104.0,
    initial_stop=math.nan,
    target=math.nan,
    gross_pnl=90.0,
    fees=1.2,
    funding=-0.3,
    net_pnl=89.1,
    exit_reason=ExitReason.SIGNAL,
    r_multiple=math.nan,
    bars_held=9,
)


def test_a_target_trade_stores_null_stop_target_r_and_regime_and_loads_them_as_nan(
    database_url: str,
) -> None:
    upgrade_to_head(database_url)
    engine = make_engine(database_url)
    cfg = load_config(TREND_YAML)
    with session_scope(engine) as s:
        run_id = repo.create_run(s, Mode.BACKTEST, cfg)
        repo.save_trades(s, run_id, [NAN_TRADE])
    with session_scope(engine) as s:
        row = s.execute(select(TradeRow.initial_stop, TradeRow.target, TradeRow.r_multiple,
                               TradeRow.regime)).one()  # fmt: skip
        assert tuple(row) == (None, None, None, None)
        (loaded,) = repo.load_trades(s, run_id)
    assert repr(loaded) == repr(NAN_TRADE)
    engine.dispose()


def test_migration_0003_keeps_old_trades_and_then_takes_null_ones(database_url: str) -> None:
    alembic_cfg = Config(str(ALEMBIC_INI))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(alembic_cfg, "0002")  # the schema before daily trend
    engine = make_engine(database_url)
    old = replace(NAN_TRADE, regime=Regime.RANGE, initial_stop=111.0, target=100.0, r_multiple=9.9)
    with session_scope(engine) as s:
        run_id = repo.create_run(s, Mode.BACKTEST, load_config())
        repo.save_trades(s, run_id, [old])
    with (
        pytest.raises(Exception, match="(?i)not null|null value"),  # NOT NULL at 0002
        session_scope(engine) as s,
    ):
        repo.save_trades(s, run_id, [NAN_TRADE])
    upgrade_to_head(database_url)
    with session_scope(engine) as s:
        repo.save_trades(s, run_id, [replace(NAN_TRADE, entry_ts=T0 + MINUTE)])
    with session_scope(engine) as s:
        first, second = repo.load_trades(s, run_id)
    assert first == old and repr(second) == repr(replace(NAN_TRADE, entry_ts=T0 + MINUTE))
    engine.dispose()


def test_a_stored_trend_backtest_comes_back_as_it_ran(database_url: str, tmp_path: Path) -> None:
    cfg = load_config(
        TREND_YAML,
        overrides={
            **FREE,
            "exchange.symbols": [BTC, ETH],
            "strategies.daily_trend.lookbacks": [2, 3],
            "strategies.daily_trend.vol_window": 2,
            "strategies.daily_trend.target_vol": 0.5,
            "paper.warmup_bars": 4,
            "costs.taker_fee_bps": 5.0,
        },
    )
    closes = [100.0 + 10 * math.sin(k / 3) for k in range(30)]
    minutes = [c for a, b in zip(closes, closes[1:], strict=False) for c in
               [a + (b - a) * (i + 1) / 1440 for i in range(1440)]]  # fmt: skip
    candles = {
        BTC: candles_from_closes(minutes),
        ETH: candles_from_closes([m / 2 for m in minutes]),
    }
    funding = {s: funding_every_8h(T0, 29 * 3, seed=i) for i, s in enumerate((BTC, ETH))}
    start, end = T0 + 5 * timedelta(days=1), T0 + 29 * timedelta(days=1)
    result = run_backtest(cfg, candles, funding, start, end, tmp_path / "KILL")
    assert len(result.trades) >= 4 and result.log.signals == []
    upgrade_to_head(database_url)
    engine = make_engine(database_url)
    with session_scope(engine) as s:
        run_id = persist_result(s, result)
    with session_scope(engine) as s:
        loaded = load_result(s, run_id, cfg)
        stored_decisions = s.execute(
            select(RiskDecisionRow.signal_id, RiskDecisionRow.rule, RiskDecisionRow.qty)
            .where(RiskDecisionRow.run_id == run_id)
            .order_by(RiskDecisionRow.id)
        ).all()
    by_entry = sorted(result.trades, key=lambda t: t.entry_ts)
    assert [repr(t) for t in loaded.trades] == [repr(t) for t in by_entry]
    assert loaded.log.fills == result.log.fills and loaded.log.equity == result.log.equity
    # decisions without a signal: stored with a NULL signal_id, in order
    assert [tuple(r) for r in stored_decisions] == [
        (None, d.rule, d.qty) for _, _, d in result.log.decisions
    ]
    engine.dispose()
