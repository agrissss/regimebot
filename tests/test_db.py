from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from regimebot.cli import app
from regimebot.config import AppConfig, config_json, load_config
from regimebot.db import repo
from regimebot.db.models import Base, Run, SignalRow
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.types import (
    ExitReason,
    Fill,
    Mode,
    OrderSide,
    Regime,
    RiskDecision,
    RiskEvent,
    Side,
    Signal,
    Trade,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = make_engine(f"sqlite:///{tmp_path / 'sub' / 'test.db'}")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with session_scope(engine) as s:
        yield s


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return load_config()


def make_candles(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC", name="ts")
    close = 100 + rng.normal(size=n).cumsum()
    return pd.DataFrame(
        {"open": close - 0.1, "high": close + 1, "low": close - 1, "close": close, "volume": 10.0},
        index=idx,
    )


def test_make_engine_creates_dir_and_enables_fk_and_wal(engine: Engine) -> None:
    with engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"


def test_session_scope_rolls_back_on_error(engine: Engine, cfg: AppConfig) -> None:
    with pytest.raises(RuntimeError), session_scope(engine) as s:
        repo.create_run(s, Mode.BACKTEST, cfg)
        raise RuntimeError("boom")
    with session_scope(engine) as s:
        assert s.scalars(select(Run)).all() == []


def test_candles_round_trip_is_utc_aware_float(session: Session) -> None:
    df = make_candles()
    assert repo.upsert_candles(session, "BTC", "1m", df) == 5
    out = repo.load_candles(session, "BTC", "1m")
    assert out.index.tz is not None and str(out.index.tz) == "UTC"
    assert out.index.name == "ts"
    assert (out.dtypes == "float64").all()
    pd.testing.assert_frame_equal(out, df, check_freq=False)


def test_candles_upsert_is_idempotent_and_updates_changed_rows(session: Session) -> None:
    df = make_candles()
    repo.upsert_candles(session, "BTC", "1m", df)
    repo.upsert_candles(session, "BTC", "1m", df)
    assert len(repo.load_candles(session, "BTC", "1m")) == 5
    changed = df.copy()
    changed.iloc[2, changed.columns.get_loc("close")] = 999.0
    repo.upsert_candles(session, "BTC", "1m", changed)
    out = repo.load_candles(session, "BTC", "1m")
    assert len(out) == 5 and out["close"].iloc[2] == 999.0


def test_candles_are_keyed_by_symbol_and_timeframe(session: Session) -> None:
    repo.upsert_candles(session, "BTC", "1m", make_candles())
    assert repo.load_candles(session, "ETH", "1m").empty
    assert repo.load_candles(session, "BTC", "15m").empty


def test_load_candles_range_is_half_open(session: Session) -> None:
    df = make_candles(10)
    repo.upsert_candles(session, "BTC", "1m", df)
    start, end = df.index[2].to_pydatetime(), df.index[5].to_pydatetime()
    out = repo.load_candles(session, "BTC", "1m", start, end)
    assert list(out.index) == list(df.index[2:5])
    assert len(repo.load_candles(session, "BTC", "1m", start=end)) == 5


def test_upserts_reject_duplicate_timestamps(session: Session) -> None:
    df = make_candles()
    with pytest.raises(ValueError, match="duplicate"):
        repo.upsert_candles(session, "BTC", "1m", pd.concat([df, df.iloc[:1]]))
    s = pd.Series([0.1, 0.2], index=[df.index[0], df.index[0]], name="rate")
    with pytest.raises(ValueError, match="duplicate"):
        repo.upsert_funding(session, "BTC", s)


def test_candles_reject_naive_index_and_nan(session: Session) -> None:
    df = make_candles()
    with pytest.raises(ValueError, match="timezone-aware"):
        repo.upsert_candles(session, "BTC", "1m", df.tz_localize(None))
    bad = df.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        repo.upsert_candles(session, "BTC", "1m", bad)


def test_candles_chunking_writes_all_rows(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo, "CHUNK_ROWS", 3)
    df = make_candles(10)
    assert repo.upsert_candles(session, "BTC", "1m", df) == 10
    pd.testing.assert_frame_equal(repo.load_candles(session, "BTC", "1m"), df, check_freq=False)


def test_naive_datetime_bound_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError, match="naive"):
        repo.load_candles(session, "BTC", "1m", start=datetime(2024, 1, 1))  # noqa: DTZ001  # naive on purpose


def test_naive_datetime_on_save_is_rejected(session: Session, cfg: AppConfig) -> None:
    run_id = repo.create_run(session, Mode.BACKTEST, cfg)
    naive_event = RiskEvent(datetime(2024, 1, 1), "k", "d", "halt")  # noqa: DTZ001  # naive on purpose
    with pytest.raises(StatementError, match="naive datetime rejected"):
        repo.save_risk_events(session, run_id, [naive_event])


def test_funding_round_trip_and_update(session: Session) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC", name="ts")
    s = pd.Series([0.0001, -0.0002, 0.0003], index=idx, name="rate")
    assert repo.upsert_funding(session, "BTC", s) == 3
    pd.testing.assert_series_equal(repo.load_funding(session, "BTC"), s, check_freq=False)
    repo.upsert_funding(session, "BTC", s * 2)
    out = repo.load_funding(session, "BTC", start=idx[1].to_pydatetime())
    assert out.tolist() == pytest.approx([-0.0004, 0.0006])
    assert repo.load_funding(session, "ETH").empty


def test_create_and_end_run(session: Session, cfg: AppConfig) -> None:
    run_id = repo.create_run(session, Mode.PAPER, cfg, note="hello")
    run = session.get(Run, run_id)
    assert run is not None
    assert run.mode == "paper" and run.note == "hello" and run.ended_at is None
    assert run.started_at.tzinfo is not None
    assert run.config_json == config_json(cfg)
    assert "database_url" not in run.config_json
    assert run.git_sha != ""
    repo.end_run(session, run_id)
    assert run.ended_at is not None and run.ended_at >= run.started_at
    with pytest.raises(ValueError, match="unknown run_id"):
        repo.end_run(session, 999)


def test_git_sha_unknown_when_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert repo._git_sha() == "unknown"


def make_trade(i: int = 0) -> Trade:
    return Trade(
        symbol="BTC",
        strategy="breakout",
        side=Side.SHORT,
        regime=Regime.EXPANSION,
        qty=0.1 + i,
        entry_ts=T0 + timedelta(hours=i),
        exit_ts=T0 + timedelta(hours=i + 2),
        entry_price=100.1,
        exit_price=98.7,
        initial_stop=101.3,
        target=97.0,
        gross_pnl=0.14,
        fees=0.02,
        funding=-0.001,
        net_pnl=0.119,
        exit_reason=ExitReason.TARGET,
        r_multiple=0.99,
        bars_held=8,
    )


def test_trades_round_trip_equal_and_ordered(session: Session, cfg: AppConfig) -> None:
    run_id = repo.create_run(session, Mode.BACKTEST, cfg)
    other = repo.create_run(session, Mode.BACKTEST, cfg)
    trades = [make_trade(0), make_trade(1), make_trade(2)]
    repo.save_trades(session, run_id, [trades[2], trades[0], trades[1]])
    repo.save_trades(session, other, [make_trade(5)])
    assert repo.load_trades(session, run_id) == trades
    assert repo.load_trades(session, 12345) == []


def test_signals_return_ids_and_decisions_link_to_them(
    session: Session, cfg: AppConfig, engine: Engine
) -> None:
    run_id = repo.create_run(session, Mode.BACKTEST, cfg)
    signals = [
        Signal(
            "breakout", "BTC", T0, Side.LONG, 100.0 + k, 98.0, 104.0, 1.5, Regime.RANGE, 10, None
        )
        for k in range(25)
    ]
    ids = repo.save_signals(session, run_id, signals)
    assert len(set(ids)) == 25
    stored = dict(session.execute(select(SignalRow.id, SignalRow.entry)).all())
    assert [stored[i] for i in ids] == [s.entry for s in signals]
    repo.save_risk_decisions(
        session,
        run_id,
        [
            (ids[1], T0, RiskDecision(True, 0.5, "ok", "")),
            (None, T0, RiskDecision(False, 0.0, "x", "d")),
        ],
    )
    session.flush()
    rows = session.execute(Base.metadata.tables["risk_decisions"].select().order_by("id")).all()
    assert [(r.signal_id, r.approved, r.qty, r.rule) for r in rows] == [
        (ids[1], True, 0.5, "ok"),
        (None, False, 0.0, "x"),
    ]
    assert repo.save_signals(session, run_id, []) == []


def test_foreign_keys_are_enforced(session: Session) -> None:
    with pytest.raises(IntegrityError):
        repo.save_risk_events(session, 999, [RiskEvent(T0, "k", "d", "halt")])
        session.flush()


def test_fills_events_regimes_equity_round_trip(session: Session, cfg: AppConfig) -> None:
    run_id = repo.create_run(session, Mode.BACKTEST, cfg)
    repo.save_fills(
        session, run_id, [Fill("BTC", T0, OrderSide.BUY, 1.0, 100.1, 100.0, 0.05, 10.0, "taker")]
    )
    repo.save_risk_events(session, run_id, [RiskEvent(T0, "daily_loss", "d", "halt")])
    repo.save_regimes(session, run_id, [("BTC", T0, Regime.RANGE, {"adx": 12.5})])
    ts = [T0 + timedelta(minutes=i) for i in range(3)]
    repo.save_equity(
        session,
        run_id,
        [(ts[2], 102.0, 0.0, 0.0), (ts[0], 100.0, 0.5, 0.0), (ts[1], 99.0, 0.5, 0.01)],
    )
    eq = repo.load_equity(session, run_id)
    assert eq.tolist() == [100.0, 99.0, 102.0] and eq.name == "equity"
    assert str(eq.index.tz) == "UTC" and eq.index[0] == ts[0]
    assert repo.load_equity(session, 999).empty
    session.flush()
    tables = Base.metadata.tables
    for name in ("fills", "risk_events", "regimes"):
        assert session.execute(tables[name].select()).all()


def test_open_positions_upsert_delete_load(session: Session, cfg: AppConfig) -> None:
    run_id = repo.create_run(session, Mode.PAPER, cfg)
    repo.save_open_position(session, run_id, "BTC", {"qty": 1.0, "stop": 95.5})
    repo.save_open_position(session, run_id, "ETH", {"qty": 2.0})
    repo.save_open_position(session, run_id, "BTC", {"qty": 0.5, "stop": 96.0})
    assert repo.load_open_positions(session, run_id) == {
        "BTC": {"qty": 0.5, "stop": 96.0},
        "ETH": {"qty": 2.0},
    }
    repo.save_open_position(session, run_id, "BTC", None)
    repo.save_open_position(session, run_id, "SOL", None)  # deleting a missing row is fine
    assert repo.load_open_positions(session, run_id) == {"ETH": {"qty": 2.0}}


def test_upsert_rejects_unsupported_dialect(cfg: AppConfig) -> None:
    eng = create_engine("sqlite://")
    with Session(eng) as s, pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(s.get_bind().dialect), "name", "mysql")
        with pytest.raises(NotImplementedError, match="mysql"):
            repo.upsert_candles(s, "BTC", "1m", make_candles())


def test_migration_matches_models(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade_to_head(url)
    insp = inspect(create_engine(url))
    tables = set(insp.get_table_names()) - {"alembic_version"}
    assert tables == set(Base.metadata.tables)
    for name, table in Base.metadata.tables.items():
        assert {c["name"] for c in insp.get_columns(name)} == {c.name for c in table.columns}
        assert {i["name"] for i in insp.get_indexes(name)} == {i.name for i in table.indexes}


def test_cli_db_upgrade_uses_database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cli" / "x.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    result = CliRunner().invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    assert "runs" in inspect(create_engine(f"sqlite:///{path}")).get_table_names()


def test_candle_timestamps_and_funding_bounds(engine: Engine) -> None:
    with session_scope(engine) as s:
        df = make_candles()
        start, end = df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()
        assert len(repo.candle_timestamps(s, "BTC", "1m", start, end)) == 0
        assert repo.funding_ts_bounds(s, "BTC") is None
        repo.upsert_candles(s, "BTC", "1m", df)
        got = repo.candle_timestamps(s, "BTC", "1m", start, end)
        assert list(got) == list(df.index)  # both ends inclusive
        assert len(repo.candle_timestamps(s, "BTC", "1m", start, end - (end - start) / 2)) < len(df)
        fs = pd.Series([0.1, 0.2], index=df.index[:2], name="rate")
        repo.upsert_funding(s, "BTC", fs)
        assert repo.funding_ts_bounds(s, "BTC") == (df.index[0], df.index[1])
