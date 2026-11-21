"""Repository functions: plain functions over a SQLAlchemy ``Session``.

Conventions: timestamps are UTC-aware. Naive datetimes are rejected: ``ValueError`` from the
``load_*`` bounds and the upsert indexes, ``sqlalchemy.exc.StatementError`` (wrapping the
``ValueError``) from the ``save_*`` functions. Every ``save_*`` function takes ``run_id``,
bulk-inserts, and does not commit (``session_scope`` does).
"""

import dataclasses
import json
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from regimebot.config import AppConfig, config_hash, config_json
from regimebot.db.models import (
    Base,
    Candle,
    EquitySnapshot,
    FillRow,
    FundingRate,
    OpenPosition,
    RegimeRow,
    RiskDecisionRow,
    RiskEventRow,
    Run,
    SignalRow,
    TradeRow,
)
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

CHUNK_ROWS = 10_000  # rows per bulk statement; keeps memory flat for 1M-row loads
CANDLE_COLUMNS = ["open", "high", "low", "close", "volume"]


# --- helpers -----------------------------------------------------------------------------


def _upsert(
    session: Session,
    model: type[Base],
    row_chunks: Iterable[list[dict[str, Any]]],
    key_columns: Sequence[str],
) -> None:
    """Bulk ``INSERT ... ON CONFLICT DO UPDATE``, one statement per chunk of rows.

    Takes an iterable of chunks so callers build each chunk lazily (SQLite and Postgres only).
    """
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        dialect_insert: Any = sqlite.insert
    elif dialect == "postgresql":
        dialect_insert = postgresql.insert
    else:
        raise NotImplementedError(f"upsert is not supported for dialect {dialect!r}")
    value_columns = [c.name for c in model.__table__.columns if c.name not in key_columns]
    for chunk in row_chunks:
        stmt = dialect_insert(model)
        stmt = stmt.on_conflict_do_update(
            index_elements=list(key_columns),
            set_={c: stmt.excluded[c] for c in value_columns},
        )
        session.execute(stmt, chunk)


def _insert_chunks(session: Session, model: type[Base], rows: list[dict[str, Any]]) -> None:
    for i in range(0, len(rows), CHUNK_ROWS):
        session.execute(insert(model), rows[i : i + CHUNK_ROWS])


def _py_timestamps(index: pd.Index) -> list[datetime]:
    """Python UTC datetimes from a tz-aware, duplicate-free ``DatetimeIndex``.

    Duplicates are rejected because Postgres cannot upsert the same row twice in one statement.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("index must be a timezone-aware DatetimeIndex")
    if index.has_duplicates:
        raise ValueError("index has duplicate timestamps")
    return list(index.tz_convert("UTC").to_pydatetime())


def _utc_index(values: Iterable[datetime]) -> pd.DatetimeIndex:
    """Microsecond-resolution UTC index named ``ts`` (pandas 3 default for datetimes)."""
    return pd.DatetimeIndex(pd.to_datetime(list(values), utc=True), name="ts").as_unit("us")


def _time_range(model: Any, start: datetime | None, end: datetime | None) -> list[Any]:
    """WHERE clauses for ``start <= ts < end``."""
    for bound in (start, end):
        if bound is not None and bound.tzinfo is None:
            raise ValueError(f"naive datetime rejected (timestamps must be UTC-aware): {bound!r}")
    clauses = []
    if start is not None:
        clauses.append(model.ts >= start)
    if end is not None:
        clauses.append(model.ts < end)
    return clauses


# --- market data -------------------------------------------------------------------------


def upsert_candles(session: Session, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Insert or update candles; idempotent. ``df`` has a UTC ``DatetimeIndex`` named ``ts``
    and columns open, high, low, close, volume. Returns the number of rows written."""
    stamps = _py_timestamps(df.index)
    values = df[CANDLE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("candles contain NaN or infinite values")

    def chunks() -> Iterator[list[dict[str, Any]]]:
        for i in range(0, len(stamps), CHUNK_ROWS):
            yield [
                {"symbol": symbol, "timeframe": timeframe, "ts": ts}
                | dict(zip(CANDLE_COLUMNS, row, strict=True))
                for ts, row in zip(
                    stamps[i : i + CHUNK_ROWS], values[i : i + CHUNK_ROWS].tolist(), strict=True
                )
            ]

    _upsert(session, Candle, chunks(), ["symbol", "timeframe", "ts"])
    return len(stamps)


def load_candles(
    session: Session,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Candles with ``start <= ts < end`` as a sorted float64 frame on a UTC ``ts`` index."""
    stmt = (
        select(Candle.ts, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume)
        .where(
            Candle.symbol == symbol, Candle.timeframe == timeframe, *_time_range(Candle, start, end)
        )
        .order_by(Candle.ts)
    )
    records = session.execute(stmt).all()
    df = pd.DataFrame([tuple(r[1:]) for r in records], columns=CANDLE_COLUMNS, dtype="float64")
    df.index = _utc_index(r[0] for r in records)
    return df


def candle_timestamps(
    session: Session, symbol: str, timeframe: str, start: datetime, end: datetime
) -> pd.DatetimeIndex:
    """Sorted stored candle ``ts`` values with ``start <= ts <= end`` (both inclusive)."""
    stmt = (
        select(Candle.ts)
        .where(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
            *_time_range(Candle, start, None),
            Candle.ts <= end,
        )
        .order_by(Candle.ts)
    )
    return _utc_index(session.scalars(stmt))


def funding_ts_bounds(session: Session, symbol: str) -> tuple[datetime, datetime] | None:
    """Earliest and latest stored funding ``ts`` for a symbol, or ``None`` when empty."""
    row = session.execute(
        select(func.min(FundingRate.ts), func.max(FundingRate.ts)).where(
            FundingRate.symbol == symbol
        )
    ).one()
    return None if row[0] is None else (row[0], row[1])


def upsert_funding(session: Session, symbol: str, series: pd.Series) -> int:
    """Insert or update funding rates; ``series`` is indexed by UTC ``ts``. Returns rows written."""
    stamps = _py_timestamps(series.index)
    rates = series.to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("funding rates contain NaN or infinite values")

    def chunks() -> Iterator[list[dict[str, Any]]]:
        for i in range(0, len(stamps), CHUNK_ROWS):
            yield [
                {"symbol": symbol, "ts": ts, "rate": rate}
                for ts, rate in zip(
                    stamps[i : i + CHUNK_ROWS], rates[i : i + CHUNK_ROWS].tolist(), strict=True
                )
            ]

    _upsert(session, FundingRate, chunks(), ["symbol", "ts"])
    return len(stamps)


def load_funding(
    session: Session, symbol: str, start: datetime | None = None, end: datetime | None = None
) -> pd.Series:
    """Funding rates with ``start <= ts < end`` as a sorted float64 Series named ``rate``."""
    stmt = (
        select(FundingRate.ts, FundingRate.rate)
        .where(FundingRate.symbol == symbol, *_time_range(FundingRate, start, end))
        .order_by(FundingRate.ts)
    )
    records = session.execute(stmt).all()
    return pd.Series(
        [r[1] for r in records],
        index=_utc_index(r[0] for r in records),
        name="rate",
        dtype="float64",
    )


# --- runs --------------------------------------------------------------------------------


_REPO_DIR = Path(__file__).resolve().parent  # inside the repo this package lives in


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=_REPO_DIR,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def create_run(session: Session, mode: Mode, cfg: AppConfig, note: str = "") -> int:
    """Record a new run (config snapshot, git sha, start time) and return its id."""
    run = Run(
        mode=str(mode),
        config_hash=config_hash(cfg),
        config_json=config_json(cfg),
        git_sha=_git_sha(),
        started_at=datetime.now(UTC),
        note=note,
    )
    session.add(run)
    session.flush()
    return run.id


def end_run(session: Session, run_id: int) -> None:
    """Stamp the run's end time."""
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"unknown run_id {run_id}")
    run.ended_at = datetime.now(UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class RunInfo:
    """A stored run's identity: what ran (config snapshot) and when."""

    id: int
    mode: str
    config_hash: str
    config_json: str
    git_sha: str
    started_at: datetime
    note: str


def load_run(session: Session, run_id: int) -> RunInfo:
    """A run's stored identity; raises ValueError for an unknown id."""
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"unknown run_id {run_id}")
    return RunInfo(
        run.id, run.mode, run.config_hash, run.config_json, run.git_sha, run.started_at, run.note
    )


def count_signals(session: Session, run_id: int) -> int:
    """Number of signals a run generated."""
    stmt = select(func.count()).select_from(SignalRow).where(SignalRow.run_id == run_id)
    return int(session.scalar(stmt) or 0)


# --- trading records ---------------------------------------------------------------------


_TRADE_FIELDS = [f.name for f in dataclasses.fields(Trade)]  # also the TradeRow column names
_TRADE_ENUMS: dict[str, type[Enum]] = {"side": Side, "regime": Regime, "exit_reason": ExitReason}


def save_trades(session: Session, run_id: int, trades: Sequence[Trade]) -> None:
    """Insert closed trades (every ``Trade`` field is stored; enums as their string values)."""
    rows = [
        {"run_id": run_id}
        | {
            name: v.value if isinstance(v := getattr(t, name), Enum) else v
            for name in _TRADE_FIELDS
        }
        for t in trades
    ]
    _insert_chunks(session, TradeRow, rows)


def load_trades(session: Session, run_id: int) -> list[Trade]:
    """A run's trades, ordered by entry time (then insertion order)."""
    stmt = (
        select(TradeRow).where(TradeRow.run_id == run_id).order_by(TradeRow.entry_ts, TradeRow.id)
    )
    trades = []
    for r in session.scalars(stmt):
        kwargs: dict[str, Any] = {name: getattr(r, name) for name in _TRADE_FIELDS}
        for name, enum_type in _TRADE_ENUMS.items():
            kwargs[name] = enum_type(kwargs[name])
        trades.append(Trade(**kwargs))
    return trades


def save_fills(session: Session, run_id: int, fills: Sequence[Fill]) -> None:
    """Insert fills."""
    rows = [
        {
            "run_id": run_id,
            "symbol": f.symbol,
            "ts": f.ts,
            "side": str(f.side),
            "qty": f.qty,
            "price": f.price,
            "ref_price": f.ref_price,
            "fee": f.fee,
            "slippage_bps": f.slippage_bps,
            "liquidity": f.liquidity,
        }
        for f in fills
    ]
    _insert_chunks(session, FillRow, rows)


def load_fills(session: Session, run_id: int) -> list[Fill]:
    """A run's fills in time order (then insertion order)."""
    stmt = select(FillRow).where(FillRow.run_id == run_id).order_by(FillRow.ts, FillRow.id)
    return [
        Fill(
            r.symbol,
            r.ts,
            OrderSide(r.side),
            r.qty,
            r.price,
            r.ref_price,
            r.fee,
            r.slippage_bps,
            r.liquidity,
        )
        for r in session.scalars(stmt)
    ]


def save_signals(session: Session, run_id: int, signals: Sequence[Signal]) -> list[int]:
    """Insert signals and return their new row ids, in input order (for ``signal_id`` links).

    ``max_hold_bars`` and ``trail_atr_mult`` are not stored (the table has no such columns).
    """
    rows = [
        {
            "run_id": run_id,
            "strategy": s.strategy,
            "symbol": s.symbol,
            "ts": s.ts,
            "side": str(s.side),
            "entry": s.entry,
            "stop": s.stop,
            "target": s.target,
            "atr": s.atr,
            "regime": str(s.regime),
        }
        for s in signals
    ]
    ids: list[int] = []
    for i in range(0, len(rows), CHUNK_ROWS):
        stmt = insert(SignalRow).returning(SignalRow.id, sort_by_parameter_order=True)
        ids.extend(session.scalars(stmt, rows[i : i + CHUNK_ROWS]))
    return ids


def save_risk_decisions(
    session: Session, run_id: int, decisions: Sequence[tuple[int | None, datetime, RiskDecision]]
) -> None:
    """Insert risk decisions given as ``(signal_id, ts, RiskDecision)`` tuples."""
    rows = [
        {
            "run_id": run_id,
            "signal_id": signal_id,
            "ts": ts,
            "approved": d.approved,
            "qty": d.qty,
            "rule": d.rule,
            "detail": d.detail,
        }
        for signal_id, ts, d in decisions
    ]
    _insert_chunks(session, RiskDecisionRow, rows)


def save_risk_events(session: Session, run_id: int, events: Sequence[RiskEvent]) -> None:
    """Insert risk events."""
    rows = [
        {"run_id": run_id, "ts": e.ts, "kind": e.kind, "detail": e.detail, "action": e.action}
        for e in events
    ]
    _insert_chunks(session, RiskEventRow, rows)


def load_risk_events(session: Session, run_id: int) -> list[RiskEvent]:
    """A run's risk events in time order (then insertion order)."""
    stmt = (
        select(RiskEventRow)
        .where(RiskEventRow.run_id == run_id)
        .order_by(RiskEventRow.ts, RiskEventRow.id)
    )
    return [RiskEvent(r.ts, r.kind, r.detail, r.action) for r in session.scalars(stmt)]


def save_regimes(
    session: Session,
    run_id: int,
    regimes: Sequence[tuple[str, datetime, Regime, dict[str, Any]]],
) -> None:
    """Insert regime labels given as ``(symbol, ts, Regime, inputs)``; inputs is stored as JSON."""
    rows = [
        {
            "run_id": run_id,
            "symbol": symbol,
            "ts": ts,
            "regime": str(regime),
            "inputs_json": json.dumps(inputs, sort_keys=True),
        }
        for symbol, ts, regime, inputs in regimes
    ]
    _insert_chunks(session, RegimeRow, rows)


def save_equity(
    session: Session, run_id: int, snapshots: Sequence[tuple[datetime, float, float, float]]
) -> None:
    """Insert equity snapshots given as ``(ts, equity, gross_exposure, drawdown)``."""
    rows = [
        {"run_id": run_id, "ts": ts, "equity": eq, "gross_exposure": gross, "drawdown": dd}
        for ts, eq, gross, dd in snapshots
    ]
    _insert_chunks(session, EquitySnapshot, rows)


def load_equity_snapshots(
    session: Session, run_id: int
) -> list[tuple[datetime, float, float, float]]:
    """A run's equity snapshots ``(ts, equity, gross_exposure, drawdown)`` in time order, as
    ``RunLog.equity`` holds them."""
    stmt = (
        select(
            EquitySnapshot.ts,
            EquitySnapshot.equity,
            EquitySnapshot.gross_exposure,
            EquitySnapshot.drawdown,
        )
        .where(EquitySnapshot.run_id == run_id)
        .order_by(EquitySnapshot.ts)
    )
    return [(ts, eq, gross, dd) for ts, eq, gross, dd in session.execute(stmt)]


def load_equity(session: Session, run_id: int) -> pd.Series:
    """A run's equity curve as a float64 Series named ``equity`` on a UTC ``ts`` index."""
    stmt = (
        select(EquitySnapshot.ts, EquitySnapshot.equity)
        .where(EquitySnapshot.run_id == run_id)
        .order_by(EquitySnapshot.ts)
    )
    records = session.execute(stmt).all()
    return pd.Series(
        [r[1] for r in records],
        index=_utc_index(r[0] for r in records),
        name="equity",
        dtype="float64",
    )


# --- open positions ----------------------------------------------------------------------


def save_open_position(
    session: Session, run_id: int, symbol: str, state: dict[str, Any] | None
) -> None:
    """Store a symbol's open-position state as JSON (upsert); ``None`` deletes it."""
    if state is None:
        session.execute(
            delete(OpenPosition).where(OpenPosition.run_id == run_id, OpenPosition.symbol == symbol)
        )
        return
    row = {
        "run_id": run_id,
        "symbol": symbol,
        "state_json": json.dumps(state, sort_keys=True),
        "updated_at": datetime.now(UTC),
    }
    _upsert(session, OpenPosition, [[row]], ["run_id", "symbol"])


def load_open_positions(session: Session, run_id: int) -> dict[str, dict[str, Any]]:
    """All open-position states of a run, keyed by symbol."""
    stmt = select(OpenPosition.symbol, OpenPosition.state_json).where(OpenPosition.run_id == run_id)
    return {symbol: json.loads(state) for symbol, state in session.execute(stmt)}
