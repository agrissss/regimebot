"""SQLAlchemy models. Every timestamp is stored and returned as timezone-aware UTC."""

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator[datetime]):
    """``DateTime(timezone=True)`` that rejects naive values and always returns UTC-aware ones.

    SQLite drops tzinfo on storage; values are written as UTC, so re-attaching UTC on the
    way out is exact. Postgres returns aware values already.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime rejected (timestamps must be UTC-aware): {value!r}")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UTCDateTime, float: Double}


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(16))
    config_hash: Mapped[str] = mapped_column(String(64))
    config_json: Mapped[str] = mapped_column(Text)
    git_sha: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]
    note: Mapped[str] = mapped_column(Text, default="")


class Candle(Base):
    __tablename__ = "candles"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(8), primary_key=True)
    ts: Mapped[datetime] = mapped_column(primary_key=True)
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[float]


class FundingRate(Base):
    __tablename__ = "funding_rates"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(primary_key=True)
    rate: Mapped[float]


# The run_id-leading primary keys below (regimes, open_positions, equity_snapshots) already
# index run_id; the tables with a surrogate id get an explicit run_id index.


class RegimeRow(Base):
    __tablename__ = "regimes"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(primary_key=True)
    regime: Mapped[str] = mapped_column(String(32))
    inputs_json: Mapped[str] = mapped_column(Text)


class SignalRow(Base):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_run_id", "run_id"),
        Index("ix_signals_symbol_ts", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    strategy: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32))
    ts: Mapped[datetime]
    side: Mapped[str] = mapped_column(String(8))
    entry: Mapped[float]
    stop: Mapped[float]
    target: Mapped[float]
    atr: Mapped[float]
    regime: Mapped[str] = mapped_column(String(32))


class RiskDecisionRow(Base):
    __tablename__ = "risk_decisions"
    __table_args__ = (
        Index("ix_risk_decisions_run_id", "run_id"),
        Index("ix_risk_decisions_signal_id", "signal_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"))
    ts: Mapped[datetime]
    approved: Mapped[bool] = mapped_column(Boolean)
    qty: Mapped[float]
    rule: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text)


class FillRow(Base):
    __tablename__ = "fills"
    __table_args__ = (
        Index("ix_fills_run_id", "run_id"),
        Index("ix_fills_symbol_ts", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    symbol: Mapped[str] = mapped_column(String(32))
    ts: Mapped[datetime]
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float]
    price: Mapped[float]
    ref_price: Mapped[float]
    fee: Mapped[float]
    slippage_bps: Mapped[float]
    liquidity: Mapped[str] = mapped_column(String(8))
    bid: Mapped[float | None]  # the quote a taker fill was priced from (migration 0002)
    ask: Mapped[float | None]


class TradeRow(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_run_id", "run_id"),
        Index("ix_trades_symbol_entry_ts", "symbol", "entry_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    symbol: Mapped[str] = mapped_column(String(32))
    strategy: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))
    regime: Mapped[str] = mapped_column(String(32))
    qty: Mapped[float]
    entry_ts: Mapped[datetime]
    exit_ts: Mapped[datetime]
    entry_price: Mapped[float]
    exit_price: Mapped[float]
    initial_stop: Mapped[float]
    target: Mapped[float]
    gross_pnl: Mapped[float]
    fees: Mapped[float]
    funding: Mapped[float]
    net_pnl: Mapped[float]
    exit_reason: Mapped[str] = mapped_column(String(16))
    r_multiple: Mapped[float]
    bars_held: Mapped[int] = mapped_column(Integer)


class OpenPosition(Base):
    __tablename__ = "open_positions"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    state_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime]


class RiskEventRow(Base):
    __tablename__ = "risk_events"
    __table_args__ = (Index("ix_risk_events_run_id", "run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    ts: Mapped[datetime]
    kind: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(64))


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    ts: Mapped[datetime] = mapped_column(primary_key=True)
    equity: Mapped[float]
    gross_exposure: Mapped[float]
    drawdown: Mapped[float]
