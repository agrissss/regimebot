"""Frozen dataclasses and enums shared by all modules.

Every timestamp is a timezone-aware UTC datetime; a bar's ``ts`` is its close time.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Regime(StrEnum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    LIQUIDITY_STRESS = "liquidity_stress"
    EVENT_RISK = "event_risk"
    NO_TRADE = "no_trade"


class Mode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"


class ExitReason(StrEnum):
    STOP = "stop"
    TARGET = "target"
    TRAIL = "trail"
    TIME = "time"
    REGIME = "regime"
    KILL = "kill"
    END_OF_DATA = "end_of_data"


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    ts: datetime
    bid: float
    ask: float


@dataclass(frozen=True, slots=True)
class Signal:
    strategy: str
    symbol: str
    ts: datetime
    side: Side
    entry: float  # reference price: decision-bar close
    stop: float
    target: float
    atr: float  # ATR at signal time, price units
    regime: Regime
    max_hold_bars: int  # decision bars
    trail_atr_mult: float | None  # None = no trailing stop


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    qty: float  # qty in base units, 0.0 when rejected
    rule: str  # machine name of the rule that decided, "ok" when approved unchanged
    detail: str


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    ts: datetime
    side: OrderSide
    qty: float
    price: float  # executed price
    ref_price: float  # intended price (decision close, stop level or target level)
    fee: float  # quote currency, positive = cost
    slippage_bps: float  # signed, positive = worse than ref_price
    liquidity: str  # "taker" | "maker"


@dataclass(frozen=True, slots=True)
class Trade:
    """One closed position."""

    symbol: str
    strategy: str
    side: Side
    regime: Regime
    qty: float
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    initial_stop: float
    target: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    exit_reason: ExitReason
    r_multiple: float  # net_pnl / (qty * |entry_price - initial_stop|)
    bars_held: int


@dataclass(frozen=True, slots=True)
class RiskEvent:
    ts: datetime
    kind: str
    detail: str
    action: str  # what the system did, e.g. "halt", "flatten", "skip_symbol"


def entry_order_side(side: Side) -> OrderSide:
    """The order that opens a ``side`` position: BUY for LONG, SELL for SHORT."""
    return OrderSide.BUY if side == Side.LONG else OrderSide.SELL


def exit_order_side(side: Side) -> OrderSide:
    """The order that closes a ``side`` position: SELL for LONG, BUY for SHORT."""
    return OrderSide.SELL if side == Side.LONG else OrderSide.BUY


def side_sign(side: Side | str) -> int:
    """Return +1 for LONG and -1 for SHORT (multiplier for price moves).

    Accepts the plain strings too (DB/pandas values); anything else raises ValueError.
    """
    return {Side.LONG: 1, Side.SHORT: -1}[Side(side)]
