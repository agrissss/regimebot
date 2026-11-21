"""Test doubles shared across test modules."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import ccxt
import pandas as pd

from regimebot.config import CostsConfig, timeframe_minutes
from regimebot.data.exchange import CLOSE_GRACE
from regimebot.execution.fill_model import quote_from_bar_open
from regimebot.types import Bar, Quote

T0 = datetime(2024, 1, 1, tzinfo=UTC)


class FakeExchange:
    """Deterministic synthetic market: one bar per timeframe step with ``ts >= T0 + step``.

    ``n_bars`` bars exist in total; pages hold at most ``page_size`` bars. Bar ``i`` (1-based)
    closes at ``T0 + i*step`` with ``close = 100 + i``. Funding events occur every 8h from
    ``T0`` (``n_funding`` of them, rate ``0.0001 * k``), except the ``k`` in ``funding_missing``
    (the exchange has no record for them). The ``fetch_candles`` (and, separately,
    ``fetch_funding``) call numbered ``fail_on_call`` (1-based) raises ``ccxt.NetworkError``
    (simulated interruption).
    ``repeat_page`` makes the exchange ignore ``since`` and always return its first page.
    """

    def __init__(
        self,
        n_bars: int,
        page_size: int = 100,
        timeframe: str = "1m",
        n_funding: int = 0,
        fail_on_call: int | None = None,
        repeat_page: bool = False,
        funding_missing: range = range(0),
    ) -> None:
        self.step = timedelta(minutes=timeframe_minutes(timeframe))
        self.n_bars = n_bars
        self.page_size = page_size
        self.n_funding = n_funding
        self.fail_on_call = fail_on_call
        self.repeat_page = repeat_page
        self.funding_missing = funding_missing
        self.candle_calls: list[datetime] = []  # `since` of every fetch_candles call
        self.funding_calls: list[datetime] = []

    def _bar(self, i: int, symbol: str) -> Bar:
        c = 100.0 + i
        return Bar(symbol, T0 + i * self.step, c, c + 1, c - 1, c, 10.0)

    def fetch_candles(self, symbol: str, timeframe: str, since: datetime, limit: int) -> list[Bar]:
        self.candle_calls.append(since)
        if len(self.candle_calls) == self.fail_on_call:
            raise ccxt.NetworkError("injected network error")
        first = 1 if self.repeat_page else max(1, -((since - T0) // -self.step))  # ceil
        last = min(self.n_bars, first + min(limit, self.page_size) - 1)
        return [self._bar(i, symbol) for i in range(first, last + 1)]

    def fetch_funding(
        self, symbol: str, since: datetime, limit: int
    ) -> list[tuple[datetime, float]]:
        self.funding_calls.append(since)
        if len(self.funding_calls) == self.fail_on_call:
            raise ccxt.NetworkError("injected network error")
        events = [
            (T0 + timedelta(hours=8 * k), 0.0001 * k)
            for k in range(self.n_funding)
            if k not in self.funding_missing
        ]
        return [e for e in events if e[0] >= since][: min(limit, self.page_size)]

    def fetch_quote(self, symbol: str) -> Quote:
        return Quote(symbol, T0, 100.0, 100.1)


@dataclass
class Clock:
    """A settable clock: tests move ``now`` between polls."""

    now: datetime

    def __call__(self) -> datetime:
        return self.now


class ReplayExchange:
    """A live market replayed from scripted base ``candles`` on a test ``clock``.

    ``fetch_candles`` returns the bars closed at least ``CLOSE_GRACE`` (plus ``lag[symbol]``, a
    slow feed) before ``clock()``. ``fetch_quote`` prices the bar still forming at ``clock()``
    exactly as the backtest prices a fill: its open -/+ ``half_spread_bps``. ``fetch_funding``
    returns the events stamped up to ``clock()``. A ``(method, symbol)`` pair in ``fail`` raises
    ``ccxt.NetworkError``; ``calls`` records every method called.
    """

    def __init__(
        self,
        candles: Mapping[str, pd.DataFrame],
        funding: Mapping[str, pd.Series],
        costs: CostsConfig,
        clock: Clock,
    ) -> None:
        self.candles = candles
        self.funding = funding
        self.costs = costs
        self.clock = clock
        self.fail: set[tuple[str, str]] = set()
        self.lag: dict[str, timedelta] = {}
        self.calls: list[str] = []

    def _call(self, method: str, symbol: str) -> None:
        self.calls.append(method)
        if (method, symbol) in self.fail:
            raise ccxt.NetworkError(f"injected {method} failure for {symbol}")

    def fetch_candles(self, symbol: str, timeframe: str, since: datetime, limit: int) -> list[Bar]:
        self._call("fetch_candles", symbol)
        cutoff = self.clock() - CLOSE_GRACE - self.lag.get(symbol, timedelta(0))
        df = self.candles[symbol]
        df = df[(df.index >= since) & (df.index <= cutoff)].iloc[:limit]
        rows = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float).tolist()
        stamps = pd.DatetimeIndex(df.index).to_pydatetime()
        return [Bar(symbol, ts, *row) for ts, row in zip(stamps, rows, strict=True)]

    def fetch_funding(
        self, symbol: str, since: datetime, limit: int
    ) -> list[tuple[datetime, float]]:
        self._call("fetch_funding", symbol)
        rates = self.funding.get(symbol)
        if rates is None:
            return []
        rates = rates[(rates.index >= since) & (rates.index <= self.clock())].iloc[:limit]
        stamps = pd.DatetimeIndex(rates.index).to_pydatetime()
        return list(zip(stamps, rates.to_numpy(dtype=float).tolist(), strict=True))

    def fetch_quote(self, symbol: str) -> Quote:
        self._call("fetch_quote", symbol)
        now = self.clock()
        df = self.candles[symbol]
        forming = df[df.index > now]
        price = forming["open"].iloc[0] if len(forming) else df["close"].iloc[-1]
        bid, ask = quote_from_bar_open(float(price), self.costs)
        return Quote(symbol, now, bid, ask)
