"""Test doubles shared across test modules."""

from datetime import UTC, datetime, timedelta

import ccxt

from regimebot.config import timeframe_minutes
from regimebot.types import Bar, Quote

T0 = datetime(2024, 1, 1, tzinfo=UTC)


class FakeExchange:
    """Deterministic synthetic market: one bar per timeframe step with ``ts >= T0 + step``.

    ``n_bars`` bars exist in total; pages hold at most ``page_size`` bars. Bar ``i`` (1-based)
    closes at ``T0 + i*step`` with ``close = 100 + i``. Funding events occur every 8h from
    ``T0`` (``n_funding`` of them, rate ``0.0001 * k``). The ``fetch_candles`` (and, separately,
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
    ) -> None:
        self.step = timedelta(minutes=timeframe_minutes(timeframe))
        self.n_bars = n_bars
        self.page_size = page_size
        self.n_funding = n_funding
        self.fail_on_call = fail_on_call
        self.repeat_page = repeat_page
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
        events = [(T0 + timedelta(hours=8 * k), 0.0001 * k) for k in range(self.n_funding)]
        return [e for e in events if e[0] >= since][: min(limit, self.page_size)]

    def fetch_quote(self, symbol: str) -> Quote:
        return Quote(symbol, T0, 100.0, 100.1)
