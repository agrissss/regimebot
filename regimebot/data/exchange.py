"""Exchange protocol and the ccxt adapter.

ccxt reports a candle by its **open** time; this module converts to the project convention
(``Bar.ts`` = close time) and never returns the still-forming bar.
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import ccxt
import structlog

from regimebot.config import timeframe_minutes
from regimebot.types import Bar, Quote

log = structlog.get_logger(__name__)

MAX_RETRIES = 5  # retries after the first attempt, so up to 6 calls in total
BACKOFF_SECONDS = 1.0  # first wait; doubles each retry


class Exchange(Protocol):
    """Public market data. All datetimes are UTC-aware."""

    def fetch_candles(self, symbol: str, timeframe: str, since: datetime, limit: int) -> list[Bar]:
        """Closed bars with ``ts >= since`` (``ts`` = close time), ascending, at most ``limit``."""
        ...

    def fetch_funding(
        self, symbol: str, since: datetime, limit: int
    ) -> list[tuple[datetime, float]]:
        """Funding events ``(ts, rate)`` with ``ts >= since``, ascending, at most ``limit``."""
        ...

    def fetch_quote(self, symbol: str) -> Quote:
        """Current best bid/ask."""
        ...


def _to_ms(ts: datetime) -> int:
    return round(ts.timestamp() * 1000)


def _from_ms(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


class CcxtExchange:
    """``Exchange`` over the synchronous ccxt client; public endpoints only, no API keys.

    ``client``, ``sleep`` and ``now`` are injectable for tests.
    """

    def __init__(
        self,
        exchange_id: str,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client or getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self._sleep = sleep
        self._now = now

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Call ``fn``, retrying ``ccxt.NetworkError`` with exponential backoff."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return fn(*args)
            except ccxt.NetworkError as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait = BACKOFF_SECONDS * 2**attempt
                log.warning("exchange_retry", attempt=attempt + 1, wait_s=wait, error=str(exc))
                self._sleep(wait)
        raise AssertionError("unreachable")

    def fetch_candles(self, symbol: str, timeframe: str, since: datetime, limit: int) -> list[Bar]:
        width = timedelta(minutes=timeframe_minutes(timeframe))
        # ccxt's `since` is an open time; our `since` is a close time.
        rows = self._call(self._client.fetch_ohlcv, symbol, timeframe, _to_ms(since - width), limit)
        now = self._now()
        bars = [Bar(symbol, _from_ms(r[0]) + width, *(float(x) for x in r[1:6])) for r in rows]
        # Drop the forming bar (not closed yet) and anything before `since`.
        return [b for b in bars if since <= b.ts <= now]

    def fetch_funding(
        self, symbol: str, since: datetime, limit: int
    ) -> list[tuple[datetime, float]]:
        rows = self._call(self._client.fetch_funding_rate_history, symbol, _to_ms(since), limit)
        return [
            (_from_ms(r["timestamp"]), float(r["fundingRate"]))
            for r in rows
            if r.get("fundingRate") is not None and _from_ms(r["timestamp"]) >= since
        ]

    def fetch_quote(self, symbol: str) -> Quote:
        # Not fetch_ticker: on Binance USD-M it returns bid/ask = None (checked live).
        t = self._call(self._client.fetch_bids_asks, [symbol])[symbol]
        return Quote(symbol, self._now(), float(t["bid"]), float(t["ask"]))
