"""Download history from an exchange into the database; safe to re-run, resumes."""

from datetime import datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from regimebot.config import timeframe_minutes
from regimebot.data.exchange import Exchange
from regimebot.db import repo

log = structlog.get_logger(__name__)

# 1000 rows per request: on Binance the request weight steps up above 1000 rows, so this is the
# largest page at the lowest weight per bar.
CANDLE_PAGE = 1000
FUNDING_PAGE = 1000  # Binance funding-rate history maximum
FUNDING_STEP = timedelta(milliseconds=1)  # ccxt timestamps are ms; `since` is inclusive


def _segments(
    bounds: tuple[datetime, datetime] | None,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> list[tuple[datetime, datetime]]:
    """Ranges still to fetch: before the stored range, then after it (or all of it if empty)."""
    if bounds is None:
        return [(start, end)]
    first, last = bounds
    return [(start, min(end, first - step)), (max(start, last + step), end)]


def download_candles(
    exchange: Exchange,
    session: Session,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> int:
    """Store closed bars with ``start <= ts <= end``; returns the number of bars written.

    ``end`` is inclusive here, while ``load_candles(start, end)`` is half-open (``ts < end``).
    Fetches only what is not stored: any gap before the earliest stored bar, then everything
    after the latest. Commits after every page (the one place outside ``session_scope``) so
    progress survives an interruption.
    """
    # ponytail: assumes the stored range is contiguous; an interior gap is reported by
    # `data check` but not backfilled. Re-download into a fresh DB to repair it.
    width = timedelta(minutes=timeframe_minutes(timeframe))
    bounds = repo.candle_ts_bounds(session, symbol, timeframe)
    total = 0
    for cursor, stop in _segments(bounds, start, end, width):
        while cursor <= stop:
            bars = [
                b
                for b in exchange.fetch_candles(symbol, timeframe, cursor, CANDLE_PAGE)
                if b.ts <= stop
            ]
            if not bars:
                break
            frame = pd.DataFrame(
                [(b.open, b.high, b.low, b.close, b.volume) for b in bars],
                columns=repo.CANDLE_COLUMNS,
                index=pd.DatetimeIndex([b.ts for b in bars], name="ts"),
            )
            total += repo.upsert_candles(session, symbol, timeframe, frame)
            session.commit()  # each page is durable, so an interruption resumes here
            log.info(
                "candles_page", symbol=symbol, timeframe=timeframe, rows=len(bars), last=bars[-1].ts
            )
            next_cursor = bars[-1].ts + width
            if next_cursor <= cursor:  # exchange keeps returning the same page
                break
            cursor = next_cursor
    return total


def download_funding(
    exchange: Exchange, session: Session, symbol: str, start: datetime, end: datetime
) -> int:
    """Store funding events with ``start <= ts <= end``; fetches only what is not stored."""
    bounds = repo.funding_ts_bounds(session, symbol)
    total = 0
    for cursor, stop in _segments(bounds, start, end, FUNDING_STEP):
        while cursor <= stop:
            rows = [r for r in exchange.fetch_funding(symbol, cursor, FUNDING_PAGE) if r[0] <= stop]
            if not rows:
                break
            series = pd.Series(
                [r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows], name="ts")
            )
            total += repo.upsert_funding(session, symbol, series)
            session.commit()
            log.info("funding_page", symbol=symbol, rows=len(rows), last=rows[-1][0])
            next_cursor = rows[-1][0] + FUNDING_STEP
            if next_cursor <= cursor:
                break
            cursor = next_cursor
    return total
