"""Download history from an exchange into the database; safe to re-run, resumes."""

from datetime import datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from regimebot.config import timeframe_minutes
from regimebot.data.exchange import Exchange
from regimebot.db import repo
from regimebot.db.models import Candle, FundingRate

log = structlog.get_logger(__name__)

CANDLE_PAGE = 1000  # Binance futures allows up to 1500 klines per request
FUNDING_PAGE = 1000  # Binance funding-rate history maximum
FUNDING_STEP = timedelta(milliseconds=1)  # ccxt timestamps are ms; `since` is inclusive


def download_candles(
    exchange: Exchange,
    session: Session,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> int:
    """Store closed bars with ``start <= ts <= end``; returns the number of bars written.

    Resumes after the last stored bar, so a re-run never fetches the stored range.
    Commits after every page (the one place outside ``session_scope``) so progress survives
    an interruption.
    """
    # ponytail: assumes the stored range is contiguous; an old interior gap is reported by
    # `data check` but not backfilled. Re-download into a fresh DB to repair it.
    width = timedelta(minutes=timeframe_minutes(timeframe))
    last = session.scalar(
        select(func.max(Candle.ts)).where(Candle.symbol == symbol, Candle.timeframe == timeframe)
    )
    cursor = start if last is None else max(start, last + width)
    total = 0
    while cursor <= end:
        bars = [
            b for b in exchange.fetch_candles(symbol, timeframe, cursor, CANDLE_PAGE) if b.ts <= end
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
    """Store funding events with ``start <= ts <= end``; resumes like ``download_candles``."""
    last = session.scalar(select(func.max(FundingRate.ts)).where(FundingRate.symbol == symbol))
    cursor = start if last is None else max(start, last + FUNDING_STEP)
    total = 0
    while cursor <= end:
        rows = [r for r in exchange.fetch_funding(symbol, cursor, FUNDING_PAGE) if r[0] <= end]
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
