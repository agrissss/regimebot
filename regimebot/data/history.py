"""Download history from an exchange into the database; safe to re-run, resumes."""

from datetime import datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from regimebot.config import timeframe_minutes
from regimebot.data.exchange import Exchange
from regimebot.data.validate import funding_gaps, gap_ranges
from regimebot.db import repo

log = structlog.get_logger(__name__)

# 1000 rows per request: the largest page before Binance's klines request weight steps up
# (weights by limit: <100 -> 1, <500 -> 2, <=1000 -> 5, <=1500 -> 10).
CANDLE_PAGE = 1000
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

    ``end`` is inclusive here, while ``load_candles(start, end)`` is half-open (``ts < end``).
    Fetches only the real holes: the epoch-aligned bar grid in ``[start, end]`` minus the stored
    timestamps, so an interrupted run (or an interior gap) is repaired by re-running with the same
    arguments. A fully stored range makes no request. A hole the exchange has no data for (outage,
    pre-listing) costs one empty request per run and never loops. Commits after every page (the
    one place outside ``session_scope``) so progress survives an interruption.
    """
    width = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    first, last = pd.Timestamp(start).ceil(width), pd.Timestamp(end).floor(width)
    grid = pd.date_range(first, last, freq=width) if first <= last else pd.DatetimeIndex([])
    # ponytail: loads every stored timestamp in the range (about 1M per symbol for two years of
    # 1m bars, ~8 MB); a SQL gap query (LAG over ts) would return only the holes if that grows.
    stored = repo.candle_timestamps(session, symbol, timeframe, start, end)
    total = 0
    for hole_start, hole_end in gap_ranges(grid.difference(stored), width):
        total += _fill_candles(
            exchange, session, symbol, timeframe, hole_start, hole_end, width.to_pytimedelta()
        )
    return total


def _fill_candles(
    exchange: Exchange,
    session: Session,
    symbol: str,
    timeframe: str,
    cursor: datetime,
    stop: datetime,
    width: timedelta,
) -> int:
    """Page ``[cursor, stop]``, committing each page; ends on an empty page or no progress."""
    total = 0
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
    exchange: Exchange,
    session: Session,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    live: bool = False,
) -> int:
    """Store funding events with ``start <= ts <= end``; returns the number of rows written.

    Fetches only the holes: every stretch of the range without a stored stamp that is longer than
    ``FUNDING_GAP_TOLERANCE`` (before the first stamp, between two, after the last), so an
    interrupted run, an interior hole or the gap behind a paper session's recent window is
    repaired by re-running with the same arguments. A range with no stored stamp is fetched
    whole; a covered range makes no request. The stretch after the last stamp commits per page
    (an interruption leaves it a shorter stretch at the end); every other hole is all-or-nothing
    (one commit, rollback on error), so an interrupted fill never leaves a new interior hole. A
    hole the exchange has no data for costs one request per run and never loops.

    ``live`` (a paper poll): the stretch after the last stamp is always fetched, so a new event is
    stored as soon as the exchange has it, not once the stretch passes the tolerance.
    """
    stamps = list(repo.funding_timestamps(session, symbol, start, end).to_pydatetime())
    if not stamps:
        return _fill_funding(exchange, session, symbol, start, end, commit_pages=True)
    # Bounds one step outside the range: each gap's fetch is then (after + step, before - step).
    gaps = funding_gaps(stamps, start - FUNDING_STEP, end + FUNDING_STEP)
    total = 0
    for after, before in gaps:
        if before > end:  # the stretch after the last stamp: below
            continue
        try:
            total += _fill_funding(
                exchange,
                session,
                symbol,
                after + FUNDING_STEP,
                before - FUNDING_STEP,
                commit_pages=False,
            )
            session.commit()
        except BaseException:
            session.rollback()
            raise
    if live or (gaps and gaps[-1][1] > end):
        total += _fill_funding(
            exchange, session, symbol, stamps[-1] + FUNDING_STEP, end, commit_pages=True
        )
    return total


def _fill_funding(
    exchange: Exchange,
    session: Session,
    symbol: str,
    cursor: datetime,
    stop: datetime,
    commit_pages: bool,
) -> int:
    total = 0
    while cursor <= stop:
        rows = [r for r in exchange.fetch_funding(symbol, cursor, FUNDING_PAGE) if r[0] <= stop]
        if not rows:
            break
        series = pd.Series(
            [r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows], name="ts")
        )
        total += repo.upsert_funding(session, symbol, series)
        if commit_pages:
            session.commit()
        log.info("funding_page", symbol=symbol, rows=len(rows), last=rows[-1][0])
        next_cursor = rows[-1][0] + FUNDING_STEP
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return total
