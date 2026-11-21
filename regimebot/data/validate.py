"""Candle data-quality checks."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from regimebot.config import timeframe_minutes

MAX_GAP_FRACTION = 0.001  # gaps tolerated when under 0.1% of expected bars


@dataclass(frozen=True, slots=True)
class ValidationReport:
    bars: int
    expected_bars: int  # bars on the regular grid from the first to the last bar
    duplicates: int
    gaps: tuple[tuple[datetime, datetime], ...]  # (first, last) missing bar ts per gap
    missing_bars: int
    bad_ohlc: int  # violates low <= min(open, close) <= max(open, close) <= high
    non_positive_prices: int
    negative_volume: int

    @property
    def ok(self) -> bool:
        """True when only small gaps exist (nothing else is wrong, under 0.1% missing)."""
        return (
            self.bars > 0
            and self.duplicates == 0
            and self.bad_ohlc == 0
            and self.non_positive_prices == 0
            and self.negative_volume == 0
            and self.missing_bars < MAX_GAP_FRACTION * self.expected_bars
        )


def validate_candles(df: pd.DataFrame, timeframe: str) -> ValidationReport:
    """Check a candle frame indexed by close time ``ts``."""
    if df.empty:
        return ValidationReport(0, 0, 0, (), 0, 0, 0, 0)
    step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    index = pd.DatetimeIndex(df.index)
    unique = index.unique().sort_values()
    grid = pd.date_range(unique[0], unique[-1], freq=step)
    missing = grid.difference(unique)
    gaps: list[tuple[datetime, datetime]] = []
    for ts in missing:
        if gaps and ts - step == gaps[-1][1]:
            gaps[-1] = (gaps[-1][0], ts.to_pydatetime())
        else:
            gaps.append((ts.to_pydatetime(), ts.to_pydatetime()))
    body_lo = df[["open", "close"]].min(axis=1)
    body_hi = df[["open", "close"]].max(axis=1)
    ok_ohlc = (df["low"] <= body_lo) & (body_hi <= df["high"])  # NaN compares False: counted bad
    return ValidationReport(
        bars=len(df),
        expected_bars=len(grid),
        duplicates=int(index.duplicated().sum()),
        gaps=tuple(gaps),
        missing_bars=len(missing),
        bad_ohlc=int((~ok_ohlc).sum()),
        non_positive_prices=int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        negative_volume=int((df["volume"] < 0).sum()),
    )
