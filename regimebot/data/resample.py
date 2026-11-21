"""Aggregate base bars into larger bars, labelled by close time."""

import pandas as pd

from regimebot.config import timeframe_minutes

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(df: pd.DataFrame, timeframe_from: str, timeframe_to: str) -> pd.DataFrame:
    """Aggregate bars indexed by close time into ``timeframe_to`` bars, labelled by close time.

    A base bar closing at ``t`` belongs to the target bar closing at ``ceil(t)`` on the
    epoch-aligned grid (1m bars closing 00:01..00:15 make the 15m bar 00:15). Buckets missing
    any base bar are dropped, never filled.
    """
    from_min, to_min = timeframe_minutes(timeframe_from), timeframe_minutes(timeframe_to)
    if to_min % from_min:
        raise ValueError(f"{timeframe_to} is not a multiple of {timeframe_from}")
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps; validate the data first")
    label = pd.DatetimeIndex(df.index).ceil(f"{to_min}min")
    grouped = df.groupby(label)
    out = grouped.agg(_AGG)
    out = out[grouped.size() == to_min // from_min]
    out.index.name = "ts"
    return out.astype("float64")
