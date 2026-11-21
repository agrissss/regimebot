"""Causal indicator features for the regime classifier and strategies.

Causality contract: row *t* of every output column depends only on input rows
``<= t`` (and funding stamped ``<= t``). Only trailing ``rolling``, ``ewm``,
``shift(k>=1)`` and ``ffill`` are used, never ``shift(-k)``,
centred windows or ``bfill``. Warm-up rows are NaN and never filled. A
prefix-invariance test (``tests/test_features.py``) enforces this per column.
"""

import numpy as np
import pandas as pd

from regimebot.config import FeaturesConfig

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate(candles: pd.DataFrame) -> pd.DatetimeIndex:
    missing = [c for c in REQUIRED_COLUMNS if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing columns: {missing}")
    idx = candles.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or str(idx.tz) != "UTC":
        raise ValueError("candles must have a UTC DatetimeIndex (bar close times)")
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("candle index must be sorted ascending without duplicates")
    return idx


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN (never inf) where den == 0."""
    return num / den.where(den != 0)


def _wilder(x: pd.Series, window: int) -> pd.Series:
    """Wilder smoothing: ``ewm(alpha=1/window, adjust=False)``.

    Seeding: the recursion is seeded with the first valid observation of ``x``
    and the output is NaN until ``window`` valid observations exist. For a
    true range series (first valid value at row 1) the first ATR is at row
    ``window``.
    """
    return x.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _pctile_rank(x: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Fraction of the trailing ``window`` values (current included) that are <= current.

    ``method="max"`` gives ties their highest rank, so the current maximum is
    1.0. Needs at least ``min_obs`` valid values in the window, else NaN.
    """
    return x.rolling(window, min_periods=min_obs).rank(method="max", pct=True)


def _asof_funding(index: pd.DatetimeIndex, funding: pd.Series) -> pd.Series:
    """Last funding value stamped at or before each candle time (NaN before the first)."""
    if not isinstance(funding.index, pd.DatetimeIndex) or funding.index.tz is None:
        raise ValueError("funding must have a tz-aware DatetimeIndex")
    f = funding.dropna().sort_index()
    f = f[~f.index.duplicated(keep="last")]
    # side="right" - 1 = last stamp <= t; never a stamp after t.
    pos = f.index.searchsorted(index, side="right") - 1
    vals = np.where(pos >= 0, f.to_numpy(dtype=float)[np.maximum(pos, 0)], np.nan)
    return pd.Series(vals, index=index, dtype=float)


def _adx(high: pd.Series, low: pd.Series, tr: pd.Series, window: int) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    tr_s = _wilder(tr, window)
    plus_di = 100.0 * _safe_div(_wilder(plus_dm, window), tr_s)
    minus_di = 100.0 * _safe_div(_wilder(minus_dm, window), tr_s)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return pd.DataFrame({"adx": _wilder(dx, window), "plus_di": plus_di, "minus_di": minus_di})


def compute_features(
    candles: pd.DataFrame,
    cfg: FeaturesConfig,
    compression_pctile: float,
    funding: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute causal feature columns on the decision-timeframe candles.

    ``candles``: UTC close-time index, float columns open/high/low/close/volume.
    ``funding``: optional funding-rate series (tz-aware index), as-of joined.
    ``vol_pctile`` / ``illiq_pctile`` need at least ``rv_window`` valid
    observations in the trailing ``pctile_window`` (not a full window).
    ``bars_since_compression`` is 0 where ``vol_pctile <= compression_pctile``.
    """
    index = _validate(candles)
    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    ret = pd.Series(np.log(close.to_numpy()), index=index).diff()
    prev_close = close.shift(1)
    # True range is NaN on the first bar (no previous close).
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1, skipna=False
    )
    atr = _wilder(tr, cfg.atr_window)
    rv = ret.rolling(cfg.rv_window).std()
    vol_pctile = _pctile_rank(rv, cfg.pctile_window, cfg.rv_window)

    prev_volume = volume.shift(1)  # current bar excluded from its own baseline
    volume_z = _safe_div(
        volume - prev_volume.rolling(cfg.volume_window).mean(),
        prev_volume.rolling(cfg.volume_window).std(),
    )
    roll_mean = close.rolling(cfg.zscore_window).mean()
    zscore = _safe_div(close - roll_mean, close.rolling(cfg.zscore_window).std())

    amihud = _safe_div(ret.abs(), close * volume)  # zero volume -> NaN, never inf
    illiq_pctile = _pctile_rank(amihud, cfg.pctile_window, cfg.rv_window)

    compressed = (vol_pctile <= compression_pctile).to_numpy()  # NaN compares False
    pos = pd.Series(np.arange(len(candles), dtype=float), index=candles.index)
    last_compressed = pos.where(compressed).ffill()  # ffill only looks backwards
    bars_since = pos - last_compressed

    out = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "ret": ret,
            "atr": atr,
            "atr_pct": _safe_div(atr, close),
            "rv": rv,
            "vol_pctile": vol_pctile,
        },
        index=candles.index,
    )
    out = out.join(_adx(high, low, tr, cfg.adx_window))
    out["ema_fast"] = close.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    out["ema_slow"] = close.ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()
    out["volume_z"] = volume_z
    out["donchian_high"] = high.shift(1).rolling(cfg.donchian_window).max()
    out["donchian_low"] = low.shift(1).rolling(cfg.donchian_window).min()
    out["zscore"] = zscore
    out["roll_mean"] = roll_mean
    out["ret_z"] = _safe_div(ret, rv.shift(1))  # previous rv: a shock bar can't dilute itself
    out["illiq_pctile"] = illiq_pctile
    out["funding"] = np.nan if funding is None else _asof_funding(index, funding)
    out["bars_since_compression"] = bars_since
    return out
