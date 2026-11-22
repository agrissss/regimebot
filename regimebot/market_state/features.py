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
# Valid once the data has a funding stamp / a compressed bar, not after a fixed warm-up.
DATA_DEPENDENT_COLUMNS = ("funding", "bars_since_compression")
_MAX_FLOAT = float(np.finfo(np.float64).max)


def warmup_bars(cfg: FeaturesConfig) -> int:
    """Decision bars after which every feature column can be valid (see ``compute_features``):
    the last row of ``compute_features`` on this many rows can be fully valid, on one fewer it
    cannot. With the default windows it is ``vol_pctile``: ``rv_window`` returns make the first
    ``rv``, then a full ``pctile_window`` of ``rv`` values is ranked."""
    first_valid_row = (
        1,  # ret
        cfg.atr_window,
        cfg.rv_window + 1,  # ret_z (rv itself at rv_window)
        cfg.rv_window + cfg.pctile_window - 1,  # vol_pctile, bars_since_compression
        cfg.pctile_window,  # illiq_pctile
        2 * cfg.adx_window - 1,  # adx (plus_di, minus_di at adx_window)
        cfg.ema_slow - 1,  # ema_fast < ema_slow
        cfg.volume_window,
        cfg.donchian_window,
        cfg.zscore_window - 1,
    )
    return max(first_valid_row) + 1


def warm_rows(features: pd.DataFrame) -> pd.Series:
    """True on the rows where every column but ``DATA_DEPENDENT_COLUMNS`` is valid (any features
    frame: daily_trend's has none of them)."""
    data_dependent = [c for c in DATA_DEPENDENT_COLUMNS if c in features.columns]
    return features.drop(columns=data_dependent).notna().all(axis=1)


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


def _pctile_rank(x: pd.Series, window: int) -> pd.Series:
    """Fraction of the trailing ``window`` values (current included) that are <= current.

    ``method="max"`` gives ties their highest rank, so the current maximum is
    1.0. Needs the full ``window`` of valid values, else NaN: a rank against a shorter history
    (a few days instead of a month) is a different, noisier feature.
    """
    return x.rolling(window, min_periods=window).rank(method="max", pct=True)


def _asof_funding(index: pd.DatetimeIndex, funding: pd.Series) -> pd.Series:
    """Last funding value stamped at or before each candle time (NaN before the first)."""
    if funding.empty:  # no stored funding yet: same as None, whatever the index type
        return pd.Series(np.nan, index=index, dtype=float)
    fidx = funding.index
    if not isinstance(fidx, pd.DatetimeIndex) or fidx.tz is None or str(fidx.tz) != "UTC":
        raise ValueError("funding must have a UTC DatetimeIndex")
    if fidx.has_duplicates:
        raise ValueError("funding index has duplicate stamps")
    f = funding.dropna().sort_index()
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
    ``bars_since_compression`` is 0 where ``vol_pctile <= compression_pctile``.

    First row (0-based) at which a column can be valid; ``warmup_bars`` is the largest plus one:

    ====================================  =========================================
    ``ret``                               1
    ``atr``, ``atr_pct``                  ``atr_window``
    ``rv``                                ``rv_window``
    ``ret_z``                             ``rv_window + 1`` (previous bar's ``rv``)
    ``vol_pctile``                        ``rv_window + pctile_window - 1`` (full
                                          ``pctile_window`` of ``rv``)
    ``illiq_pctile``                      ``pctile_window`` (full window of Amihud; a
                                          zero-volume bar ranks as the most illiquid)
    ``plus_di``, ``minus_di`` / ``adx``   ``adx_window`` / ``2 * adx_window - 1``
    ``ema_fast``, ``ema_slow``            ``ema_fast - 1``, ``ema_slow - 1``
    ``volume_z``                          ``volume_window``
    ``donchian_high``, ``donchian_low``   ``donchian_window``
    ``zscore``, ``roll_mean``             ``zscore_window - 1``
    ``funding``                           the first funding stamp (data, not config)
    ``bars_since_compression``            the first compressed bar (data, not config)
    ====================================  =========================================

    A NaN input keeps a percentile NaN while it is in the window. Zero volume does not make
    Amihud NaN: a bar with zero volume and a valid return gets the largest finite float.
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
    vol_pctile = _pctile_rank(rv, cfg.pctile_window)

    prev_volume = volume.shift(1)  # current bar excluded from its own baseline
    volume_z = _safe_div(
        volume - prev_volume.rolling(cfg.volume_window).mean(),
        prev_volume.rolling(cfg.volume_window).std(),
    )
    roll_mean = close.rolling(cfg.zscore_window).mean()
    zscore = _safe_div(close - roll_mean, close.rolling(cfg.zscore_window).std())

    # Zero volume with a valid return is maximally illiquid: the largest finite float ranks it so
    # and keeps the window full (rolling rank treats inf as missing, which would blank it).
    amihud = _safe_div(ret.abs(), close * volume).mask((volume == 0) & ret.notna(), _MAX_FLOAT)
    illiq_pctile = _pctile_rank(amihud, cfg.pctile_window)

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
