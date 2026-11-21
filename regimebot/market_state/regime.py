"""Rule-based regime classifier and playbook lookup.

Rules apply in priority order, first match wins (see ``classify``). ``classify_frame`` is the
vectorised form of the same rules and must agree with ``classify`` row by row.
"""

import math
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd

from regimebot.config import AppConfig, PlaybookEntry, RegimeConfig
from regimebot.types import Regime

REQUIRED_FEATURES = (
    "atr_pct",
    "vol_pctile",
    "adx",
    "ema_fast",
    "ema_slow",
    "volume_z",
    "ret_z",
    "illiq_pctile",
)


def classify(
    row: Mapping[str, float],
    cfg: RegimeConfig,
    ts: datetime,
    spread_bps: float | None = None,
) -> Regime:
    """Label one bar. Priority: NaN features, event risk, liquidity stress, expansion,
    trend, compression, range, else no-trade. Thresholds are inclusive as documented."""
    if any(math.isnan(row[k]) for k in REQUIRED_FEATURES):
        return Regime.NO_TRADE
    if any(w.start <= ts < w.end for w in cfg.event_windows) or abs(row["ret_z"]) >= cfg.shock_z:
        return Regime.EVENT_RISK
    if (spread_bps is not None and spread_bps > cfg.max_spread_bps) or (
        row["illiq_pctile"] >= cfg.illiq_pctile
    ):
        return Regime.LIQUIDITY_STRESS
    if row["vol_pctile"] >= cfg.expansion_pctile:
        return Regime.EXPANSION
    if row["adx"] >= cfg.adx_trend:
        if row["ema_fast"] > row["ema_slow"] and row["close"] > row["ema_fast"]:
            return Regime.TREND_UP
        if row["ema_fast"] < row["ema_slow"] and row["close"] < row["ema_fast"]:
            return Regime.TREND_DOWN
    if row["vol_pctile"] <= cfg.compression_pctile:
        return Regime.COMPRESSION
    if row["adx"] <= cfg.adx_range:
        return Regime.RANGE
    return Regime.NO_TRADE


def classify_frame(features: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    """Vectorised ``classify`` (no spread) over a features frame; returns Regime string values."""
    f = features
    idx = pd.DatetimeIndex(f.index)
    nan = f[list(REQUIRED_FEATURES)].isna().any(axis=1).to_numpy()
    in_event = np.zeros(len(f), dtype=bool)
    for w in cfg.event_windows:
        in_event |= (idx >= pd.Timestamp(w.start)) & (idx < pd.Timestamp(w.end))
    adx_trend = (f["adx"] >= cfg.adx_trend).to_numpy()
    up = adx_trend & ((f["ema_fast"] > f["ema_slow"]) & (f["close"] > f["ema_fast"])).to_numpy()
    down = adx_trend & ((f["ema_fast"] < f["ema_slow"]) & (f["close"] < f["ema_fast"])).to_numpy()
    conditions = [
        nan,
        in_event | (f["ret_z"].abs() >= cfg.shock_z).to_numpy(),
        (f["illiq_pctile"] >= cfg.illiq_pctile).to_numpy(),
        (f["vol_pctile"] >= cfg.expansion_pctile).to_numpy(),
        up,
        down,
        (f["vol_pctile"] <= cfg.compression_pctile).to_numpy(),
        (f["adx"] <= cfg.adx_range).to_numpy(),
    ]
    choices = [
        Regime.NO_TRADE,
        Regime.EVENT_RISK,
        Regime.LIQUIDITY_STRESS,
        Regime.EXPANSION,
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.COMPRESSION,
        Regime.RANGE,
    ]
    labels = np.select(conditions, [c.value for c in choices], default=Regime.NO_TRADE.value)
    return pd.Series(labels, index=f.index, dtype=object)


def playbook_for(regime: Regime, cfg: AppConfig) -> PlaybookEntry:
    """What the config allows in ``regime``."""
    return cfg.playbook[Regime(regime)]
