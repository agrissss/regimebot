"""Breakout: close beyond the Donchian channel on volume, shortly after compression."""

import pandas as pd

from regimebot.config import BreakoutConfig
from regimebot.strategy.base import all_finite, build_signal
from regimebot.types import Regime, Side, Signal


class Breakout:
    name = "breakout"

    def __init__(self, cfg: BreakoutConfig) -> None:
        self.cfg = cfg

    def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None:
        if window.empty:
            return None
        last = window.iloc[-1]
        close, atr = float(last["close"]), float(last["atr"])
        hi, lo = float(last["donchian_high"]), float(last["donchian_low"])
        vol_z, since = float(last["volume_z"]), float(last["bars_since_compression"])
        if not all_finite(close, atr, hi, lo, vol_z, since):
            return None
        c = self.cfg
        if vol_z < c.volume_z_min or since > c.compression_lookback:
            return None
        if close > hi:
            side, stop = Side.LONG, close - c.stop_atr_mult * atr
        elif close < lo:
            side, stop = Side.SHORT, close + c.stop_atr_mult * atr
        else:
            return None
        return build_signal(
            name=self.name, symbol=symbol, window=window, regime=regime, side=side, stop=stop, cfg=c
        )
