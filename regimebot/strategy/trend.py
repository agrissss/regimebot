"""Trend continuation: a pullback to the fast EMA, then a close through the prior bar."""

import pandas as pd

from regimebot.config import TrendConfig
from regimebot.strategy.base import all_finite, build_signal
from regimebot.types import Regime, Side, Signal


class Trend:
    name = "trend"

    def __init__(self, cfg: TrendConfig) -> None:
        self.cfg = cfg

    def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None:
        c = self.cfg
        if regime == Regime.TREND_UP:
            side = Side.LONG
        elif regime == Regime.TREND_DOWN:
            side = Side.SHORT
        else:
            return None
        if len(window) < c.pullback_lookback + 1:
            return None
        last = window.iloc[-1]
        prior = window.iloc[-1 - c.pullback_lookback : -1]  # lookback bars, current excluded
        close, atr, vol_z = float(last["close"]), float(last["atr"]), float(last["volume_z"])
        cols = prior[["high", "low", "ema_fast", "atr"]]
        if not all_finite(close, atr, vol_z) or not cols.notna().all().all():
            return None
        if vol_z < c.volume_z_min:
            return None
        band = c.pullback_atr_dist * prior["atr"]
        if side == Side.LONG:
            pulled_back = (prior["low"] <= prior["ema_fast"] + band).any()
            resumed = close > float(prior["high"].iloc[-1])
            stop = float(prior["low"].min()) - c.stop_atr_mult * atr
        else:
            pulled_back = (prior["high"] >= prior["ema_fast"] - band).any()
            resumed = close < float(prior["low"].iloc[-1])
            stop = float(prior["high"].max()) + c.stop_atr_mult * atr
        if not (pulled_back and resumed):
            return None
        return build_signal(
            name=self.name, symbol=symbol, window=window, regime=regime, side=side, stop=stop, cfg=c
        )
