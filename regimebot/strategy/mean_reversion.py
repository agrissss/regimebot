"""Mean reversion: fade a stretched z-score in a range, targeting the rolling mean."""

import pandas as pd

from regimebot.config import MeanReversionConfig
from regimebot.strategy.base import all_finite, build_signal
from regimebot.types import Regime, Side, Signal


class MeanReversion:
    name = "mean_reversion"

    def __init__(self, cfg: MeanReversionConfig) -> None:
        self.cfg = cfg

    def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None:
        if regime != Regime.RANGE or window.empty:
            return None
        last = window.iloc[-1]
        close, atr = float(last["close"]), float(last["atr"])
        z, mean = float(last["zscore"]), float(last["roll_mean"])
        if not all_finite(close, atr, z, mean):
            return None
        c = self.cfg
        if z <= -c.entry_z:
            side, stop = Side.LONG, close - c.stop_atr_mult * atr
        elif z >= c.entry_z:
            side, stop = Side.SHORT, close + c.stop_atr_mult * atr
        else:
            return None
        return build_signal(
            name=self.name,
            symbol=symbol,
            window=window,
            regime=regime,
            side=side,
            stop=stop,
            cfg=c,
            target=mean,
        )
