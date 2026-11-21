"""Signal generators: breakout, trend continuation, mean reversion."""

from regimebot.config import AppConfig
from regimebot.strategy.base import Strategy
from regimebot.strategy.breakout import Breakout
from regimebot.strategy.mean_reversion import MeanReversion
from regimebot.strategy.trend import Trend

__all__ = ["Breakout", "MeanReversion", "Strategy", "Trend", "build_strategies"]


def build_strategies(cfg: AppConfig) -> list[Strategy]:
    """Instantiate the enabled strategies, in a fixed order."""
    s = cfg.strategies
    out: list[Strategy] = []
    if s.breakout.enabled:
        out.append(Breakout(s.breakout))
    if s.trend.enabled:
        out.append(Trend(s.trend))
    if s.mean_reversion.enabled:
        out.append(MeanReversion(s.mean_reversion))
    return out
