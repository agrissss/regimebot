"""Signal generators (breakout, trend continuation, mean reversion) and daily trend following."""

from regimebot.config import AppConfig
from regimebot.strategy.base import AnyStrategy, Strategy, TargetStrategy
from regimebot.strategy.breakout import Breakout
from regimebot.strategy.daily_trend import DailyTrend
from regimebot.strategy.mean_reversion import MeanReversion
from regimebot.strategy.trend import Trend

__all__ = [
    "AnyStrategy",
    "Breakout",
    "DailyTrend",
    "MeanReversion",
    "Strategy",
    "TargetStrategy",
    "Trend",
    "build_strategies",
]


def build_strategies(cfg: AppConfig) -> list[AnyStrategy]:
    """Instantiate the enabled strategies, in a fixed order: daily trend alone when it is enabled
    (config validation keeps the others off), else the enabled playbook strategies."""
    s = cfg.strategies
    if s.daily_trend.enabled:
        return [DailyTrend(s.daily_trend, cfg.risk, cfg.exchange.symbols)]
    out: list[AnyStrategy] = []
    if s.breakout.enabled:
        out.append(Breakout(s.breakout))
    if s.trend.enabled:
        out.append(Trend(s.trend))
    if s.mean_reversion.enabled:
        out.append(MeanReversion(s.mean_reversion))
    return out
