"""Baselines the strategy must beat to claim an edge: buy-and-hold and random entry.

Both pay the same modelled costs as the strategy. Random entry reruns the real engine, playbook
and risk manager with only the entry decision randomised, so it isolates what the strategy's
signal adds over coin flips with the same stop/target geometry and similar trade frequency.
"""

import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from regimebot.backtest.runner import BacktestResult, run_backtest
from regimebot.config import AppConfig, BreakoutConfig, CostsConfig
from regimebot.data.resample import resample
from regimebot.reporting.metrics import active_period
from regimebot.strategy.base import build_signal
from regimebot.types import Regime, Side, Signal

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Baselines:
    """What the strategy is compared with; missing parts show as "n/a" in the report.

    ``buy_and_hold`` covers the strategy's active period (up to a permanent halt), and
    ``buy_and_hold_full`` the whole backtest. ``random_entry`` holds one result per seed, all run
    with the calibrated entry probability ``random_p``.
    """

    buy_and_hold: pd.Series | None = None
    buy_and_hold_full: pd.Series | None = None
    random_entry: tuple[BacktestResult, ...] = ()
    random_seeds: tuple[int, ...] = ()
    random_p: float | None = None
    reference_signals: int | None = None


def buy_and_hold(
    candles: Mapping[str, pd.DataFrame], initial_equity: float, costs: CostsConfig
) -> pd.Series:
    """Equity of holding equal notional (``initial_equity / n_symbols``) of every symbol at 1x.

    ``candles`` are decision-timeframe frames with a ``close`` column (UTC close-time index). Each
    symbol is bought at its first close paying half spread, slippage and the taker fee, held, and
    marked at every close; the last point is net of the same costs to sell. Cash waits until a
    symbol's first bar. The result is indexed by the union of the symbols' decision times.

    ponytail: no volume impact and no funding (longs usually pay funding, so this flatters
    buy-and-hold a little); add both if the comparison gets close.
    """
    if not candles:
        raise ValueError("buy_and_hold needs at least one symbol")
    if empty := [s for s, df in candles.items() if df.empty]:
        raise ValueError(f"no decision bars in the window for {', '.join(empty)}")
    slip = (costs.half_spread_bps + costs.slippage_bps) / 1e4
    fee = costs.taker_fee_bps / 1e4
    index = pd.DatetimeIndex(sorted(set().union(*(df.index for df in candles.values()))))
    share = initial_equity / len(candles)
    total = pd.Series(0.0, index=index)
    sell_cost = 0.0  # what selling every position at the last close costs
    for df in candles.values():
        close = df["close"].astype(float)
        qty = share / (close.iloc[0] * (1 + slip) * (1 + fee))  # buy price and fee cost the share
        total += (qty * close).reindex(index).ffill().fillna(share)
        sell_cost += qty * close.iloc[-1] * (1 - (1 - slip) * (1 - fee))
    total.iloc[-1] -= sell_cost
    return total.rename("equity")


class RandomEntry:
    """Stub strategy: enters long or short with probability ``p`` on each call.

    Named "breakout" so the playbook allows it wherever breakout is allowed; stop and target
    use breakout's geometry (ATR stop, ``build_signal`` target). Two draws per call regardless
    of outcome, so a seed gives the same sequence whatever the data does.
    """

    name = "breakout"

    def __init__(self, cfg: BreakoutConfig, p: float, seed: int) -> None:
        self.cfg, self.p = cfg, p
        self.rng = np.random.default_rng(seed)

    def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None:
        enter, long = self.rng.random() < self.p, self.rng.random() < 0.5
        if not enter or window.empty:
            return None
        close, atr = float(window["close"].iloc[-1]), float(window["atr"].iloc[-1])
        if not math.isfinite(atr):
            return None
        side = Side.LONG if long else Side.SHORT
        stop = close - self.cfg.stop_atr_mult * atr * (1 if long else -1)
        return build_signal(
            name=self.name,
            symbol=symbol,
            window=window,
            regime=regime,
            side=side,
            stop=stop,
            cfg=self.cfg,
        )


def random_entry(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    seed: int,
    *,
    p: float,
) -> BacktestResult:
    """Rerun the real engine with entries at random, probability ``p`` per call.

    The engine only asks a strategy while a symbol is flat, trading is not halted and the regime
    allows "breakout", so ``p`` per call is not a per-bar rate; ``calibrate_p`` finds the value
    that reproduces a reference signal count.
    """
    strategy = RandomEntry(cfg.strategies.breakout, min(max(p, 0.0), 1.0), seed)
    with tempfile.TemporaryDirectory() as tmp:
        return run_backtest(
            cfg, candles, funding, start, end, Path(tmp) / "KILL", strategies=[strategy]
        )


def calibrate_p(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    *,
    n_reference_signals: int,
    n_decision_bars: int,
    seed: int = 0,
) -> float:
    """Entry probability that makes random entry signal about as often as the reference.

    Start from the naive p0 = signals / (decision bars * symbols), run once, count the realised
    signals N1 and rescale: p1 = min(1, p0 * reference / max(N1, 1)). One pass is enough
    because signals grow roughly in proportion to p.
    """
    calls = max(n_decision_bars * len(cfg.exchange.symbols), 1)
    p0 = min(max(n_reference_signals / calls, 0.0), 1.0)
    n1 = len(random_entry(cfg, candles, funding, start, end, seed, p=p0).log.signals)
    p1 = min(1.0, p0 * n_reference_signals / max(n1, 1))
    log.info("random_entry_calibrated", p0=p0, n1=n1, reference=n_reference_signals, p1=p1)
    return p1


def compute_baselines(
    result: BacktestResult,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    n_signals: int,
    seeds: Sequence[int] = (0, 1, 2),
) -> Baselines:
    """Baselines for ``result`` from base-timeframe ``candles`` (``n_signals`` = the reference
    run's signal count). Buy and hold is given for the active period (up to the reference's
    first permanent halt, cut as in ``active_period``) and the full period; random entry runs once
    per seed at a calibrated probability, under the same risk limits, over that same active
    period (so frequency and window match the strategy's)."""
    cfg = result.cfg
    decision = {
        s: resample(df, cfg.exchange.base_timeframe, cfg.exchange.decision_timeframe)
        for s, df in candles.items()
    }
    lo, hi = pd.Timestamp(result.start), pd.Timestamp(result.end)
    window = {s: df[(df.index >= lo) & (df.index < hi)] for s, df in decision.items()}
    full = buy_and_hold(window, cfg.backtest.initial_equity, cfg.costs)
    _, active_equity, halt = active_period(result.trades, result.equity, result.log.risk_events)
    active, end = full, result.end
    if halt is not None:
        cut = active_equity.index[-1]  # the same cut as the strategy's headline metrics
        active = _hold_until(window, cut, cfg, full)
        end = cut.to_pydatetime() + timedelta(microseconds=1)  # decisions are in [start, end)
    seeds = tuple(seeds)
    runs: tuple[BacktestResult, ...] = ()
    p = None
    if seeds:
        p = calibrate_p(
            cfg,
            candles,
            funding,
            result.start,
            end,
            n_reference_signals=n_signals,
            n_decision_bars=len(active_equity.index.unique()),
            seed=seeds[0],
        )
        runs = tuple(random_entry(cfg, candles, funding, result.start, end, s, p=p) for s in seeds)
    return Baselines(active, full, runs, seeds, p, n_signals)


def _hold_until(
    window: Mapping[str, pd.DataFrame], cut: pd.Timestamp, cfg: AppConfig, full: pd.Series
) -> pd.Series:
    """Buy and hold up to ``cut``; a symbol with no bar by then stays cash (its share of equity)."""
    until = {s: df[df.index <= cut] for s, df in window.items()}
    held = {s: df for s, df in until.items() if not df.empty}
    initial = cfg.backtest.initial_equity
    cash = initial * (len(until) - len(held)) / len(until)
    if not held:
        return pd.Series(initial, index=full.index[full.index <= cut], name="equity")
    equity = buy_and_hold(held, initial * len(held) / len(until), cfg.costs)
    return (equity + cash).rename("equity")
