"""Performance metrics from a backtest's trades and equity curve.

Every ratio is NaN (never 0 or a made-up number) when the data cannot support it, and the report
prints NaN as "n/a". Daily statistics use the last equity value of each UTC day.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regimebot.risk.manager import PERMANENT_HALTS
from regimebot.types import ExitReason, Fill, RiskEvent, Trade, entry_order_side, exit_order_side

DAYS_PER_YEAR = 365  # crypto trades every day
BEST_TRADES_DROPPED = 0.05
_FLAT_STD = 1e-12  # a deviation below this is rounding noise, not risk


@dataclass(frozen=True, slots=True)
class Metrics:
    """Headline numbers of one run. Units: ``net_pnl``, ``avg_win``, ``avg_loss``, ``fees``,
    ``funding`` and ``net_pnl_without_best_5pct`` are currency; ``return_pct``, ``max_drawdown``,
    ``win_rate``, ``exposure_pct``, ``target_hit_rate`` and ``stop_out_rate`` are fractions
    (0.1 = 10%); ``avg_*_slippage_bps`` are basis points; ``expectancy_r`` is in R multiples;
    ``turnover`` and ``max_gross_leverage_seen`` (largest gross exposure / equity at a
    snapshot) are multiples of equity; ``sharpe``, ``sortino`` and ``profit_factor`` are plain
    ratios. NaN means "not computable from this data"."""

    net_pnl: float
    return_pct: float  # fraction of initial equity
    max_drawdown: float  # fraction, >= 0
    sharpe: float
    sortino: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float  # negative
    expectancy_r: float
    n_trades: int
    avg_bars_held: float
    exposure_pct: float
    turnover: float
    fees: float
    funding: float
    target_hit_rate: float
    stop_out_rate: float
    net_pnl_without_best_5pct: float
    avg_entry_slippage_bps: float
    avg_exit_slippage_bps: float
    max_gross_leverage_seen: float


def daily_equity(equity: pd.Series) -> pd.Series:
    """Last equity value of each UTC day."""
    if equity.empty:
        return pd.Series(dtype=float)
    days = pd.DatetimeIndex(equity.index).tz_convert("UTC").normalize()
    return equity.groupby(days).last()


def daily_returns(equity: pd.Series) -> pd.Series:
    """Percentage change of the last equity value of each UTC day."""
    return daily_equity(equity).pct_change().dropna()


def sharpe_ratio(equity: pd.Series) -> float:
    """mean / std (ddof=1) of daily returns * sqrt(365); NaN with under 2 returns or no spread."""
    r = daily_returns(equity)
    if len(r) < 2:
        return math.nan
    std = float(r.std(ddof=1))
    return math.nan if std < _FLAT_STD else float(r.mean()) / std * math.sqrt(DAYS_PER_YEAR)


def sortino_ratio(equity: pd.Series) -> float:
    """mean / downside deviation * sqrt(365), downside deviation = sqrt(mean(min(r, 0)^2));
    NaN with under 2 returns or no losing day."""
    r = daily_returns(equity)
    if len(r) < 2:
        return math.nan
    downside = math.sqrt(float(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    return (
        math.nan if downside < _FLAT_STD else float(r.mean()) / downside * math.sqrt(DAYS_PER_YEAR)
    )


def max_drawdown(equity: pd.Series, initial_equity: float) -> float:
    """Largest peak-to-trough fall as a fraction of the peak, starting from ``initial_equity``."""
    values = np.concatenate([[initial_equity], equity.to_numpy(dtype=float)])
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))


def best_trades_dropped(n_trades: int) -> int:
    """ceil(5% of n) in integer arithmetic (0.05 * n can round the wrong way)."""
    return -(-n_trades * 5 // 100)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _profit_factor(pnls: Sequence[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses > 0:
        return wins / losses
    return math.inf if wins > 0 else math.nan


def _exposure(trades: Sequence[Trade], equity: pd.Series) -> float:
    """Share of decision timestamps inside any trade's [entry_ts, exit_ts)."""
    stamps = np.unique(pd.DatetimeIndex(equity.index).as_unit("ns").asi8)
    if len(stamps) == 0:
        return math.nan
    entries = np.sort(pd.DatetimeIndex([t.entry_ts for t in trades]).as_unit("ns").asi8)
    exits = np.sort(pd.DatetimeIndex([t.exit_ts for t in trades]).as_unit("ns").asi8)
    # trades open at ts = entries at or before ts minus exits at or before ts
    open_count = np.searchsorted(entries, stamps, "right") - np.searchsorted(exits, stamps, "right")
    return float(np.mean(open_count > 0))


def _slippage(trades: Sequence[Trade], fills: Sequence[Fill]) -> tuple[float, float]:
    """Mean taker-fill slippage (bps) of entry fills and of exit fills, matched to the trades by
    (symbol, ts, side)."""
    entry_keys = {(t.symbol, t.entry_ts, entry_order_side(t.side)) for t in trades}
    exit_keys = {(t.symbol, t.exit_ts, exit_order_side(t.side)) for t in trades}
    taker = [f for f in fills if f.liquidity == "taker"]
    entries = [f.slippage_bps for f in taker if (f.symbol, f.ts, f.side) in entry_keys]
    exits = [f.slippage_bps for f in taker if (f.symbol, f.ts, f.side) in exit_keys]
    return _mean(entries), _mean(exits)


def _max_leverage(equity: pd.Series, gross_exposure: pd.Series | None) -> float:
    if gross_exposure is None or equity.empty:
        return math.nan
    leverage = gross_exposure.reindex(equity.index) / equity
    return float(leverage.max()) if leverage.notna().any() else math.nan


def compute_metrics(
    trades: Sequence[Trade],
    equity: pd.Series,
    initial_equity: float,
    fills: Sequence[Fill] = (),
    gross_exposure: pd.Series | None = None,
) -> Metrics:
    """All headline metrics. ``equity`` is the decision-time equity curve on a UTC index;
    ``gross_exposure`` is the gross exposure at each equity snapshot (only the stamps in
    ``equity`` count; without it ``max_gross_leverage_seen`` is NaN)."""
    pnls = [t.net_pnl for t in trades]
    n = len(trades)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    drop = best_trades_dropped(n)
    kept = sorted(pnls, reverse=True)[drop:]
    stopped = sum(
        t.exit_reason == ExitReason.STOP or (t.exit_reason == ExitReason.TRAIL and t.net_pnl < 0)
        for t in trades
    )
    net_pnl = float(sum(pnls))
    final = float(equity.iloc[-1]) if len(equity) else initial_equity + net_pnl
    entry_slip, exit_slip = _slippage(trades, fills)
    return Metrics(
        net_pnl=net_pnl,
        return_pct=final / initial_equity - 1.0,
        max_drawdown=max_drawdown(equity, initial_equity),
        sharpe=sharpe_ratio(equity),
        sortino=sortino_ratio(equity),
        profit_factor=_profit_factor(pnls),
        win_rate=len(winners) / n if n else math.nan,
        avg_win=_mean(winners),
        avg_loss=_mean(losers),
        expectancy_r=_mean([t.r_multiple for t in trades]),
        n_trades=n,
        avg_bars_held=_mean([t.bars_held for t in trades]),
        exposure_pct=_exposure(trades, equity) if n else (0.0 if len(equity) else math.nan),
        turnover=sum(t.qty * t.entry_price for t in trades) / initial_equity,
        fees=float(sum(t.fees for t in trades)),
        funding=float(sum(t.funding for t in trades)),
        target_hit_rate=(
            sum(t.exit_reason == ExitReason.TARGET for t in trades) / n if n else math.nan
        ),
        stop_out_rate=stopped / n if n else math.nan,
        net_pnl_without_best_5pct=float(sum(kept)),
        avg_entry_slippage_bps=entry_slip,
        avg_exit_slippage_bps=exit_slip,
        max_gross_leverage_seen=_max_leverage(equity, gross_exposure),
    )


def breakdown(
    trades: Sequence[Trade], by: Literal["regime", "symbol", "strategy", "exit_reason"]
) -> pd.DataFrame:
    """Per-group trade count, win rate, net pnl and mean R, indexed by the group value."""
    columns = ["n", "win_rate", "net_pnl", "expectancy_r"]
    if not trades:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=by), dtype=float)
    frame = pd.DataFrame(
        {
            by: [str(getattr(t, by)) for t in trades],
            "net_pnl": [t.net_pnl for t in trades],
            "r": [t.r_multiple for t in trades],
        }
    )
    frame["win"] = frame["net_pnl"] > 0
    grouped = frame.groupby(by)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "win_rate": grouped["win"].mean(),
            "net_pnl": grouped["net_pnl"].sum(),
            "expectancy_r": grouped["r"].mean(),
        }
    )
    return out.sort_index()


def first_permanent_halt(events: Sequence[RiskEvent]) -> RiskEvent | None:
    """The earliest halt that ends trading (``max_drawdown`` or ``manual_kill``), if any."""
    halts = [e for e in events if e.kind in PERMANENT_HALTS and e.action == "halt"]
    return min(halts, key=lambda e: e.ts, default=None)


def active_period(
    trades: Sequence[Trade], equity: pd.Series, events: Sequence[RiskEvent]
) -> tuple[list[Trade], pd.Series, RiskEvent | None]:
    """Trades and equity up to the first permanent halt (everything when there is none).

    After such a halt the strategy sits flat, and averaging that into Sharpe or exposure would
    flatter it, so headline metrics use this window. The flatten that follows a halt fills at
    the next price, so its realised loss first shows in the first snapshot strictly after both
    the halt and the last kept trade's exit; equity is cut right after that snapshot, so return
    and drawdown agree with the trades' net P&L and the flat stretch after it is excluded.
    """
    halt = first_permanent_halt(events)
    if halt is None:
        return list(trades), equity, None
    kept = [t for t in trades if t.entry_ts <= halt.ts]
    after = max([halt.ts, *(t.exit_ts for t in kept)])
    later = equity.index[equity.index > after]
    cut = later[0] if len(later) else equity.index[-1] if len(equity) else after
    return kept, equity[equity.index <= cut], halt


def median_or_nan(values: Sequence[float]) -> float:
    """Median ignoring NaN; NaN when nothing is left."""
    kept = [v for v in values if not math.isnan(v)]
    return statistics.median(kept) if kept else math.nan
